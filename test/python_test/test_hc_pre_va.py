"""hc_pre_va upstream accuracy cases, determinism checks, and tiling regressions.

Source: vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py
at commit 8fde6d4e335fe300e2b949aab0e31659cce0811d.
"""

import ctypes
import gc
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

DETERMINISTIC_ENV = {
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
}


HC_MULT = 4
DSV4_FLASH_HIDDEN_SIZE = 4096
HIDDEN_SIZE = DSV4_FLASH_HIDDEN_SIZE
EXTENDED_HIDDEN_SIZE = 7168
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995


def _check_deterministic_env() -> None:
    for name, expected in DETERMINISTIC_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"Set {name}={expected} before starting Python")


def _bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    )


def _repeatability_cases() -> list[tuple[tuple[int, ...], bool, str]]:
    cases = [
        ((tokens, 4, 4096), False, "normal") for tokens in (1, 2, 3, 16, 17, 31, 32, 33, 127, 128, 129, 512, 1024, 2048)
    ]
    cases += [((tokens, 4, 7168), False, "normal") for tokens in (1, 17, 129)]
    cases += [
        ((2, 3, 4, 4096), False, "normal"),
        ((2, 65, 4, 4096), False, "normal"),
        ((1, 4, 4096), True, "normal"),
        ((33, 4, 4096), True, "normal"),
        ((2, 3, 4, 4096), True, "normal"),
        ((17, 4, 7168), True, "normal"),
        ((2, 4, 4096), False, "positive"),
        ((2, 4, 4096), False, "zeros"),
        ((2, 4, 4096), False, "constant"),
    ]
    return cases


@pytest.mark.parametrize("shape,external_pre_mix,distribution", _repeatability_cases())
@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_repeatability(shape: tuple[int, ...], external_pre_mix: bool, distribution: str) -> None:
    hidden, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    hidden.normal_()
    hc_fn.normal_().div_((HC_MULT * shape[-1]) ** 0.5)
    if distribution == "positive":
        hidden.uniform_(0, 2)
    elif distribution == "zeros":
        hidden.zero_()
    elif distribution == "constant":
        hidden.fill_(1)
    pre_mix = torch.rand(shape[:-1], device="npu") if external_pre_mix else None
    arguments = tuple(tensor.npu() for tensor in (hidden, hc_fn, hc_scale, hc_base))
    batch_shape = shape[:-2]
    mix_shape = (*batch_shape, HC_MULT)
    expected_shapes = ((*batch_shape, shape[-1]), mix_shape, (*mix_shape, HC_MULT), mix_shape)
    baseline = None
    for _ in range(3):
        outputs = custom_ops.hc_pre_va_npu(*arguments, pre_mix=pre_mix)
        torch.npu.synchronize()
        actual = tuple(output.cpu() for output in outputs)
        for output, expected_shape, expected_dtype in zip(
            actual, expected_shapes, (torch.bfloat16,) + (torch.float32,) * 3
        ):
            assert output.shape == expected_shape and output.dtype == expected_dtype
            assert torch.isfinite(output).all()
        if baseline is not None:
            assert all(_bitwise_equal(output, reference) for output, reference in zip(actual, baseline))
        baseline = actual


@pytest.mark.usefixtures("_prepare_custom_op")
def test_hc_pre_va_invalid_inputs() -> None:
    valid = {
        "hidden": torch.zeros((2, 4, 4096), dtype=torch.bfloat16, device="npu"),
        "hc_fn": torch.zeros((24, 16384), dtype=torch.float32, device="npu"),
        "hc_scale": torch.ones(3, dtype=torch.float32, device="npu"),
        "hc_base": torch.zeros(24, dtype=torch.float32, device="npu"),
    }
    invalid_inputs = (
        {name: tensor.cpu() for name, tensor in valid.items()},
        {"hc_mult": 3},
        {"hc_sinkhorn_iters": 0},
        {"norm_eps": 0.0},
        {"hc_eps": float("nan")},
        {"hidden": valid["hidden"].float()},
        {"hidden": valid["hidden"][0, 0]},
        {"hidden": valid["hidden"][:0]},
        {"hc_fn": valid["hc_fn"].cpu()},
        {"hc_fn": valid["hc_fn"][:23]},
        {"hc_scale": valid["hc_scale"][:2]},
        {"hc_base": valid["hc_base"].to(torch.bfloat16)},
        {"pre_mix": torch.zeros((2, 3), dtype=torch.float32, device="npu")},
        {"pre_mix": torch.zeros((2, 4), dtype=torch.bfloat16, device="npu")},
        {"pre_mix": torch.zeros((2, 4), dtype=torch.float32)},
    )
    for invalid in invalid_inputs:
        with pytest.raises(RuntimeError):
            custom_ops.hc_pre_va_npu(**{**valid, **invalid})


@pytest.fixture(scope="module")
def _prepare_custom_op() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("An Ascend NPU is required")
    if torch_npu._C._npu_get_soc_version() != 253:
        pytest.skip("HcPreVa tests require an Ascend A3")
    _check_deterministic_env()
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.config.allow_internal_format = True


def _make_hc_pre_inputs(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1024)
    hidden_size = shape[-1]
    fan_in = HC_MULT * hidden_size
    hidden = (torch.rand(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = (
        torch.rand(
            MIX_HC,
            fan_in,
            dtype=torch.float32,
        )
        / fan_in
    )
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return hidden, hc_fn, hc_scale, hc_base


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


def _hc_pre_cpu(
    hidden: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_float = hidden.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)

    # HcPre performs this matmul in HF32 round-toward-zero mode.
    mixes = functional.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
    pre, post, comb_frag = mixes.split(
        [HC_MULT, HC_MULT, HC_MULT * HC_MULT],
        dim=-1,
    )
    comb_frag = comb_frag.unflatten(-1, (HC_MULT, HC_MULT))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb_frag = comb_frag * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)

    comb_frag = comb_frag.softmax(-1) + HC_EPS
    comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + HC_EPS)
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)

    mix_for_y = pre_mix.float() if pre_mix is not None else pre
    output = (mix_for_y.unsqueeze(-1) * x_float).sum(dim=-2).to(hidden.dtype)
    return output, post, comb_frag, pre


def _assert_close_with_pass_rate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diff_threshold: float,
    required_pass_rate: float,
) -> None:
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    assert pass_rate >= required_pass_rate, (
        f"pass rate {pass_rate:.2%} is below {required_pass_rate:.2%}; max absolute difference: {max_abs_diff}"
    )


def _compare_hc_pre_with_cpu(shape: tuple[int, ...]) -> None:
    hidden, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    expected_y, expected_post, expected_comb_frag, _ = _hc_pre_cpu(
        hidden,
        hc_fn,
        hc_scale,
        hc_base,
    )
    output, post, comb_frag, _ = custom_ops.hc_pre_va_npu(
        hidden.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        pre_mix=None,
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=HC_SINKHORN_ITERS,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
    )

    batch_shape = shape[:-2]
    assert output.shape == (*batch_shape, shape[-1])
    assert post.shape == (*batch_shape, HC_MULT)
    assert comb_frag.shape == (*batch_shape, HC_MULT, HC_MULT)
    assert output.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb_frag.dtype == torch.float32
    _assert_close_with_pass_rate(
        output,
        expected_y,
        diff_threshold=Y_DIFF_THRESHOLD,
        required_pass_rate=Y_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        post,
        expected_post,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        comb_frag,
        expected_comb_frag,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )


@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_bf16_3d_input() -> None:
    _compare_hc_pre_with_cpu((2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_bf16_4d_input() -> None:
    _compare_hc_pre_with_cpu((1, 2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_bf16_dsv4_flash_hidden_size() -> None:
    _compare_hc_pre_with_cpu((4, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_bf16_extended_hidden_size() -> None:
    _compare_hc_pre_with_cpu((2, HC_MULT, EXTENDED_HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_uses_external_pre_mix() -> None:
    shape = (2, HC_MULT, HIDDEN_SIZE)
    hidden, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    pre_mix = torch.rand(shape[:-1], dtype=torch.float32)
    expected_y, expected_post, expected_comb_frag, expected_pre = _hc_pre_cpu(hidden, hc_fn, hc_scale, hc_base, pre_mix)
    output, post, comb_frag, pre = custom_ops.hc_pre_va_npu(
        hidden.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        pre_mix.npu(),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=HC_SINKHORN_ITERS,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
    )
    assert output.shape == (shape[0], shape[-1])
    assert post.shape == (shape[0], HC_MULT)
    assert comb_frag.shape == (shape[0], HC_MULT, HC_MULT)
    assert pre.shape == (shape[0], HC_MULT)
    assert output.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb_frag.dtype == torch.float32
    assert pre.dtype == torch.float32
    for actual, expected, diff_threshold, required_pass_rate in (
        (output, expected_y, Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
        (post, expected_post, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        (comb_frag, expected_comb_frag, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        (pre, expected_pre, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
    ):
        _assert_close_with_pass_rate(
            actual,
            expected,
            diff_threshold=diff_threshold,
            required_pass_rate=required_pass_rate,
        )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "shape",
    (
        (129, 4, 128),
        (129, 4, 127),
        (129, 4, 129),
        (513, 4, 64),
        (513, 4, 128),
        (4097, 4, 128),
        (1025, 4, 256),
        (1025, 4, 512),
        (513, 4, 1024),
        (257, 4, 1536),
        (129, 4, 2048),
        (129, 4, 3072),
        (129, 4, 4100),
        (2, 65, 4, 128),
    ),
)
@pytest.mark.parametrize("external_pre_mix", (False, True))
@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_multirow_and_tail_tiles(shape: tuple[int, ...], external_pre_mix: bool) -> None:
    hidden, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    pre_mix = torch.rand(shape[:-1]) if external_pre_mix else None
    expected = _hc_pre_cpu(hidden, hc_fn, hc_scale, hc_base, pre_mix)
    arguments = tuple(tensor.npu() for tensor in (hidden, hc_fn, hc_scale, hc_base))
    device_pre_mix = pre_mix.npu() if pre_mix is not None else None
    baseline = None
    for _ in range(3):
        outputs = custom_ops.hc_pre_va_npu(*arguments, pre_mix=device_pre_mix)
        torch.npu.synchronize()
        actual = tuple(output.cpu() for output in outputs)
        for index, (output, reference) in enumerate(zip(actual, expected)):
            assert output.shape == reference.shape and output.dtype == reference.dtype
            assert torch.isfinite(output).all()
            _assert_close_with_pass_rate(
                output,
                reference,
                diff_threshold=Y_DIFF_THRESHOLD if index == 0 else AUX_DIFF_THRESHOLD,
                required_pass_rate=Y_REQUIRED_PASS_RATE if index == 0 else AUX_REQUIRED_PASS_RATE,
            )
        if baseline is not None:
            assert all(_bitwise_equal(output, reference) for output, reference in zip(actual, baseline))
        baseline = actual
    if not external_pre_mix:
        explicit = custom_ops.hc_pre_va_npu(*arguments, pre_mix=outputs[3])
        torch.npu.synchronize()
        assert all(_bitwise_equal(output.cpu(), reference) for output, reference in zip(explicit, baseline))


@pytest.mark.parametrize("shape", ((129, 4, 128), (1024, 4, 4096)))
@pytest.mark.usefixtures("_prepare_custom_op")
@torch.inference_mode()
def test_hc_pre_va_pre_output_buffer_reuse(shape: tuple[int, ...]) -> None:
    hidden, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    hidden.normal_()
    hc_fn.normal_().div_((HC_MULT * shape[-1]) ** 0.5)
    arguments = tuple(tensor.npu() for tensor in (hidden, hc_fn, hc_scale, hc_base))
    baseline = None
    for _ in range(20):
        outputs = custom_ops.hc_pre_va_npu(*arguments, hc_sinkhorn_iters=1)
        torch.npu.synchronize()
        actual = tuple(output.cpu() for output in outputs)
        assert all(torch.isfinite(output).all() for output in actual)
        if baseline is not None:
            assert all(_bitwise_equal(output, reference) for output, reference in zip(actual, baseline))
        baseline = actual


def _workspace_status(
    shape: tuple[int, ...],
    hc_mult: int = 4,
    iterations: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> int:
    vendor_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    libraries = [Path(path) / "op_api/lib/libcust_opapi.so" for path in vendor_paths if path]
    library = next((path for path in libraries if path.is_file()), None)
    if library is None:
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH must select the isolated test operator package")
    metadata = ctypes.CDLL("libopapi.so")
    operation = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    dimensions_pointer = ctypes.POINTER(ctypes.c_int64)
    metadata.aclCreateTensor.argtypes = [
        dimensions_pointer,
        ctypes.c_uint64,
        ctypes.c_int,
        dimensions_pointer,
        ctypes.c_int64,
        ctypes.c_int,
        dimensions_pointer,
        ctypes.c_uint64,
        pointer,
    ]
    metadata.aclCreateTensor.restype = pointer
    metadata.aclDestroyTensor.argtypes = [pointer]
    metadata.aclDestroyTensor.restype = ctypes.c_int
    workspace_size = operation.aclnnHcPreVaGetWorkspaceSize
    workspace_size.argtypes = (
        [pointer] * 5
        + [ctypes.c_int64] * 2
        + [ctypes.c_double] * 2
        + [pointer] * 4
        + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(pointer)]
    )
    workspace_size.restype = ctypes.c_int
    tensors = (
        torch.empty(shape, dtype=torch.bfloat16, device="npu"),
        torch.empty((24, shape[-2] * shape[-1]), dtype=torch.float32, device="npu"),
        torch.empty(3, dtype=torch.float32, device="npu"),
        torch.empty(24, dtype=torch.float32, device="npu"),
        torch.empty((*shape[:-2], shape[-1]), dtype=torch.bfloat16, device="npu"),
        torch.empty((*shape[:-2], HC_MULT), dtype=torch.float32, device="npu"),
        torch.empty((*shape[:-2], HC_MULT, HC_MULT), dtype=torch.float32, device="npu"),
        torch.empty((*shape[:-2], HC_MULT), dtype=torch.float32, device="npu"),
    )
    descriptors = []
    try:
        for tensor in tensors:
            dimensions = (ctypes.c_int64 * tensor.dim())(*tensor.shape)
            strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
            descriptor = metadata.aclCreateTensor(
                dimensions,
                tensor.dim(),
                27 if tensor.dtype == torch.bfloat16 else 0,
                strides,
                tensor.storage_offset(),
                2,
                dimensions,
                tensor.dim(),
                tensor.data_ptr(),
            )
            assert descriptor is not None
            descriptors.append(descriptor)
        size = ctypes.c_uint64()
        executor = pointer()
        return workspace_size(
            *descriptors[:4],
            None,
            hc_mult,
            iterations,
            hc_eps,
            norm_eps,
            *descriptors[4:],
            ctypes.byref(size),
            ctypes.byref(executor),
        )
    finally:
        for descriptor in descriptors:
            metadata.aclDestroyTensor(descriptor)


@pytest.mark.parametrize(
    "invalid",
    (
        {"shape": (2, 3, 128)},
        {"shape": (2, 4, 128), "hc_mult": 3},
        {"shape": (2, 4, 128), "iterations": 0},
        {"shape": (2, 4, 128), "iterations": -1},
        {"shape": (2, 4, 128), "norm_eps": 0.0},
        {"shape": (2, 4, 128), "hc_eps": float("nan")},
    ),
)
@pytest.mark.usefixtures("_prepare_custom_op")
def test_hc_pre_va_direct_aclnn_rejects_invalid_attributes(invalid: dict[str, object]) -> None:
    assert _workspace_status(**invalid) != 0


@pytest.mark.parametrize("shape", ((2, 4, 128), (1, 2, 4, 128)))
@pytest.mark.usefixtures("_prepare_custom_op")
def test_hc_pre_va_direct_aclnn_accepts_valid_attributes(shape: tuple[int, ...]) -> None:
    assert _workspace_status(shape) == 0
