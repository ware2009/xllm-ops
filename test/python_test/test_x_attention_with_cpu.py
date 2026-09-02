import math
import random
import numpy as np
import pytest
import torch
import os


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

torch.manual_seed(1)
random.seed(1)

device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
torch_npu.npu.set_device(device_id)

def compare_tnd_outputs(cpu_out, npu_out, atol, rtol, max_bad_tokens=10):
    """Compare [T, N, D] outputs and report mismatches by token and head."""
    if cpu_out.shape != npu_out.shape:
        print(
            "TND output shape mismatch: "
            f"CPU={tuple(cpu_out.shape)}, NPU={tuple(npu_out.shape)}"
        )
        return False
    if cpu_out.ndim != 3:
        raise ValueError(
            f"expected TND outputs with 3 dimensions, got {tuple(cpu_out.shape)}"
        )

    close = torch.isclose(
        cpu_out, npu_out, atol=atol, rtol=rtol, equal_nan=True
    )
    mismatch = ~close
    bad_token_mask = mismatch.any(dim=(1, 2))
    bad_token_indices = bad_token_mask.nonzero(as_tuple=False).flatten()
    bad_element_count = int(mismatch.sum().item())

    print(
        "TND precision summary: "
        f"bad_tokens={bad_token_indices.numel()}/{cpu_out.shape[0]}, "
        f"bad_elements={bad_element_count}/{cpu_out.numel()}, "
        f"atol={atol}, rtol={rtol}"
    )
    if bad_token_indices.numel() == 0:
        return True

    abs_diff = torch.nan_to_num(
        (npu_out - cpu_out).abs(), nan=float("inf"), posinf=float("inf")
    )
    rel_diff = torch.nan_to_num(
        abs_diff / cpu_out.abs().clamp_min(torch.finfo(cpu_out.dtype).eps),
        nan=float("inf"),
        posinf=float("inf"),
    )

    print(f"First {min(max_bad_tokens, bad_token_indices.numel())} bad tokens:")
    for token_idx_tensor in bad_token_indices[:max_bad_tokens]:
        token_idx = int(token_idx_tensor.item())
        bad_head_indices = mismatch[token_idx].any(dim=1).nonzero(
            as_tuple=False
        ).flatten()
        print(
            f"  token={token_idx}, "
            f"bad_heads={bad_head_indices.numel()}/{cpu_out.shape[1]}"
        )
        for head_idx_tensor in bad_head_indices:
            head_idx = int(head_idx_tensor.item())
            head_mismatch = mismatch[token_idx, head_idx]
            head_abs_diff = abs_diff[token_idx, head_idx].masked_fill(
                ~head_mismatch, -1.0
            )
            worst_dim = int(head_abs_diff.argmax().item())
            print(
                f"    head={head_idx}, "
                f"bad_dims={int(head_mismatch.sum().item())}/{cpu_out.shape[2]}, "
                f"max_abs_diff={abs_diff[token_idx, head_idx, worst_dim].item():.6g}, "
                f"max_rel_diff={rel_diff[token_idx, head_idx][head_mismatch].max().item():.6g}, "
                f"worst_dim={worst_dim}, "
                f"cpu={cpu_out[token_idx, head_idx, worst_dim].item():.6g}, "
                f"npu={npu_out[token_idx, head_idx, worst_dim].item():.6g}"
            )
    return False

def get_max_block_num(shared_kv_lens, unshared_kv_len, block_size):
    max_block_num = 0
    for shared_kv_len in shared_kv_lens:
        kv_len = shared_kv_len + unshared_kv_len
        block_num = (kv_len + block_size - 1) // block_size
        max_block_num = max(max_block_num, block_num)
    return max_block_num


def _expand_gqa_kv(query, key, value):
    query = query.float()
    key = key.float()
    value = value.float()

    q_head_num = query.shape[1]
    kv_head_num = key.shape[1]
    if q_head_num % kv_head_num != 0:
        raise ValueError("q_head_num must be divisible by kv_head_num")

    group_size = q_head_num // kv_head_num
    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)
    return query, key, value


def online_softmax_update(
    running_numerator,
    running_max,
    running_sum,
    block_scores,
    block_value,
    input_dtype,
):
    """Merge one KV block using the NPU online-softmax update order."""
    block_max = block_scores.amax(dim=-1, keepdim=True)
    new_max = torch.maximum(running_max, block_max)
    running_scale = torch.exp(running_max - new_max)

    # The NPU updates the global max before ExpSub and casts the resulting
    # probabilities to the input dtype before PV.
    block_exp = torch.exp(block_scores - new_max)
    block_sum = block_exp.sum(dim=-1, keepdim=True)
    block_exp = block_exp.to(input_dtype).float()
    block_numerator = torch.matmul(
        block_exp.permute(1, 0, 2),
        block_value.float().permute(1, 0, 2),
    ).permute(1, 0, 2)

    new_sum = running_sum * running_scale + block_sum
    new_numerator = running_numerator * running_scale + block_numerator
    return new_numerator, new_max, new_sum


def shared_attention_fa_cpu(query, key, value, scale, kv_block_size=128):
    """Shared FA with KV-blocked online softmax."""
    input_dtype = query.dtype
    query, key, value = _expand_gqa_kv(query, key, value)
    if key.shape[0] == 0:
        raise ValueError("shared KV length must be positive")

    stats_shape = (query.shape[0], query.shape[1], 1)
    running_max = torch.full(stats_shape, -torch.inf, dtype=torch.float32)
    running_sum = torch.zeros(stats_shape, dtype=torch.float32)
    running_numerator = torch.zeros_like(query, dtype=torch.float32)

    for kv_start in range(0, key.shape[0], kv_block_size):
        kv_end = min(kv_start + kv_block_size, key.shape[0])
        key_block = key[kv_start:kv_end]
        value_block = value[kv_start:kv_end]

        scores = torch.matmul(
            query.permute(1, 0, 2),
            key_block.permute(1, 2, 0),
        ).permute(1, 0, 2) * scale
        running_numerator, running_max, running_sum = online_softmax_update(
            running_numerator,
            running_max,
            running_sum,
            scores,
            value_block,
            input_dtype,
        )

    return running_numerator, running_max, running_sum


def unshared_self_attention_cpu(query, key, value, scale):
    """Unshared self-attention for one beam."""
    input_dtype = query.dtype
    query, key, value = _expand_gqa_kv(query, key, value)
    if key.shape[0] == 0:
        raise ValueError("unshared KV length must be positive")

    # [H, Sq, D] @ [H, D, Skv] -> [Sq, H, Skv]
    scores = torch.matmul(
        query.permute(1, 0, 2),
        key.permute(1, 2, 0),
    ).permute(1, 0, 2) * scale
    row_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - row_max)
    exp_sum = exp_scores.sum(dim=-1, keepdim=True)
    exp_scores = exp_scores.to(input_dtype).float()
    numerator = torch.matmul(
        exp_scores.permute(1, 0, 2),
        value.float().permute(1, 0, 2),
    ).permute(1, 0, 2)
    return numerator, row_max, exp_sum


def x_attention_cpu_golden(
    query,
    shared_key_block,
    shared_value_block,
    unshared_key_block,
    unshared_value_block,
    shared_kv_lens,
    decode_step,
    unshared_block_tables,
    scale_value=None,
):
    """CPU golden for continuous shared KV and paged unshared KV.

    Shared attention treats all beam queries belonging to one request as the Q
    sequence of an FA invocation. Unshared attention is evaluated independently
    for every beam, as self-attention. The two softmax partitions are then
    merged with their max/sum statistics.
    """
    if query.device.type != "cpu":
        query = query.cpu()
    shared_key_block = shared_key_block.cpu()
    shared_value_block = shared_value_block.cpu()
    unshared_key_block = unshared_key_block.cpu()
    unshared_value_block = unshared_value_block.cpu()
    shared_kv_lens = shared_kv_lens.cpu()
    decode_step = decode_step.cpu()
    unshared_block_tables = unshared_block_tables.cpu()

    request_num = shared_kv_lens.numel()
    if request_num == 0 or query.shape[0] % request_num != 0:
        raise ValueError("query token count must be divisible by request count")
    if int(shared_kv_lens.sum().item()) > shared_key_block.shape[0]:
        raise ValueError("sum(shared_kv_lens) exceeds shared KV token count")
    beam_size = query.shape[0] // request_num
    unshared_kv_len = int(decode_step.reshape(-1)[0].item())
    if not 0 < unshared_kv_len <= unshared_key_block.shape[-2]:
        raise ValueError("decode_step is outside the unshared KV cache")

    scale = scale_value
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(query.shape[-1])

    output = torch.empty_like(query, device="cpu")
    shared_offset = 0
    for request_idx, shared_kv_len_tensor in enumerate(shared_kv_lens):
        shared_kv_len = int(shared_kv_len_tensor.item())
        query_start = request_idx * beam_size
        query_end = query_start + beam_size
        request_query = query[query_start:query_end]

        # Shared path: one FA over [beam_size, q_heads, head_dim].
        shared_key = shared_key_block[shared_offset:shared_offset + shared_kv_len]
        shared_value = shared_value_block[shared_offset:shared_offset + shared_kv_len]
        shared_num, shared_max, shared_sum = shared_attention_fa_cpu(
            request_query, shared_key, shared_value, scale
        )

        # dump shared res
        # for i in range(0, 8):
        #     print(f"token {i} shared_golden")
        #     print(shared_num[i, :, :8])
        #     print(f"token {i} shared_max_golden")
        #     print(shared_max[i])
        #     print(f"token {i} shared_sum_golden")
        #     print(shared_sum[i])

        cache_idx = int(unshared_block_tables.reshape(request_num, -1)[request_idx, 0].item())
        for beam_idx in range(beam_size):
            # Unshared path: each beam owns an independent self-attention cache.
            unshared_key = unshared_key_block[
                cache_idx, beam_idx, :, :unshared_kv_len, :
            ].permute(1, 0, 2)
            unshared_value = unshared_value_block[
                cache_idx, beam_idx, :, :unshared_kv_len, :
            ].permute(1, 0, 2)
            unshared_num, unshared_max, unshared_sum = unshared_self_attention_cpu(
                request_query[beam_idx:beam_idx + 1],
                unshared_key,
                unshared_value,
                scale,
            )

            # if 63 < beam_idx < 80:
            #     print(f"token {beam_idx} unshared_golden")
            #     print(unshared_num[0, :, :8])
            #     print(f"token {beam_idx} unshared_max_golden")
            #     print(unshared_max)
            #     print(f"token {beam_idx} unshared_sum_golden")
            #     print(unshared_sum)

            # Merge two independently reduced softmax partitions stably.
            global_max = torch.maximum(
                shared_max[beam_idx:beam_idx + 1], unshared_max
            )
            shared_weight = torch.exp(
                shared_max[beam_idx:beam_idx + 1] - global_max
            )
            unshared_weight = torch.exp(unshared_max - global_max)
            denominator = (
                shared_sum[beam_idx:beam_idx + 1] * shared_weight
                + unshared_sum * unshared_weight
            )
            combined = (
                shared_num[beam_idx:beam_idx + 1] * shared_weight
                + unshared_num * unshared_weight
            )[0] / denominator[0]
            output[query_start + beam_idx] = combined.to(query.dtype)

        shared_offset += shared_kv_len

    return output


# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# @pytest.mark.parametrize("q_head_num,kv_head_num", [(16, 8), (32, 8)])
# @pytest.mark.parametrize("head_dim", [128])
# @pytest.mark.parametrize("batch", [1, 6, 8])
# @pytest.mark.parametrize("beam_size", [32, 64, 128, 256, 512])
# @pytest.mark.parametrize("max_decode_step", [2, 4, 8, 12, 16, 20, 50, 100])
# @pytest.mark.parametrize("prompt_length", [128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("q_head_num, kv_head_num", [(16, 4), (16, 8), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch", [1, 6, 8])
@pytest.mark.parametrize("beam_size", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("max_decode_step", [3, 8, 12])
@pytest.mark.parametrize("prompt_length", [60, 256, 300, 500, 1000])

# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("q_head_num, kv_head_num", [(32, 8)])
# @pytest.mark.parametrize("head_dim", [128])
# @pytest.mark.parametrize("batch", [6])
# @pytest.mark.parametrize("beam_size", [256])
# @pytest.mark.parametrize("max_decode_step", [8])
# @pytest.mark.parametrize("prompt_length", [500])
def test_x_attention_with_cpu(
    dtype,
    q_head_num,
    kv_head_num,
    head_dim,
    batch,
    beam_size,
    max_decode_step,
    prompt_length
):
    shared_block_tokens = batch * prompt_length
    unshared_block_num = batch + 209
    unshared_kv_len = random.randint(1, max_decode_step)
    total_tokens = batch * beam_size

    query = torch.empty(total_tokens, q_head_num, head_dim, dtype=dtype).uniform_(-1, 1)
    shared_key_block = torch.empty(
        shared_block_tokens, kv_head_num, head_dim, dtype=dtype
    ).uniform_(-1, 1)
    shared_value_block = torch.empty_like(shared_key_block).uniform_(-1, 1)
    shared_kv_lens = torch.full((batch,), prompt_length, dtype=torch.int32)

    unshared_block_tables = torch.randperm(unshared_block_num)[:batch].to(torch.int32)
    unshared_key_block = torch.zeros(
        unshared_block_num,
        beam_size,
        kv_head_num,
        max_decode_step,
        head_dim,
        dtype=dtype,
    )
    unshared_value_block = torch.zeros_like(unshared_key_block)
    for block_idx in unshared_block_tables.tolist():
        unshared_key_block[block_idx, :, :, :unshared_kv_len, :].uniform_(-500, 500)
        unshared_value_block[block_idx, :, :, :unshared_kv_len, :].uniform_(-3, 3)

    decode_step = torch.tensor([unshared_kv_len], dtype=torch.int32)
    cpu_out = x_attention_cpu_golden(
        query,
        shared_key_block,
        shared_value_block,
        unshared_key_block,
        unshared_value_block,
        shared_kv_lens,
        decode_step,
        unshared_block_tables,
    )

    # print(golden.cpu().shape)
    # print(golden.cpu())
    npu_out = custom_ops.x_attention_npu(
        query.npu(),
        shared_key_block.npu(),
        shared_value_block.npu(),
        unshared_key_block.npu(),
        unshared_value_block.npu(),
        shared_kv_lens.npu(),
        decode_step.npu(),
        None,
        unshared_block_tables.npu(),
    )
    torch.npu.synchronize()

    # atol, rtol = ((0.01, 0.01) if dtype == torch.bfloat16 else (0.001, 0.001))
    # assert torch.allclose(actual.float().cpu(), golden, atol=atol, rtol=rtol, equal_nan=True)

    if dtype == torch.bfloat16:
        atol = 0.01
        rtol = 0.01
    else:
        atol = 0.001
        rtol = 0.001
    
    # atol = 0.001
    # rtol = 0.001

    npu_out = npu_out.cpu().float()
    cpu_out = cpu_out.float()

    # flag = torch.allclose(cpu_out, npu_out, atol=atol, rtol=rtol, equal_nan=True)
    flag = compare_tnd_outputs(cpu_out, npu_out, atol=atol, rtol=rtol)
    assert flag
