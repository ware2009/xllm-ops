#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Independent FP32 reductions can land on adjacent BF16 values.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    return tolerance, tolerance


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
def test_rms_norm_cast_va(dtype: torch.dtype, num_tokens: int):
    torch.manual_seed(7)
    hidden_size = 7168
    epsilon = 1e-6
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="npu")
    gamma = torch.randn(hidden_size, dtype=dtype, device="npu")

    expected, _ = torch_npu.npu_rms_norm(x, gamma, epsilon)
    actual, actual_fp32 = custom_ops.rms_norm_cast_va_npu(x, gamma, epsilon)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # Routing must consume the widened, already-rounded RMSNorm result.
    torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_cast_va_npu_graph(dtype: torch.dtype):
    torch.manual_seed(11)
    x = torch.randn(16, 7168, dtype=dtype, device="npu")
    gamma = torch.randn(7168, dtype=dtype, device="npu")
    expected, _ = torch_npu.npu_rms_norm(x, gamma, 1e-6)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        actual, actual_fp32 = custom_ops.rms_norm_cast_va_npu(
            x,
            gamma,
            1e-6,
        )
    graph.replay()

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)