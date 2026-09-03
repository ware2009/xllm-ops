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
"""Tests for the x_flash_attention_infer operator (paged-KV flash decoding).

x_flash_attention_infer is a paged-KV flash-decoding attention kernel with a
TND layout and (kernel-internal) causal masking. This test targets the
DECODE scenario (qSeqlen == 1 per batch), which is both the simplest and the
most common inference case:

  * query        : [numTokens, qHead, headDim]  (numTokens = batch * 1)  fp16/bf16
  * key_cache /   : [numBlocks, blockSize, kvHead, headDim]  (paged, ND/TND)
    value_cache
  * block_table  : [batch, maxBlocksPerBatch]   int32
  * actual_q_lens: int32 cumulative prefix-sum (TND);  qSeqlen==1 -> [1,2,...,batch]
  * actual_kv_lens: int32 per-batch kv length (PAGED, NOT prefix-summed)

Single-core extra_tiling layout is built by custom_ops.x_flash_attention_infer_npu.

Single-core constraints (see custom_ops._build_xfa_extra_tiling):
  * curKSBlockNum must be uniform across batches (holds while every
    actual kv len <= 512 == MAX_KV_STACK_LEN)
  * maxQSeqlen(=1) * qHead < 128 -> guarantees the FD kernel branch

For qSeqlen == 1 the causal triangle is empty (the single query sees every kv
position), so the golden is a plain paged full-attention.

The parametrization covers BOTH:
  * equal-length batches (actual == paged capacity, no padding), and
  * UNEQUAL per-batch actual kv lens within one shared paged capacity.
    Every batch owns a [maxBlocksPerBatch * blockSize] capacity; batches whose
    actual_kv_lens[b] < capacity have a padded tail in the cache that MUST be
    shielded by the causal mask. The mask boundary is derived per batch from
    actual_kv_lens[b] (the A3 path reads gActualKvseqlen.GetValue(BIdx)), so a
    kernel that reuses one batch's length for all batches (e.g. GetValue(0))
    fails these cases: short batches attend random padding, or long batches
    lose valid kv. Cases [256,128]/[128,256] catch a batch-0 leak in either
    direction; [256,192,64] also exercises partially- and fully-masked
    128-wide kv tiles (mid-block tail and a half-empty block).
"""

import math
import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def _paged_gather_kv(cache, block_table, actual_kv_lens, block_size):
    """Reassemble per-batch contiguous KV from a paged cache.

    cache          : [numBlocks, blockSize, kvHead, headDim]
    block_table    : [batch, maxBlocksPerBatch] int32
    actual_kv_lens : per-batch valid kv length (positions beyond it are padding)
    returns        : [batch, max(actual_kv_lens), kvHead, headDim]; positions
                     beyond a batch's actual length stay zero (never read).
    """
    batch = block_table.shape[0]
    kv_head = cache.shape[2]
    head_dim = cache.shape[3]
    max_len = max(int(l) for l in actual_kv_lens)
    out = torch.zeros((batch, max_len, kv_head, head_dim), dtype=cache.dtype)
    for b in range(batch):
        pos = 0
        blk = 0
        remain = int(actual_kv_lens[b])
        while remain > 0:
            block_id = int(block_table[b, blk].item())
            take = min(block_size, remain)
            out[b, pos:pos + take] = cache[block_id, :take]
            pos += take
            remain -= take
            blk += 1
    return out


def _xfa_decode_golden(query, key, value, actual_kv_lens, q_head, kv_head, scale):
    """Reference decode attention (qSeqlen == 1) in fp32, per-batch kv length.

    query : [batch, q_head, head_dim]      (one query token per batch)
    key   : [batch, max_kv, kv_head, head_dim] (zero-padded beyond each batch's length)
    value : [batch, max_kv, kv_head, head_dim]
    returns attn_out [batch, q_head, head_dim] fp32
    """
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    batch = q.shape[0]
    group = q_head // kv_head
    head_dim = q.shape[-1]
    out = torch.zeros((batch, q_head, head_dim), dtype=torch.float32)
    for b in range(batch):
        kv_len = int(actual_kv_lens[b])
        for h in range(q_head):
            kvh = h // group
            qh = q[b, h]                       # [head_dim]
            kh = k[b, :kv_len, kvh]            # [kv_len, head_dim]
            vh = v[b, :kv_len, kvh]            # [kv_len, head_dim]
            scores = (kh @ qh) * scale         # [kv_len]
            scores = scores - scores.max()
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = probs @ vh
    return out


def _build_causal_mask():
    """Kernel expects an int8 [2048, 2048] causal mask table (upper-tri masked)."""
    m = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1)
    return m


@pytest.mark.parametrize(
    "dtype, batch, q_head, kv_head, head_dim, block_size, kv_capacity, actual_kv_lens",
    [
        # equal-length regressions: actual == capacity, no padded tail
        (torch.float16, 1, 8, 8, 128, 128, 128, [128]),
        (torch.float16, 2, 8, 8, 128, 128, 128, [128, 128]),
        (torch.float16, 2, 16, 8, 128, 128, 256, [256, 256]),
        # unequal per-batch actual kv lens, one shared capacity (256 = 2 blocks):
        # batch 1 has a 128-slot padded tail that must be masked out
        (torch.float16, 2, 8, 8, 128, 128, 256, [256, 128]),
        # reversed: a short batch 0 must not leak its mask boundary to batch 1
        (torch.float16, 2, 8, 8, 128, 128, 256, [128, 256]),
        # full / mid-block tail (192) / half block (64): exercises partial- and
        # fully-masked 128-wide kv tiles within the same launch
        (torch.float16, 3, 8, 8, 128, 128, 256, [256, 192, 64]),
    ],
    ids=[
        "equal_b1_kv128",
        "equal_b2_kv128",
        "equal_b2_h16_kv256",
        "unequal_b2_kv256_128",
        "unequal_b2_kv128_256",
        "unequal_b3_kv256_192_64",
    ],
)
def test_x_flash_attention_infer(dtype, batch, q_head, kv_head, head_dim,
                                 block_size, kv_capacity, actual_kv_lens):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(head_dim)
    q_seqlen = 1
    num_tokens = batch * q_seqlen

    # every batch owns the same paged capacity; only actual_kv_lens[b] slots are valid
    blocks_per_batch = (kv_capacity + block_size - 1) // block_size
    assert blocks_per_batch * block_size >= max(actual_kv_lens)
    assert all(0 < l <= 512 for l in actual_kv_lens)  # keep curKSBlockNum uniform (A3 FD)
    num_blocks = batch * blocks_per_batch

    query = torch.randn(num_tokens, q_head, head_dim, dtype=dtype)
    key_cache = torch.randn(num_blocks, block_size, kv_head, head_dim, dtype=dtype)
    value_cache = torch.randn(num_blocks, block_size, kv_head, head_dim, dtype=dtype)

    block_table = torch.zeros(batch, blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for j in range(blocks_per_batch):
            block_table[b, j] = b * blocks_per_batch + j

    # actual_q_lens: cumulative prefix-sum of qSeqlen(=1) per batch.
    actual_q_lens = torch.arange(1, batch + 1, dtype=torch.int32)
    # actual_kv_lens: per-batch kv length (PAGED, not prefix-summed).
    actual_kv_lens_t = torch.tensor(actual_kv_lens, dtype=torch.int32)

    mask = _build_causal_mask()

    # ---- golden (fp32): attend only the first actual_kv_lens[b] kv positions ----
    gathered_k = _paged_gather_kv(key_cache, block_table, actual_kv_lens, block_size)
    gathered_v = _paged_gather_kv(value_cache, block_table, actual_kv_lens, block_size)
    query_bhd = query.view(batch, q_seqlen, q_head, head_dim)[:, 0]  # [batch, q_head, head_dim]
    golden = _xfa_decode_golden(query_bhd, gathered_k, gathered_v,
                                actual_kv_lens, q_head, kv_head, scale)

    # ---- device ----
    out = custom_ops.x_flash_attention_infer_npu(
        query.npu(), key_cache.npu(), value_cache.npu(), block_table.npu(),
        actual_q_lens.npu(), actual_kv_lens_t.npu(),
        q_head, kv_head, scale, batch, kv_capacity,
        mask=mask.npu(), layout="TND",
    )
    out = out.cpu().view(batch, q_head, head_dim).to(torch.float32)

    torch.testing.assert_close(out, golden, atol=6e-2, rtol=6e-2)