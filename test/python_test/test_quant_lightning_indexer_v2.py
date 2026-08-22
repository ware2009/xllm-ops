#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest test for quant_lightning_indexer_v2 (two-stage: metadata + main, PA paged indexer).

Two test cases:
  1. test_small: minimal shape for basic functionality debugging
  2. test_full:  user-specified production shape

User-specified production shape:
  query:  [1, 64, 128]            int8     (TND: T=1, N=64, D=128)
  key:    [block_num, 128, 1, 128] int8     (PA_BSND: block_size=128, K_N=1, D=128)
  weights:[1, 64]                  float16  (TND)
  q_scale:[1, 64]                  float16
  k_scale:[block_num, 128, 1]     float16
  block_table: [1, 1]             int32
  topk=512, mask_mode=3, cmp_ratio=4, quant_mode=2 (INT8)
  output: [1, 1, 512] int32 (TND)
"""
import math
import os

import numpy as np
import pytest
import torch
import torch_npu

from custom_ops import (
    quant_lightning_indexer_v2_metadata_npu,
    quant_lightning_indexer_v2_npu,
)

INVALID_IDX = -1


def _get_platform():
    soc = torch_npu._C._npu_get_soc_version()
    if soc in (200, 260):
        return "A5"
    return "910C"


PLATFORM = _get_platform()


def golden_quant_lightning_indexer_v2_tnd(
    query_raw, key_pa_raw,
    weights, q_scale, k_scale_pa,
    act_seq_q, act_seq_k,
    q_head_num, k_head_num, head_dim,
    block_table, block_size,
    topk, mask_mode, cmp_ratio):
    """
    Golden for quant_lightning_indexer_v2 with TND query + PA key.
    query_raw:   [T, qHead, hd]             float32 (dequantized)
    key_pa_raw:  [block_num, block_size, kHead, hd] float32 (dequantized)
    weights:     [T, qHead]                 float32
    q_scale:     [T, qHead]                 float32
    k_scale_pa:  [block_num, block_size, kHead] float32
    act_seq_q:   list[int] per-batch actual query length
    act_seq_k:   list[int] per-batch compressed key length (after cmp_ratio)
    Returns:     [T, kHead, topk]           int32
    """
    B = len(act_seq_q)
    T = sum(act_seq_q)
    out = np.full((T, k_head_num, topk), INVALID_IDX, dtype=np.int32)

    t_offset = 0
    for b in range(B):
        actual_q = act_seq_q[b]
        actual_k = act_seq_k[b]
        # Gather key tokens via block_table
        blocks_per_b = block_table.shape[1]
        k_tokens = []
        k_scale_tokens = []
        for blk in range(blocks_per_b):
            gblk = block_table[b, blk]
            for tok in range(block_size):
                k_tokens.append(key_pa_raw[gblk, tok, 0, :])
                k_scale_tokens.append(k_scale_pa[gblk, tok, 0])
        max_avail = len(k_tokens)
        valid_k = min(actual_k, max_avail)
        k_all = np.stack(k_tokens[:valid_k])   # [valid_k, hd]
        ks_all = np.array(k_scale_tokens[:valid_k])  # [valid_k]

        for s1 in range(actual_q):
            t_idx = t_offset + s1
            for kh in range(k_head_num):
                if valid_k <= 0:
                    continue

                q = query_raw[t_idx, :, :]   # [qHead, hd]
                k = k_all[:valid_k, :]       # [valid_k, hd]

                dots = q @ k.T               # [qHead, valid_k]

                w = weights[t_idx, :]       # [qHead]
                qs = q_scale[t_idx, :]      # [qHead]
                ks = ks_all[:valid_k]       # [valid_k]

                coeff = w * qs
                score = (coeff[:, None] * dots).sum(axis=0) * ks

                order = np.argsort(-score)
                take = min(topk, valid_k)
                out[t_idx, kh, :take] = order[:take].astype(np.int32)

        t_offset += actual_q

    return out


def _valid_set(row):
    return set(int(x) for x in row if int(x) != INVALID_IDX)


def _run_test(B, Q_SEQ, Q_HEAD, K_HEAD, HD, BLOCK_SIZE, BLOCK_NUM, TOPK,
              MASK_MODE, CMP_RATIO, QUANT_MODE):
    """Shared test body for both small and full cases."""
    np.random.seed(2026)
    torch.manual_seed(2026)

    # Data ranges from user spec
    query_f32 = np.random.uniform(-100, 100, (Q_SEQ, Q_HEAD, HD)).astype(np.float32)
    key_pa_f32 = np.random.uniform(-100, 100, (BLOCK_NUM, BLOCK_SIZE, K_HEAD, HD)).astype(np.float32)
    weights_f32 = np.random.uniform(-25, 25, (Q_SEQ, Q_HEAD)).astype(np.float32)
    q_scale_f32 = np.random.uniform(0, 255, (Q_SEQ, Q_HEAD)).astype(np.float32)
    k_scale_pa_f32 = np.random.uniform(0, 65504, (BLOCK_NUM, BLOCK_SIZE, K_HEAD)).astype(np.float32)

    # Quantize to int8
    query_i8 = np.clip(np.round(query_f32), -127, 127).astype(np.int8)
    key_pa_i8 = np.clip(np.round(key_pa_f32), -127, 127).astype(np.int8)
    query_deq = query_i8.astype(np.float32)
    key_pa_deq = key_pa_i8.astype(np.float32)

    # Cast weights/scales to fp16 (simulate precision loss)
    weights_fp16 = torch.from_numpy(weights_f32).to(torch.float16).float().numpy()
    q_scale_fp16 = torch.from_numpy(q_scale_f32).to(torch.float16).float().numpy()
    k_scale_pa_fp16 = torch.from_numpy(k_scale_pa_f32).to(torch.float16).float().numpy()

    # Actual sequence lengths
    act_seq_q = [Q_SEQ] * B
    act_seq_k = [BLOCK_SIZE] * B   # 128 compressed key tokens (1 full block)

    # Block table: [B, 1] → block 0
    block_table = np.zeros((B, 1), dtype=np.int32)

    # Golden (CPU)
    golden_idx = golden_quant_lightning_indexer_v2_tnd(
        query_deq, key_pa_deq,
        weights_fp16, q_scale_fp16, k_scale_pa_fp16,
        act_seq_q, act_seq_k,
        Q_HEAD, K_HEAD, HD,
        block_table, BLOCK_SIZE,
        TOPK, MASK_MODE, CMP_RATIO)

    # Prepare NPU tensors
    query_npu = torch.from_numpy(query_i8).npu()
    key_npu = torch.from_numpy(key_pa_i8).npu()
    weights_npu = torch.from_numpy(weights_fp16).to(torch.float16).npu()
    q_scale_npu = torch.from_numpy(q_scale_fp16).to(torch.float16).npu()
    k_scale_npu = torch.from_numpy(k_scale_pa_fp16).to(torch.float16).npu()
    aslq_npu = torch.tensor(act_seq_q, dtype=torch.int32).npu()
    aslk_npu = torch.tensor(act_seq_k, dtype=torch.int32).npu()
    bt_npu = torch.from_numpy(block_table).npu()

    # Stage 1: Generate metadata
    metadata = quant_lightning_indexer_v2_metadata_npu(
        aslq_npu, aslk_npu,
        num_heads_q=Q_HEAD, num_heads_k=K_HEAD, head_dim=HD,
        batch_size=B, max_seqlen_q=Q_SEQ, max_seqlen_k=act_seq_k[0],
        layout_q="TND", layout_k="PA_BSND",
        topk=TOPK, quant_mode=QUANT_MODE,
        mask_mode=MASK_MODE, cmp_ratio=CMP_RATIO,
    )

    # Stage 2: Run main op
    npu_out = quant_lightning_indexer_v2_npu(
        query_npu, key_npu, weights_npu, q_scale_npu, k_scale_npu,
        aslq_npu, aslk_npu, bt_npu, metadata,
        num_heads_q=Q_HEAD, num_heads_k=K_HEAD, head_dim=HD,
        topk=TOPK, quant_mode=QUANT_MODE,
        layout_q="TND", layout_k="PA_BSND",
        mask_mode=MASK_MODE, cmp_ratio=CMP_RATIO,
        return_value=0,
    )
    npu_result = npu_out.cpu().numpy()

    # Compare
    expected_shape = (Q_SEQ, K_HEAD, TOPK)
    assert npu_result.shape == expected_shape, (
        f"Shape mismatch: {npu_result.shape} vs {expected_shape}")

    for t in range(Q_SEQ):
        for kh in range(K_HEAD):
            got = _valid_set(npu_result[t, kh])
            exp = _valid_set(golden_idx[t, kh])
            overlap = len(got & exp)
            min_match = max(1, int(len(exp) * 0.5))
            assert overlap >= min_match, (
                f"t={t} kh={kh}: got {sorted(got)} exp {sorted(exp)}")


def test_small():
    """Small shape for basic functionality debugging."""
    _run_test(
        B=1, Q_SEQ=1, Q_HEAD=64, K_HEAD=1, HD=128,
        BLOCK_SIZE=128, BLOCK_NUM=1,
        TOPK=8, MASK_MODE=3, CMP_RATIO=1, QUANT_MODE=2,
    )


def test_full():
    """Full user-specified production shape."""
    _run_test(
        B=1, Q_SEQ=1, Q_HEAD=64, K_HEAD=1, HD=128,
        BLOCK_SIZE=128, BLOCK_NUM=1,
        TOPK=512, MASK_MODE=3, CMP_RATIO=4, QUANT_MODE=2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
