#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest test for quant_lightning_indexer_v2 (two-stage: metadata + main, PA paged indexer).
Golden: weighted dot-product score -> topk indices (set comparison).
NPU: metadata + main two-stage with paged key.

Platform adaptation:
  - A5 (ascend950): quant_mode=1 (fp8), weights/scale fp32; also quant_mode=2 (int8), weights/scale fp16
  - 910C (Ascend910C): quant_mode=2 (int8), weights/scale fp16
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


def golden_quant_lightning_indexer_v2(query_raw, key_bnsd_raw,
                                      weights, q_scale, k_scale_bns,
                                      act_seq_q, act_seq_k,
                                      q_head_num, k_head_num, head_dim,
                                      topk, mask_mode, cmp_ratio):
    """
    Golden for quant_lightning_indexer_v2 (simplified).
    query_raw: [B, qSeq, qHead, hd] float32 (dequantized)
    key_bnsd_raw: [B, kHead, kSeq, hd] float32 (dequantized)
    weights: [B, qSeq, qHead] float32
    q_scale: [B, qSeq, qHead] float32
    k_scale_bns: [B, kHead, kSeq] float32
    act_seq_q: list[int] per-batch actual query length
    act_seq_k: list[int] per-batch actual key length (before cmp_ratio)
    Returns: indices [B, qSeq, kHead, topk] int32
    """
    B = query_raw.shape[0]
    q_seq = query_raw.shape[1]
    k_seq = key_bnsd_raw.shape[2]

    out = np.full((B, q_seq, k_head_num, topk), INVALID_IDX, dtype=np.int32)

    for b in range(B):
        actual_q = act_seq_q[b]
        actual_k = int(math.floor(act_seq_k[b] / cmp_ratio))
        for s1 in range(actual_q):
            for kh in range(k_head_num):
                if mask_mode == 3:
                    valid_k = actual_k
                else:
                    valid_k = actual_k

                if valid_k <= 0:
                    continue

                q = query_raw[b, s1, :, :]
                k = key_bnsd_raw[b, kh, :valid_k, :]

                dots = q @ k.T  # [qHead, valid_k]

                w = weights[b, s1, :]
                qs = q_scale[b, s1, :]
                ks = k_scale_bns[b, kh, :valid_k]

                coeff = w * qs
                score = (coeff[:, None] * dots).sum(axis=0) * ks

                order = np.argsort(-score)
                take = min(topk, valid_k)
                out[b, s1, kh, :take] = order[:take].astype(np.int32)

    return out


def to_paged_key(key_bnsd, k_scale_bns, B, k_head_num, k_seq, head_dim, block_size):
    blocks_per_b = (k_seq + block_size - 1) // block_size
    block_num = B * blocks_per_b
    key_pa = np.zeros((block_num, block_size, k_head_num, head_dim), dtype=key_bnsd.dtype)
    kscale_pa = np.zeros((block_num, block_size, k_head_num), dtype=k_scale_bns.dtype)
    block_table = np.zeros((B, blocks_per_b), dtype=np.int32)

    for b in range(B):
        for blk in range(blocks_per_b):
            gblk = b * blocks_per_b + blk
            block_table[b, blk] = gblk
            for tok in range(block_size):
                t = blk * block_size + tok
                if t < k_seq:
                    for kh in range(k_head_num):
                        key_pa[gblk, tok, kh, :] = key_bnsd[b, kh, t, :]
                        kscale_pa[gblk, tok, kh] = k_scale_bns[b, kh, t]
    return key_pa, kscale_pa, block_table


def _valid_set(row):
    return set(int(x) for x in row if int(x) != INVALID_IDX)


CASES = [
    # (B, q_seq, k_seq, q_head, k_head, head_dim, block_size, topk, mask_mode, cmp_ratio)
    (1, 4, 128, 64, 1, 128, 128, 8, 3, 1),
]


@pytest.mark.parametrize("B,Q_SEQ,K_SEQ,Q_HEAD,K_HEAD,HD,BLOCK_SIZE,TOPK,MASK_MODE,CMP_RATIO", CASES)
def test_quant_lightning_indexer_v2(B, Q_SEQ, K_SEQ, Q_HEAD, K_HEAD, HD, BLOCK_SIZE, TOPK, MASK_MODE, CMP_RATIO):
    np.random.seed(2026)
    torch.manual_seed(2026)

    # Platform-dependent dtype and quant_mode selection
    if PLATFORM == "A5":
        # A5: test int8 path (quant_mode=2)
        qk_dtype = torch.int8
        scale_dtype = torch.float16
        quant_mode = 2
    else:
        # 910C: int8 only
        qk_dtype = torch.int8
        scale_dtype = torch.float16
        quant_mode = 2

    # Generate data via float -> quantize -> dequantize for golden
    query_f32 = np.random.uniform(-100, 100, (B, Q_SEQ, Q_HEAD, HD)).astype(np.float32)
    key_bnsd_f32 = np.random.uniform(-100, 100, (B, K_HEAD, K_SEQ, HD)).astype(np.float32)
    weights_f32 = np.random.uniform(-25, 25, (B, Q_SEQ, Q_HEAD)).astype(np.float32)
    q_scale_f32 = np.random.uniform(0, 255, (B, Q_SEQ, Q_HEAD)).astype(np.float32)
    k_scale_f32 = np.random.uniform(0, 65504, (B, K_HEAD, K_SEQ)).astype(np.float32)

    if qk_dtype == torch.float8_e4m3fn:
        query_quant = torch.from_numpy(query_f32).to(torch.float8_e4m3fn)
        key_quant = torch.from_numpy(key_bnsd_f32).to(torch.float8_e4m3fn)
        query_deq = query_quant.float().numpy()
        key_deq = key_quant.float().numpy()
    else:
        query_i8 = np.clip(np.round(query_f32), -127, 127).astype(np.int8)
        key_i8 = np.clip(np.round(key_bnsd_f32), -127, 127).astype(np.int8)
        query_quant = torch.from_numpy(query_i8)
        key_quant = torch.from_numpy(key_i8)
        query_deq = query_i8.astype(np.float32)
        key_deq = key_i8.astype(np.float32)

    if scale_dtype == torch.float16:
        weights_f32 = torch.from_numpy(weights_f32).to(torch.float16).float().numpy()
        q_scale_f32 = torch.from_numpy(q_scale_f32).to(torch.float16).float().numpy()
        k_scale_f32 = torch.from_numpy(k_scale_f32).to(torch.float16).float().numpy()

    act_seq_q = [Q_SEQ] * B
    act_seq_k = [K_SEQ * CMP_RATIO] * B

    # Golden (CPU)
    golden_idx = golden_quant_lightning_indexer_v2(
        query_deq, key_deq, weights_f32, q_scale_f32, k_scale_f32,
        act_seq_q, act_seq_k, Q_HEAD, K_HEAD, HD,
        TOPK, MASK_MODE, CMP_RATIO)

    # Page the key and k_scale for NPU
    key_pa, kscale_pa, block_table = to_paged_key(
        key_deq, k_scale_f32, B, K_HEAD, K_SEQ, HD, BLOCK_SIZE)

    # Prepare NPU tensors
    query_npu = query_quant.npu()
    if qk_dtype == torch.float8_e4m3fn:
        key_npu = torch.from_numpy(key_pa).to(torch.float8_e4m3fn).npu()
    else:
        key_npu = torch.from_numpy(key_pa.astype(np.int8)).npu()
    weights_npu = torch.from_numpy(weights_f32).to(scale_dtype).npu()
    q_scale_npu = torch.from_numpy(q_scale_f32).to(scale_dtype).npu()
    k_scale_npu = torch.from_numpy(kscale_pa).to(scale_dtype).npu()
    aslq_npu = torch.tensor(act_seq_q, dtype=torch.int32).npu()
    aslk_npu = torch.tensor(act_seq_k, dtype=torch.int32).npu()
    bt_npu = torch.from_numpy(block_table).int().npu()

    # Stage 1: Generate metadata
    metadata = quant_lightning_indexer_v2_metadata_npu(
        aslq_npu, aslk_npu,
        num_heads_q=Q_HEAD, num_heads_k=K_HEAD, head_dim=HD,
        batch_size=B, max_seqlen_q=Q_SEQ, max_seqlen_k=K_SEQ,
        layout_q="BSND", layout_k="PA_BSND",
        topk=TOPK, quant_mode=quant_mode,
        mask_mode=MASK_MODE, cmp_ratio=CMP_RATIO,
    )

    # Stage 2: Run main op
    npu_out = quant_lightning_indexer_v2_npu(
        query_npu, key_npu, weights_npu, q_scale_npu, k_scale_npu,
        aslq_npu, aslk_npu, bt_npu, metadata,
        num_heads_q=Q_HEAD, num_heads_k=K_HEAD, head_dim=HD,
        topk=TOPK, quant_mode=quant_mode,
        layout_q="BSND", layout_k="PA_BSND",
        mask_mode=MASK_MODE, cmp_ratio=CMP_RATIO,
        return_value=0,
    )
    npu_result = npu_out.cpu().numpy()

    # Compare using set overlap
    assert npu_result.shape == golden_idx.shape, f"Shape mismatch: {npu_result.shape} vs {golden_idx.shape}"
    for b in range(B):
        for s1 in range(Q_SEQ):
            for kh in range(K_HEAD):
                got = _valid_set(npu_result[b, s1, kh])
                exp = _valid_set(golden_idx[b, s1, kh])
                overlap = len(got & exp)
                min_match = max(1, int(len(exp) * 0.5))
                assert overlap >= min_match, (
                    f"b={b} s1={s1} kh={kh}: got {sorted(got)} exp {sorted(exp)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
