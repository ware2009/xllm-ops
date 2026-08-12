/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// ============================================================================
// multi_latent_attention_bs.h
//
// A5 (ascend950 / DAV_3510) InnerRunCubeMLA building-block sub-functions.
//
// This header holds the sub-functions extracted from the monolithic
// InnerRunCubeMLA so that the main function keeps only the business-flow
// skeleton. Each sub-function receives:
//   - a reference to the owning Cube class instance (aic) to reach the
//     GlobalTensor / LocalTensor members and tiling parameters
//   - an MLAContext (ctx) that carries the per-call scalar state
//
// Pipeline synchronization semantics are preserved 1:1 from arch32, with the
// A3 -> A5 event-channel mapping applied (A3's MTE2 UB<->L1 events become
// A5's MTE3 on-chip UB->L1 channel).
//
// NOTE: The sub-function bodies below are skeletons. The concrete data-move /
// mmad implementation is migrated from arch32 in follow-up steps; the
// signatures and MLAContext are the stable contract used by InnerRunCubeMLA.
// ============================================================================

#ifndef MULTI_LATENT_ATTENTION_BS_ARCH35_H
#define MULTI_LATENT_ATTENTION_BS_ARCH35_H

#include "kernel_operator.h"
#include "multi_latent_attention_utils.h"
// Platform (arch-specific) primitives. Business flow below calls arch-neutral
// names (CopyGmToL1Nd2Nz / PlatformSetQLoadComplete); swap this include for an
// arch32 header to retarget the same business logic to A3.
#include "multi_latent_attention_arch35.h"

namespace XllmOps {
namespace MlaArch35 {

// ============================================================================
// MLAContext 鈥?per-call scalar state shared across the MLA sub-functions.
//
// Populated by InitMLAContext() and threaded through LoadQData / LoadKVData /
// ComputeQK / ComputePV. ComputeQK/ComputePV take it by non-const reference
// because they refresh the tail-block n/round-n values as the loop advances.
// ============================================================================
struct MLAContext {
    // --- Q / Q_rope GM offsets (base addresses in elements) ---
    uint64_t q_offset{0};
    uint64_t q_rope_offset{0};

    // --- KV block loop control ---
    uint32_t pp_n_scalar{0};   // per ping-pong block N (== block_size)
    uint32_t sub_n_loop{0};    // pp_n_scalar / block_size
    uint32_t n_loop{0};        // number of KV blocks

    // --- QK GEMM1 shape (current + tail-updated) ---
    uint32_t qk_n{0};          // current N for QK
    uint32_t qk_round_n{0};    // RoundUp<BLOCK_SIZE>(qk_n)
    uint32_t qk_n_2{0};        // current N for PV
    uint32_t qk_round_n_2{0};  // RoundUp<BLOCK_SIZE>(qk_n_2)
    uint32_t qk_round_n_l1{0};   // RoundUp<T_BLOCK_SIZE>(qk_n)
    uint32_t qk_round_n_2_l1{0}; // RoundUp<T_BLOCK_SIZE>(qk_n_2)

    // --- hidden / K dims ---
    uint64_t hidden_size{576}; // 576 (FP16/BF16) or 512 (INT8)
    uint64_t k_round_n{0};     // rounded N used for KV L1 layout

    // --- M dimension (rows) ---
    uint32_t row_num{0};       // cur_head_num * cur_q_seqlen
    uint32_t m{0};             // RoundUp<16>(row_num)

    // --- Per-call task descriptors (echoed from InnerRunCubeMLA args) ---
    uint32_t cur_batch{0};
    uint32_t start_head{0};
    uint32_t cur_head_num{0};
    uint32_t start_kv{0};
    uint32_t cur_q_seqlen{0};
    uint32_t cur_kv_seqlen{0};
    uint32_t offset_tiling{0};
};

// ============================================================================
// Sub-function contract
//
// All sub-functions are templated on CubeT 鈥?the owning
// MLAttentionDecoderAic<...> instance type. Using a template parameter avoids
// a circular include (bs.h does not need the full class definition) while
// still giving the body direct access to the class' tensor/tiling members.
//
// The owning class declares these as friends so they can reach its private
// members:
//     template <typename CubeT> friend __aicore__ void InitMLAContext(...);
//     ...
// ============================================================================

// ----------------------------------------------------------------------------
// InitMLAContext 鈥?arch32 L696-719
//
// Computes Q / Q_rope GM offsets from the per-batch tiling address, derives the
// KV block loop count, the QK/PV N + rounded-N values, hidden_size (576 for
// FP16/BF16, 512 for INT8) and the padded row count m. Pure scalar setup: no
// data movement, no sync events.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void InitMLAContext(
    CubeT &aic, MLAContext &ctx,
    uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num,
    uint32_t start_kv, uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t offset_tiling)
{
    // Echo per-call descriptors first.
    ctx.cur_batch = cur_batch;
    ctx.start_head = start_head;
    ctx.cur_head_num = cur_head_num;
    ctx.start_kv = start_kv;
    ctx.cur_q_seqlen = cur_q_seqlen;
    ctx.cur_kv_seqlen = cur_kv_seqlen;
    ctx.offset_tiling = offset_tiling;

    // --- Q / Q_rope GM offsets (arch32 L696-700) ---
    // Per-batch base token index is stored as a hi/lo uint32 pair at
    // tiling_gm[2 + offset_tiling] / [3 + offset_tiling] (uint32 stride).
    uint32_t addr_q_high32 =
        (uint32_t)(*((__gm__ uint32_t *)aic.tiling_gm + 2 + offset_tiling));
    uint32_t addr_q_loww32 =
        (uint32_t)(*((__gm__ uint32_t *)aic.tiling_gm + 3 + offset_tiling));
    uint64_t addr_q_scalar =
        (uint64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);
    ctx.q_offset = addr_q_scalar * 512 + (uint64_t)start_head * 512;
    ctx.q_rope_offset = addr_q_scalar * 64 + (uint64_t)start_head * 64;

    // --- KV block loop control (arch32 L702-705) ---
    ctx.pp_n_scalar = aic.block_size;
    ctx.sub_n_loop = ctx.pp_n_scalar / aic.block_size;
    ctx.n_loop = (cur_kv_seqlen + ctx.pp_n_scalar - 1) / ctx.pp_n_scalar;

    // --- QK / PV N + rounded-N (arch32 L707-712) ---
    ctx.qk_n = ctx.pp_n_scalar;
    ctx.qk_round_n = RoundUp<BLOCK_SIZE_16>(ctx.qk_n);
    ctx.qk_n_2 = ctx.pp_n_scalar;
    ctx.qk_round_n_2 = RoundUp<BLOCK_SIZE_16>(ctx.qk_n_2);
    ctx.qk_round_n_l1 = RoundUp<CubeT::T_BLOCK_SIZE>(ctx.qk_n);
    ctx.qk_round_n_2_l1 = RoundUp<CubeT::T_BLOCK_SIZE>(ctx.qk_n_2);

    // --- hidden / K dims (arch32 L713-717) ---
    ctx.hidden_size = CubeT::kIsInt8 ? 512 : 576;
    ctx.k_round_n = ctx.qk_round_n;

    // --- M dimension (arch32 L718-719) ---
    ctx.row_num = cur_head_num * cur_q_seqlen;
    ctx.m = RoundUp<16>(ctx.row_num);
    aic.m = ctx.m;  // arch32 stores m as a class member consumed downstream
}

// ----------------------------------------------------------------------------
// LoadQMainFromGMToL1 鈥?Q main (compressed latent, dim 512) GM -> L1.
// arch32 L721-766. Business layer: owns the three-way branch selection and
// fully builds each Nd2NzParams, handing it to the arch-neutral
// CopyGmToL1Nd2Nz primitive (platform layer owns only the raw DataCopy).
//   - cur_q_seqlen == 1 : single ND matrix, contiguous
//   - q_heads < 128     : batch all tokens in one instruction
//   - q_heads >= 128    : per-token loop (regulates stride bit-width limit)
//
// CONSTRAINT (A5 tile sizing): total bytes written into the l1q region =
//   RoundUp<16>(cur_head_num * cur_q_seqlen) * 512 * sizeof(IN_DTYPE), and
//   must stay <= l1q region capacity (147456 bytes), else it overruns l1kv.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void LoadQMainFromGMToL1(
    CubeT &aic, const MLAContext &ctx)
{
    const uint32_t cur_head_num = ctx.cur_head_num;
    const uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    const uint32_t q_heads = aic.q_heads;

    if (cur_q_seqlen == 1) {
        // Single-token path (arch32 L721-733): one ND matrix, contiguous.
        CopyGmToL1Nd2Nz(
            aic.l1q_buf_addr_tensor,
            aic.q_gm_tensor[ctx.q_offset],
            AscendC::Nd2NzParams(
                /* ndNum            */ 1,
                /* nValue           */ cur_head_num,
                /* dValue           */ 512,
                /* srcNdMatrixStride*/ 0,
                /* srcDValue        */ 512,
                /* dstNzC0Stride    */ RoundUp<16>(cur_head_num),
                /* dstNzNStride     */ 1,
                /* dstNzMatrixStride*/ 0));
    } else if (q_heads < 128) {
        // Batch all tokens in one instruction (arch32 L734-748).
        CopyGmToL1Nd2Nz(
            aic.l1q_buf_addr_tensor,
            aic.q_gm_tensor[ctx.q_offset],
            AscendC::Nd2NzParams(
                /* ndNum            */ cur_q_seqlen,
                /* nValue           */ cur_head_num,
                /* dValue           */ 512,
                /* srcNdMatrixStride*/ 512 * q_heads,
                /* srcDValue        */ 512,
                /* dstNzC0Stride    */ RoundUp<16>(cur_head_num * cur_q_seqlen),
                /* dstNzNStride     */ cur_q_seqlen,
                /* dstNzMatrixStride*/ 16));
    } else {
        // Per-token loop for large head counts (arch32 L749-766).
        for (uint32_t ii = 0; ii < cur_q_seqlen; ii++) {
            CopyGmToL1Nd2Nz(
                aic.l1q_buf_addr_tensor[ii * 16],
                aic.q_gm_tensor[ctx.q_offset + (uint64_t)ii * q_heads * 512],
                AscendC::Nd2NzParams(
                    /* ndNum            */ 1,
                    /* nValue           */ cur_head_num,
                    /* dValue           */ 512,
                    /* srcNdMatrixStride*/ 0,
                    /* srcDValue        */ 512,
                    /* dstNzC0Stride    */ RoundUp<16>(cur_q_seqlen * cur_head_num),
                    /* dstNzNStride     */ cur_q_seqlen,
                    /* dstNzMatrixStride*/ 16));
        }
    }
}

// ----------------------------------------------------------------------------
// BUSINESS LOGIC: Q RoPE (decoupled positional, dim 64) GM -> L1.
//
// INT8: RoPE (NOT quantized, dtype IN_ROPE_DTYPE=half/bf16) goes into its
//   OWN L1 buffer, contiguous per head (arch32 L769-781).
// non-INT8: RoPE appended right after Q main on the SHARED buffer, at
//   offset RoundUp<16>(head*seqlen)*512, per-head q_heads stride
//   (arch32 L782-795).
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void LoadQRopeFromGMToL1(
    CubeT &aic, const MLAContext &ctx)
{
    const uint32_t cur_head_num = ctx.cur_head_num;
    const uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    const uint32_t q_heads = aic.q_heads;

    if constexpr (CubeT::kIsInt8) {
        CopyGmToL1Nd2Nz(
            aic.l1q_rope_buf_addr_tensor,
            aic.q_rope_gm_tensor[ctx.q_rope_offset],
            AscendC::Nd2NzParams(
                /* ndNum            */ 1,
                /* nValue           */ cur_head_num,
                /* dValue           */ 64,
                /* srcNdMatrixStride*/ 0,
                /* srcDValue        */ 64,
                /* dstNzC0Stride    */ RoundUp<16>(cur_head_num),
                /* dstNzNStride     */ 1,
                /* dstNzMatrixStride*/ 0));
    } else {
        CopyGmToL1Nd2Nz(
            aic.l1q_buf_addr_tensor[RoundUp<16>(cur_head_num * cur_q_seqlen) * 512],
            aic.q_rope_gm_tensor[ctx.q_rope_offset],
            AscendC::Nd2NzParams(
                /* ndNum            */ cur_head_num,
                /* nValue           */ cur_q_seqlen,
                /* dValue           */ 64,
                /* srcNdMatrixStride*/ 64,
                /* srcDValue        */ 64 * q_heads,
                /* dstNzC0Stride    */ RoundUp<16>(cur_head_num * cur_q_seqlen),
                /* dstNzNStride     */ 1,
                /* dstNzMatrixStride*/ 16 * cur_q_seqlen));
    }
}

// ----------------------------------------------------------------------------
// LoadQData 鈥?orchestration layer.
//
// Pure control flow: loads Q main + Q RoPE from GM to L1, then issues the
// terminal producer/consumer sync. Platform-specific details live in the
// business functions (Nd2Nz branches) and the platform layer
// (PlatformSetQLoadComplete encapsulates the A3 MTE2 / A5 MTE3 mapping).
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void LoadQData(
    CubeT &aic, const MLAContext &ctx)
{
    LoadQMainFromGMToL1(aic, ctx);
    LoadQRopeFromGMToL1(aic, ctx);
    PlatformSetQLoadComplete();
}

// ----------------------------------------------------------------------------
// LoadKVData 鈥?arch32 L799-896
//
// For KV block n_idx: resolves block_table_id -> kv_offset / kv_offset_rope,
// then loads K (main) and K_rope into the ping-pong L1 KV region. Three data
// paths: ND_FORMAT / TILING_INT8_DATA / NZ. Producer/consumer events that were
// MTE2 on A3 become MTE3 on A5.
//
// Returns the L1 KV ping-pong flag (n_idx % 2) for the QK stage to consume.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline uint32_t LoadKVData(
    CubeT &aic, MLAContext &ctx, uint32_t n_idx)
{
    const uint32_t flag = n_idx % 2;  // L1 KV ping-pong slot

    // ---- tail-block N refresh (arch32 L801-810) ----
    // The last KV block may be shorter than pp_n_scalar; recompute the real /
    // rounded N so the DataCopy shapes below cover exactly the live rows.
    if (n_idx == (ctx.n_loop - 1)) {
        ctx.qk_n = ctx.cur_kv_seqlen - n_idx * ctx.pp_n_scalar;
        ctx.qk_round_n = RoundUp<BLOCK_SIZE_16>(ctx.qk_n);
        ctx.qk_round_n_l1 = RoundUp<CubeT::T_BLOCK_SIZE>(ctx.qk_n);
    }
    // k_round_n mirrors A3: INT8 rounds to the L1 fractal (T_BLOCK_SIZE), others
    // to the 16-wide fractal. Consumed downstream by ComputeQK.
    ctx.k_round_n = CubeT::kIsInt8 ? ctx.qk_round_n_l1 : ctx.qk_round_n;

    // ---- paged KV addressing (arch32 L817-820) ----
    // block_tables_gm[cur_batch, start_kv/block_size + n_idx] -> physical block,
    // then scale by block_size * per-token stride to reach this block's base.
    uint32_t block_table_id = (uint32_t)(*(aic.block_tables_gm +
        ctx.cur_batch * aic.max_num_blocks_per_query +
        ctx.start_kv / aic.block_size + n_idx));
    int64_t kv_offset = (int64_t)block_table_id * aic.block_size * aic.stride_kv;
    int64_t kv_offset_rope =
        (int64_t)block_table_id * aic.block_size * aic.stride_kv_rope;

    const uint32_t qk_n = ctx.qk_n;
    const uint32_t qk_round_n = ctx.qk_round_n;
    const uint32_t qk_round_n_l1 = ctx.qk_round_n_l1;
    const uint32_t block_size = aic.block_size;

    // Wait for the previous consumer (Cube L1->L0B) to release this ping-pong
    // slot. A3 used MTE2 for the GM/UB->L1 producer edge; A5 maps it to MTE3.
    PlatformWaitKVLoadReady(flag);

    if constexpr (CubeT::kInputFormat == InputFormat::ND_FORMAT) {
        // ---- ND -> NZ : row-major KV cache, transform on the fly (L824-846) ----
        // K main into the shared KV buffer; K main width 512, src row = stride_kv.
        LoadKVMainFromGMToL1(
            false,
            aic.l1kv_buf_addr_tensor[flag * 128 * 576],
            aic.k_gm_tensor[kv_offset],
            qk_n,           // nTileActual
            qk_round_n,     // nTileCeil
            512,            // dTileActual (K main column dim)
            aic.stride_kv); // srcDValue
        PlatformSetKVMainLoadComplete(flag);
        // K RoPE appended after K main (offset 512 * qk_round_n); width 64.
        LoadKVRopeFromGMToL1(
            false,
            aic.l1kv_buf_addr_tensor[flag * 128 * 576 + 512 * qk_round_n],
            aic.k_rope_gm_tensor[kv_offset_rope],
            qk_n,
            qk_round_n,
            aic.stride_kv_rope);
    } else if constexpr (CubeT::kIsInt8) {
        // ---- INT8 NZ -> NZ : paged NZ cache, K RoPE in its OWN buffer (L847-869) ----
        LoadKVMainFromGMToL1(
            true,
            aic.l1kv_buf_addr_tensor[flag * 128 * 512],
            aic.k_gm_tensor[kv_offset],
            qk_round_n_l1,  // nTileActual
            block_size,     // nTileCeil (NZ src pitch)
            512,            // dTileActual
            0);             // srcDValue unused for NZ->NZ
        PlatformSetKVMainLoadComplete(flag);
        LoadKVRopeFromGMToL1(
            true,
            aic.l1kv_rope_buf_addr_tensor[flag * 128 * 64],
            aic.k_rope_gm_tensor[kv_offset_rope],
            qk_round_n,
            block_size,
            0);
    } else {
        // ---- NZ -> NZ : paged NZ cache, K RoPE appended on shared buf (L870-893) ----
        LoadKVMainFromGMToL1(
            true,
            aic.l1kv_buf_addr_tensor[flag * 128 * 576],
            aic.k_gm_tensor[kv_offset],
            qk_round_n,     // nTileActual
            block_size,     // nTileCeil (NZ src pitch)
            512,            // dTileActual
            0);             // srcDValue unused for NZ->NZ
        PlatformSetKVMainLoadComplete(flag);
        LoadKVRopeFromGMToL1(
            true,
            aic.l1kv_buf_addr_tensor[flag * 128 * 576 + 512 * qk_round_n],
            aic.k_rope_gm_tensor[kv_offset_rope],
            qk_round_n,
            block_size,
            0);
    }

    // Publish the KV block to the Cube L1->L0B consumer (arch32 L896: MTE2->A5 MTE3).
    PlatformSetKVRopeLoadComplete(flag);
    return flag;
}

// ============================================================================
// ComputeQK sub-functions (business layer). Each owns its Flag handshakes and
// INT8/tail branches; they call the pure-AscendC *Raw helpers in utils.h for
// the actual LoadData / Mmad / Fixpipe. ComputeQK below is pure orchestration.
// ============================================================================

// ----------------------------------------------------------------------------
// LoadQFromL1ToL0A 鈥?Q (L1) -> L0A for one embedding split (arch32 L905-918).
// Owns: WAIT_FLAG(M,MTE1,esi%2), the loa_load_idx fractal loop, SET_FLAG(MTE1,M,esi%2).
// ----------------------------------------------------------------------------
// Passed by parameter (arch35 policy): the private L0A / L1-Q tensors and the
// T_CUBE_MATRIX_SIZE / T_BLOCK_SIZE tiling constants are supplied by the friend
// orchestrator ComputeQK, so this sub-function needs no access to CubeT privates.
template <typename L0aT, typename L1qT>
__aicore__ __attribute__((always_inline)) inline void LoadQFromL1ToL0A(
    const AscendC::LocalTensor<L0aT> &l0a_buf_tensor,
    const AscendC::LocalTensor<L1qT> &l1q_buf_addr_tensor,
    uint32_t esi, uint32_t m, uint64_t round_embed_split_size,
    uint32_t t_cube_matrix_size, uint32_t t_block_size)
{
    const uint32_t q_load_coeff = m;

    PlatformWaitL0AReady(esi % 2);
    for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE_16; ++loa_load_idx) {
        LoadQL1ToL0ARaw(
            l0a_buf_tensor[esi % 2 * 16384 + loa_load_idx * round_embed_split_size * BLOCK_SIZE_16],
            l1q_buf_addr_tensor[esi * m * 128 + loa_load_idx * t_cube_matrix_size],
            (uint32_t)(round_embed_split_size / t_block_size),   // repeat
            q_load_coeff / BLOCK_SIZE_16);                       // srcStride
    }
    PlatformSetL0ALoadComplete(esi % 2);
}

// ----------------------------------------------------------------------------
// LoadKVFromL1ToL0B 鈥?KV (L1) -> L0B for one embedding split (arch32 L921-941).
// Owns: the esi==0 / esi==4 special WAIT/SET_FLAG(MTE3,MTE1,...) KV-publish
// handshakes plus WAIT_FLAG(M,MTE1,esi%2+2)/SET_FLAG(MTE1,M,esi%2+2). A3 MTE2
// (UB<->L1) maps to A5 MTE3.
// ----------------------------------------------------------------------------
// Passed by parameter (arch35 policy): private L0B / L1-KV tensors and the
// T_CUBE_MATRIX_SIZE tiling constant come from the friend orchestrator ComputeQK.
template <typename L0bT, typename L1kvT>
__aicore__ __attribute__((always_inline)) inline void LoadKVFromL1ToL0B(
    const AscendC::LocalTensor<L0bT> &l0b_buf_tensor,
    const AscendC::LocalTensor<L1kvT> &l1kv_buf_addr_tensor,
    uint32_t esi, uint32_t l1_kv_pingpong_flag,
    uint64_t round_embed_split_size, uint64_t k_round_n, uint64_t hidden_size,
    uint32_t t_cube_matrix_size)
{
    if (esi == 0) {
        PlatformWaitKVMainSlot(l1_kv_pingpong_flag);
    }
    if (esi == 4) {
        PlatformWaitKVRopeSlot(l1_kv_pingpong_flag);
    }
    PlatformWaitL0BReady(esi % 2);
    LoadKVL1ToL0BRaw(
        l0b_buf_tensor[esi % 2 * 16384],
        l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size+ esi * k_round_n * 128],
        (uint32_t)(round_embed_split_size * k_round_n / t_cube_matrix_size)); // repeat
    if (esi == 4) {
        PlatformSetKVRopeSlotFree(l1_kv_pingpong_flag);
    }
    PlatformSetL0BLoadComplete(esi % 2);
    PlatformWaitL0AB(esi % 2);
    if (esi == 0) {
        PlatformWaitL0cReleased(l1_kv_pingpong_flag);
    }
}

// ----------------------------------------------------------------------------
// ComputeQKMMad 鈥?L0A x L0B -> mm1 L0C for one embedding split (arch32 L947-967).
// Owns: INT8 vs non-INT8 n selection (qk_round_n_l1 vs qk_n), cmatrixInitVal
// (esi==0), and the trailing PIPE_BARRIER(M) + SET_FLAG(M,MTE1,esi%2 / +2).
// ----------------------------------------------------------------------------
// Passed by parameter (arch35 policy): private mm1-L0C / L0A / L0B tensors and
// the kIsInt8 flag come from the friend orchestrator ComputeQK.
template <typename L0cT, typename L0aT, typename L0bT>
__aicore__ __attribute__((always_inline)) inline void ComputeQKMMad(
    const AscendC::LocalTensor<L0cT> &mm1_l0c_buf_tensor,
    const AscendC::LocalTensor<L0aT> &l0a_buf_tensor,
    const AscendC::LocalTensor<L0bT> &l0b_buf_tensor,
    bool is_int8, uint32_t esi, uint32_t l1_kv_pingpong_flag,
    uint32_t m, uint32_t qk_n, uint32_t qk_round_n_l1, uint64_t embed_split_size)
{
    ComputeQKMMadRaw(
        mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
        l0a_buf_tensor[esi % 2 * 16384],
        l0b_buf_tensor[esi % 2 * 16384],
        m,
        is_int8 ? qk_round_n_l1 : qk_n,   // n
        (uint32_t)embed_split_size,       // k
        esi == 0);                        // cmatrixInitVal
    PlatformSetMmadComplete(esi % 2);
}

// ----------------------------------------------------------------------------
// CopyQKResultToUB - A5 shared-UB variant that replaces the A3 CopyQKResultToGM. Instead of the A3
// L0C->GM(S) round-trip it FixPipes the QK score straight into the UB region the
// AIC shares with its two paired AIV cores (s_ubuf_tensor == AIV ls32_ubuf,
// offset 0). Same flush-idx trigger (INT8 esi==3 / non-INT8 esi==4) and the same
// M<->FIX fence as the GM variant. The UB region is per-AI-Core private, so there
// is NO block_idx stride; only the n_idx double-buffer half stays.
// ----------------------------------------------------------------------------
template <typename SubufT, typename L0cT>
__aicore__ __attribute__((always_inline)) inline void CopyQKResultToUB(
    const AscendC::LocalTensor<SubufT> &s_ubuf_tensor,
    const AscendC::LocalTensor<L0cT> &mm1_l0c_buf_tensor,
    bool is_int8, uint32_t esi, uint32_t n_idx, uint32_t l1_kv_pingpong_flag,
    uint32_t m, uint32_t qk_n, uint32_t qk_round_n)
{
    const uint32_t flush_idx = is_int8 ? 3 : 4;
    const uint32_t nSize = is_int8 ? qk_n : qk_round_n;

    if (esi != flush_idx) {
        return;
    }
    PlatformWaitL0cForFix(l1_kv_pingpong_flag);
    // A5 FixPipe supports L0C(float)->UB(float) (FP16 main path). INT8 L0C is
    // int32 and A5 has NO Fixpipe<int,float> combo; the INT8 QK score is instead
    // finalised by ComputeQRope's L0C(float)->s_rope_gm round-trip. So only the
    // FP16 (float L0C) path FixPipes into the shared UB here; INT8 is a TODO stub.
    if constexpr (AscendC::IsSameType<L0cT, float>::value) {
        CopyQKResultToUBRaw(
            s_ubuf_tensor[(uint64_t)(n_idx % 2) * TMP_SIZE_DECODER_A5 / 2],
            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
            m,               // MSize
            nSize,           // NSize
            RoundUp<16>(m),  // srcStride
            qk_round_n);     // dstStride
    }
    // TODO(INT8): route the int32 L0C QK score into the shared UB (DEQ to half)
    // once the INT8 decode path is enabled.
    PlatformSetFixComplete(l1_kv_pingpong_flag);
}

// ----------------------------------------------------------------------------
// ComputeQRope 鈥?INT8-only decoupled RoPE mmad tail (arch32 L1005-1081).
// Q_rope/KV_rope L1->L0A/L0B (reinterpreted as IN_ROPE_DTYPE), a mmad that
// accumulates (initC=true) onto the main QK score in mm1 L0C (viewed as float),
// then L0C->GM into s_rope_gm (float). Owns all its own Flag handshakes.
// ----------------------------------------------------------------------------
// Passed by parameter (arch35 policy): the RoPE-related private tensors and
// block_idx come from the friend orchestrator ComputeQK. IN_ROPE_DTYPE is the
// reinterpret element type for the L0A/L0B RoPE views (passed as a template arg).
template <typename IN_ROPE_DTYPE, typename L0aT, typename L0bT, typename L0cT,
          typename L1qRopeT, typename L1kvRopeT, typename SRopeGmT>
__aicore__ __attribute__((always_inline)) inline void ComputeQRope(
    const AscendC::LocalTensor<L0aT> &l0a_buf_tensor,
    const AscendC::LocalTensor<L0bT> &l0b_buf_tensor,
    const AscendC::LocalTensor<L0cT> &mm1_l0c_buf_tensor,
    const AscendC::LocalTensor<L1qRopeT> &l1q_rope_buf_addr_tensor,
    const AscendC::LocalTensor<L1kvRopeT> &l1kv_rope_buf_addr_tensor,
    const AscendC::GlobalTensor<SRopeGmT> &s_rope_gm_tensor,
    uint32_t block_idx, MLAContext &ctx, uint32_t n_idx, uint32_t l1_kv_pingpong_flag)
{
    const uint32_t m = ctx.m;
    const uint32_t qk_n = ctx.qk_n;
    const uint32_t qk_round_n = ctx.qk_round_n;
    const uint32_t q_load_coeff = m;

    const uint64_t esi = 4;                 // arch32 explicitly resets to 4
    const uint64_t embed_split_size = 64;
    const uint64_t round_embed_split_size = 64;

    PlatformWaitL0AReady(esi % 2);
    // Q_rope: L1 -> L0A, reinterpreted as IN_ROPE_DTYPE (arch32 L1011-1022).
    auto l0aRope = l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>();
    for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE_16; ++loa_load_idx) {
        LoadQL1ToL0ARaw(
            l0aRope[esi % 2 * 16384 * 2 + loa_load_idx * round_embed_split_size * BLOCK_SIZE_16],
            l1q_rope_buf_addr_tensor[loa_load_idx * CUBE_MATRIX_SIZE_A5],
            (uint32_t)(round_embed_split_size / BLOCK_SIZE_16),  // repeat
            q_load_coeff / BLOCK_SIZE_16);                       // srcStride
    }
    PlatformSetL0ALoadComplete(esi % 2);

    // KV_rope: L1 -> L0B (arch32 L1026-1037). A3 MTE2 -> A5 MTE3.
    PlatformWaitKVRopeSlot(l1_kv_pingpong_flag);
    PlatformWaitL0BReady(esi % 2);
    auto l0bRope = l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>();
    LoadKVL1ToL0BRaw(
        l0bRope[esi % 2 * 16384 * 2],
        l1kv_rope_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 64],
        (uint32_t)(round_embed_split_size * qk_round_n / CUBE_MATRIX_SIZE_A5)); // repeat
    PlatformSetKVRopeSlotFree(l1_kv_pingpong_flag);
    PlatformSetL0BLoadComplete(esi % 2);
    PlatformWaitL0AB(esi % 2);
    PlatformWaitL0cReleased(l1_kv_pingpong_flag);

    // RoPE Mmad accumulates (initC=1) onto the main QK score, viewed as float.
    auto mm1L0cFloat = mm1_l0c_buf_tensor.template ReinterpretCast<float>();
    ComputeQKMMadRaw(
        mm1L0cFloat[l1_kv_pingpong_flag * 16384],
        l0aRope[esi % 2 * 16384 * 2],
        l0bRope[esi % 2 * 16384 * 2],
        m,
        qk_n,
        (uint32_t)embed_split_size,
        /* initC */ true);
    PlatformSetMmadComplete(esi % 2);

    // RoPE S write-back (arch32 L1059-1080). INT8 -> s_rope_gm (float).
    PlatformWaitL0cForFix(l1_kv_pingpong_flag);
    CopyQKResultToGMRaw(
        s_rope_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER_A5 +
                         (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER_A5 / 2],
        mm1L0cFloat[l1_kv_pingpong_flag * 16384],
        m,               // MSize
        qk_round_n,      // NSize
        RoundUp<16>(m),  // srcStride
        qk_round_n);     // dstStride
    PlatformSetFixComplete(l1_kv_pingpong_flag);
}

// ----------------------------------------------------------------------------
// ComputeQK 鈥?arch32 L897-1081  (CUBE1 stage1). Pure orchestration: runs the
// hidden_split_time embedding-split loop over the four sub-functions, then the
// INT8 RoPE tail (ComputeQRope), then publishes the QK result. ctx is a
// non-const reference because qk_n / qk_round_n are refreshed on the tail block.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void ComputeQK(
    CubeT &aic, MLAContext &ctx, uint32_t n_idx, uint32_t l1_kv_pingpong_flag)
{
    constexpr bool kIsInt8 = CubeT::kIsInt8;
    constexpr uint32_t T_BLOCK_SIZE = CubeT::T_BLOCK_SIZE;
    constexpr uint32_t T_CUBE_MATRIX_SIZE = CubeT::T_CUBE_MATRIX_SIZE;

    const uint32_t m = ctx.m;
    const uint32_t qk_n = ctx.qk_n;
    const uint32_t qk_round_n = ctx.qk_round_n;
    const uint32_t qk_round_n_l1 = ctx.qk_round_n_l1;
    const uint64_t hidden_size = ctx.hidden_size;
    const uint64_t k_round_n = ctx.k_round_n;

// arch32 L897: embedding-split loop count.
    const uint64_t hidden_split_time = (hidden_size + 128 - 1) / 128;
    uint64_t embed_split_size = 128;
    uint64_t round_embed_split_size = RoundUp<T_BLOCK_SIZE>(embed_split_size);

    // ---- Main embedding-split QK GEMM (arch32 L899-1004) ----
    // arch35 policy: extract CubeT privates here (ComputeQK is a friend) and pass
    // them to the sub-functions by parameter, so the sub-functions stay decoupled.
    for (uint64_t esi = 0; esi < hidden_split_time; ++esi) {
        if (esi == 4) {
            embed_split_size = 64;
            round_embed_split_size = 64;
        }
        LoadQFromL1ToL0A(aic.l0a_buf_tensor, aic.l1q_buf_addr_tensor,
                         (uint32_t)esi, m, round_embed_split_size,
                         T_CUBE_MATRIX_SIZE, T_BLOCK_SIZE);
        LoadKVFromL1ToL0B(aic.l0b_buf_tensor, aic.l1kv_buf_addr_tensor,
                          (uint32_t)esi, l1_kv_pingpong_flag,
                          round_embed_split_size, k_round_n, hidden_size,
                          T_CUBE_MATRIX_SIZE);
        ComputeQKMMad(aic.mm1_l0c_buf_tensor, aic.l0a_buf_tensor, aic.l0b_buf_tensor,
                      kIsInt8, (uint32_t)esi, l1_kv_pingpong_flag,
                      m, qk_n, qk_round_n_l1, embed_split_size);
        // A5 shared-UB: FixPipe the QK score straight into the UB region shared
        // with the paired AIV cores (== AIV ls32_ubuf), no S GM round-trip.
        CopyQKResultToUB(aic.s_ubuf_tensor, aic.mm1_l0c_buf_tensor,
                         kIsInt8, (uint32_t)esi, n_idx,
                         l1_kv_pingpong_flag, m, qk_n, qk_round_n);
    }

    // ---- INT8 RoPE mmad tail segment (arch32 L1005-1081) ----
    if constexpr (kIsInt8) {
      using IN_ROPE_DTYPE = typename CubeT::IN_ROPE_DTYPE_ALIAS;
      ComputeQRope<IN_ROPE_DTYPE>(
          aic.l0a_buf_tensor, aic.l0b_buf_tensor, aic.mm1_l0c_buf_tensor,
          aic.l1q_rope_buf_addr_tensor, aic.l1kv_rope_buf_addr_tensor,
          aic.s_rope_gm_tensor, (uint32_t)block_idx, ctx, n_idx, l1_kv_pingpong_flag);
    }

    // Publish QK result to the softmax (Vector) consumer (arch32 L1082).
    FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_DECODER);
}


// ============================================================================
// ComputePV business sub-functions (arch32 L1084-1208, CUBE2 stage1).
// arch35 policy (mirrors the QK layer): each sub-function is templated on the
// tensor element types, takes the CubeT privates by parameter, composes the
// arch35.h FLAG-only platform helpers with the utils.h AscendC Raw primitives,
// and owns no cross-function state. block_idx is directly visible (no aic.).
// ============================================================================

// ----------------------------------------------------------------------------
// LoadVTransposeToL0B 鈥?KV (L1) -> L0B transpose load for one embed split
// (arch32 PlatformLoadKVTranspose{SmallN,LargeN} L211-253). Selects SmallN /
// LargeN by k_round_n <= round_embed_split_size, builds the
// LoadData2dTransposeParams here (arch35 platform math lives in the caller),
// and publishes the L1 KV slot on the last split. IN_DTYPE==1 (INT8) sets the
// per-fractal dstGap to 1 (SmallN only, matching arch32 L223).
// ----------------------------------------------------------------------------
template <typename L0bT, typename L1kvT>
__aicore__ __attribute__((always_inline)) inline void LoadVTransposeToL0B(
    const AscendC::LocalTensor<L0bT> &l0b_buf_tensor,
    const AscendC::LocalTensor<L1kvT> &l1kv_buf_addr_tensor,
    uint32_t l0b_pingpong_flag, uint32_t l1_kv_pingpong_flag,
    uint64_t l1kv_offset, uint64_t k_round_n, uint64_t hidden_size,
    uint32_t qk_round_n_2, uint32_t embed_split_size,
    uint32_t round_embed_split_size, bool is_last_split,
    uint32_t t_block_size)
{
    constexpr bool kIsInt8 = (sizeof(L1kvT) == 1);
    PlatformWaitPVL0BReady(l0b_pingpong_flag);

    if (k_round_n <= round_embed_split_size) {
        // SmallN (arch32 L211-231): loop k_round_n / T_BLOCK_SIZE, per-fractal
        // L1 stride = T_BLOCK_SIZE * T_BLOCK_SIZE, L0B stride =
        // RoundUp<16>(embed_split_size) * T_BLOCK_SIZE.
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.dstFracGap = 0;
        params.repeatTimes = round_embed_split_size / t_block_size;
        params.srcStride = k_round_n / t_block_size;
        params.dstGap = kIsInt8 ? 1 : 0;
        for (uint32_t l0b_load_idx = 0; l0b_load_idx < k_round_n / t_block_size; ++l0b_load_idx) {
            LoadVTransposeToL0BRaw(
                l0b_buf_tensor[l0b_pingpong_flag * 16384 +
                               l0b_load_idx * RoundUp<16>(embed_split_size) * t_block_size],
                l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + l1kv_offset +
                                     l0b_load_idx * t_block_size * t_block_size],
                params);
        }
    } else {
        // LargeN (arch32 L234-253): loop round_embed_split_size / T_BLOCK_SIZE,
        // per-fractal L1 stride = qk_round_n_2 * T_BLOCK_SIZE, L0B stride =
        // T_BLOCK_SIZE * T_BLOCK_SIZE, dstGap = round_embed_split_size/BLOCK-1.
        AscendC::LoadData2dTransposeParams params;
        params.startIndex = 0;
        params.dstFracGap = 0;
        params.repeatTimes = qk_round_n_2 / t_block_size;
        params.srcStride = 1;
        params.dstGap = round_embed_split_size / BLOCK_SIZE_16 - 1;
        for (uint32_t l0b_load_idx = 0; l0b_load_idx < round_embed_split_size / t_block_size; ++l0b_load_idx) {
            LoadVTransposeToL0BRaw(
                l0b_buf_tensor[l0b_pingpong_flag * 16384 +
                               l0b_load_idx * t_block_size * t_block_size],
                l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + l1kv_offset +
                                     l0b_load_idx * qk_round_n_2 * t_block_size],
                params);
        }
    }
    PlatformSetPVKVLoadComplete(is_last_split, l1_kv_pingpong_flag);
}

// ----------------------------------------------------------------------------
// LoadPDataToL0A 鈥?P (softmax output) UB -> L1 -> L0A (arch32
// PlatformLoadPFromGMToL1 L265-282 + PlatformLoadPToL0A{Int8,General} L285-308).
// Runs only on the first embed split.
//
// A5 SHARED-UB: the softmax result P is produced by the paired AIV Vector core
// directly into the shared UB (p_ubuf), so the A3 P GM->L1 (MTE2) stage becomes
// a UB->L1 move over the MTE3 producer channel (CopyUbToL1Nd2Nz). UB is per-AI-
// -Core private, so the block_idx GM stride is dropped; only the (n_idx-1)%2
// double-buffer half survives, matching the QK score half in CopyQKResultToUB.
// L1->L0A uses the INT8 (NZ->ZZ) or General (VECTOR) Raw, unchanged.
// ----------------------------------------------------------------------------
template <typename L0aT, typename L1pT, typename PubufT>
__aicore__ __attribute__((always_inline)) inline void LoadPDataToL0A(
    const AscendC::LocalTensor<L0aT> &l0a_buf_tensor,
    const AscendC::LocalTensor<L1pT> &l1p_buf_addr_tensor,
    const AscendC::LocalTensor<PubufT> &p_ubuf_tensor,
    bool is_int8, uint32_t n_idx, uint32_t l0_p_pingpong_flag,
    uint32_t row_num, uint64_t k_round_n, uint32_t qk_round_n_2,
    uint32_t qk_round_n_2_l1, uint32_t t_cube_matrix_size, uint32_t t_block_offset,
    uint32_t t_block_size)
{
    // --- P UB -> L1 (A5 shared-UB, MTE3; same ND->NZ params as the GM path) ---
    PlatformWaitPBeforeLoad();
    CopyUbToL1Nd2Nz(
        l1p_buf_addr_tensor,
        p_ubuf_tensor[(uint64_t)((n_idx - 1) % 2) * TMP_SIZE_DECODER_A5 / 2],
        AscendC::Nd2NzParams(
            /* ndNum            */ 1,
            /* nValue           */ row_num,
            /* dValue           */ (uint32_t)k_round_n,
            /* srcNdMatrixStride*/ 0,
            /* srcDValue        */ qk_round_n_2 * 2 / (uint32_t)sizeof(L1pT),
            /* dstNzC0Stride    */ RoundUp<BLOCK_SIZE_16>(row_num),
            /* dstNzNStride     */ 1,
            /* dstNzMatrixStride*/ 0));
    PlatformSetPL1LoadComplete();

    // --- P L1 -> L0A (INT8 NZ->ZZ / General VECTOR) ---
    PlatformWaitPVL0AReady(l0_p_pingpong_flag);
    if (is_int8) {
        LoadPL1ToL0AInt8Raw(
            l0a_buf_tensor[l0_p_pingpong_flag * 16384],
            l1p_buf_addr_tensor,
            RoundUp<BLOCK_SIZE_16>(row_num),  // mTileCeil
            qk_round_n_2_l1);                 // kPartCeil
    } else {
        const uint32_t p_load_coeff = RoundUp<16>(row_num);
        for (uint64_t loa_load_idx = 0; loa_load_idx < p_load_coeff / BLOCK_SIZE_16; ++loa_load_idx) {
            LoadPL1ToL0AGeneralRaw(
                l0a_buf_tensor[l0_p_pingpong_flag * 16384 + loa_load_idx * qk_round_n_2 * BLOCK_SIZE_16],
                l1p_buf_addr_tensor[loa_load_idx * t_cube_matrix_size],
                qk_round_n_2 / t_block_size,     // repeat (arch32 qk_round_n_2 / T_BLOCK_SIZE)
                p_load_coeff / BLOCK_SIZE_16);   // srcStride
        }
    }
    PlatformSetPVPLoadComplete();
}

// ----------------------------------------------------------------------------
// ComputePVMmad 鈥?P (L0A) x V (L0B) -> mm2 L0C for one embed split
// (arch32 PlatformComputePVMmad L317-334). Fences L0B into the mmad, waits the
// mm2 L0C release, runs the PV mmad, then releases L0B (and, on the last split,
// the L0A P slot).
// ----------------------------------------------------------------------------
template <typename L0cT, typename L0aT, typename L0bT>
__aicore__ __attribute__((always_inline)) inline void ComputePVMmad(
    const AscendC::LocalTensor<L0cT> &mm2_l0c_buf_tensor,
    const AscendC::LocalTensor<L0aT> &l0a_buf_tensor,
    const AscendC::LocalTensor<L0bT> &l0b_buf_tensor,
    uint32_t l0b_pingpong_flag, uint32_t l0c_pingpong_flag, uint32_t l0_p_pingpong_flag,
    uint32_t m, uint32_t embed_split_size, uint32_t qk_n_2, bool is_last_split)
{
    PlatformWaitPVMmad(l0b_pingpong_flag, l0c_pingpong_flag);
    ComputeQKMMadRaw(
        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
        l0a_buf_tensor[l0_p_pingpong_flag * 16384],
        l0b_buf_tensor[l0b_pingpong_flag * 16384],
        m,
        embed_split_size,  // n
        qk_n_2,            // k
        true);             // cmatrixInitVal (arch32 last mmad arg = 1)
    PlatformSetPVMmadComplete(l0b_pingpong_flag, l0_p_pingpong_flag, is_last_split);
}

// ----------------------------------------------------------------------------
// CopyPVResultToGM 鈥?mm2 L0C -> O GM for one embed split (arch32
// PlatformCopyPVResultToGM L337-348). Fences the L0C block against FIX, issues
// the L0C->GM (Fixpipe) with the round_v column count, then releases L0C.
// ----------------------------------------------------------------------------
template <typename OgmT, typename L0cT>
__aicore__ __attribute__((always_inline)) inline void CopyPVResultToGM(
    const AscendC::GlobalTensor<OgmT> &o_tmp_gm_tensor,
    const AscendC::LocalTensor<L0cT> &mm2_l0c_buf_tensor,
    uint32_t l0c_pingpong_flag, uint32_t block_idx, uint32_t embed_split_idx,
    uint32_t n_idx, uint32_t m, uint32_t round_embed_split_size, uint32_t round_v)
{
    PlatformWaitPVL0cForFix(l0c_pingpong_flag);
    CopyQKResultToGMRaw(
        o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER_A5 * 2 +
                        (uint64_t)embed_split_idx * round_embed_split_size +
                        (uint64_t)((n_idx - 1) % 2) * TMP_SIZE_DECODER_A5],
        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
        m,                                // MSize
        round_v,                       // NSize (arch32 passes round_v as N)
        RoundUp<16>(m),                   // srcStride
        RoundUp<16>(round_embed_split_size));  // dstStride
    PlatformSetPVFixComplete(l0c_pingpong_flag);
}

// ----------------------------------------------------------------------------
// ComputePV 鈥?arch32 L1084-1208  (CUBE2 stage1). Pure orchestration (mirrors
// ComputeQK): refresh qk_n_2 on the tail block, run the embed_split_loop_v (=4)
// loop over the sub-functions (V transpose -> P load on split 0 -> PV mmad ->
// O write-back), then publish the PV result to the softmax (Vector) consumer.
// Runs only when n_idx != 0. ctx is non-const: qk_n_2 / qk_round_n_2 refresh.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void ComputePV(
    CubeT &aic, MLAContext &ctx, uint32_t n_idx)
{
    constexpr bool kIsInt8 = CubeT::kIsInt8;
    constexpr uint32_t T_BLOCK_SIZE = CubeT::T_BLOCK_SIZE;
    constexpr uint32_t T_BLOCK_OFFSET = CubeT::T_BLOCK_OFFSET;
    constexpr uint32_t T_CUBE_MATRIX_SIZE = CubeT::T_CUBE_MATRIX_SIZE;

    // ---- InitPVParams (arch32 L236-257): copy from ctx, refresh on tail ----
    uint32_t qk_n_2 = ctx.qk_n_2;
    uint32_t qk_round_n_2 = ctx.qk_round_n_2;
    uint32_t qk_round_n_2_l1 = ctx.qk_round_n_2_l1;
    uint64_t k_round_n = ctx.k_round_n;
    const uint32_t row_num = ctx.row_num;
    const uint64_t hidden_size = ctx.hidden_size;
    const uint32_t m = ctx.m;
    const uint32_t round_v = aic.round_v;

    if (n_idx == ctx.n_loop) {
        qk_n_2 = ctx.cur_kv_seqlen - (n_idx - 1) * ctx.pp_n_scalar;
        qk_round_n_2 = RoundUp<BLOCK_SIZE_16>(qk_n_2);
        qk_round_n_2_l1 = RoundUp<T_BLOCK_SIZE>(qk_n_2);
    }
    k_round_n = qk_round_n_2_l1;

    const uint32_t l1_kv_pingpong_flag = (n_idx - 1) % 2;
    const uint32_t l0_p_pingpong_flag = (n_idx - 1) % 2;
    const uint32_t embed_split_size = 128;
    const uint32_t round_embed_split_size = RoundUp<T_BLOCK_SIZE>(embed_split_size);
    constexpr uint32_t embed_split_loop_v = 4;

    for (uint32_t esi = 0; esi < embed_split_loop_v; ++esi) {
        const uint32_t l0c_pingpong_flag = (n_idx + esi) % 2;
        const uint32_t l0b_pingpong_flag = (esi + 1) % 2;
        const uint64_t l1kv_offset = (uint64_t)esi * k_round_n * round_embed_split_size;
        const bool is_last_split = (esi == embed_split_loop_v - 1);

        LoadVTransposeToL0B(aic.l0b_buf_tensor, aic.l1kv_buf_addr_tensor,
                            l0b_pingpong_flag, l1_kv_pingpong_flag, l1kv_offset,
                            k_round_n, hidden_size, qk_round_n_2, embed_split_size,
                            round_embed_split_size, is_last_split, T_BLOCK_SIZE);

        if (esi == 0) {
            // A5 shared-UB: P comes from p_ubuf (written by the paired AIV
            // softmax), moved UB->L1 over MTE3 instead of read back from p_gm.
            LoadPDataToL0A(aic.l0a_buf_tensor, aic.l1p_buf_addr_tensor, aic.p_ubuf_tensor,
                           kIsInt8, n_idx, l0_p_pingpong_flag,
                         row_num, k_round_n, qk_round_n_2, qk_round_n_2_l1,
                           T_CUBE_MATRIX_SIZE, T_BLOCK_OFFSET, T_BLOCK_SIZE);
        }

        ComputePVMmad(aic.mm2_l0c_buf_tensor, aic.l0a_buf_tensor, aic.l0b_buf_tensor,
                      l0b_pingpong_flag, l0c_pingpong_flag, l0_p_pingpong_flag,
                      m, embed_split_size, qk_n_2, is_last_split);

        CopyPVResultToGM(aic.o_tmp_gm_tensor, aic.mm2_l0c_buf_tensor,
                         l0c_pingpong_flag, (uint32_t)block_idx, esi, n_idx,
                         m, round_embed_split_size, round_v);
    }

    // Publish PV result to the softmax (Vector) consumer (arch32 L1299).
    FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_DECODER);
}

} // namespace MlaArch35
} // namespace XllmOps

#endif // MULTI_LATENT_ATTENTION_BS_ARCH35_H
