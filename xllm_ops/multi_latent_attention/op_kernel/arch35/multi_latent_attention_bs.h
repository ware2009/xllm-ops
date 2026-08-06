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

namespace XllmOps {
namespace MlaArch35 {

// ============================================================================
// MLAContext — per-call scalar state shared across the MLA sub-functions.
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
// All sub-functions are templated on CubeT — the owning
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
// InitMLAContext — arch32 L696-719
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
    // TODO(migrate L696-719): fill ctx.q_offset / q_rope_offset / n_loop /
    // qk_n / qk_round_n / hidden_size / m from aic tiling members.
    (void)aic;
    ctx.cur_batch = cur_batch;
    ctx.start_head = start_head;
    ctx.cur_head_num = cur_head_num;
    ctx.start_kv = start_kv;
    ctx.cur_q_seqlen = cur_q_seqlen;
    ctx.cur_kv_seqlen = cur_kv_seqlen;
    ctx.offset_tiling = offset_tiling;
}

// ----------------------------------------------------------------------------
// LoadQData — arch32 L721-797
//
// Moves Q and Q_rope from GM into L1. Three Q paths:
//   - cur_q_seqlen == 1        : direct gm_to_l1 (ND -> NZ)
//   - q_heads < 128            : single DataCopy with Nd2NzParams
//   - q_heads >= 128           : per-seq DataCopy loop
// Q_rope: INT8 uses gm_to_l1, FP16/BF16 uses DataCopy(Nd2Nz).
// Terminal sync: A3 issued SET/WAIT_FLAG(MTE2, MTE1, EVENT_ID0); on A5 the
// UB<->L1 producer channel maps to MTE3, so this becomes
// SET/WAIT_FLAG(MTE3, MTE1, EVENT_ID0).
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void LoadQData(
    CubeT &aic, const MLAContext &ctx)
{
    // TODO(migrate L721-797): Q + Q_rope GM->L1 with 3-way branch.
    // Terminal event (A3 MTE2 -> A5 MTE3):
    //   SET_FLAG(MTE3, MTE1, EVENT_ID0);
    //   WAIT_FLAG(MTE3, MTE1, EVENT_ID0);
    (void)aic;
    (void)ctx;
}

// ----------------------------------------------------------------------------
// LoadKVData — arch32 L799-896
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
    // TODO(migrate L799-896): K + K_rope GM->L1, 3-way path.
    // A3 SET_FLAG(MTE2, MTE1, ...) / WAIT_FLAG(MTE1, MTE2, ...)
    //   -> A5 uses MTE3 for the UB<->L1 producer edge.
    (void)aic;
    (void)ctx;
    return n_idx % 2;
}

// ----------------------------------------------------------------------------
// ComputeQK — arch32 L897-1081  (CUBE1 stage1)
//
// hidden_split_time embedding-split loop: L1->L0A (Q), L1->L0B (KV), mmad ->
// mm1 L0C, conditional L0C->GM S write-back (INT8 at embed_split_idx==3, others
// at ==4). INT8 has an extra RoPE mmad tail segment (L1005-1081). ctx is a
// non-const reference because qk_n / qk_round_n are refreshed on the tail block.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void ComputeQK(
    CubeT &aic, MLAContext &ctx, uint32_t n_idx, uint32_t l1_kv_pingpong_flag)
{
    // TODO(migrate L897-1081): embedding-split QK GEMM + S write-back.
    (void)aic;
    (void)ctx;
    (void)n_idx;
    (void)l1_kv_pingpong_flag;
}

// ----------------------------------------------------------------------------
// ComputePV — arch32 L1084-1208  (CUBE2 stage1)
//
// Runs when n_idx != 0. Refreshes qk_n_2 on the tail block. embed_split_loop_v
// (=4) loop: V via LoadDataWithTranspose (two branches on k_round_n vs
// round_embed_split_size), P GM->L1->L0A (first iter waits SOFTMAX_READY_DECODER
// cross-core sync), mmad (PV) -> mm2 L0C, L0C->GM O write-back. l0c ping-pong =
// (n_idx + embed_split_idx) % 2.
// ----------------------------------------------------------------------------
template <typename CubeT>
__aicore__ __attribute__((always_inline)) inline void ComputePV(
    CubeT &aic, MLAContext &ctx, uint32_t n_idx)
{
    // TODO(migrate L1084-1208): V transpose load + PV GEMM + O write-back.
    (void)aic;
    (void)ctx;
    (void)n_idx;
}

} // namespace MlaArch35
} // namespace XllmOps

#endif // MULTI_LATENT_ATTENTION_BS_ARCH35_H