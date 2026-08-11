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

#ifndef MULTI_LATENT_ATTENTION_ARCH35_H
#define MULTI_LATENT_ATTENTION_ARCH35_H

#include "kernel_operator.h"
#include "multi_latent_attention_npu.h"
#include "../common/common_utils.h"
#include "../common/common.h"
// NOTE: multi_latent_attention_bs.h is intentionally included later (after the
// BLOCK_SIZE_16 / CONST_* constants below), because its migrated sub-function
// bodies reference BLOCK_SIZE_16, RoundUp (common_utils.h) and SET_FLAG/MTEx
// (common.h) as non-dependent names, which must be visible at template
// definition time (two-phase lookup).

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

// ============================================================================
// Common const values (mirrors arch32 for tiling index compatibility)
// ============================================================================

constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int64_t TMP_SIZE = 65536;
constexpr int32_t BIT_SHIFT = 8;

// FFTS Flags
constexpr int32_t QK_READY = 0;
constexpr int32_t SOFTMAX_READY = 1;
constexpr int32_t UPDATE_READY = 2;
constexpr int32_t QK_READY_DECODER = 3;
constexpr int32_t SOFTMAX_READY_DECODER = 4;
constexpr int32_t UPDATE_READY_DECODER = 5;
constexpr int32_t TAIL_OPTIMIZATION_SYNC = 6;
constexpr int32_t TAIL_OPTIMIZATION_SYNC2 = 7;

// Tiling parameter indices (must match op_host tiling_impl_a5.cpp)
const int32_t TILING_BATCH = 0;
const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_HEADSIZE = 8;
const int32_t TILING_PARASIZE = 9;
const int32_t TILING_MTP_HEAD_SPLIT_SIZE = 10;
const int32_t TILING_TOTAL_BLOCK_NUM = 11;
const int32_t TILING_MASK_TYPE_ND = 12;
const int32_t TILING_MAX_KV_SEQ_LEN = 14;
const int32_t TILING_BLOCKSIZE_CALC = 25;

constexpr uint32_t CONST_16 = 16;
constexpr int32_t MASK_COLUMNS = 128;

template <typename T>
using GlobalT = AscendC::GlobalTensor<T>;

template <typename T>
using LocalT = AscendC::LocalTensor<T>;

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_DATA = 2
};

enum class InputFormat {
    ND_FORMAT = 0,
    NZ_FORMAT = 1
};

enum class BlockStack {
    ONE_FLOW = 0,
    FOUR_FLOW = 1
};

// ============================================================================
// A5 (ascend950 / DAV_3510) hardware constants
//
// Key changes from A3 (C220):
//   - UB:  192 KB -> 248 KB  (UB_UINT8_MAX_SIZE_A5 = 253952)
//   - L0C:  64 KB -> 256 KB (L0C_FLOAT_BUF_SIZE_A5 = 65536 elements)
//   - L0A/B: 64 KB (unchanged)
//   - L1:  512 KB (unchanged)
//
// New on-chip data paths enabled on A5:
//   - FixPipe: L0C -> UB  (GEMM output can flow directly to UB for Vector processing)
//   - MTE3:    UB  -> L1  (UB data can be pushed to L1 for Cube consumption)
//
// These new paths eliminate intermediate GM round-trips that were required
// on A3:  L0C -> GM -> UB (for softmax) and  UB -> GM -> L1 (for GEMM2 input).
//
// A5 sync event mapping (vs A3):
//   A3: SET_FLAG(MTE1, MTE2, ...)  — MTE1 loads L1, MTE2 consumes UB->L1
//   A5: SET_FLAG(MTE1, MTE3, ...)  — MTE1 loads L1, MTE3 handles UB->L1
//   A3: SET_FLAG(MTE2, FIX, ...)   — MTE2 completes, FIX pipe consumes L0C->UB
//   A5: SET_FLAG(MTE3, FIX, ...)   — MTE3 completes, FIX pipe consumes L0C->UB
// ============================================================================

// --- L0A/L0B buffer sizes (64 KB, same as A3) ---
constexpr int32_t L0AB_HALF_BUF_SIZE_A5 = 16384;     // 128 * 128 = 16K elements
constexpr int32_t L0AB_UINT8_BUF_SIZE_A5 = 32768;   // 32 KB

// --- L0C buffer sizes (256 KB, 4x expansion from A3's 64 KB) ---
constexpr int32_t L0C_FLOAT_BUF_SIZE_A5 = 65536;    // 256K / 4B = 64K elements
constexpr int32_t L0C_UINT8_BUF_SIZE_A5 = 262144;  // 256 KB

// --- L1 buffer sizes (512 KB, same as A3) ---
constexpr int32_t L1_UINT8_TOTAL_SIZE_A5 = 524288;  // 512 KB total L1
constexpr int32_t L1_HALF_BUF_SIZE_A5 = 65536;      // 256 * 256
constexpr int32_t L1_HALF_BUF_SIZE_DECODER_A5 = 16384;  // 128 * 128
constexpr int32_t L1_KV_HALF_BUF_SIZE_A5 = 73728;   // 2 * 128 * 256

// --- UB (Unified Buffer) sizes (248 KB, expansion from A3's 192 KB) ---
constexpr uint32_t MAX_UB_SIZE_A5 = 253952;         // 248 * 1024 bytes

// UB block and line sizes (BF16 oriented for MLA decoder)
// UB_UINT8_BLOCK_SIZE_MLA_A5: tile for QK/PV GEMM output staging
//   = 128 rows * 96 cols * 2B(BF16) = 24576 bytes
constexpr int64_t UB_UINT8_BLOCK_SIZE_MLA_A5 = 24576;
// UB_UINT8_LINE_SIZE_A5: single vector line for scalar/temp data
//   = 64 * 4B = 256 bytes (double buffer -> 512)
constexpr int64_t UB_UINT8_LINE_SIZE_A5 = 512;
constexpr int64_t UB_HALF_LINE_SIZE_A5 = 256;
constexpr int64_t UB_FLOAT_LINE_SIZE_A5 = 64;

// Cube matrix dimensions
constexpr int32_t CUBE_MATRIX_SIZE_A5 = 256;        // 16 * 16
constexpr int32_t CUBE_MATRIX_SIZE_512_A5 = 512;
constexpr int32_t BLOCK_SIZE_16 = 16;
constexpr int64_t CONST_4 = 4;
constexpr int64_t CONST_32 = 32;
constexpr int64_t CONST_64 = 64;
constexpr int64_t CONST_128 = 128;

// Decoder-specific constants (must be defined BEFORE including bs.h, because the
// bs.h sub-functions CopyQKResultToGM / ComputeQRope reference TMP_SIZE_DECODER_A5).
constexpr int32_t TMP_SIZE_DECODER_A5 = 32768;

// Now that BLOCK_SIZE_16 / CONST_* / TMP_SIZE_DECODER_A5 and the common headers
// (RoundUp, SET_FLAG, MTEx HardEvents) are all visible, pull in the
// InnerRunCubeMLA sub-functions.
#include "multi_latent_attention_bs.h"

// ============================================================================
// UbufAllocA5 — UB 248KB space layout for A5 MLA Decoder (Cube side)
//
// Layout strategy:
//   [0,     2*BLK)   : ls32 (GEMM1 QK score, float, double buffer)      48 KB
//   [2*BLK, 4*BLK)   : lp / mask / lo (softmax staging area)            48 KB
//   [4*BLK, 6*BLK)   : ls16 / lm32 (low-precision scores)                48 KB
//   [6*BLK, 6*BLK+8*LINE) : hm32/pm32/dm32/ll/gm32/gl (scalar & temp)   ~4 KB
//   [8*BLK, 10*BLK)  : go32 (GEMM2 PV output via FixPipe L0C->UB)       48 KB
//   [10*BLK, 12*BLK) : tv32 (temp buffer for V transpose / staging)      48 KB
//
// Total used: ~244 KB (within 248 KB limit, 4 KB reserved for alignment)
//
// Key difference from A3:
//   - go32 uses FixPipe (L0C->UB) instead of L0C->GM->UB round-trip
//   - tv32 can optionally use MTE3 (UB->L1) to feed GEMM2 without GM hop
// ============================================================================
template<BlockStack blockStack>
struct UbufAllocA5 {
};

template<>
struct UbufAllocA5<BlockStack::ONE_FLOW> {
    // GEMM1 QK score (float, double buffer)
    const uint32_t ls32_ubuf_offset = 0;
    const uint32_t ls32_quant_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA_A5;

    // Softmax staging
    const uint32_t lp_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t lp32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t mask32_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA_A5;

    // Mid-precision and scalar region
    const uint32_t lm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t hm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 1 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t pm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 2 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t pm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 3 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t descale1_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 4 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t descale2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 5 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t dm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 6 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t dm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 7 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t ll_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 9 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t ll_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 11 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t gm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 13 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t gl_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 15 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t gl32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 15 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t p_scale_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 17 * UB_UINT8_LINE_SIZE_A5;

    // GEMM2 PV output: staged via FixPipe (L0C -> UB), no GM round-trip
    const uint32_t go_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t go32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA_A5;

    // Temp / V staging: can use MTE3 (UB -> L1) to feed GEMM2 without GM hop
    const uint32_t tv32_ubuf_offset = 10 * UB_UINT8_BLOCK_SIZE_MLA_A5;
};

namespace XllmOps {
namespace MlaArch35 {

// ============================================================================
// AttentionType — type aliases for GEMM intermediate dtypes (A5 BF16 path)
// ============================================================================
template<TilingKeyType tilingKeyType>
struct AttentionTypeA5 {
};

template<>
struct AttentionTypeA5<TilingKeyType::TILING_HALF_DATA> {
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

template<>
struct AttentionTypeA5<TilingKeyType::TILING_BF16_DATA> {
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

// INT8 path: GEMM outputs are int32_t (quantized matmul)
// BF16 end-to-end first; INT8 support will be enabled after BF16 validation
template<>
struct AttentionTypeA5<TilingKeyType::TILING_INT8_DATA> {
    using mm1OutputType = int32_t;
    using mm1CopyType = int32_t;
    using mm2OutputType = int32_t;
    using mm2CopyType = int32_t;
};

// ============================================================================
// MLAttentionDecoderAic — A5 (ascend950 / DAV_3510) Cube core
//
// Template parameters mirror the A3 class for transparent dispatch.
// A5 adaptation highlights (SetArgs level):
//   - Retains INT8 L1 buffer path for future INT8 support (BF16 prioritized first)
//   - Uses A5 hardware constants (L0C 256KB, UB 248KB)
//   - L1 buffer offsets same layout as A3 (Q/KV/P regions unchanged)
//   - embed_split_size_qk reads from tiling param [25] (TILING_BLOCKSIZE_CALC)
//     which A5 tiling fills with 256 (up from A3's hardcoded 128)
// ============================================================================
// NOTE: template param order MUST match arch32's
// <IN_DTYPE, IN_ROPE_DTYPE, OUT_DTYPE, IN_KVDTYPE> so that the cpp entry-point
// instantiations (kept identical to arch32) resolve to the correct semantics.
// INT8 path: IN_DTYPE=int8, IN_ROPE_DTYPE=float(RoPE NOT quantized),
//            OUT_DTYPE=float, IN_KVDTYPE=int8.
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half,
          typename IN_ROPE_DTYPE = half,
          typename OUT_DTYPE = half,
          typename IN_KVDTYPE = half,
          InputFormat KInputType = InputFormat::ND_FORMAT,
          bool EnableOptimization = false>
class MLAttentionDecoderAic {
    // Type aliases for GEMM intermediates
    using mm1OutputType = typename AttentionTypeA5<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionTypeA5<tilingKeyType>::mm1CopyType;
    using mm2OutputType = typename AttentionTypeA5<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionTypeA5<tilingKeyType>::mm2CopyType;

    static constexpr uint32_t T_CUBE_MATRIX_SIZE = CUBE_MATRIX_SIZE_512_A5 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_SIZE = BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);
    static constexpr int32_t L1_KV_HALF_SIZE = L1_KV_HALF_BUF_SIZE_A5;  // 73728

public:
    // Expose template tiling key so bs.h helper sub-functions can branch on INT8 path
    static constexpr TilingKeyType kTilingKeyType = tilingKeyType;
    static constexpr bool kIsInt8 = (tilingKeyType == TilingKeyType::TILING_INT8_DATA);
    // Expose the KV input layout so bs.h LoadKVData can pick the ND->NZ vs
    // NZ->NZ helper path at compile time (mirrors arch32's KInputType branch).
    static constexpr InputFormat kInputFormat = KInputType;

    // Expose template dtype parameters so bs.h helper sub-functions can name
    // them (e.g. ReinterpretCast<IN_ROPE_DTYPE>) via CubeT::*_ALIAS.
    // (mm1OutputType/mm1CopyType are private but reachable by friends directly,
    //  so no alias is needed for them.)
    using IN_DTYPE_ALIAS = IN_DTYPE;
    using IN_KVDTYPE_ALIAS = IN_KVDTYPE;
    using IN_ROPE_DTYPE_ALIAS = IN_ROPE_DTYPE;

    __aicore__ __attribute__((always_inline)) inline MLAttentionDecoderAic() {}

    // ========================================================================
    // SetArgs — Bind GM pointers, set up GlobalTensor/L1 buffers, read tiling
    //
    // A5 changes from A3:
    //   1. Retains INT8 L1 buffer allocation branch (BF16 prioritized, INT8 deferred)
    //   2. L1 buffer offsets use same layout as A3 (Q/KV/P regions unchanged)
    //   3. embed_split_size_qk reads from TILING_BLOCKSIZE_CALC (index 25),
    //      which A5 tiling fills with 256 (vs A3 hardcoded 128)
    //   4. stride_kv / stride_kv_rope unchanged (512 / 64 are MLA architectural)
    // ========================================================================
    __aicore__ __attribute__((always_inline)) inline void SetArgs(
        __gm__ uint8_t *__restrict__ q_in_gm,
        __gm__ uint8_t *__restrict__ q_rope_in_gm,
        __gm__ uint8_t *__restrict__ k_in_gm,
        __gm__ uint8_t *__restrict__ k_rope_in_gm,
        __gm__ uint8_t *__restrict__ block_tables_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ s_out_gm,
        __gm__ uint8_t *__restrict__ s_rope_out_gm,
        __gm__ uint8_t *__restrict__ p_out_gm,
        __gm__ uint8_t *__restrict__ o_temp_gm,
        __gm__ uint8_t *__restrict__ tiling_para_gm)
    {
        // --- A5 hardware state setup ---
        SetPadding<uint64_t>(0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        SetMasknorm();

        // --- Bind raw GM pointers ---
        q_gm = reinterpret_cast<__gm__ IN_DTYPE *>(q_in_gm);
        q_rope_gm = reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(q_rope_in_gm);
        k_gm = reinterpret_cast<__gm__ IN_KVDTYPE *>(k_in_gm);
        k_rope_gm = reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(k_rope_in_gm);
        block_tables_gm = reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm);
        s_gm = reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm);
        p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm);
        o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm);
        tiling_gm = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);

        // --- Bind GlobalTensor wrappers (for Nd2Nz / DataCopy operations) ---
        q_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(q_in_gm));
        q_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(q_rope_gm));
        k_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_KVDTYPE *>(k_gm));
        k_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_ROPE_DTYPE *>(k_rope_gm));
        s_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm));
        p_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm));
        o_tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm));
        block_tables_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm));

        // --- A5 Buffer initialization (TPipe + TBuf + InitBuffer + Get) ---
        // L1: single TBuf<TPosition::A1>, sub-tensors via tensor[offset]
        // L0A/L0B/L0C: separate TBuf, single Get (no offset needed)
        pipe.InitBuffer(l1TBuf, L1_UINT8_TOTAL_SIZE_A5);   // L1 = 512KB
        pipe.InitBuffer(l0aTBuf, L0AB_UINT8_BUF_SIZE_A5);  // L0A = 32KB
        pipe.InitBuffer(l0bTBuf, L0AB_UINT8_BUF_SIZE_A5);  // L0B = 32KB
        pipe.InitBuffer(l0cTBuf, L0C_UINT8_BUF_SIZE_A5);   // L0C = 256KB

        // Get base tensors (required before sub-buffer offset assignment)
        auto l1BaseTensor = l1TBuf.Get<uint8_t>();
        l0a_buf_tensor = l0aTBuf.Get<IN_DTYPE>();
        l0b_buf_tensor = l0bTBuf.Get<IN_DTYPE>();
        // L0C is shared between mm1 and mm2 via ping-pong offset
        auto l0cBaseTensor = l0cTBuf.Get<uint8_t>();
        // L0C total size in bytes = L0C_UINT8_BUF_SIZE_A5; each half for ping-pong
        constexpr uint32_t l0cHalfSize = L0C_UINT8_BUF_SIZE_A5 / 2;
        mm1_l0c_buf_tensor = l0cBaseTensor.ReinterpretCast<mm1OutputType>();
        mm2_l0c_buf_tensor = l0cBaseTensor[l0cHalfSize].ReinterpretCast<mm2OutputType>();

        // --- L1 sub-buffer allocation ---
        // BF16/HALF path: Q+RoPE contiguous in l1q, KV+RoPE contiguous in l1kv
        // INT8 path: separate Q/Q_rope/KV/KV_rope buffers with dedicated offsets
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            s_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(s_rope_out_gm));
            l1q_buf_addr_tensor = l1BaseTensor.ReinterpretCast<IN_DTYPE>();
            l1q_rope_buf_addr_tensor = l1BaseTensor[128 * 512 * 2].ReinterpretCast<IN_ROPE_DTYPE>();
            l1kv_buf_addr_tensor = l1BaseTensor[128 * 576 * 2].ReinterpretCast<IN_KVDTYPE>();
            l1kv_rope_buf_addr_tensor = l1BaseTensor[128 * 576 * 2 + 128 * 512 * 2].ReinterpretCast<IN_ROPE_DTYPE>();
            l1p_buf_addr_tensor = l1BaseTensor[128 * 576 * 6].ReinterpretCast<IN_DTYPE>();
        } else {
            l1q_buf_addr_tensor = l1BaseTensor[l1q_buf_addr_offset].ReinterpretCast<IN_DTYPE>();
            l1kv_buf_addr_tensor = l1BaseTensor[l1kv_buf_addr_offset].ReinterpretCast<IN_KVDTYPE>();
            l1p_buf_addr_tensor = l1BaseTensor[l1p_buf_addr_offset].ReinterpretCast<IN_DTYPE>();
        }

        // --- Read tiling parameters from GM ---
        num_batches = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm));
        q_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_NUMHEADS));
        embedding_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM));
        block_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE));
        max_num_blocks_per_query = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAXBLOCKS));
        kv_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVHEADS));
        tiling_head_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADSIZE));
        tiling_para_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_PARASIZE));
        cur_qn_blk_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MTP_HEAD_SPLIT_SIZE));
        block_size_calc = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE_CALC));
        mask_type = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MASK_TYPE_ND));
        totalTaskNum = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 13));  // TILING_TASK_NUM
        maxKVSeqLen = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAX_KV_SEQ_LEN));

        num_batches_pad = RoundUp<16>(num_batches);

        // --- MLA architectural constants ---
        // stride_kv = kv_heads * 512 (compressed KV dim = 512 for MLA)
        stride_kv = static_cast<uint64_t>(kv_heads) * 512;
        // stride_kv_rope = kv_heads * 64 (decoupled RoPE dim = 64 for MLA)
        stride_kv_rope = static_cast<uint64_t>(kv_heads) * 64;

        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);
        __v = embedding_size;
        stride_vo = static_cast<uint64_t>(kv_heads) * embedding_size;
        round_v = RoundUp<BLOCK_SIZE_16>(__v);

        // A5: embed_split_size_qk comes from tiling param [25] (TILING_BLOCKSIZE_CALC)
        // A5 tiling fills this with 256 (up from A3's hardcoded 128)
        // If tiling didn't fill it, default to 128 for safety
        embed_split_size_qk = (block_size_calc > 0) ? block_size_calc : 128;
        embed_split_loop_qk = (embedding_size + embed_split_size_qk - 1) / embed_split_size_qk;
    }

    // ========================================================================
    // Run — A5 Cube main loop (Normal / non-TP1 path)
    //
    // Sync event layout (8-event ping-pong pipeline):
    //
    //   Init phase (pre-set all flags for double-buffered pipeline):
    //     M -> MTE1  x8  (Scalar releases MTE1 to load L1 from GM)
    //     FIX -> M   x2  (FixPipe releases Scalar after L0C->UB)
    //     MTE1 -> MTE3 x8 (MTE1 releases MTE3 for UB->L1 data transfer)
    //     FIX -> MTE1 x6 (FixPipe releases MTE1 for next iteration L1 load)
    //     MTE3 -> FIX x1 (MTE3 releases FixPipe for L0C->UB consumption)
    //
    //   A5 change from A3:
    //     A3 uses MTE2 for UB->L1:  SET_FLAG(MTE1, MTE2, ...) / SET_FLAG(MTE2, FIX, ...)
    //     A5 uses MTE3 for UB->L1:  SET_FLAG(MTE1, MTE3, ...) / SET_FLAG(MTE3, FIX, ...)
    //     FixPipe (FIX) path remains identical: L0C -> UB
    //
    //   Task loop:
    //     Each block processes: for (process = block_idx; process < process_num;
    //                              process += block_num)
    //     Reads per-batch tiling params (qSeqLen, kvSeqlen) from GM
    //     Calls InnerRunCubeMLA (TODO: A5 inner implementation)
    //
    //   Cleanup phase (wait all flags + barrier):
    //     Mirror of init phase WAIT_FLAGs + PIPE_BARRIER(ALL)
    // ========================================================================
    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        // --- Init: pre-set sync flags for 8-event ping-pong pipeline ---
        // M -> MTE1: Scalar tells MTE1 it can start loading L1 from GM
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID4);
        SET_FLAG(M, MTE1, EVENT_ID5);
        SET_FLAG(M, MTE1, EVENT_ID6);
        SET_FLAG(M, MTE1, EVENT_ID7);

        // FIX -> M: FixPipe (L0C->UB) releases Scalar for next GEMM setup
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);

        // MTE1 -> MTE3: MTE1 (L1 load) releases MTE3 for UB->L1 transfer
        // (A3 used MTE2 here; A5 uses MTE3 for the new UB->L1 on-chip path)
        SET_FLAG(MTE1, MTE3, EVENT_ID0);
        SET_FLAG(MTE1, MTE3, EVENT_ID1);
        SET_FLAG(MTE1, MTE3, EVENT_ID2);
        SET_FLAG(MTE1, MTE3, EVENT_ID3);
        SET_FLAG(MTE1, MTE3, EVENT_ID4);
        SET_FLAG(MTE1, MTE3, EVENT_ID5);
        SET_FLAG(MTE1, MTE3, EVENT_ID6);
        SET_FLAG(MTE1, MTE3, EVENT_ID7);

        // FIX -> MTE1: FixPipe releases MTE1 for next iteration's L1 reload
        SET_FLAG(FIX, MTE1, EVENT_ID0);
        SET_FLAG(FIX, MTE1, EVENT_ID1);
        SET_FLAG(FIX, MTE1, EVENT_ID2);
        SET_FLAG(FIX, MTE1, EVENT_ID3);
        SET_FLAG(FIX, MTE1, EVENT_ID4);
        SET_FLAG(FIX, MTE1, EVENT_ID5);

        // M -> FIX: Scalar releases FixPipe for L0C->UB consumption
        // (A3 used MTE2->FIX; A5 has no MTE3->FIX, use M->FIX instead)
        SET_FLAG(M, FIX, EVENT_ID0);

        // --- Task loop ---
        uint64_t cur_batch = 0;
        uint32_t q_block_num_per_batch = (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
        uint32_t process_num = q_block_num_per_batch * num_batches;

        for (uint32_t process = block_idx; process < process_num; process += (uint32_t)block_num) {
            cur_batch = process / q_block_num_per_batch;
            if (cur_batch >= num_batches) {
                break;
            }

            // Read per-batch tiling parameters from GM
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            uint32_t q_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }

            uint32_t start_head = (process % q_block_num_per_batch) * cur_qn_blk_size;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = cur_qn_blk_size;

            InnerRunCubeMLA(cur_batch, start_head, cur_head_num, start_kv,
                            cur_q_seq_len, cur_kv_seqlen, offset_tiling);
        }

        // --- Cleanup: wait all sync flags ---
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID4);
        WAIT_FLAG(M, MTE1, EVENT_ID5);
        WAIT_FLAG(M, MTE1, EVENT_ID6);
        WAIT_FLAG(M, MTE1, EVENT_ID7);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID7);
        WAIT_FLAG(FIX, MTE1, EVENT_ID0);
        WAIT_FLAG(FIX, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, MTE1, EVENT_ID2);
        WAIT_FLAG(FIX, MTE1, EVENT_ID3);
        WAIT_FLAG(FIX, MTE1, EVENT_ID4);
        WAIT_FLAG(FIX, MTE1, EVENT_ID5);
        WAIT_FLAG(M, FIX, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

    // ========================================================================
    // RunTP1 — A5 Cube main loop (MTP / TP1 path)
    //
    // Structure mirrors Run() with TP1-specific task scheduling:
    //   - Uses totalTaskNum instead of q_block_num_per_batch * num_batches
    //   - Per-task tiling: seqIdx, prevTaskNum, effectiveKVLen
    //   - Tail optimization for uneven task/core distribution
    //
    // Sync events are identical to Run() (MTE3 replaces A3's MTE2).
    // InnerRunCubeMLATP1 is left as TODO.
    // ========================================================================
    __aicore__ __attribute__((always_inline)) inline void RunTP1()
    {
        // --- Init: pre-set sync flags (identical to Run) ---
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID4);
        SET_FLAG(M, MTE1, EVENT_ID5);
        SET_FLAG(M, MTE1, EVENT_ID6);
        SET_FLAG(M, MTE1, EVENT_ID7);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);
        SET_FLAG(MTE1, MTE3, EVENT_ID0);
        SET_FLAG(MTE1, MTE3, EVENT_ID1);
        SET_FLAG(MTE1, MTE3, EVENT_ID2);
        SET_FLAG(MTE1, MTE3, EVENT_ID3);
        SET_FLAG(MTE1, MTE3, EVENT_ID4);
        SET_FLAG(MTE1, MTE3, EVENT_ID5);
        SET_FLAG(MTE1, MTE3, EVENT_ID6);
        SET_FLAG(MTE1, MTE3, EVENT_ID7);
        SET_FLAG(FIX, MTE1, EVENT_ID0);
        SET_FLAG(FIX, MTE1, EVENT_ID1);
        SET_FLAG(FIX, MTE1, EVENT_ID2);
        SET_FLAG(FIX, MTE1, EVENT_ID3);
        SET_FLAG(FIX, MTE1, EVENT_ID4);
        SET_FLAG(FIX, MTE1, EVENT_ID5);
        SET_FLAG(M, FIX, EVENT_ID0);

        // --- Tail optimization setup ---
        uint32_t tail = totalTaskNum % block_num;
        if constexpr (EnableOptimization) {
            // tail optimization enabled — keep computed tail
        } else {
            tail = 0;  // disable tail optimization
        }
        uint32_t totalTaskNumRound = totalTaskNum - tail;

        // --- Main task loop ---
        for (uint32_t process = block_idx; process < totalTaskNumRound; process += (uint32_t)block_num) {
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
            uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

            uint32_t q_seqlen = 1;
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }

            uint32_t start_head = 0;
            uint32_t start_kv = 0;
            uint32_t cur_q_seq_len = q_seqlen;
            uint32_t cur_kv_seqlen = kv_seqlen;
            uint32_t cur_head_num = q_heads;

            // TODO: A5 InnerRunCubeMLATP1 implementation
            // InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv,
            //                    cur_q_seq_len, cur_kv_seqlen, offset_tiling);
        }

        // --- Tail optimization: remaining tasks after round ---
        if (tail > 0) {
            // Process remaining tail tasks (simplified: single-core per task)
            uint32_t process = totalTaskNumRound + block_idx;
            if (process < totalTaskNum) {
                uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

                uint32_t q_seqlen = 1;
                uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                if (kv_seqlen > 0) {
                    uint32_t start_head = 0;
                    uint32_t start_kv = 0;
                    uint32_t cur_q_seq_len = q_seqlen;
                    uint32_t cur_kv_seqlen = kv_seqlen;
                    uint32_t cur_head_num = q_heads;

                    // TODO: A5 InnerRunCubeMLATP1 implementation
                    // InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv,
                    //                    cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                }
            }
        }

        // --- Cleanup: wait all sync flags (identical to Run) ---
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID4);
        WAIT_FLAG(M, MTE1, EVENT_ID5);
        WAIT_FLAG(M, MTE1, EVENT_ID6);
        WAIT_FLAG(M, MTE1, EVENT_ID7);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE3, EVENT_ID7);
        WAIT_FLAG(FIX, MTE1, EVENT_ID0);
        WAIT_FLAG(FIX, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, MTE1, EVENT_ID2);
        WAIT_FLAG(FIX, MTE1, EVENT_ID3);
        WAIT_FLAG(FIX, MTE1, EVENT_ID4);
        WAIT_FLAG(FIX, MTE1, EVENT_ID5);
        WAIT_FLAG(M, FIX, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

private:
    // ========================================================================
    // Friend declarations — MLA building-block sub-functions (bs.h).
    //
    // These free-function templates take *this (as CubeT&) so they can reach
    // the private tensor / tiling members below. Signatures MUST stay in sync
    // with multi_latent_attention_bs.h.
    // ========================================================================
    template <typename CubeT>
    friend __aicore__ void InitMLAContext(CubeT &, MLAContext &, uint32_t, uint32_t,
                                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    template <typename CubeT>
    friend __aicore__ void LoadQData(CubeT &, const MLAContext &);
    template <typename CubeT>
    friend __aicore__ void LoadQMainFromGMToL1(CubeT &, const MLAContext &);
    template <typename CubeT>
    friend __aicore__ void LoadQRopeFromGMToL1(CubeT &, const MLAContext &);
    template <typename CubeT>
    friend __aicore__ uint32_t LoadKVData(CubeT &, MLAContext &, uint32_t);
    template <typename CubeT>
    friend __aicore__ void ComputeQK(CubeT &, MLAContext &, uint32_t, uint32_t);
    template <typename CubeT>
    friend __aicore__ void ComputePV(CubeT &, MLAContext &, uint32_t);

    // ========================================================================
    // InnerRunCubeMLA — A5 Cube MLA per-task business-flow skeleton.
    //
    // Mirrors arch32 InnerRunCubeMLA (L693-1212) but keeps ONLY the control
    // flow: init context, load Q, then iterate KV blocks issuing QK (stage1)
    // and PV (stage2) with the pipelined cross-core sync. All heavy lifting is
    // delegated to the bs.h sub-functions. n_idx runs [0, n_loop]:
    //   - n_idx <  n_loop : LoadKVData + ComputeQK for block n_idx
    //   - n_idx >= 1      : ComputePV consumes the block prepared last round
    // so LoadKV/QK of block k overlaps with PV of block k-1 (software pipeline).
    // ========================================================================
    __aicore__ __attribute__((always_inline)) inline void InnerRunCubeMLA(
        uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num, uint32_t start_kv,
        uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t offset_tiling)
    {
        MLAContext ctx;
        InitMLAContext(*this, ctx, cur_batch, start_head, cur_head_num, start_kv,
                       cur_q_seqlen, cur_kv_seqlen, offset_tiling);

        // --- Load Q / Q_rope into L1 (blocks until Q is resident) ---
        LoadQData(*this, ctx);

        // --- KV-block software pipeline ---
        for (uint32_t n_idx = 0; n_idx < ctx.n_loop + 1; ++n_idx) {
            if (n_idx != ctx.n_loop) {
                // Stage1: load this KV block, run QK GEMM1, publish QK result.
                uint32_t l1_kv_pingpong_flag = LoadKVData(*this, ctx, n_idx);
                ComputeQK(*this, ctx, n_idx, l1_kv_pingpong_flag);
                // QK_READY_DECODER is now published inside ComputeQK (bs.h).
            }

            // Stage2: PV GEMM2 for the previously prepared block (n_idx != 0
            // guarded inside ComputePV, which also publishes UPDATE_READY).
            ComputePV(*this, ctx, n_idx);
        }
    }

    // --- Raw GM pointers ---
    __gm__ IN_DTYPE *__restrict__ q_gm{nullptr};
    __gm__ IN_ROPE_DTYPE *__restrict__ q_rope_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ k_gm{nullptr};
    __gm__ IN_ROPE_DTYPE *__restrict__ k_rope_gm{nullptr};
    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ int32_t *__restrict__ block_tables_gm{nullptr};
    __gm__ uint8_t *__restrict__ tiling_gm{nullptr};

    // --- GlobalTensor wrappers ---
    AscendC::GlobalTensor<IN_DTYPE> q_gm_tensor;
    AscendC::GlobalTensor<IN_ROPE_DTYPE> q_rope_gm_tensor;
    AscendC::GlobalTensor<IN_KVDTYPE> k_gm_tensor;
    AscendC::GlobalTensor<IN_ROPE_DTYPE> k_rope_gm_tensor;
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor;
    AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor;
    AscendC::GlobalTensor<int32_t> block_tables_gm_tensor;
    AscendC::GlobalTensor<float> s_rope_gm_tensor;  // INT8 path: QK RoPE score output

    // --- L1 buffer offsets ---
    // BF16/HALF path: Q+RoPE contiguous, KV+RoPE contiguous (3 regions)
    // INT8 path: Q/Q_rope/KV/KV_rope separate (5 regions)
    // CONSTRAINT: l1q region capacity = (l1kv_buf_addr_offset - l1q_buf_addr_offset)
    //   = 147456 bytes (~144 rows of half). LoadQData writes up to
    //   RoundUp<16>(cur_head_num * cur_q_seqlen) * 512 * sizeof(IN_DTYPE) bytes here.
    //   Host tiling MUST keep head_num * seqlen block so this upper bound stays
    //   <= 147456; otherwise A5's enlarged tile silently overruns into l1kv region.
    const uint32_t l1q_buf_addr_offset = 0;
    const uint32_t l1q_rope_buf_addr_offset = 65536;
    const uint32_t l1kv_buf_addr_offset = 147456;
    const uint32_t l1kv_rope_buf_addr_offset = 278528;
    const uint32_t l1p_buf_addr_offset = 442368;

    // --- A5 Buffer management (TPipe + TBuf<TPosition>) ---
    // A5 replaces A2's AsdopsBuffer<ArchType::ASCEND_V220> with the standard
    // TPipe + TBuf<TPosition::xxx> pattern. TPosition enum mapping:
    //   A1=L1, A2=L0A, B2=L0B, CO1=L0C
    // L1 is a single large TBuf; sub-buffers are carved out via tensor[offset].
    // L0A/L0B/L0C each get their own TBuf (separate address spaces on A5).
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> l1TBuf;    // L1 unified buffer
    AscendC::TBuf<AscendC::TPosition::A2> l0aTBuf;   // L0A
    AscendC::TBuf<AscendC::TPosition::B2> l0bTBuf;    // L0B
    AscendC::TBuf<AscendC::TPosition::CO1> l0cTBuf;  // L0C (shared by mm1/mm2)

    // L1 sub-tensors (assigned in SetArgs after pipe.InitBuffer)
    AscendC::LocalTensor<IN_DTYPE> l1q_buf_addr_tensor;
    AscendC::LocalTensor<IN_ROPE_DTYPE> l1q_rope_buf_addr_tensor;
    AscendC::LocalTensor<IN_KVDTYPE> l1kv_buf_addr_tensor;
    AscendC::LocalTensor<IN_ROPE_DTYPE> l1kv_rope_buf_addr_tensor;
    AscendC::LocalTensor<IN_DTYPE> l1p_buf_addr_tensor;

    // L0A/L0B/L0C tensors (assigned in SetArgs after pipe.InitBuffer)
    AscendC::LocalTensor<IN_DTYPE> l0a_buf_tensor;
    AscendC::LocalTensor<IN_DTYPE> l0b_buf_tensor;
    AscendC::LocalTensor<mm1OutputType> mm1_l0c_buf_tensor;
    AscendC::LocalTensor<mm2OutputType> mm2_l0c_buf_tensor;

    // --- Tiling parameters (read from GM in SetArgs) ---
    uint32_t num_batches{0};
    uint32_t q_heads{0};
    uint32_t kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t block_size{0};
    uint32_t max_num_blocks_per_query{0};
    uint32_t stride_kv{0};
    uint32_t stride_kv_rope{0};
    uint32_t stride_vo{0};
    uint32_t m{0};
    uint32_t __k{0};
    uint32_t __v{0};
    uint32_t round_k{0};
    uint32_t round_v{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t block_size_calc{0};
    uint32_t mask_type{0};
    uint32_t totalTaskNum{0};
    uint32_t maxKVSeqLen{0};

    uint32_t cur_qn_blk_size{0};
    uint32_t num_batches_pad{0};

    uint32_t embed_split_size_qk{0};
    uint32_t embed_split_loop_qk{1};

    // --- Ping-pong flags ---
    uint32_t l1_pingpong_flag = 0;
    uint32_t l1b_pingpong_flag = 0;
    uint32_t l0_pingpong_flag = 0;
    uint32_t l0b_pingpong_flag = 0;
    uint32_t l0c_pingpong_flag = 0;
    uint32_t l1p_pingpong_flag = 0;

    // --- Ping-pong offsets ---
    uint32_t l1_offset = l1_pingpong_flag * L1_HALF_BUF_SIZE_DECODER_A5 / sizeof(IN_DTYPE);
    uint32_t l1b_offset = l1b_pingpong_flag * L1_KV_HALF_BUF_SIZE_A5 / sizeof(IN_DTYPE);
    uint32_t l0_offset = l0_pingpong_flag * L0AB_UINT8_BUF_SIZE_A5 / sizeof(IN_DTYPE);
    uint32_t l0c_offset = l0c_pingpong_flag * L0C_FLOAT_BUF_SIZE_A5;
    uint32_t l0b_offset = l0b_pingpong_flag * L0AB_UINT8_BUF_SIZE_A5 / sizeof(IN_DTYPE);
    uint32_t l1p_start_offset = l1p_pingpong_flag * L1_HALF_BUF_SIZE_A5 / sizeof(IN_DTYPE);
};

// ============================================================================
// MLADecoderAiv — A5 (ascend950 / DAV_3510) Vector core
//
// A5 adaptation for Vector side:
//   - UB 248KB allows larger softmax/p staging tiles
//   - Uses UbufAllocA5<BlockStack> for offset management
//   - FixPipe L0C->UB enables direct GEMM output consumption in Vector
// ============================================================================
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half,
          typename OUT_DTYPE = half,
          bool IsRing = false,
          BlockStack BlockFlow = BlockStack::ONE_FLOW,
          bool IsTP1 = false>
class MLADecoderAiv {
    using mm1OutputType = typename AttentionTypeA5<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionTypeA5<tilingKeyType>::mm1CopyType;
    using mm2OutputType = typename AttentionTypeA5<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionTypeA5<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_BLOCK_SIZE = BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);

public:
    __aicore__ __attribute__((always_inline)) inline MLADecoderAiv() {}

    __aicore__ __attribute__((always_inline)) inline void SetArgs(
        __gm__ uint8_t *__restrict__ gm_block_table,
        __gm__ uint8_t *__restrict__ deq_qk_in_gm,
        __gm__ uint8_t *__restrict__ deq_pv_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ s_out_gm,
        __gm__ uint8_t *__restrict__ s_rope_out_gm,
        __gm__ uint8_t *__restrict__ p_out_gm,
        __gm__ uint8_t *__restrict__ o_temp_gm,
        __gm__ uint8_t *__restrict__ globalo_gm,
        __gm__ uint8_t *__restrict__ tmp_gm,
        __gm__ uint8_t *__restrict__ tiling_para_gm,
        __gm__ uint8_t *__restrict__ mask_input_gm)
    {
        // A5 Vector SetArgs: bind output/staging GM buffers and load tiling params.
        // Mapped from arch32 (A3) MLADecoderAiv::SetArgs. GM binding + tiling read are
        // platform-agnostic; vector-mask/atomic setup uses A5 (dav-3510) intrinsics.
        sub_block_idx = static_cast<uint64_t>(GetSubBlockidx());
        SetAtomicnone();
        SetMasknorm();
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        o_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(o_out_gm);
        s_gm = reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm);
        p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm);
        o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm);
        go_gm = reinterpret_cast<__gm__ float *>(globalo_gm);
        tiling_gm = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);
        gm_block_tables_ = reinterpret_cast<__gm__ int32_t *>(gm_block_table);
        o_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(o_gm));
        mask_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(mask_input_gm));
        s_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm1CopyType *>(s_gm));
        p_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(p_gm));
        o_tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ mm2CopyType *>(o_tmp_gm));
        go_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(go_gm));
        tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tmp_gm));
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            deq_scale_gm_tensor_q1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(deq_qk_in_gm));
            deq_scale_gm_tensor_k1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(deq_pv_in_gm));
            s_rope_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(s_rope_out_gm));
        }

        num_batches = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm));
        q_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_NUMHEADS));
        embedding_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM));
        block_size = (int32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE));
        max_num_blocks_per_query = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAXBLOCKS));
        tor = (float)(*((__gm__ float *)tiling_para_gm + TILING_TOR));
        num_kv_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVHEADS));
        tiling_head_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADSIZE));
        tiling_para_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_PARASIZE));
        totalTaskNum = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + 13));
        maxKVSeqLen = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAX_KV_SEQ_LEN));

        cur_qn_blk_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MTP_HEAD_SPLIT_SIZE));
        block_size_calc = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE_CALC));
        mask_type = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MASK_TYPE_ND));

        go_flag_scalar = 1;
        gl_flag_scalar = 1;

        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);
        __v = embedding_size;
        round_v = RoundUp<BLOCK_SIZE>(__v);
    }

    __aicore__ __attribute__((always_inline)) inline void SetArgs2(
        __gm__ uint8_t *__restrict__ lse_out_gm)
    {
        // A5 Vector SetArgs2: bind LSE (log-sum-exp) output GM for ring mode.
        // Mapped from arch32 (A3) MLADecoderAiv::SetArgs2 — platform-agnostic GM binding.
        lse_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(lse_out_gm);
        lse_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(lse_gm));
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        // ====================================================================
        // A5 Vector Run — orchestration layer (mirrors arch32 A3 Run).
        //
        // Three-layer split for the A5 port (same as A3 aiv_* files):
        //   - orchestration (here): 3-line flow — init sync / schedule / wait sync
        //   - business  (aiv_bs.h ScheduleVectorTasks): batch/head task dispatch
        //   - platform  (aiv_arch35.h): SET/WAIT_FLAG MTE3/V/MTE2 pipe sync
        // ====================================================================
        PlatformInitVectorPipeSync();
        ScheduleVectorTasks();
        PlatformWaitVectorPipeSync();
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1()
    {
        // TODO: A5 Vector RunTP1 implementation (MTP / TP1 path)
    }

    // ------------------------------------------------------------------------
    // Vector-core business layer (bs.h friend, mirrors the Cube-core policy).
    // InnerRunVectorChange holds the double-buffered n_loop softmax pipeline for
    // one task; it is a free function in multi_latent_attention_bs.h and receives
    // this AIV instance by reference so it can reach the private GM/UB members.
    // ------------------------------------------------------------------------

private:
    // [AIV 第三层平台函数 + 第二层业务函数] 迁移至 aiv_arch35.h / aiv_bs.h。
    // 类内 include 展开为成员函数，可直接访问下方私有 GM/UB/tiling 成员。
    // 与 A3(arch32) 类内 include aiv_arch32.h / aiv_bs.h 范式一致。
#include "multi_latent_attention_aiv_arch35.h"
#include "multi_latent_attention_aiv_bs.h"

    // LSE (log-sum-exp) output GM — bound by SetArgs2 for ring mode.
    __gm__ OUT_DTYPE *__restrict__ lse_gm{nullptr};
    AscendC::GlobalTensor<OUT_DTYPE> lse_gm_tensor;

    // ---- GM raw pointers (bound in SetArgs) ----
    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ float *__restrict__ go_gm{nullptr};
    __gm__ int32_t *__restrict__ gm_block_tables_{nullptr};
    __gm__ OUT_DTYPE *__restrict__ o_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ mask_gm{nullptr};
    __gm__ uint8_t *__restrict__ tiling_gm{nullptr};

    // ---- GlobalTensor views (bound in SetArgs) ----
    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor;
    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor;
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor;
    AscendC::GlobalTensor<float> s_rope_gm_tensor;
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor;
    AscendC::GlobalTensor<mm2OutputType> o_tmp_gm_tensor;
    AscendC::GlobalTensor<float> go_gm_tensor;
    AscendC::GlobalTensor<float> tmp_gm_tensor;
    AscendC::GlobalTensor<float> deq_scale_gm_tensor_q1;
    AscendC::GlobalTensor<float> deq_scale_gm_tensor_k1;

    // ---- scalar tiling params (loaded in SetArgs) ----
    uint32_t go_flag_scalar{1};
    uint32_t gl_flag_scalar{1};
    uint32_t num_batches{0};
    uint32_t q_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t block_size{0};
    uint32_t __k{0};
    uint32_t round_k{0};
    uint32_t __v{0};
    uint32_t round_v{0};
    float tor{0};
    uint64_t sub_block_idx{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t block_size_calc{0};
    uint32_t mask_type{0};
    uint32_t max_num_blocks_per_query{0};
    uint32_t totalTaskNum{0};
    uint32_t maxKVSeqLen{0};
    uint32_t cur_qn_blk_size{0};

    // Embedding-split loop count for the PV (V) former segment. arch32 L4454
    // default = 1; passed to InnerRunVectorChange from Run.
    uint32_t embed_split_loop_v_former{1};
    // Double-buffer softmax pm flags (ping/pong), toggled per n_idx parity.
    uint32_t pm_flag_scalar1{0};
    uint32_t pm_flag_scalar2{0};

    UbufAllocA5<BlockFlow> ubufAlloc;
};

} // namespace MlaArch35
} // namespace XllmOps

// ============================================================================
// UbufAllocA5<BlockStack::FOUR_FLOW> — 4-buffer variant for Ring / multi-block
// (Must be at global scope to match the primary template at line 170)
// ============================================================================
template<>
struct UbufAllocA5<BlockStack::FOUR_FLOW> {
    const uint32_t ls32_ubuf_offset = 0;
    const uint32_t ls32_quant_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t lp_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t lp32_ubuf_offset = 0 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t lo_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t mask32_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t ls16_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t lm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t hm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 1 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t pm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 2 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t pm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 3 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t descale1_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 4 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t descale2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 5 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t dm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 6 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t dm32_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 7 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t ll_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 10 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t ll_ubuf_stage2_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 12 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t gm32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 14 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t gl_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 16 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t gl32_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_MLA_A5 + 16 * UB_UINT8_LINE_SIZE_A5;
    const uint32_t go_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t go32_ubuf_offset = 8 * UB_UINT8_BLOCK_SIZE_MLA_A5;
    const uint32_t tv32_ubuf_offset = 10 * UB_UINT8_BLOCK_SIZE_MLA_A5;
};


#endif // MULTI_LATENT_ATTENTION_ARCH35_H