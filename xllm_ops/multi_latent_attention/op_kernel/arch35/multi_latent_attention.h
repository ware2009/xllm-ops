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

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

// Reuse common const values and enums from the A3 header
// (these are defined at file scope in multi_latent_attention.h and will be
//  available because the arch35 header is conditionally included *instead of*
//  the A3 header — so we must re-declare the minimal set needed here).

constexpr int32_t BLOCK_SIZE_32 = 32;

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

namespace XllmOps {
namespace MlaArch35 {

// ============================================================================
// MLAttentionDecoderAic — A5 (ascend950 / DAV_3510) Cube core skeleton
// Template signature mirrors the A3 class for transparent dispatch.
// ============================================================================
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half,
          typename IN_KVDTYPE = half,
          typename OUT_DTYPE = half,
          typename IN_ROPE_DTYPE = half,
          InputFormat KInputType = InputFormat::ND_FORMAT,
          bool EnableOptimization = false>
class MLAttentionDecoderAic {
public:
    __aicore__ __attribute__((always_inline)) inline MLAttentionDecoderAic() {}

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
        // TODO: A5 Cube SetArgs implementation
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        // TODO: A5 Cube Run implementation (Normal / non-TP1 path)
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1()
    {
        // TODO: A5 Cube RunTP1 implementation (MTP / TP1 path)
    }
};

// ============================================================================
// MLADecoderAiv — A5 (ascend950 / DAV_3510) Vector core skeleton
// Template signature mirrors the A3 class for transparent dispatch.
// ============================================================================
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half,
          typename OUT_DTYPE = half,
          bool IsRing = false,
          BlockStack BlockFlow = BlockStack::ONE_FLOW,
          bool IsTP1 = false>
class MLADecoderAiv {
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
        // TODO: A5 Vector SetArgs implementation
    }

    __aicore__ __attribute__((always_inline)) inline void SetArgs2(
        __gm__ uint8_t *__restrict__ lse_out_gm)
    {
        // TODO: A5 Vector SetArgs2 (lse for ring mode) implementation
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        // TODO: A5 Vector Run implementation (Normal / non-TP1 path)
    }

    __aicore__ __attribute__((always_inline)) inline void RunTP1()
    {
        // TODO: A5 Vector RunTP1 implementation (MTP / TP1 path)
    }
};

} // namespace MlaArch35
} // namespace XllmOps

#endif // MULTI_LATENT_ATTENTION_ARCH35_H