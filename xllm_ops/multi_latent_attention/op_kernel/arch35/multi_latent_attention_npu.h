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

#ifndef MULTI_LATENT_ATTENTION_NPU_ARCH35_H
#define MULTI_LATENT_ATTENTION_NPU_ARCH35_H

#include "kernel_operator.h"

// A5 (ascend950 / DAV_3510) Vector-side constants and helpers
constexpr int32_t ROW_OPS_SPEC_MASK_32 = 32;
constexpr int32_t ROW_OPS_SPEC_MASK_8 = 8;
constexpr int32_t ROW_OPS_SPEC_MASK_4 = 4;
constexpr int32_t REDUCE_UB_SIZE = 1024;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t VECTOR_SIZE = 128;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t S_DB_SIZE = 8192;

enum class RowCalcTile {
    TAIL_TILE = 0,
    SPEC_TILE_256,
    SPEC_TILE_512
};

enum ScaleType {
    SCALE_TOR = 0,
    SCALE_LOGN = 1,
    SCALE_LOGN_FP32 = 2
};

enum class MaskType {
    MASK_TYPE_NONE = 0,
    MASK_TYPE_TRIU = 1,
    MASK_TYPE_ALIBI = 2,
    MASK_TYPE_ALIBI_COMPRESS = 6,
    MASK_TYPE_ALIBI_COMPRESS_SQRT = 7,
    MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 8,
    MASK_TYPE_ALIBI_COMPRESS_128 = 9
};

// TODO: Port vector-side helper functions (SetVecMask, Softmax, etc.)
//       from A3 multi_latent_attention_npu.h, adapted for A5 architecture.

#endif // MULTI_LATENT_ATTENTION_NPU_ARCH35_H