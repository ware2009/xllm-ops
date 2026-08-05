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

#include "kernel_operator.h"

// ----------------------------------------------------------------------------
// Architecture dispatch entry point
// ----------------------------------------------------------------------------
// This file only performs architecture detection and delegates to the
// corresponding arch-specific implementation:
//   - arch32: A2 / A3 (V220) — compile-time #ifdef __DAV_C220_CUBE__/VEC__
//   - arch35: A5 (ascend950 / DAV_3510) — runtime ASCEND_IS_AIC/AIV
// ----------------------------------------------------------------------------

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510))
#define MLA_ARCH35 1
#endif

#if defined(MLA_ARCH35)
#if !defined(CATLASS_ARCH)
#define CATLASS_ARCH 3510
#endif
#include "arch35/multi_latent_attention.cpp"
#else
#include "arch32/multi_latent_attention.cpp"
#endif

// The arch-specific .cpp file defines `mla_archXX_entry()` with the full
// TilingKey dispatch body. The kernel entry point simply delegates.
extern "C" __global__ __aicore__ void multi_latent_attention(GM_ADDR query, GM_ADDR queryRope, GM_ADDR kvCache,
                                                             GM_ADDR kvCacheRope, GM_ADDR block_tables,
                                                             GM_ADDR contextLens, GM_ADDR mask, GM_ADDR qSeqlen,
                                                             GM_ADDR qkDescale, GM_ADDR pvDescale, GM_ADDR attenOut,
                                                             GM_ADDR lseOut, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(MLA_ARCH35)
    mla_arch35_entry(query, queryRope, kvCache, kvCacheRope, block_tables,
                     contextLens, mask, qSeqlen, qkDescale, pvDescale,
                     attenOut, lseOut, workspace, tiling);
#else
    mla_arch32_entry(query, queryRope, kvCache, kvCacheRope, block_tables,
                     contextLens, mask, qSeqlen, qkDescale, pvDescale,
                     attenOut, lseOut, workspace, tiling);
#endif
}