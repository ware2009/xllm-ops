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
#include "multi_latent_attention_tiling_impl.h"
#include "multi_latent_attention_tiling.h"
#include "mla_arch_config.h"  // CMake-generated: defines CATLASS_ARCH=3510 on A5
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

// A5 (ascend950 / DAV_3510) 架构专用 Tiling 分支
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510)
#include "multi_latent_attention_tiling_impl_a5.h"
#endif

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510)
        // A5 平台：使用 A5 专用 Tiling 实现
        // - 动态 blockDim（28核，无硬编码）
        // - tile size 256（利用 L0A/B 64KB, L0C 256KB）
        // - TILING_BLOCKSIZE_CALC=256 字段填充
        // - 简化 TilingKey 编码（3种 dataType，无 kNz/ring）
        return AtbOps::MLATilingA5(context);
#else
        // A3/A2 平台：使用原始 Tiling 实现
        // - 硬编码 blockDim (batch==32 ? 20 : GetCoreNumAic)
        // - tile size 128（L0A/B 32KB, L0C 128KB）
        // - TilingKey: dataType + kNz + mtpTp1Flag + isRing
        return AtbOps::MLATiling(context);
#endif
    }
    IMPL_OP_OPTILING(MultiLatentAttention)
    .Tiling(TilingFunc);
}
