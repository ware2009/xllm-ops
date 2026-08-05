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

#ifndef MLA_TILING_IMPL_A5_H
#define MLA_TILING_IMPL_A5_H

#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510)

#include "multi_latent_attention_tiling_dependency.h"
#include "mla.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"

namespace AtbOps {

// ============================================================================
// A5 (ascend950 / DAV_3510) Tiling 实现接口
//
// 与 A3 (arch32) Tiling 的核心差异：
// 1. 动态 blockDim：使用 GetCoreNumAic() 获取实际核数（A5=28），去除硬编码 20
// 2. Tile size 扩大：TILING_BLOCKSIZE_CALC=25 填充 256（GEMM1/GEMM2 tile 从 128 提升到 256）
// 3. Workspace 重估：A5 tile 256 后 WORKSPACE_BLOCK_SIZE_DB 相应调整
// 4. 新增 A5 专用 tiling 字段：TILING_BLOCKSIZE_CALC（索引25）填充 embed_split_size_qk/v
// 5. TilingKey 编码简化：A5 仅支持 3 种 dataType（HALF/BF16/INT8），无 kNz/ring
// ============================================================================

// A5 Tiling 主入口
ge::graphStatus MLATilingA5(gert::TilingContext *context);

// A5 Tiling 参数生成
ge::graphStatus GetMLATilingParamA5(OpParam::MLA param, const MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize);

} // namespace AtbOps

#endif // CATLASS_ARCH == 3510

#endif // MLA_TILING_IMPL_A5_H