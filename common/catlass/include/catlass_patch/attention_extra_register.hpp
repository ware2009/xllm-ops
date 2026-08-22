/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_PATCH_ATTENTION_EXTRA_REGISTER_HPP
#define CATLASS_PATCH_ATTENTION_EXTRA_REGISTER_HPP

// Project-owned registration point for the non-XA attention templates that
// were relocated out of the third_party/catlass submodule into
// common/catlass. These cover: XFAI (x_flash_attention_infer A3
// path), FA/Unshared-FA/CombineScale/CopyGLM/NoDivRowSum (x_attention_bak A2
// path), and FD flash-decoding epilogues (x_flash_attention_infer FD path).
//
// The dispatch-policy STRUCTS stay defined in the submodule's
// catlass/{gemm,epilogue}/dispatch_policy.hpp (they are not standalone files,
// so they are not part of this file-level migration). This header only pulls
// them in and then mounts the relocated block-specialization files, mirroring
// the exact arch guards that the submodule aggregation headers used to apply.
//
// It MUST be included AFTER the catlass aggregation headers
// (catlass/gemm/block/block_mmad.hpp and catlass/epilogue/block/block_epilogue.hpp)
// so the primary templates BlockMmadTla / BlockEpilogue are already visible.

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

// ---- A2/A3 (CATLASS_ARCH == 2201): unshared-FA / FAI-split-row mmad ----
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
#include "catlass/gemm/block/block_mmad_unshared_fa_pv.hpp"
#include "catlass/gemm/block/block_mmad_unshared_fa_qk.hpp"
#include "catlass/gemm/block/block_mmad_fai_qk_split_row.hpp"
#include "catlass/gemm/block/block_mmad_fai_pv_split_row.hpp"
#endif

// ---- Cross-arch: XFAI mmad ----
#include "catlass/gemm/block/block_mmad_xfai_qk.hpp"
#include "catlass/gemm/block/block_mmad_xfai_pv.hpp"

// ---- Cross-arch: FA unshared softmax / rescaleO-no-div-rowsum / combine-scale / copy-glm ----
#include "catlass/epilogue/block/block_epilogue_fa_unshared_softmax.hpp"
#include "catlass/epilogue/block/block_epilogue_rescale_o_no_div_rowsum.hpp"
#include "catlass/epilogue/block/block_epilogue_combine_scale.hpp"
#include "catlass/epilogue/block/block_epilogue_online_softmax_copy_glm.hpp"

// ---- Cross-arch: FD flash-decoding + XFAI softmax/rescaleO epilogues ----
#include "catlass/epilogue/block/block_epilogue_online_softmax_FD.hpp"
#include "catlass/epilogue/block/block_epilogue_rescale_o_FD.hpp"
#include "catlass/epilogue/block/block_epilogue_xfai_online_softmax.hpp"
#include "catlass/epilogue/block/block_epilogue_xfai_rescale_o.hpp"

// ---- CombineScale helper class (used directly by x_flash_attention_infer FD path) ----
#include "catlass/epilogue/block/block_epilogue_xfai_combine_scale.hpp"

#endif // CATLASS_PATCH_ATTENTION_EXTRA_REGISTER_HPP
