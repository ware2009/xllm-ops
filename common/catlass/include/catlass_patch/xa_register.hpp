/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_PATCH_XA_REGISTER_HPP
#define CATLASS_PATCH_XA_REGISTER_HPP

// This header is the project-owned registration point for the x_attention (XA)
// custom catlass templates. It MUST be included AFTER the catlass aggregation
// headers (catlass/gemm/block/block_mmad.hpp and catlass/epilogue/block/block_epilogue.hpp)
// so that the primary templates (BlockMmadTla / BlockEpilogue) and MmadBase are visible.
//
// It re-introduces exactly the dispatch-policy structs and block-specialization
// includes that used to live as uncommitted patches in the third_party/catlass
// submodule, so the submodule can stay on its committed (xa-free) state and the
// XA templates are owned by common/catlass.

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

namespace Catlass::Gemm {

// For Ascend950, XA Shared Infer QK (s1 x d * d x s2)
template <class ArchTag_>
struct MmadXASharedQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

// For Ascend950, XA Unshared KV Infer QK
template <class ArchTag_>
struct MmadXAUnsharedQK : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

// For Ascend950, XA Unshared KV Infer PV
template <class ArchTag_>
struct MmadXAUnsharedPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

// For Ascend950, XA Shared Infer PV
template <class ArchTag_>
struct MmadXASharedPV : public MmadBase<ArchTag_, false> {
    static constexpr uint32_t STAGES = 2;
};

} // namespace Catlass::Gemm

namespace Catlass::Epilogue {

// For Ascend950, XA Shared Infer online Softmax
struct EpilogueAscend950XASharedSoftmax {
    using ArchTag = Arch::Ascend950;
};

// For Ascend950, XA unshared KV infer softmax
struct EpilogueAscend950XAUnsharedSoftmax {
    using ArchTag = Arch::Ascend950;
};

// For Ascend950, XA Shared Infer RescaleO
struct EpilogueAscend950XASharedRescaleO {
    using ArchTag = Arch::Ascend950;
};

// For Ascend950, XA Combine scale
struct EpilogueAscend950XACombineScale {
    using ArchTag = Arch::Ascend950;
};

} // namespace Catlass::Epilogue

// XA block specializations (Ascend950 only). These resolve to the copies under
// common/catlass/include/catlass/ via the -I search order in the op
// CMakeLists (-I.../common/catlass/include precedes -I${CANN_3RD_LIB_PATH}/catlass/include).
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass/gemm/block/block_mmad_xa_shared_qk_tla.hpp"
#include "catlass/gemm/block/block_mmad_xa_unshared_qk_tla.hpp"
#include "catlass/gemm/block/block_mmad_xa_shared_pv_tla.hpp"
#include "catlass/gemm/block/block_mmad_xa_unshared_pv_tla.hpp"
#include "catlass/epilogue/block/block_epilogue_xa_shared_softmax_ascend950.hpp"
#include "catlass/epilogue/block/block_epilogue_xa_shared_rescale_ascend950.hpp"
#include "catlass/epilogue/block/block_epilogue_xa_unshared_softmax_ascend950.hpp"
#include "catlass/epilogue/block/block_epilogue_xa_combine_scale_ascend950.hpp"
#endif

#endif // CATLASS_PATCH_XA_REGISTER_HPP
