/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_SHARED_RESCALE_O_ASCEND950
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_SHARED_RESCALE_O_ASCEND950

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Block {

// fp16o probe variant (override of the CANN-vendored shared rescale epilogue):
// - ElementOTmp (the PV GEMM output, fixpiped to UB as half) is fp16; the running
//   O accumulator stays fp32 (attnTmpBuf, ElementStat = float) so 32 kv-block
//   rescales do not accumulate fp16 rounding.
// - pvRes rows are widened in-register via Cast<float, half> (RegLayout::ZERO
//   unpack of the low 64 lanes of the 128-lane half register).
// - isLastKv stores the fp32 accumulator straight to the fp32 sharedO GM
//   (combine contract unchanged, no extra cast pass).
// Numerics: the only new quantization vs the fp32-OTmp baseline is the fixpipe
// fp32->half of each PV block result; the accumulate/rescale chain stays fp32.
template <class L1TileShape_, class OTmpType_>
class BlockEpilogue<EpilogueAscend950XASharedRescaleO, L1TileShape_, OTmpType_> {
public:
    using DispatchPolicy = EpilogueAscend950XASharedRescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOTmp = typename OTmpType_::Element;
    using ElementStat = float;
    using LayoutTagOTmp = typename OTmpType_::Layout;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t S1_BASE_SIZE = tla::get<0>(L1TileShape{});
    static constexpr uint32_t D_BASE_SIZE = tla::get<2>(L1TileShape{});
    static constexpr uint32_t HALF_S1_BASE_SIZE = S1_BASE_SIZE / 2;
    static constexpr uint32_t VEC2_UB_SIZE = HALF_S1_BASE_SIZE * D_BASE_SIZE * sizeof(ElementStat);

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, uint32_t& ubBufAddrStart)
    {
        attnTmpBuf = resource.ubBuf.template GetBufferByByte<ElementStat>(ubBufAddrStart);
        ubBufAddrStart += VEC2_UB_SIZE;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst& attenOutGm, const AscendC::LocalTensor<ElementStat>& expMaxUb, TensorSrc& pvRes, bool isFirstKv,
        bool isLastKv, uint16_t PV_RELEASE_FLAG)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
        uint32_t m = tla::get<0>(pvRes.shape());
        uint32_t n = tla::get<1>(pvRes.shape());
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementStat));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;

        __ubuf__ ElementOTmp* pvResUbAddr = (__ubuf__ ElementOTmp*)pvRes.data().GetPhyAddr();
        __ubuf__ ElementStat* attnTmpAddr = (__ubuf__ ElementStat*)attnTmpBuf.GetPhyAddr();
        __ubuf__ ElementStat* expMaxUbAddr = (__ubuf__ ElementStat*)expMaxUb.GetPhyAddr();

        if (isFirstKv) {
            // attnTmp = (float)pvRes, one in-register widen pass (V pipe, ordered).
            CastFirstKv<D_BASE_SIZE>(attnTmpAddr, pvResUbAddr, static_cast<uint16_t>(m), nLoops, tailN);
        } else {
            FlashUpdateNew<D_BASE_SIZE>(
                attnTmpAddr, pvResUbAddr, expMaxUbAddr, static_cast<uint16_t>(m), nLoops, tailN);
        }

        if (isLastKv) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventOVMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventOVMTE3);
            auto layoutUb = tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(D_BASE_SIZE, tla::Int<1>{}));
            auto attenOutUb = tla::MakeTensor(attnTmpBuf, layoutUb, Arch::PositionUB{});
            using CopyUbToGmO = Tile::CopyUb2GmTla<ArchTag, decltype(attenOutUb), TensorDst>;
            CopyUbToGmO copyUbToGmO;
            copyUbToGmO(attenOutGm, attenOutUb);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(PV_RELEASE_FLAG);
    }

private:
    AscendC::LocalTensor<ElementStat> attnTmpBuf;
    static constexpr int32_t SYNC_MODE = 4;
    static constexpr uint16_t FLOAT_REP_SIZE = 64;
    static constexpr int32_t eventOVMTE3 = 3;
    static constexpr int32_t eventOMTE3V = 3;

    // fp16 unpack traits: the low 64 halves of a 128-lane half register widen to
    // 64 floats with RegLayout::ZERO (RegLayout::ONE would take the upper 64).
    constexpr static CastTrait castTraitHalfZero = {
        RegLayout::ZERO,
        SatMode::SAT,
        MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND,
    };

    // attnTmp = (float)pvRes. Each LoadAlign pulls 128 halves (the requested 64-lane
    // segment plus the next one, possibly past the row into UB scratch); only the
    // low 64 lanes are consumed via the ZERO cast, so the over-read is discarded.
    template <uint16_t DBaseSize>
    __simd_vf__ inline void CastFirstKv(
        __ubuf__ ElementStat* dstUb, __ubuf__ ElementOTmp* srcUb, uint16_t m, uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::Reg;
        RegTensor<half> curHalf;
        RegTensor<float> curSrcVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(curHalf, srcUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                Cast<float, half, castTraitHalfZero>(curSrcVreg, curHalf, pregFull);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(dstUb + i * DBaseSize + j * FLOAT_REP_SIZE, curSrcVreg, pregFull);
            }
            LoadAlign(curHalf, srcUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            Cast<float, half, castTraitHalfZero>(curSrcVreg, curHalf, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                dstUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, curSrcVreg, pregTailN);
        }
    }

    // attnTmp = attnTmp * expMax + (float)pvRes, fp32 accumulate / fp16 cur input.
    template <uint16_t DBaseSize>
    __simd_vf__ inline void FlashUpdateNew(
        __ubuf__ ElementStat* updateUb, __ubuf__ ElementOTmp* curUb, __ubuf__ ElementStat* expMaxUb, uint16_t m,
        uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::Reg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> preSrcVreg;
        RegTensor<float> curSrcVreg;
        RegTensor<half> curHalf;
        RegTensor<float> mulVreg;
        RegTensor<float> addVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementStat, LoadDist::DIST_BRC_B32>(expMaxVreg, expMaxUb + i);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(preSrcVreg, updateUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                LoadAlign(curHalf, curUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                Cast<float, half, castTraitHalfZero>(curSrcVreg, curHalf, pregFull);
                Mul(mulVreg, expMaxVreg, preSrcVreg, pregFull);
                Add(addVreg, mulVreg, curSrcVreg, pregFull);
                StoreAlign<ElementStat, StoreDist::DIST_NORM_B32>(
                    updateUb + i * DBaseSize + j * FLOAT_REP_SIZE, addVreg, pregFull);
            }
            LoadAlign(preSrcVreg, updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            LoadAlign(curHalf, curUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            Cast<float, half, castTraitHalfZero>(curSrcVreg, curHalf, pregFull);
            Mul(mulVreg, expMaxVreg, preSrcVreg, pregTailN);
            Add(addVreg, mulVreg, curSrcVreg, pregTailN);
            StoreAlign<ElementStat, StoreDist::DIST_NORM_B32>(
                updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, addVreg, pregTailN);
        }
    }
};
} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ASCEND950
