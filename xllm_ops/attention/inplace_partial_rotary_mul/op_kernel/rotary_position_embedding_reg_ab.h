/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotary_position_embedding_reg_ab.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_AB_H
#define ROTARY_POSITION_EMBEDDING_REG_AB_H

// #include "op_kernel/math_util.h"
// #include "op_kernel/platform_util.h"
// #include "op_kernel/load_store_utils.h"
#include "apply_rotary_pos_emb_common.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

// round3 (kept from round2, feature B): flattened-pointer variant of InterleaveModeVF.
// Per-row instruction sequence is identical to InterleaveModeVF (bit-exact); differences:
//  - row base addresses advance by dAlignLen via pointer increments (removes the per-row
//    (sIdx*currDNum+idxD)*dAlignLen re-multiply of in/out bases and the tail-part base recomputation);
//    the in/out pointers keep advancing across the sIdx loop (no per-row full recompute);
//  - the ALL loop mask and the tail mask are hoisted out of the row loops (both are row-invariant:
//    tailLen < VL holds for every dLen, so UpdateMask(tailLen) yields the same mask every time).
// Loop skeleton keeps canonical uint16_t induction loops (CCE requirement): single flat loop with a
// conditionally-reset induction counter crashed the bisheng frontend (stack smashing), so the sIdx/idxD
// nesting is kept and only the address arithmetic is flattened.
template <typename T>
__aicore__ inline void InterleaveModeVFFlat(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor, const LocalTensor<T>& inTensor,
    const LocalTensor<T>& outTensor, uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __local_mem__ T* sinUb = (__local_mem__ T*)sinTensor.GetPhyAddr();
    __local_mem__ T* cosUb = (__local_mem__ T*)cosTensor.GetPhyAddr();
    __local_mem__ T* inUb = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ T* outUb = (__local_mem__ T*)outTensor.GetPhyAddr();
    uint16_t repeatTimes = dLen / VL_FLOAT32_SIZE;
    uint32_t dAlignLen = ops::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t loopNum = repeatTimes / 2;
    uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;
    uint16_t tailTwoVL = tailNum / VL_FLOAT32_SIZE;
    uint16_t tailOneVL = (tailTwoVL == 1) ? 0 : 1;
    uint32_t tailLen = tailNum % VL_FLOAT32_SIZE;
    __local_mem__ T* currInUb = inUb;
    __local_mem__ T* currOutUb = outUb;
    __local_mem__ T* currSinUb = sinUb;
    __local_mem__ T* currCosUb = cosUb;
    __local_mem__ T* tailInUb;
    __local_mem__ T* tailOutUb;
    __local_mem__ T* tailSinUb;
    __local_mem__ T* tailCosUb;
    uint32_t tailBodyOff = loopNum * 2 * VL_FLOAT32_SIZE;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregFormerCos;
        MicroAPI::RegTensor<float> vregLatterCos;
        MicroAPI::RegTensor<float> vregFormerSin;
        MicroAPI::RegTensor<float> vregLatterSin;
        MicroAPI::RegTensor<float> vregFormerIn;
        MicroAPI::RegTensor<float> vregLatterIn;
        MicroAPI::RegTensor<float> vregOdd;
        MicroAPI::RegTensor<float> vregEven;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregTail;
        pregLoop = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        uint32_t tailCnt = tailLen;
        pregTail = MicroAPI::UpdateMask<float>(tailCnt);
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
                for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                uint32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, evenOffSet);
                ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregLoop, oddOffset);
                ops::LoadOneTensorForDtypeT<T>(currCosUb, vregFormerCos, pregLoop, evenOffSet);
                ops::LoadOneTensorForDtypeT<T>(currCosUb, vregLatterCos, pregLoop, oddOffset);
                ops::LoadOneTensorForDtypeT<T>(currSinUb, vregFormerSin, pregLoop, evenOffSet);
                ops::LoadOneTensorForDtypeT<T>(currSinUb, vregLatterSin, pregLoop, oddOffset);
                Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregLoop);
                MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregLoop);
                Add(vregLatterCos, vregLatterCos, vregLatterSin, pregLoop);
                ops::StoreOneTensorForDtypeT<T>(currOutUb, vregFormerCos, pregLoop, evenOffSet);
                ops::StoreOneTensorForDtypeT<T>(currOutUb, vregLatterCos, pregLoop, oddOffset);
            }

            // 尾块大于VL时,读取一个VL，读取尾块
            tailInUb = currInUb + tailBodyOff;
            tailOutUb = currOutUb + tailBodyOff;
            tailSinUb = currSinUb + tailBodyOff;
            tailCosUb = currCosUb + tailBodyOff;
            for (uint16_t i = 0; i < tailTwoVL; i++) {
                ops::LoadOneTensorForDtypeT<T>(tailInUb, vregFormerIn, pregLoop, 0);
                ops::LoadOneTensorForDtypeT<T>(tailInUb, vregLatterIn, pregTail, VL_FLOAT32_SIZE);
                ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregLoop, 0);
                ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregLoop, 0);
                ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregLatterSin, pregTail, VL_FLOAT32_SIZE);
                Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                ops::StoreOneTensorForDtypeT<T>(tailOutUb, vregFormerCos, pregLoop, 0);
                ops::StoreOneTensorForDtypeT<T>(tailOutUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
            }

            // 尾块小于VL时,只读取VL
            for (uint16_t i = 0; i < tailOneVL; i++) {
                ops::LoadOneTensorForDtypeT<T>(tailInUb, vregFormerIn, pregTail, 0);
                ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregTail, 0);
                ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregTail, 0);
                Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                ops::StoreOneTensorForDtypeT<T>(tailOutUb, vregFormerCos, pregTail, 0);
            }

            currInUb += dAlignLen;
            currOutUb += dAlignLen;
            }
            currSinUb += dAlignLen;
            currCosUb += dAlignLen;
        }
    }
}

template <typename T>
class RotaryPositionEmbeddingAB
{
public:
    __aicore__ inline RotaryPositionEmbeddingAB(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace, const RopeRegbaseTilingData* tilingData,
        TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(
        int64_t xGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer, int64_t ubIdx, int64_t bsCount,
        int64_t nCount);
    // round3 (kept from round2, feature C): legacy schedule (original per-iteration copy/compute), used by
    // all modes including INTERLEAVE; cos/sin are served from the merged queue via two views
    __aicore__ inline void ProcessLegacy(int64_t bsLoopIdx, int64_t xGmOffset, int64_t currBSNum);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> csInQueue_;  // round3 (kept from r2): merged cos|sin queue (cos at [0], sin at [csHalfElems_])
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> yGm_;
    const RopeRegbaseTilingData* tilingData_;
    DataCopyPadExtParams<T> padParams_ = {false, 0, 0, static_cast<T>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSize_ = 0;
    int64_t bsBlockCount_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t sliceAlign_ = 0;
    bool fullRow_ = false;  // full-row-copy: INTERLEAVE full-row copy (UB budget gated, falls back when over budget)
    int64_t rowWidth_ = 0;  // x/y UB row width (elements): D when fullRow else sliceAlign_
    int64_t csHalfElems_ = 0;  // round3 (kept from r2): one cos (or sin) region in elements = sliceAlign_ * ubFactorBS
    uint32_t srcStrideBytes_ = 0;  // round3 (kept from r2): loop-invariant x copy-in row gap = (D - sliceLength) * sizeof(T)
    uint32_t dstStrideBytes_ = 0;  // round3 (kept from r2): loop-invariant y copy-out row gap = (D - sliceLength) * sizeof(T)
};

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::Init(
    GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace, const RopeRegbaseTilingData* tilingData,
    TPipe* pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    dSplitSize_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(T);
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->sliceLength;
    int64_t offset = blockDimBS * tilingData_->blockFactorBS * tilingData_->D;

    sliceAlign_ = ops::CeilDiv(tilingData_->sliceLength * sizeof(T), GetUbBlockSize()) * GetUbBlockSize() / sizeof(T);
    // full-row-copy (INTERLEAVE): full-row continuous copy removes small-block+gap; GM base moves to row start, UB offsets sliceStart
    fullRow_ = FULLROW_ENABLE &&
               (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) &&
               (tilingData_->dSplitCoef == 1) &&
               ((tilingData_->D * sizeof(T)) % GetUbBlockSize() == 0);  // whole-row UB copy block alignment
    if (fullRow_) {
        int64_t rows = tilingData_->ubFactorBS * tilingData_->ubFactorN;
        // guard: shapes whose four-queue DB total UB exceeds budget fall back to original copy path
        int64_t totalUb = 4 * rows * (tilingData_->D + sliceAlign_) * sizeof(T);
        if (totalUb > FULLROW_UB_BUDGET) {
            fullRow_ = false;
        }
    }
    rowWidth_ = fullRow_ ? tilingData_->D : sliceAlign_;

    int64_t xOffset = offset * tilingData_->N + blockDimN * tilingData_->blockFactorN * tilingData_->D;
    if (!fullRow_) {
        xOffset += tilingData_->sliceStart;
    }
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin + cosOffset);
    this->xGm_.SetGlobalBuffer((__gm__ T*)x + xOffset);
    this->yGm_.SetGlobalBuffer((__gm__ T*)y + xOffset);

    int64_t xRowBytes = rowWidth_ * sizeof(T) * tilingData_->ubFactorBS;
    int64_t csRowBytes = sliceAlign_ * sizeof(T) * tilingData_->ubFactorBS;
    pipe_->InitBuffer(xInQueue_, DB_FLAG, xRowBytes * tilingData_->ubFactorN);
    // round3 (kept from r2): merged cos|sin queue, 2*csRowBytes per buffer; UB total identical to the former
    // cos/sin queue pair (2 * DB * csRowBytes)
    pipe_->InitBuffer(csInQueue_, DB_FLAG, 2 * csRowBytes);
    pipe_->InitBuffer(yOutQueue_, DB_FLAG, xRowBytes * tilingData_->ubFactorN);
    csHalfElems_ = sliceAlign_ * tilingData_->ubFactorBS;
    // round3 (kept from r2, scalar reduction): row-gap params are loop-invariant, computed once
    srcStrideBytes_ = static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(T));
    dstStrideBytes_ = srcStrideBytes_;
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::Process()
{
    // round3: sub-block pipeline (round2 feat-pipe-subblock-ab) removed; every mode takes the legacy
    // schedule so the INTERLEAVE path regains the original per-iteration copy-in/VF/copy-out rhythm
    uint32_t bsLoopCnt = ops::CeilDiv(bsBlockCount_, tilingData_->ubFactorBS);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t xGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->N * tilingData_->D;
        int64_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ? tilingData_->ubFactorBS :
                                                           bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);
        ProcessLegacy(bsLoopIdx, xGmOffset, currBSNum);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::ProcessLegacy(int64_t bsLoopIdx, int64_t xGmOffset, int64_t currBSNum)
{
    uint32_t nLoopCnt = ops::CeilDiv(nBlockCount_, tilingData_->ubFactorN);
    DataCopyExtParams csParams = {
        static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSize_, 0, 0, 0};
    LocalTensor<T> csBuffer = csInQueue_.AllocTensor<T>();
    int64_t csGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength;
    DataCopyPad(csBuffer, cosGm_[csGmOffset], csParams, padParams_);
    DataCopyPad(csBuffer[csHalfElems_], sinGm_[csGmOffset], csParams, padParams_);
    csInQueue_.EnQue(csBuffer);
    csBuffer = csInQueue_.DeQue<T>();
    LocalTensor<T> cosBuffer = csBuffer;
    LocalTensor<T> sinBuffer = csBuffer[csHalfElems_];

    for (int64_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
        int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? tilingData_->ubFactorN :
                                                        nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
        ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
    }

    csInQueue_.FreeTensor(csBuffer);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::ProcessLoop(
    int64_t xGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer, int64_t ubIdx, int64_t bsCount,
    int64_t nCount)
{
    int64_t totalCount = bsCount * nCount;
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef), dSplitSize_, srcStrideBytes_, 0, 0};
    DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount * tilingData_->dSplitCoef), dSplitSize_, 0, dstStrideBytes_, 0};
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        inParams = {static_cast<uint16_t>(totalCount), tilingData_->D * sizeof(T), static_cast<uint32_t>((tilingData_->D - tilingData_->sliceLength) * sizeof(T)), 0, 0};
    }
    if (fullRow_) {
        // full-row-copy (INTERLEAVE): full-row continuous copy, removes 128B small-block + row-gap MTE address overhead
        inParams = {static_cast<uint16_t>(totalCount), static_cast<uint32_t>(tilingData_->D * sizeof(T)), 0, 0, 0};
        outParams = {static_cast<uint16_t>(totalCount), static_cast<uint32_t>(tilingData_->D * sizeof(T)), 0, 0, 0};
    }

    LocalTensor<T> inBuffer = xInQueue_.AllocTensor<T>();
    LocalTensor<T> outBuffer = yOutQueue_.AllocTensor<T>();

    DataCopyPad(inBuffer, xGm_[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D], inParams, padParams_);

    xInQueue_.EnQue(inBuffer);
    inBuffer = xInQueue_.DeQue<T>();

    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfAlignVF(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->sliceLength, sliceAlign_, bsCount, nCount);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        if (fullRow_) {
            // out-of-segment elements written back as read (bit-exact), rotated segment covered by VF; UB row layout = GM row layout
            DataCopy(outBuffer, inBuffer, static_cast<uint32_t>(totalCount * tilingData_->D));
            InterleaveModeVFRowStrided(sinBuffer, cosBuffer, inBuffer[tilingData_->sliceStart],
                outBuffer[tilingData_->sliceStart], tilingData_->sliceLength, rowWidth_, bsCount, nCount);
        } else {
            // round3 (feature B): flattened-pointer VF, bit-exact per-row instruction sequence vs InterleaveModeVF
            InterleaveModeVFFlat(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->sliceLength, bsCount, nCount);
        }
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        QuarterAlignVF(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->sliceLength, sliceAlign_, bsCount, nCount);
    } else {
        DeepSeekInterleaveModeVF<T>(sinBuffer, cosBuffer, inBuffer, outBuffer, tilingData_->sliceLength, bsCount, nCount);
    }

    yOutQueue_.EnQue(outBuffer);
    outBuffer = yOutQueue_.DeQue<T>();
    xInQueue_.FreeTensor(inBuffer);

    DataCopyPad(yGm_[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D], outBuffer, outParams);

    yOutQueue_.FreeTensor(outBuffer);
}

} // namespace InplacePartialRotaryMul

#endif // ROTARY_POSITION_EMBEDDING_REG_AB_H
