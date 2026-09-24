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
 * \file rotary_position_embedding_reg_a_and_b.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_A_AND_B_H
#define ROTARY_POSITION_EMBEDDING_REG_A_AND_B_H

// #include "op_kernel/math_util.h"
#include "apply_rotary_pos_emb_common.h"

namespace InplacePartialRotaryMul {
using namespace AscendC;

template <typename T, bool IsBoardCast>
class RotaryPositionEmbeddingAAndB
{
public:
    __aicore__ inline RotaryPositionEmbeddingAAndB(){};

    __aicore__ inline ~RotaryPositionEmbeddingAAndB(){};

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace, const RopeRegbaseTilingData* tilingData,
        TPipe* pipe);

    __aicore__ inline void Process();

private:
    // Init过程中使用的内部函数
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    // 各个层级的Process函数
    __aicore__ inline void ProcessInLoop(LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bStart, int64_t bLength, bool qPreLoaded = false);
    // 拷入拷出函数
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInQ(GlobalTensor<T>& source, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyOutQ(GlobalTensor<T>& target, int64_t bStart, int64_t bLength);

    // 计算函数
    __aicore__ inline void Compute(LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bLength);

private:
    constexpr static uint32_t COS_DB_BUFFER = IsBoardCast ? 1 : DOUBLE_BUFFER;

    TPipe* pipe_;

    // GlobalMemory
    GlobalTensor<T> qGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> qOutGm_;

    // UB
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> qInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> sinInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> qOutQueue_;

    // Split core info
    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;

    // TilingData
    const RopeRegbaseTilingData* tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;

    // 拷贝参数
    uint8_t dSplitCoef_ = 1;
    uint8_t copyInQSplitCoef_ = 1; // 拷贝q时使用的splitCoef
    uint64_t ubCopyInStride = 0;   // 输入在ub中的stride，deepseek_interleave中不为0
    bool fullRow_ = false;  // full-row-copy: broadcast INTERLEAVE full-row copy (UB budget gated)
};

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Init(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace, const RopeRegbaseTilingData* tilingData,
    TPipe* pipe)
{
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->InitAllGlobalBuffer(q, cos, sin, qOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitAllGlobalBuffer(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ T*)q);
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ T*)qOut);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->D_ = this->tilingData_->D;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->copyInQSplitCoef_ = dSplitCoef_;
    this->dAlign_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_, BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->copyInQSplitCoef_ = 1;
        // 非boardcast时，使用批量计算API，需要拷贝时添加stride
        if constexpr (!IsBoardCast) {
            this->ubCopyInStride =
                (this->dAlign_ * sizeof(T) - ops::CeilAlign<int64_t>(tilingData_->sliceLength * sizeof(T), BLOCK_TYPE_SIZE)) /
                BLOCK_TYPE_SIZE;
        }
    }

    // full-row (broadcast INTERLEAVE): q full-row copy, UB row width extended to D; over-budget shapes fall back
    if constexpr (IsBoardCast) {
        fullRow_ = FULLROW_ENABLE &&
                   (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) &&
                   (dSplitCoef_ == 1) &&
                   ((tilingData_->D * sizeof(T)) % 32 == 0) &&  // whole-row UB copy block alignment
                   (4 * ubFactorB_ * (tilingData_->D + dAlign_) * sizeof(T) <= FULLROW_UB_BUDGET);
    }
    int64_t qRowWidth = fullRow_ ? tilingData_->D : dAlign_;
    this->pipe_->InitBuffer(this->qInQueue_, DOUBLE_BUFFER, ubFactorB_ * qRowWidth * sizeof(T));
    this->pipe_->InitBuffer(this->qOutQueue_, DOUBLE_BUFFER, ubFactorB_ * qRowWidth * sizeof(T));
    if constexpr (IsBoardCast) {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, dAlign_ * sizeof(T));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, ubFactorB_ * dAlign_ * sizeof(T));
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitLoopParams()
{
    this->bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    this->bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Process()
{
    // 在B轴进行循环
    int64_t ubLoopCount = ops::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBoardCast) {
        this->CopyInCosAndSin(0, 1);
        // parallel copy-in: q first-round copy-in issued concurrently with cos/sin MTE2, DeQue waits moved to Compute
        if (ubLoopCount > 0) {
            this->CopyInQ(qGm_, bBlockStart_, ubLoopCount > 1 ? ubFactorB_ : bBlockLength_);
        }
        LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
        LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInLoop(
                cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_,
                ubLoopIdx == 0);
        }
        this->cosInQueue_.FreeTensor(cosUb);
        this->sinInQueue_.FreeTensor(sinUb);
    } else {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            int64_t bStart = bBlockStart_ + ubLoopIdx * ubFactorB_;
            int64_t bLen = ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_;
            this->CopyInCosAndSin(bStart, bLen);
            // parallel copy-in: q issued concurrently with cos/sin MTE2
            this->CopyInQ(qGm_, bStart, bLen);
            LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
            LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
            this->ProcessInLoop(cosUb, sinUb, bStart, bLen, true);
            this->cosInQueue_.FreeTensor(cosUb);
            this->sinInQueue_.FreeTensor(sinUb);
        }
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::ProcessInLoop(
    LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bUbStart, int64_t bUbLength, bool qPreLoaded)
{
    if (!qPreLoaded) {
        CopyInQ(qGm_, bUbStart, bUbLength);
    }
    Compute(cos, sin, bUbLength);
    CopyOutQ(qOutGm_, bUbStart, bUbLength);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyInCosAndSin(int64_t bStart, int64_t bLength)
{
    LocalTensor<T> cosUb = this->cosInQueue_.template AllocTensor<T>();
    LocalTensor<T> sinUb = this->sinInQueue_.template AllocTensor<T>();
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPad(cosUb, this->cosGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb, this->sinGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyInQ(
    GlobalTensor<T>& source, int64_t bStart, int64_t bLength)
{
    LocalTensor<T> target = this->qInQueue_.template AllocTensor<T>();
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    if (fullRow_) {
        // full-row: continuous whole-row copy-in (removes small-block+gap), GM base at row start
        copyExtParams.blockCount = bLength;
        copyExtParams.blockLen = D_ * sizeof(T);
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = 0;
        DataCopyPad(target, source[bStart * D_], copyExtParams, copyPadExtparams);
    } else {
        copyExtParams.blockCount = bLength * copyInQSplitCoef_;
        copyExtParams.blockLen = tilingData_->sliceLength * sizeof(T) / copyInQSplitCoef_;
        copyExtParams.srcStride = (tilingData_->D - tilingData_->sliceLength) * sizeof(T);
        copyExtParams.dstStride = ubCopyInStride;
        DataCopyPad(target, source[bStart * D_ + tilingData_->sliceStart], copyExtParams, copyPadExtparams);
    }
    this->qInQueue_.template EnQue(target);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyOutQ(
    GlobalTensor<T>& target, int64_t bStart, int64_t bLength)
{
    LocalTensor<T> source = this->qOutQueue_.template DeQue<T>();
    DataCopyExtParams copyExtParams;
    if (fullRow_) {
        // full-row: whole-row write-back (out-of-segment = read-back original values, bit-exact), GM base at row start
        copyExtParams.blockCount = bLength;
        copyExtParams.blockLen = D_ * sizeof(T);
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = 0;
        DataCopyPad(target[bStart * D_], source, copyExtParams);
    } else {
        copyExtParams.blockCount = bLength * dSplitCoef_;
        copyExtParams.blockLen = tilingData_->sliceLength * sizeof(T) / dSplitCoef_;
        copyExtParams.srcStride = 0;
        copyExtParams.dstStride = (tilingData_->D - tilingData_->sliceLength) * sizeof(T);
        DataCopyPad(target[bStart * D_+ tilingData_->sliceStart], source, copyExtParams);
    }
    this->qOutQueue_.FreeTensor(source);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Compute(
    LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bLength)
{
    LocalTensor<T> inUb = this->qInQueue_.template DeQue<T>();
    LocalTensor<T> outUb = this->qOutQueue_.template AllocTensor<T>();
    if constexpr (IsBoardCast) {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            HalfAlignVF<T>(sin, cos, inUb, outUb, tilingData_->sliceLength, dAlign_, 1, bLength);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            if (fullRow_) {
                // full-row: whole-row UB copy keeps out-of-segment original values on write-back; VF uses D row stride offset sliceStart
                DataCopy(outUb, inUb, static_cast<uint32_t>(bLength * D_));
                InterleaveModeVFRowStrided<T>(sin, cos, inUb[tilingData_->sliceStart], outUb[tilingData_->sliceStart],
                    tilingData_->sliceLength, static_cast<uint32_t>(D_), 1, static_cast<uint16_t>(bLength));
            } else {
                InterleaveModeVF<T>(sin, cos, inUb, outUb, tilingData_->sliceLength, 1, bLength);
            }
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
            QuarterAlignVF<T>(sin, cos, inUb, outUb, tilingData_->sliceLength, dAlign_, 1, bLength);
        } else {
            DeepSeekInterleaveModeVF<T>(sin, cos, inUb, outUb, tilingData_->sliceLength, 1, bLength);
        }
    } else {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            BatchHalfAlignVF<T, IsBoardCast>(
                (__local_mem__ T*)inUb.GetPhyAddr(), (__local_mem__ T*)cos.GetPhyAddr(),
                (__local_mem__ T*)sin.GetPhyAddr(), (__local_mem__ T*)outUb.GetPhyAddr(), bLength, 1, 1, tilingData_->sliceLength, dAlign_,
                ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            BatchInterleaveModeVF<T, IsBoardCast>(
                (__local_mem__ T*)inUb.GetPhyAddr(), (__local_mem__ T*)cos.GetPhyAddr(),
                (__local_mem__ T*)sin.GetPhyAddr(), (__local_mem__ T*)outUb.GetPhyAddr(), bLength, 1, 1, tilingData_->sliceLength, dAlign_,
                ubFactorB_, 1);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
            BatchQuarterAlignVF<T, IsBoardCast>(
                (__local_mem__ T*)inUb.GetPhyAddr(), (__local_mem__ T*)cos.GetPhyAddr(),
                (__local_mem__ T*)sin.GetPhyAddr(), (__local_mem__ T*)outUb.GetPhyAddr(), bLength, 1, 1, tilingData_->sliceLength, dAlign_,
                ubFactorB_, 1);
        } else {
            BatchDeepSeekInterleaveModeVF<T, IsBoardCast>(
                (__local_mem__ T*)inUb.GetPhyAddr(), (__local_mem__ T*)cos.GetPhyAddr(),
                (__local_mem__ T*)sin.GetPhyAddr(), (__local_mem__ T*)outUb.GetPhyAddr(), bLength, 1, 1, tilingData_->sliceLength, dAlign_,
                ubFactorB_, 1);
        }
    }

    this->qInQueue_.FreeTensor(inUb);
    this->qOutQueue_.template EnQue(outUb);
}
} // namespace InplacePartialRotaryMul

#endif