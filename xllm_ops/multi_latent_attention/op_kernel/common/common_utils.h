#ifndef OP_KERNEL_COMMON_UTILS_H
#define OP_KERNEL_COMMON_UTILS_H

// ============================================================================
// common_utils.h: Platform-independent AscendC API wrappers and math utilities
//
// Extracted from arch32/mixkernels/include/utils.h and common_func.h.
// Contains ONLY platform-independent functions that work across A2/A3/A5.
// A2-specific functions (CopyCbufToFbuf/CopyCbufToBt using address_.logicPos,
// and SetQuantPreAddr using ArchType) are NOT included here.
// ============================================================================

#include <limits>
#include <type_traits>

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#else
#include "kernel_macros.h"
#endif

// ============================================================================
// Math utility functions (from common_func.h)
// ============================================================================

template <uint32_t ALIGN, typename T = uint32_t>
inline __aicore__ T RoundUp(const T val)
{
    static_assert(ALIGN != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T align = ALIGN;
    if (val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <uint32_t DIVISOR, typename T = uint32_t>
inline __aicore__ T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T divisor = DIVISOR;
    if (dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return T_MAX<T>;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline T Min(const T lhs, const T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint32_t BlockSize()
{
    return 32 / sizeof(Dtype);
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint32_t MatrixSize()
{
    return 512 / sizeof(Dtype);
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint64_t BlockSizeRoundUp(uint64_t num)
{
    return (num + BlockSize<Dtype>() - 1) / BlockSize<Dtype>() * BlockSize<Dtype>();
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint64_t NumBlocksRoundUp(uint64_t num)
{
    return (num + BlockSize<Dtype>() - 1) / BlockSize<Dtype>();
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint64_t MatrixSizeRoundUp(uint64_t num)
{
    return (num + MatrixSize<Dtype>() - 1) / MatrixSize<Dtype>() * MatrixSize<Dtype>();
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint64_t NumMatrixsRoundUp(uint64_t num)
{
    return (num + MatrixSize<Dtype>() - 1) / MatrixSize<Dtype>();
}

template <typename Dtype> __aicore__ __attribute__((always_inline)) inline uint64_t L0HalfSize()
{
    return 32 * 1024 / sizeof(Dtype);
}

// ============================================================================
// AscendC API wrapper functions (from utils.h, platform-independent)
// ============================================================================

template <typename IN_DTYPE>
__aicore__ inline void CreateCaMatrix(const AscendC::LocalTensor<IN_DTYPE> &dst,
                                      const uint16_t repeats, const uint16_t blockNum,
                                      const uint16_t dstGap, const IN_DTYPE initValue)
{
    AscendC::InitConstValue<IN_DTYPE>(dst,
        AscendC::InitConstValueParams<IN_DTYPE>(repeats, blockNum, dstGap, initValue));
}

__aicore__ inline void SetFftsBaseAddr(uint64_t config) { AscendC::SetSyncBaseAddr(config); }

template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue) { AscendC::SetLoadDataPaddingValue<IN_DTYPE>(padValue); }

__aicore__ inline void SetAtomicnone() { AscendC::SetAtomicNone(); }

__aicore__ inline void SetMasknorm() {
#if __CCE_AICORE__ == 100
    return;
#endif
    AscendC::SetMaskNorm();
}

__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride) {
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);
}

template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow) {
    AscendC::SetVectorMask<IN_DTYPE>(maskHigh, maskLow);
}

__aicore__ inline int64_t GetSubBlockidx() { return AscendC::GetSubBlockIdx(); }

__aicore__ inline void WaitFlagDev(uint16_t flagId) { AscendC::WaitEvent(flagId); }

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId) { AscendC::CrossCoreSetFlag<mode, pipe>(flagId); }

template <typename IN_DTYPE, bool setRelu = false>
__aicore__ inline void SetFpc(const AscendC::LocalTensor<IN_DTYPE> &preTensor, bool isUnitFlag = false) {
    AscendC::SetFixPipeConfig<IN_DTYPE, setRelu>(preTensor, isUnitFlag);
}

#endif // OP_KERNEL_COMMON_UTILS_H