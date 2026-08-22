
#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_UNSHARED_SOFTMAX_ASCEND950
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_UNSHARED_SOFTMAX_ASCEND950

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Epilogue::Block {

template <
    class L1TileShape_, 
    class PType_, 
    class SType_>
class BlockEpilogue<
    EpilogueAscend950XAUnsharedSoftmax, L1TileShape_, PType_, SType_> {
public:
    using DispatchPolicy = EpilogueAscend950XAUnsharedSoftmax;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using ElementP = typename PType_::Element;
    using ElementS = typename SType_::Element;
    using LayoutTagP = typename PType_::Layout;

    static constexpr uint32_t S1_BASE_SIZE = tla::get<0>(L1TileShape{});
    static constexpr uint32_t S2_BASE_SIZE = tla::get<1>(L1TileShape{});
    static constexpr int32_t HALF_S1_BASE_SIZE = S1_BASE_SIZE >> 1;
    static constexpr int32_t HALF_VEC_SIZE = HALF_S1_BASE_SIZE * sizeof(ElementS);
    static constexpr int32_t HALF_SCM_BLOCK_SIZE = HALF_S1_BASE_SIZE * S2_BASE_SIZE * sizeof(ElementP);
    static constexpr int32_t HALF_MASK_BLOCK_SIZE = HALF_S1_BASE_SIZE * S2_BASE_SIZE * sizeof(uint8_t);
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementP);

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, uint32_t& ubBufAddrStart, ElementS scaleValue_, uint32_t unsharedKvLen_,
        uint32_t maxDecodeStep_, uint32_t groupCountPerLoop_, uint32_t groupSize_)
    {
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        scaleValue = scaleValue_;
        unsharedKvLen = unsharedKvLen_;
        maxDecodeStep = maxDecodeStep_;
        groupCountPerLoop = groupCountPerLoop_;
        groupSize = groupSize_;
        
        for (int32_t i = 0; i < TASK_NUM2; i++) {
            pNzOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementP>(ubBufAddrStart);
            ubBufAddrStart += HALF_SCM_BLOCK_SIZE;
        }
        maskUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(ubBufAddrStart);
        ubBufAddrStart += HALF_MASK_BLOCK_SIZE;
        maxUbTensor = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
        ubBufAddrStart += HALF_VEC_SIZE;
        sumUbTensor = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
        ubBufAddrStart += HALF_VEC_SIZE;
        
        InitUnsharedMask(groupCountPerLoop, groupSize, unsharedKvLen);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < TASK_NUM3; i++) {
            eventUbPMTE3VList[i] = eventMTE3V++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbPMTE3VList[i]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < TASK_NUM3; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbPMTE3VList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst& pOutL1, 
        TensorSrc& qkRes,
        AscendC::GlobalTensor<ElementS>& unSharedMaxGm,
        AscendC::GlobalTensor<ElementS>& unSharedSumGm,
        uint16_t SYNC_QK_READY_FLAG,
        uint16_t SYNC_SOFTMAX_READY_FLAG,
        uint16_t QK_UB_RELEASE_FLAG,
        int8_t taskIdMod2,
        int8_t taskIdMod3)
    {
        uint32_t m = tla::get<0>(qkRes.shape());
        uint32_t n = tla::get<1>(qkRes.shape());
        uint32_t blockStride = AscendC::CeilDivision(m, ELE_NUM_PER_C0) * ELE_NUM_PER_C0;
        auto pNzUbLayout = tla::MakeLayout(
            tla::MakeShape(
                tla::MakeShape(m, tla::Int<1>{}),
                tla::MakeShape(
                    tla::Int<ELE_NUM_PER_C0>{}, AscendC::CeilDivision(S2_BASE_SIZE, tla::Int<ELE_NUM_PER_C0>{}))),
            tla::MakeStride(
                tla::MakeStride(tla::Int<ELE_NUM_PER_C0>{}, tla::Int<ELE_NUM_PER_C0>{} * m),
                tla::MakeStride(tla::Int<1>{}, blockStride * ELE_NUM_PER_C0)));
        auto pOutUb = tla::MakeTensor(pNzOutTensorList[taskIdMod2], pNzUbLayout, Arch::PositionUB{});

        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementS));
        uint32_t tailN = (n - 1) % vlSize + 1;
        uint32_t tailM = m;

        __ubuf__ ElementP* outputAddr = (__ubuf__ ElementP*)pOutUb.data().GetPhyAddr();
        __ubuf__ ElementS* inputAddr = (__ubuf__ ElementS*)qkRes.data().GetPhyAddr();
        __ubuf__ ElementS* maxUbAddr = (__ubuf__ ElementS*)maxUbTensor.GetPhyAddr();
        __ubuf__ ElementS* sumUbAddr = (__ubuf__ ElementS*)sumUbTensor.GetPhyAddr();
        __ubuf__ uint8_t* maskUbUnRoll1 = (__ubuf__ uint8_t*)maskUbTensor.GetPhyAddr();
        __ubuf__ uint8_t* maskUbUnRoll2 = maskUbUnRoll1 + FLOAT_REP_SIZE;

        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_QK_READY_FLAG);
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        if (n == 128) {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N128>(
                inputAddr, maskUbUnRoll1, maskUbUnRoll2, maxUbAddr,
                static_cast<uint16_t>(m), tailN, scaleValue);
        } else if (n <= 64) {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N0_64>(
                inputAddr, maskUbUnRoll1, maskUbUnRoll2, maxUbAddr,
                static_cast<uint16_t>(m), tailN, scaleValue);
        } else {
            ComputeMaskandScale<ElementS, S2_BASE_SIZE, NRangeIndex::N65_127>(
                inputAddr, maskUbUnRoll1, maskUbUnRoll2, maxUbAddr,
                static_cast<uint16_t>(m), tailN, scaleValue);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        if (unlikely(n > 64)) {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N128>(
                outputAddr, inputAddr, maxUbAddr, sumUbAddr, static_cast<uint16_t>(m), blockStride);
        } else {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N0_64>(
                outputAddr, inputAddr, maxUbAddr, sumUbAddr, static_cast<uint16_t>(m), blockStride);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(taskIdMod2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(taskIdMod2);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(QK_UB_RELEASE_FLAG);

        if (likely(m != 0)) {
            using CopyUbToL1P = Tile::CopyUb2L1Tla<ArchTag, decltype(pOutUb), TensorDst>;
            CopyUbToL1P copyUb2L1P;
            copyUb2L1P(pOutL1, pOutUb);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_SOFTMAX_READY_FLAG);

        if (likely(m != 0)) {
            CopyOutMaxAndSum(unSharedMaxGm, unSharedSumGm, m);
        }
        
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);

    }

private:
    static constexpr uint32_t TASK_NUM2 = 2;
    static constexpr uint32_t TASK_NUM3 = 3;
    static constexpr int32_t SYNC_MODE = 4;
    static constexpr int64_t BLOCK_BYTES = 32;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr float MIN_VALUE = -3e38f;
    static constexpr uint16_t FLOAT_REP_SIZE = 64;
    static constexpr uint16_t DOUBLE_FLOAT_REP_SIZE = 128;
    static constexpr uint32_t UINT8_BLOCK_SIZE = 256;

    int32_t eventUbPMTE3VList[TASK_NUM3];

    uint32_t groupSize;
    uint32_t groupCountPerLoop;
    uint32_t unsharedKvLen;
    uint32_t maxDecodeStep;

    ElementS scaleValue;
    AscendC::LocalTensor<ElementP> pNzOutTensorList[TASK_NUM2];
    AscendC::LocalTensor<uint8_t> maskUbTensor;
    AscendC::LocalTensor<ElementS> maxUbTensor;
    AscendC::LocalTensor<ElementS> sumUbTensor;

    enum class NRangeIndex
    {
        N0_64 = 0,
        N65_127,
        N128,
        N128_INF
    };

    CATLASS_DEVICE
    void InitUnsharedMask(uint32_t groupCountPerLoop, uint32_t groupSize, uint32_t unsharedKvSeqLen)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t curGroupCount = (groupCountPerLoop + subBlockNum - 1) / subBlockNum;
        uint32_t curGroupThisSubBlock = (subBlockIdx == 0) ? curGroupCount : (groupCountPerLoop - curGroupCount);
        uint32_t groupOffset = (subBlockIdx == 0) ? 0 : curGroupCount;
        // init mask
        
        uint8_t repeatStride = S2_BASE_SIZE * sizeof(uint8_t);

        AscendC::Duplicate(maskUbTensor, uint8_t(0), HALF_MASK_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t round = 0; round < curGroupThisSubBlock; ++round) {
            uint32_t colOffset = (groupOffset + round) * maxDecodeStep;
            uint32_t colOffsetFloor = colOffset / UINT8_BLOCK_SIZE * UINT8_BLOCK_SIZE;
            uint32_t floorSub = colOffset - colOffsetFloor;
            uint64_t rowOffset = round * groupSize * S2_BASE_SIZE;
            uint64_t totalOffset = rowOffset + colOffsetFloor;
            auto totalDupLen = unsharedKvSeqLen + floorSub;
            for (int groupId = 0; groupId < groupSize; groupId++) {
                AscendC::Duplicate(maskUbTensor[totalOffset + groupId * repeatStride], uint8_t(1), totalDupLen);
            }
            // AscendC::Duplicate(maskUbTensor[totalOffset], uint8_t(1), static_cast<uint64_t>(totalDupLen), groupSize, 1, repeatStride);
            AscendC::PipeBarrier<PIPE_V>();

            if (floorSub > 0) {
                for (int groupId = 0; groupId < groupSize; groupId++) {
                    AscendC::Duplicate(maskUbTensor[totalOffset + groupId * repeatStride], uint8_t(0), floorSub);
                }
                // AscendC::Duplicate(maskUbTensor[totalOffset], uint8_t(0), static_cast<uint64_t>(floorSub), groupSize, 1, repeatStride);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // AscendC::printf("groupCountPerLoop %d unsharedKvLen %d curGroupCount %d S2_BASE_SIZE %d repeatStride %d\n", groupCountPerLoop, unsharedKvSeqLen, curGroupCount, S2_BASE_SIZE, repeatStride);

        // for (int i = 0; i < 8; i++) {
        //     AscendC::printf("token %d unshared mask\n", i);
        //     AscendC::DumpTensor(maskUbTensor[i * repeatStride], 10, 8);
        // }
    }

    template <typename ElementS, uint16_t S2BaseSize, NRangeIndex NRange>
    __simd_vf__ inline void ComputeMaskandScale(
        __ubuf__ ElementS* srcUb, 
        __ubuf__ uint8_t* maskUbUnRoll1, 
        __ubuf__ uint8_t* maskUbUnRoll2,
        __ubuf__ ElementS* maxUb,
        uint16_t m, uint32_t tailN, ElementS scale
    )
    {
        using namespace AscendC::Reg;
        static_assert(NRange <= NRangeIndex::N128, "ComputeMaskandScale only supports N <= 128");
        RegTensor<ElementS> minVreg;
        RegTensor<ElementS> srcVreg0;
        RegTensor<ElementS> srcVreg1;
        RegTensor<ElementS> maxVreg;
        RegTensor<ElementS> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare0;
        MaskReg pregCompare1;
        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementS>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            if constexpr (NRange == NRangeIndex::N0_64) {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                Muls(srcVreg0, srcVreg0, scale, pregTailN);
                LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                    pregCompare0, (__ubuf__ uint32_t*&)maskUbUnRoll1, S2BaseSize);
                Select(srcVreg0, srcVreg0, minVreg, pregCompare0);
                Select(srcVreg0, srcVreg0, minVreg, pregTailN);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize, srcVreg0, pregFull);
                ReduceMax(maxVreg, srcVreg0, pregFull);
            } else {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                LoadAlign(srcVreg1, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
                Muls(srcVreg0, srcVreg0, scale, pregFull);
                Muls(srcVreg1, srcVreg1, scale,
                    NRange == NRangeIndex::N128 ? pregFull : pregTailN);
                LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                    pregCompare0, (__ubuf__ uint32_t*&)maskUbUnRoll1, S2BaseSize);
                LoadAlign<uint32_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_DS>(
                    pregCompare1, (__ubuf__ uint32_t*&)maskUbUnRoll2, S2BaseSize);
                Select(srcVreg0, srcVreg0, minVreg, pregCompare0);
                Select(srcVreg1, srcVreg1, minVreg, pregCompare1);
                if constexpr (NRange == NRangeIndex::N65_127) {
                    Select(srcVreg1, srcVreg1, minVreg, pregTailN);
                }
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg1, pregFull);
                Max(maxTmpVreg, srcVreg0, srcVreg1, pregFull);
                ReduceMax(maxVreg, maxTmpVreg, pregFull);
            }
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(maxUb, maxVreg, maxUreg, 1);
        }
        StoreUnAlignPost(maxUb, maxUreg, 0);
        
    }

    template <typename ElementP, typename ElementS, uint16_t S2BaseSize, NRangeIndex NRange>
    __simd_vf__ inline void ComputeExpSubSum(
        __ubuf__ ElementP* expUb, __ubuf__ ElementS* srcUb, __ubuf__ ElementS* nowMaxUb, __ubuf__ ElementS* expSumUb,
        uint16_t m, uint32_t blockStride)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        RegTensor<ElementS> srcVreg0;
        RegTensor<ElementS> srcVreg1;
        RegTensor<ElementS> expVreg;
        RegTensor<ElementS> expVreg0;
        RegTensor<ElementS> expVreg1;
        RegTensor<ElementS> expSumVreg;
        RegTensor<ElementS> maxVreg;

        RegTensor<ElementP> expDstVreg0;
        RegTensor<ElementP> expDstVreg1;
        RegTensor<ElementP> expDstVreg;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregFull16 = CreateMask<uint16_t, MaskPattern::ALL>();
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementS, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            if constexpr (NRange > NRangeIndex::N0_64) {
                LoadAlign<ElementS, LoadDist::DIST_DINTLV_B32>(srcVreg0, srcVreg1, srcUb + i * S2BaseSize);
                ExpSub(expVreg0, srcVreg0, maxVreg, pregFull);
                ExpSub(expVreg1, srcVreg1, maxVreg, pregFull);
                Add(expVreg, expVreg0, expVreg1, pregFull);
                Cast<ElementP, ElementS, castTraitZero>(expDstVreg0, expVreg0, pregFull);
                Cast<ElementP, ElementS, castTraitOne>(expDstVreg1, expVreg1, pregFull);
                Or((RegTensor<uint16_t>&)expDstVreg, (RegTensor<uint16_t>&)expDstVreg0,
                   (RegTensor<uint16_t>&)expDstVreg1, pregFull16);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, expDstVreg, blockStride, REPEAT_STRIDE, pregFull16);
            } else {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                ExpSub(expVreg, srcVreg0, maxVreg, pregFull);
                Cast<ElementP, ElementS, castTraitZero>(expDstVreg, expVreg, pregFull);
                DeInterleave(expDstVreg0, expDstVreg1, expDstVreg, expDstVreg);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    expUb, expDstVreg0, blockStride, REPEAT_STRIDE, pregFull16);
            }
            ReduceSum(expSumVreg, expVreg, pregFull);
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        StoreUnAlignPost(expSumUb, expSumUreg, 0);
    }

    template <typename ElementS>
    CATLASS_DEVICE 
    void CopyOutMaxAndSum(
        AscendC::GlobalTensor<ElementS> maxGm,
        AscendC::GlobalTensor<ElementS> sumGm,
        uint16_t tailM)
    {
        AscendC::DataCopyExtParams copyOutParams(
            1,
            static_cast<uint32_t>(tailM) * sizeof(ElementS),
            0,
            0,
            0
        );

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::DataCopyPad(maxGm, maxUbTensor, copyOutParams);
        AscendC::DataCopyPad(sumGm, sumUbTensor, copyOutParams);
    }

};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_SHARED_SOFTMAX_ASCEND950
