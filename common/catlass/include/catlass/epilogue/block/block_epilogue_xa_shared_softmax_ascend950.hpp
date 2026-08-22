
#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_SHARED_SOFTMAX_ASCEND950
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_SHARED_SOFTMAX_ASCEND950

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
    EpilogueAscend950XASharedSoftmax, L1TileShape_, PType_, SType_> {
public:
    using DispatchPolicy = EpilogueAscend950XASharedSoftmax;
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
    static constexpr int32_t HALF_BRCB_SIZE = S1_BASE_SIZE * 32;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementP);

    CATLASS_DEVICE
    BlockEpilogue(
        Arch::Resource<ArchTag>& resource, uint32_t& ubBufAddrStart, ElementS scaleValue_, uint32_t qHeads_)
    {
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        scaleValue = scaleValue_;
        qHeads = qHeads_;
        for (int32_t i = 0; i < TASK_NUM2; i++) {
            pNzOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementP>(ubBufAddrStart);
            ubBufAddrStart += HALF_SCM_BLOCK_SIZE;
        }

        for (int32_t i = 0; i < TASK_NUM3; i++) {
            eventUbPMTE3VList[i] = eventMTE3V++;
            eventUbPVMTE3List[i] = eventVMTE3++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbPMTE3VList[i]);
        }

        maxBrcbTensor = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
        ubBufAddrStart += HALF_BRCB_SIZE;
        sumBrcbTensor = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
        ubBufAddrStart += HALF_BRCB_SIZE;
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
        AscendC::LocalTensor<ElementS>& lastExpSum,
        AscendC::LocalTensor<ElementS>& nowExpSum,
        AscendC::LocalTensor<ElementS>& nowExpMax,
        AscendC::LocalTensor<ElementS>& lastMax,
        AscendC::LocalTensor<ElementS>& nowMax,
        AscendC::GlobalTensor<ElementS>& sharedMaxGm,
        AscendC::GlobalTensor<ElementS>& sharedSumGm,
        bool isUpdate,
        bool isLastKv,
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
        __ubuf__ ElementS* lastMaxUbAddr = (__ubuf__ ElementS*)lastMax.GetPhyAddr();
        __ubuf__ ElementS* nowMaxUbAddr = (__ubuf__ ElementS*)nowMax.GetPhyAddr();
        __ubuf__ ElementS* lastExpSumUbAddr = (__ubuf__ ElementS*)lastExpSum.GetPhyAddr();
        __ubuf__ ElementS* nowExpSumUbAddr = (__ubuf__ ElementS*)nowExpSum.GetPhyAddr();
        __ubuf__ ElementS* nowExpMaxUbAddr = (__ubuf__ ElementS*)nowExpMax.GetPhyAddr();

        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_QK_READY_FLAG);

        if (n == 128) {
            ComputeScaleandMax<ElementS, S2_BASE_SIZE, NRangeIndex::N128>(
                inputAddr, nowMaxUbAddr, static_cast<uint16_t>(m), tailN, scaleValue);
        } else if (n <= 64) {
            ComputeScaleandMax<ElementS, S2_BASE_SIZE, NRangeIndex::N0_64>(
                inputAddr, nowMaxUbAddr, static_cast<uint16_t>(m), tailN, scaleValue);
        } else {
            ComputeScaleandMax<ElementS, S2_BASE_SIZE, NRangeIndex::N65_127>(
                inputAddr, nowMaxUbAddr, static_cast<uint16_t>(m), tailN, scaleValue);
        }

        if (likely(isUpdate)) {
            UpdateMax<ElementS>(nowMaxUbAddr, lastMaxUbAddr, tailM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(taskIdMod3);
        if (unlikely(n > 64)) {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N128>(
                outputAddr, inputAddr, nowMaxUbAddr, nowExpSumUbAddr, static_cast<uint16_t>(m), blockStride);
        } else {
            ComputeExpSubSum<ElementP, ElementS, S2_BASE_SIZE, NRangeIndex::N0_64>(
                outputAddr, inputAddr, nowMaxUbAddr, nowExpSumUbAddr, static_cast<uint16_t>(m), blockStride);
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
        
        if (likely(isUpdate)) {
            UpdateExpSumAndExpMax<ElementS>(lastExpSumUbAddr, nowExpMaxUbAddr, lastMaxUbAddr, nowExpSumUbAddr, nowMaxUbAddr, tailM);
        }

        if (unlikely(isLastKv) ) {
            CopyOutMaxAndSum(sharedMaxGm, sharedSumGm, nowMax, nowExpSum, m);
        }

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
    static constexpr uint32_t FLOATS_PER_BLOCK = 32 / sizeof(ElementS);


    int32_t eventUbPMTE3VList[TASK_NUM3];
    int32_t eventUbPVMTE3List[TASK_NUM3];
    uint32_t qHeads;

    ElementS scaleValue;
    AscendC::LocalTensor<ElementP> pNzOutTensorList[TASK_NUM2];
    AscendC::LocalTensor<ElementS> maxBrcbTensor;
    AscendC::LocalTensor<ElementS> sumBrcbTensor;

    enum class NRangeIndex
    {
        N0_64 = 0,
        N65_127,
        N128,
        N128_INF
    };

    template <typename ElementS, uint16_t S2BaseSize, NRangeIndex NRange>
    __simd_vf__ inline void ComputeScaleandMax(
        __ubuf__ ElementS* srcUb, __ubuf__ ElementS* newMaxUb,
        uint16_t m, uint32_t tailN, ElementS dScale)
    {
        using namespace AscendC::Reg;
        RegTensor<ElementS> minVreg;
        RegTensor<ElementS> srcVreg0;
        RegTensor<ElementS> srcVreg1;
        RegTensor<ElementS> maxVreg;
        RegTensor<ElementS> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementS>(tailN);

        Duplicate(minVreg, MIN_VALUE);

        for (uint16_t i = 0; i < m; ++i) {
            if constexpr (NRange > NRangeIndex::N0_64) {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                LoadAlign(srcVreg1, srcUb + i * S2BaseSize + FLOAT_REP_SIZE);
                Muls(srcVreg0, srcVreg0, dScale, pregFull);
                Muls(srcVreg1, srcVreg1, dScale, pregTailN);
                if constexpr (NRange < NRangeIndex::N128) {
                    Select(srcVreg1, srcVreg1, minVreg, pregTailN);
                }
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(
                    srcUb + i * S2BaseSize + FLOAT_REP_SIZE, srcVreg1, pregFull);
                Max(maxTmpVreg, srcVreg0, srcVreg1, pregFull);
                ReduceMax(maxVreg, maxTmpVreg, pregFull);
            } else {
                LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
                Muls(srcVreg0, srcVreg0, dScale, pregTailN);
                Select(srcVreg0, srcVreg0, minVreg, pregTailN);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(srcUb + i * S2BaseSize, srcVreg0, pregFull);
                ReduceMax(maxVreg, srcVreg0, pregFull);
            }
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxVreg, maxUreg, 1);
        }
        StoreUnAlignPost(newMaxUb, maxUreg, 0);
    }

    template <typename ElementS>
    __simd_vf__ inline void UpdateMax(__ubuf__ ElementS* nowMaxUb, __ubuf__ ElementS* lastMaxUb, uint32_t tailM)
    {
        using namespace AscendC::Reg;
        RegTensor<ElementS> nowMaxVreg;
        RegTensor<ElementS> lastMaxVreg;
        RegTensor<ElementS> maxVreg;

        MaskReg pregTailM = UpdateMask<ElementS>(tailM);
        LoadAlign(lastMaxVreg, lastMaxUb);
        LoadAlign(nowMaxVreg, nowMaxUb);
        Max(maxVreg, nowMaxVreg, lastMaxVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(nowMaxUb, maxVreg, pregTailM);
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
    __simd_vf__ inline void UpdateExpSumAndExpMax(
        __ubuf__ ElementS* sumUb, __ubuf__ ElementS* expMaxUb, __ubuf__ ElementS* maxUb, __ubuf__ ElementS* expSumUb,
        __ubuf__ ElementS* nowMaxUb, uint32_t tailM)
    {
        using namespace AscendC::Reg;
        RegTensor<ElementS> nowMaxVreg;
        RegTensor<ElementS> lastMaxVreg;
        RegTensor<ElementS> expMaxVreg;
        RegTensor<ElementS> lastExpSumVreg;
        RegTensor<ElementS> brcExpSumVreg;
        MaskReg pregTailM = UpdateMask<ElementS>(tailM);
        LoadAlign(lastMaxVreg, maxUb);
        LoadAlign(nowMaxVreg, nowMaxUb);
        ExpSub(expMaxVreg, lastMaxVreg, nowMaxVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(maxUb, nowMaxVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb);
        LoadAlign(brcExpSumVreg, expSumUb);
        MulDstAdd(expMaxVreg, lastExpSumVreg, brcExpSumVreg, pregTailM);
        StoreAlign<ElementS, StoreDist::DIST_NORM_B32>(expSumUb, expMaxVreg, pregTailM);
    }

    CATLASS_DEVICE
    void CopyOutMaxAndSum(
        AscendC::GlobalTensor<ElementS> maxGm,
        AscendC::GlobalTensor<ElementS> sumGm,
        AscendC::LocalTensor<ElementS> maxUb,
        AscendC::LocalTensor<ElementS> sumUb,
        uint16_t tailM
    )
    {

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);

        uint8_t repeatTimes = (tailM + 7) / 8;
        AscendC::Brcb(
            maxBrcbTensor,
            maxUb,
            repeatTimes,
            AscendC::BrcbRepeatParams(1, FLOATS_PER_BLOCK));

        AscendC::Brcb(
            sumBrcbTensor,
            sumUb,
            repeatTimes,
            AscendC::BrcbRepeatParams(1, FLOATS_PER_BLOCK));
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);

        AscendC::DataCopyExtParams copyOutParams(
            static_cast<uint16_t>(tailM),
            static_cast<uint32_t>(sizeof(ElementS)),
            0,
            static_cast<uint32_t>((qHeads - 1) * sizeof(ElementS)),
            0
        );

        AscendC::DataCopyPad(maxGm, maxBrcbTensor, copyOutParams);
        AscendC::DataCopyPad(sumGm, sumBrcbTensor, copyOutParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);

    }
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_SHARED_SOFTMAX_ASCEND950
