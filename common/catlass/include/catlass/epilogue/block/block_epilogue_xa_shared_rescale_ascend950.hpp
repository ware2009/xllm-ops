
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

template <class L1TileShape_, class OTmpType_>
class BlockEpilogue<
    EpilogueAscend950XASharedRescaleO, 
    L1TileShape_, 
    OTmpType_> {
public:
    using DispatchPolicy = EpilogueAscend950XASharedRescaleO;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using ElementOTmp = typename OTmpType_::Element;
    using LayoutTagOTmp = typename OTmpType_::Layout;
    using L1TileShape = L1TileShape_;

    static constexpr uint32_t S1_BASE_SIZE = tla::get<0>(L1TileShape{});
    static constexpr uint32_t D_BASE_SIZE = tla::get<2>(L1TileShape{});
    static constexpr uint32_t HALF_S1_BASE_SIZE = S1_BASE_SIZE / 2;
    static constexpr uint32_t VEC2_UB_SIZE = HALF_S1_BASE_SIZE * D_BASE_SIZE * sizeof(ElementOTmp);

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, uint32_t& ubBufAddrStart)
    {
        attnTmpBuf = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubBufAddrStart);
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
        TensorDst& attenOutGm, 
        const AscendC::LocalTensor<ElementOTmp>& expMaxUb,
        TensorSrc& pvRes, 
        bool isFirstKv, 
        bool isLastKv,
        uint16_t PV_RELEASE_FLAG)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOMTE3V);
        uint32_t m = tla::get<0>(pvRes.shape());
        uint32_t n = tla::get<1>(pvRes.shape());
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementOTmp));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;

        __ubuf__ ElementOTmp* pvResUbAddr = (__ubuf__ ElementOTmp*)pvRes.data().GetPhyAddr();
        __ubuf__ ElementOTmp* attnTmpAddr = (__ubuf__ ElementOTmp*)attnTmpBuf.GetPhyAddr();
        __ubuf__ ElementOTmp* expMaxUbAddr = (__ubuf__ ElementOTmp*)expMaxUb.GetPhyAddr();

        if (isFirstKv) {
            DataCopy(attnTmpBuf, pvRes.data(), m * n);
        } else {
            FlashUpdateNew<ElementOTmp, D_BASE_SIZE>(attnTmpAddr, pvResUbAddr, expMaxUbAddr, static_cast<uint16_t>(m), nLoops, tailN);
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
    AscendC::LocalTensor<ElementOTmp> attnTmpBuf;
    static constexpr int32_t SYNC_MODE = 4;
    static constexpr uint16_t FLOAT_REP_SIZE = 64;
    static constexpr int32_t eventOVMTE3 = 3;
    static constexpr int32_t eventOMTE3V = 3;

    template <class T, uint16_t DBaseSize>
    __simd_vf__ inline void FlashUpdateNew(
        __ubuf__ T* updateUb, __ubuf__ T* curUb, __ubuf__ T* expMaxUb, uint16_t m, uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::Reg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> preSrcVreg;
        RegTensor<float> curSrcVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> addVreg;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<T, LoadDist::DIST_BRC_B32>(expMaxVreg, expMaxUb + i);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(preSrcVreg, updateUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                LoadAlign(curSrcVreg, curUb + i * DBaseSize + j * FLOAT_REP_SIZE);
                Mul(mulVreg, expMaxVreg, preSrcVreg, pregFull);
                Add(addVreg, mulVreg, curSrcVreg, pregFull);
                StoreAlign<T, StoreDist::DIST_NORM_B32>(
                    updateUb + i * DBaseSize + j * FLOAT_REP_SIZE, addVreg, pregFull);
            }
            LoadAlign(preSrcVreg, updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            LoadAlign(curSrcVreg, curUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE);
            Mul(mulVreg, expMaxVreg, preSrcVreg, pregTailN);
            Add(addVreg, mulVreg, curSrcVreg, pregTailN);
            StoreAlign<T, StoreDist::DIST_NORM_B32>(
                updateUb + i * DBaseSize + nLoops * FLOAT_REP_SIZE, addVreg, pregTailN);
        }
    }

};
} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ASCEND950
