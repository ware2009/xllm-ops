

 #ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_COMBINE_SCALE_ASCEND950_HPP
 #define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_COMBINE_SCALE_ASCEND950_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/arch/resource.hpp"
 #include "catlass/epilogue/dispatch_policy.hpp"
 #include "catlass/epilogue/tile/tile_copy.hpp"
 #include "catlass/gemm_coord.hpp"
 #include "catlass/matrix_coord.hpp"
 
 namespace Catlass::Epilogue::Block {
 
 template <class OutputType_, class InputType_>
 class BlockEpilogue<EpilogueAscend950XACombineScale, OutputType_, InputType_> {
 public:
    // Type aliases
    using DispatchPolicy = EpilogueAscend950XACombineScale;
    using ArchTag = typename DispatchPolicy::ArchTag; 

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;

    using LayoutOutput = typename OutputType_::Layout; 
    using LayoutInput = typename InputType_::Layout;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, uint32_t &ubBufAddrStart, uint32_t rowNumPerLoop_, uint32_t headDim_)
    {
        rowNumPerLoop = rowNumPerLoop_;
        headDim = headDim_;
        uint32_t reduceUbSize = rowNumPerLoop * sizeof(ElementInput);
        uint32_t inputUbSize = rowNumPerLoop * headDim * sizeof(ElementInput);
        
        for (int i = 0; i < 2; i++) {
            sharedAttnUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += inputUbSize;
            sharedGmUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += reduceUbSize;
            sharedGlUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += reduceUbSize;
            unsharedAttnUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += inputUbSize;
            unsharedGmUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += reduceUbSize;
            unsharedGlUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += reduceUbSize;
            finalAttnUbTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
            ubBufAddrStart += inputUbSize;
            eventAttnVMTE2[i] = i;
            eventGmGlVMTE2[i] = i + 2;
            eventOutMTE3V[i] = i;
        }

        finalGlUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
        ubBufAddrStart += reduceUbSize;
        expMaxSharedUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
        ubBufAddrStart += reduceUbSize;
        expMaxUnSharedUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(ubBufAddrStart);
        ubBufAddrStart += reduceUbSize;

        for (int i = 0; i < 2; i++) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventAttnVMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventGmGlVMTE2[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutMTE3V[i]);
        }
        
    }
 
    CATLASS_DEVICE
    ~BlockEpilogue() {
    for (int i = 0; i < 2; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventAttnVMTE2[i]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventGmGlVMTE2[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutMTE3V[i]);
    }
    }
 
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementInput> gSharedGm,
        AscendC::GlobalTensor<ElementInput> gUnsharedGm,
        AscendC::GlobalTensor<ElementInput> gSharedGl,
        AscendC::GlobalTensor<ElementInput> gUnsharedGl,
        AscendC::GlobalTensor<ElementInput> gSharedOut,
        AscendC::GlobalTensor<ElementInput> gUnsharedOut,
        AscendC::GlobalTensor<ElementOutput> gFinalOutput,
        uint32_t m, int8_t &taskId)
    {
        uint32_t attnCount = m * headDim;
        uint32_t gmglCount = m;
        // copyIn
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventAttnVMTE2[taskId]);
        AscendC::DataCopy(sharedAttnUbTensorList[taskId], gSharedOut, attnCount);
        AscendC::DataCopy(unsharedAttnUbTensorList[taskId], gUnsharedOut, attnCount);
        
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventGmGlVMTE2[taskId]);
        AscendC::DataCopy(sharedGmUbTensorList[taskId], gSharedGm, gmglCount);
        AscendC::DataCopy(sharedGlUbTensorList[taskId], gSharedGl, gmglCount);
        AscendC::DataCopy(unsharedGmUbTensorList[taskId], gUnsharedGm, gmglCount);
        AscendC::DataCopy(unsharedGlUbTensorList[taskId], gUnsharedGl, gmglCount);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(taskId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(taskId);

        __ubuf__ ElementInput* sharedAttnUbAddr = (__ubuf__ ElementInput*)sharedAttnUbTensorList[taskId].GetPhyAddr();
        __ubuf__ ElementInput* unsharedAttnUbAddr = (__ubuf__ ElementInput*)unsharedAttnUbTensorList[taskId].GetPhyAddr();
        __ubuf__ ElementInput* sharedGmUbAddr = (__ubuf__ ElementInput*)sharedGmUbTensorList[taskId].GetPhyAddr();
        __ubuf__ ElementInput* sharedGlUbAddr = (__ubuf__ ElementInput*)sharedGlUbTensorList[taskId].GetPhyAddr();
        __ubuf__ ElementInput* unsharedGmUbAddr = (__ubuf__ ElementInput*)unsharedGmUbTensorList[taskId].GetPhyAddr();
        __ubuf__ ElementInput* unsharedGlUbAddr = (__ubuf__ ElementInput*)unsharedGlUbTensorList[taskId].GetPhyAddr();
        __ubuf__ ElementInput* finalGlUbAddr = (__ubuf__ ElementInput*)finalGlUbTensor.GetPhyAddr();
        __ubuf__ ElementInput* expMaxSharedUbAddr = (__ubuf__ ElementInput*)expMaxSharedUbTensor.GetPhyAddr();
        __ubuf__ ElementInput* expMaxUnSharedUbAddr = (__ubuf__ ElementInput*)expMaxUnSharedUbTensor.GetPhyAddr();
        
        __ubuf__ ElementInput* finalAttnFloatUbAddr = (__ubuf__ ElementInput*)finalAttnUbTensorList[taskId].GetPhyAddr();

        ComputeExpSumAndExpMax(sharedGmUbAddr, sharedGlUbAddr, unsharedGmUbAddr, unsharedGlUbAddr,
                               finalGlUbAddr, expMaxSharedUbAddr, expMaxUnSharedUbAddr, m);
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventGmGlVMTE2[taskId]);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventOutMTE3V[taskId]);
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(headDim, vlSize) - 1;
        uint32_t tailN = (headDim - 1) % vlSize + 1;
        ComputeFinalAttn(sharedAttnUbAddr, unsharedAttnUbAddr, expMaxSharedUbAddr, expMaxUnSharedUbAddr,
                         finalGlUbAddr, finalAttnFloatUbAddr, static_cast<uint16_t>(m),
                         headDim, nLoops, tailN);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventAttnVMTE2[taskId]);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<ElementOutput> finalAttnUb = finalAttnUbTensorList[taskId].template ReinterpretCast<ElementOutput>();
        AscendC::Cast(finalAttnUb, finalAttnUbTensorList[taskId], AscendC::RoundMode::CAST_ROUND, attnCount);
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(taskId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(taskId);
        AscendC::DataCopy(gFinalOutput, finalAttnUb, attnCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventOutMTE3V[taskId]);
        taskId = 1 - taskId;
    }

private:
    template <typename ElementInput>
    __simd_vf__ inline void ComputeExpSumAndExpMax(
        __ubuf__ ElementInput* sharedGm, __ubuf__ ElementInput* sharedGl,
        __ubuf__ ElementInput* unsharedGm, __ubuf__ ElementInput* unsharedGl,
        __ubuf__ ElementInput* finalGl, __ubuf__ ElementInput* expMaxShared,
        __ubuf__ ElementInput* expMaxUnShared, uint32_t tailM
    )
    {
        using namespace AscendC::Reg;
        RegTensor<ElementInput> sharedMaxVreg;
        RegTensor<ElementInput> sharedSumVreg;
        RegTensor<ElementInput> unsharedMaxVreg;
        RegTensor<ElementInput> unsharedSumVreg;
        RegTensor<ElementInput> finalGmVreg;
        RegTensor<ElementInput> finalGlVreg;
        RegTensor<ElementInput> expMaxSharedVreg;
        RegTensor<ElementInput> expMaxUnSharedVreg;
        MaskReg pregTailM = UpdateMask<ElementInput>(tailM);

        LoadAlign(sharedMaxVreg, sharedGm);
        LoadAlign(unsharedMaxVreg, unsharedGm);
        Max(finalGmVreg, sharedMaxVreg, unsharedMaxVreg, pregTailM);
        ExpSub(expMaxSharedVreg, sharedMaxVreg, finalGmVreg, pregTailM);
        ExpSub(expMaxUnSharedVreg, unsharedMaxVreg, finalGmVreg, pregTailM);
        StoreAlign<ElementInput, StoreDist::DIST_NORM_B32>(expMaxShared, expMaxSharedVreg, pregTailM);
        StoreAlign<ElementInput, StoreDist::DIST_NORM_B32>(expMaxUnShared, expMaxUnSharedVreg, pregTailM);

        LoadAlign(sharedSumVreg, sharedGl);
        LoadAlign(unsharedSumVreg, unsharedGl);
        Mul(finalGlVreg, sharedSumVreg, expMaxSharedVreg, pregTailM);
        MulAddDst(finalGlVreg, unsharedSumVreg, expMaxUnSharedVreg, pregTailM);
        StoreAlign<ElementInput, StoreDist::DIST_NORM_B32>(finalGl, finalGlVreg, pregTailM);
    }

    template <class ElementInput>
    __simd_vf__ inline void ComputeFinalAttn(
        __ubuf__ ElementInput* sharedO, __ubuf__ ElementInput* unsharedO,
        __ubuf__ ElementInput* expMaxShared, __ubuf__ ElementInput* expMaxUnShared,
        __ubuf__ ElementInput* finalGl, __ubuf__ ElementInput* finalAttn,
        uint16_t m, uint32_t headDim, uint16_t nLoops, uint32_t tailN)
    {
        using namespace AscendC::Reg;
        RegTensor<ElementInput> sharedOVreg;
        RegTensor<ElementInput> unsharedOVreg;
        RegTensor<ElementInput> expMaxSharedVreg;
        RegTensor<ElementInput> expMaxUnsharedVreg;
        RegTensor<ElementInput> finalGlVreg;

        RegTensor<ElementInput> finalAttnVreg;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B32>(expMaxSharedVreg, expMaxShared + i);
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B32>(expMaxUnsharedVreg, expMaxUnShared + i);
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B32>(finalGlVreg, finalGl + i);
            for (uint16_t j = 0; j < nLoops; ++j) {
                LoadAlign(sharedOVreg, sharedO + i * headDim + j * FLOAT_REP_SIZE);
                LoadAlign(unsharedOVreg, unsharedO + i * headDim + j * FLOAT_REP_SIZE);
                Mul(sharedOVreg, sharedOVreg, expMaxSharedVreg, pregFull);
                Mul(unsharedOVreg, unsharedOVreg, expMaxUnsharedVreg, pregFull);
                Add(finalAttnVreg, sharedOVreg, unsharedOVreg, pregFull);
                Div(finalAttnVreg, finalAttnVreg, finalGlVreg, pregFull);
                StoreAlign<ElementInput, StoreDist::DIST_NORM_B32>(
                    finalAttn + i * headDim + j * FLOAT_REP_SIZE, finalAttnVreg, pregFull);
            }
            LoadAlign(sharedOVreg, sharedO + i * headDim + nLoops * FLOAT_REP_SIZE);
            LoadAlign(unsharedOVreg, unsharedO + i * headDim + nLoops * FLOAT_REP_SIZE);
            Mul(sharedOVreg, sharedOVreg, expMaxSharedVreg, pregTailN);
            Mul(unsharedOVreg, unsharedOVreg, expMaxUnsharedVreg, pregTailN);
            Add(finalAttnVreg, sharedOVreg, unsharedOVreg, pregTailN);
            Div(finalAttnVreg, finalAttnVreg, finalGlVreg, pregTailN);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B32>(
                finalAttn + i * headDim + nLoops * FLOAT_REP_SIZE, finalAttnVreg, pregTailN);
        }
    }
private:
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    AscendC::LocalTensor<ElementInput> sharedAttnUbTensorList[2];
    AscendC::LocalTensor<ElementInput> sharedGmUbTensorList[2];
    AscendC::LocalTensor<ElementInput> sharedGlUbTensorList[2];
    AscendC::LocalTensor<ElementInput> unsharedAttnUbTensorList[2];
    AscendC::LocalTensor<ElementInput> unsharedGmUbTensorList[2];
    AscendC::LocalTensor<ElementInput> unsharedGlUbTensorList[2];

    // tmp buffer
    AscendC::LocalTensor<ElementInput> finalGlUbTensor;
    AscendC::LocalTensor<ElementInput> expMaxSharedUbTensor;
    AscendC::LocalTensor<ElementInput> expMaxUnSharedUbTensor;

    AscendC::LocalTensor<ElementInput> finalAttnUbTensorList[2];
    int32_t eventAttnVMTE2[2];
    int32_t eventGmGlVMTE2[2];
    int32_t eventOutMTE3V[2];
    uint32_t headDim;
    uint32_t rowNumPerLoop;
 };
 
 } // namespace Catlass::Epilogue::Block
 
 #endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_XA_COMBINE_SCALE_ASCEND950_HPP
