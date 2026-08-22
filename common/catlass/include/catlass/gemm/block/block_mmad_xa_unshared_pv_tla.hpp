#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_XA_UNSHARED_PV_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_XA_UNSHARED_PV_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class L1TileShape_, class L0TileShape_, class ElementA_,
    class ElementB_, class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadXAUnsharedPV<Arch::Ascend950>,
    L1TileShape_, L0TileShape_, ElementA_, ElementB_,
    ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadXAUnsharedPV<Arch::Ascend950>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;

    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using LayoutTagA = typename TileCopy::LayoutTagA;
    using LayoutTagB = typename TileCopy::LayoutTagB;
    using LayoutTagC = typename TileCopy::LayoutTagC;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    // PV shape is M x N x K. QK's N axis maps to PV's K axis.
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_K * L1_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);
    // QK owns a 128 x 256 L0C stage. PV reuses its first 128 x 128 result region.
    static constexpr uint32_t SHARED_L0C_STAGE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementAccumulator);

    static constexpr uint32_t BLOCK_L1_SIZE = L1B_TILE_SIZE * STAGES;
    static constexpr uint32_t BLOCK_L0C_SIZE = SHARED_L0C_STAGE_SIZE * STAGES;

    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "L1 and L0 tile shapes must match on the M and N axes");
    static_assert(
        L1_TILE_K >= L0_TILE_K && L1_TILE_K % L0_TILE_K == 0,
        "The L1 K tile must be an integer multiple of the L0 K tile");
    static_assert(BLOCK_L1_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(BLOCK_L0C_SIZE <= ArchTag::L0C_SIZE, "Shared L0C buffers exceeding the L0C space!");

public:
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t& l1BufAddrStart, uint32_t& l0CBufAddrStart)
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1BTensorList_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart);
            l1BufAddrStart += L1B_TILE_SIZE;
            l0ATensorList_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0BTensorList_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0CTensorList_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                l0CBufAddrStart + SHARED_L0C_STAGE_SIZE * i);

            l1BEventList_[i] = BLOCK_EVENT_ID + i + STAGES;
            l0AEventList_[i] = i;
            l0BEventList_[i] = i + STAGES;
            l0CEventList_[i] = i;

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
        }
    }

    template <class TensorA, class TensorB, class TensorC, class Shape>
    CATLASS_DEVICE void operator()(
        TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, Shape& actualShape,
        uint32_t& taskIdL0A, uint32_t& taskIdL0B, uint32_t& taskIdL0C)
    {
        uint32_t mActual = tla::get<0>(actualShape);
        uint32_t nActual = tla::get<1>(actualShape);
        uint32_t kActual = tla::get<2>(actualShape);

        auto layoutBInL1 = tla::MakeLayout<ElementB, LayoutTagL1B>(kActual, nActual);
        auto tensorBInL1 = tla::MakeTensor(l1BTensorList_[l1BListId_], layoutBInL1, Arch::PositionL1{});
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BListId_]);
        CopyGmToL1B copyGmToL1B;
        copyGmToL1B(tensorBInL1, tensorB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BListId_]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BListId_]);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[taskIdL0C]);
        auto layoutInL0C = tla::MakeLayoutL0C(mActual, nActual);
        auto tensorInL0C =
            tla::MakeTensor(l0CTensorList_[taskIdL0C], layoutInL0C, Arch::PositionL0C{});

        uint32_t kLoops = (kActual + L0_TILE_K - 1) / L0_TILE_K;
        for (uint32_t kIdx = 0; kIdx < kLoops; ++kIdx) {
            uint32_t kOffset = kIdx * L0_TILE_K;
            uint32_t tileK = kIdx + 1 == kLoops ? kActual - kOffset : L0_TILE_K;

            auto tensorATileInL1 = GetTile(
                tensorA, tla::MakeCoord(0, kOffset), tla::MakeShape(mActual, tileK));
            auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mActual, tileK);
            auto tensorAInL0 =
                tla::MakeTensor(l0ATensorList_[taskIdL0A], layoutAInL0, Arch::PositionL0A{});
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[taskIdL0A]);
            copyL1ToL0A_(tensorAInL0, tensorATileInL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[taskIdL0A]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[taskIdL0A]);

            auto tensorBTileInL1 = GetTile(
                tensorBInL1, tla::MakeCoord(kOffset, 0), tla::MakeShape(tileK, nActual));
            auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(tileK, nActual);
            auto tensorBInL0 =
                tla::MakeTensor(l0BTensorList_[taskIdL0B], layoutBInL0, Arch::PositionL0B{});
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[taskIdL0B]);
            copyL1ToL0B_(tensorBInL0, tensorBTileInL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList_[taskIdL0B]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList_[taskIdL0B]);

            tileMmad_(tensorInL0C, tensorAInL0, tensorBInL0, mActual, nActual, tileK, kIdx == 0);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[taskIdL0A]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[taskIdL0B]);
            taskIdL0A = 1 - taskIdL0A;
            taskIdL0B = 1 - taskIdL0B;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BListId_]);
        l1BListId_ = 1 - l1BListId_;

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[taskIdL0C]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList_[taskIdL0C]);

        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(tensorC, tensorInL0C);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList_[taskIdL0C]);
        taskIdL0C = 1 - taskIdL0C;
    }

private:
    static constexpr uint32_t SYNC_MODE = 4;
    static constexpr uint32_t AIV1_EVENT_OFFSET = 16;
    static constexpr uint32_t BLOCK_EVENT_ID = 4;

    AscendC::LocalTensor<ElementB> l1BTensorList_[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList_[STAGES];

    int32_t l1BEventList_[STAGES];
    int32_t l0AEventList_[STAGES];
    int32_t l0BEventList_[STAGES];
    int32_t l0CEventList_[STAGES];

    uint32_t l1BListId_{0};

    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_XA_UNSHARED_PV_TLA_HPP
