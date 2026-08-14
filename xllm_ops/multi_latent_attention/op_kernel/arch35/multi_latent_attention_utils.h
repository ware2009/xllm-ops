/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// ============================================================================
// multi_latent_attention_utils.h
//
// A5 (ascend950 / DAV_3510) MLA low-level data-move helpers implemented with
// pure AscendC APIs (NO catlass / NO mixkernels gm_to_l1 template).
//
// These helpers wrap AscendC::DataCopy + Nd2NzParams to move Q / Q_rope from
// GM into L1 with an ND -> NZ layout transform. They are the A5 replacement
// for arch32's gm_to_l1<ArchType::ASCEND_V220, ...> template calls.
//
// The ND -> NZ semantics mirror arch32 gm_to_l1<ND, NZ>, whose body is exactly
//   AscendC::DataCopy(l1, gm, Nd2NzParams(1, nValue, dValue, 0,
//                                         srcDValue, dstNzC0Stride, 1, 0));
// Here we additionally expose ndNum / srcNdMatrixStride / dstNzNStride /
// dstNzMatrixStride so the multi-matrix (multi-token) batch path can be
// expressed with a single DataCopy instruction.
// ============================================================================

#ifndef MULTI_LATENT_ATTENTION_UTILS_ARCH35_H
#define MULTI_LATENT_ATTENTION_UTILS_ARCH35_H

#include "kernel_operator.h"
#include "../common/common_utils.h"  // RoundUp<>

namespace XllmOps {
namespace MlaArch35 {

// ----------------------------------------------------------------------------
// STRIDE_LIMIT -matches arch32 gm_to_l1_iterator.inc. When the per-row source
// stride (ND->NZ) or the inter-block source stride (NZ->NZ) reaches this bound,
// a single DataCopy can no longer encode it, so the transfer degrades into a
// per-row / per-block loop.
// ----------------------------------------------------------------------------
constexpr uint32_t STRIDE_LIMIT = 65536;

// ----------------------------------------------------------------------------
// A5(3510) FixPipe L0C->UB direct-write config.
// The Fixpipe<T, U, config> template takes `config` as a `const FixpipeConfig&`
// NON-TYPE template parameter, which requires a variable with LINKAGE. A
// function-local constexpr has NO linkage and is therefore rejected
// ("invalid explicitly-specified argument for template parameter 'config'").
// It MUST be defined at namespace scope.
//
// The built-in AscendC::CFG_ROW_MAJOR is {ROW_MAJOR, false} (isToUB=false =>
// L0C->GM). For a UB destination the second field (isToUB) MUST be true so the
// FIX unit routes to the UB address space; using the GM config against a UB
// address triggers error171 (FIXP L0C ECC) / 507015.
// ----------------------------------------------------------------------------
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
#endif

// ----------------------------------------------------------------------------
// LoadKVMainFromGMToL1 -K main (compressed latent, dim 512) GM -> L1.
//
// Pure-AscendC replacement for arch32 gm_to_l1<..., kNzIn?NZ:ND, NZ>. It owns
// BOTH layout modes plus the STRIDE_LIMIT degradation the iterator template
// implemented, selected at runtime by the kNzIn argument:
//
//   - kNzIn == false (ND->NZ) : source is a row-major ND block. Emits a single
//     Nd2Nz DataCopy when srcDValue < STRIDE_LIMIT, otherwise falls back to a
//     per-row loop (arch32 gm_to_l1<ND,NZ>, gm_to_l1_iterator.inc L67-105).
//   - kNzIn == true  (NZ->NZ) : source is already NZ (paged INT8/NZ KV cache).
//     Emits a single fractal DataCopy when the inter-fractal src stride
//     (nTileCeil - nTileActual) < STRIDE_LIMIT, otherwise a per-fractal loop
//     (arch32 gm_to_l1<NZ,NZ>, gm_to_l1_iterator.inc L31-62).
//
// dTileActual is the column dim (dValue): 512 for K main, 64 for K RoPE -so
// LoadKVRopeFromGMToL1 forwards to this same primitive with dTileActual = 64.
// kBlockSize (L1 fractal width) is derived from the L1 element type: 16 for
// half/bf16, 32 for int8 -matching HardwareInfo::l1l0BlockSize / sizeof(T).
//
// Parameters (all element units):
//   l1Tensor     : destination L1 sub-tensor (already offset by the caller)
//   gmTensor     : source GM sub-tensor (already offset to kv_offset)
//   nTileActual  : real row count for this block   (arch32 nTileActual)
//   nTileCeil    : rounded row count / NZ src pitch (arch32 nTileCeil)
//   dTileActual  : column dim (512 main / 64 rope)  (arch32 dTileActual)
//   srcDValue    : ND source row stride (stride_kv / stride_kv_rope); ignored
//                  by the NZ->NZ branch.
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadKVMainFromGMToL1(
    bool kNzIn,
    const AscendC::LocalTensor<DstT> &l1Tensor,
    const AscendC::GlobalTensor<SrcT> &gmTensor,
    uint32_t nTileActual,
    uint32_t nTileCeil,
    uint32_t dTileActual,
    uint32_t srcDValue)
{
    constexpr uint32_t kBlockSize = 32 / sizeof(DstT);  // half/bf16=16, int8=32
    if (kNzIn) {
        // NZ -> NZ : fractal-major copy, inter-fractal src gap = nTileCeil - nTileActual.
        uint32_t srcStride = nTileCeil - nTileActual;
        if (srcStride < STRIDE_LIMIT) {
            AscendC::DataCopy(
                l1Tensor,
                gmTensor,
                AscendC::DataCopyParams(
                    /* nBurst  */ dTileActual / kBlockSize,
                    /* lenBurst*/ nTileActual,
                    /* srcGap  */ nTileCeil - nTileActual,
                    /* dstGap  */ 0));
        } else {
            for (uint32_t i = 0; i < dTileActual / kBlockSize; i++) {
                AscendC::DataCopy(
                    l1Tensor[i * nTileActual * kBlockSize],
                    gmTensor[(uint64_t)i * nTileCeil * kBlockSize],
                    AscendC::DataCopyParams(
                        /* nBurst  */ 1,
                        /* lenBurst*/ nTileActual,
                        /* srcGap  */ 0,
                        /* dstGap  */ 0));
            }
        }
    } else {
        // ND -> NZ : row-major source, transform to NZ via Nd2Nz.
        if (srcDValue < STRIDE_LIMIT) {
            AscendC::DataCopy(
                l1Tensor,
                gmTensor,
                AscendC::Nd2NzParams(
                    /* ndNum            */ 1,
                    /* nValue           */ nTileActual,
                    /* dValue           */ dTileActual,
                    /* srcNdMatrixStride*/ 0,
                    /* srcDValue        */ srcDValue,
                    /* dstNzC0Stride    */ nTileCeil,
                    /* dstNzNStride     */ 1,
                    /* dstNzMatrixStride*/ 0));
        } else {
            for (uint32_t i = 0; i < nTileActual; i++) {
                AscendC::DataCopy(
                    l1Tensor[i * kBlockSize],
                    gmTensor[(uint64_t)i * srcDValue],
                    AscendC::Nd2NzParams(
                        /* ndNum            */ 1,
                        /* nValue           */ 1,
                        /* dValue           */ dTileActual,
                        /* srcNdMatrixStride*/ 0,
                        /* srcDValue        */ 0,
                        /* dstNzC0Stride    */ nTileCeil,
                        /* dstNzNStride     */ 0,
                        /* dstNzMatrixStride*/ 0));
            }
        }
    }
}

// ----------------------------------------------------------------------------
// LoadKVRopeFromGMToL1 -K RoPE (decoupled positional, dim 64) GM -> L1.
//
// Thin wrapper over LoadKVMainFromGMToL1: RoPE differs from K main ONLY in the
// column dim (dTileActual = 64 vs 512). The layout mode (kNzIn) and stride
// arguments are forwarded unchanged, so RoPE inherits the same ND->NZ / NZ->NZ
// selection and STRIDE_LIMIT degradation.
//
// Parameters mirror LoadKVMainFromGMToL1 (dTileActual is fixed to 64 here):
//   srcDValue : ND source row stride (stride_kv_rope); ignored when kNzIn.
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadKVRopeFromGMToL1(
    bool kNzIn,
    const AscendC::LocalTensor<DstT> &l1Tensor,
    const AscendC::GlobalTensor<SrcT> &gmTensor,
    uint32_t nTileActual,
    uint32_t nTileCeil,
    uint32_t srcDValue)
{
    LoadKVMainFromGMToL1(
        kNzIn,
        l1Tensor,
        gmTensor,
        nTileActual,
        nTileCeil,
        /* dTileActual */ 64,
        srcDValue);
}

// ----------------------------------------------------------------------------
// LoadQFromL1ToL0A -Q (L1) -> L0A, pure-AscendC replacement for arch32
//   l1_to_l0_a<..., DataFormat::VECTOR, DataFormat::VECTOR>.
//
// The arch32 VECTOR/VECTOR specialization body (l1_to_l0_iterator.inc L22-30)
// is exactly:
//   AscendC::LoadData(l0, l1,
//       LoadData2dParams(0, kPartCeil, kSrcStride, 0, kDstStride, IsTranspose, 0));
// The QK-stage call (arch32 L907-916) passes positional args
//   (l0, l1, mTileCeil=0, kPartCeil=repeat, mSrcStride=0, kSrcStride=srcStride,
//    mDstStride=0, kDstStride=0), so this helper reduces to a single LoadData
// with transpose == false.
//
// Parameters (all element units; caller pre-offsets both tensors):
//   l0aTensor : destination L0A sub-tensor
//   l1Tensor  : source L1 sub-tensor
//   repeat    : fractal repeat count  (arch32 round_embed_split_size / T_BLOCK_SIZE
//               for the main segment, round_embed_split_size / BLOCK_SIZE_16 for
//               the INT8 RoPE tail)
//   srcStride : inter-fractal L1 stride (arch32 q_load_coeff / BLOCK_SIZE_16)
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadQL1ToL0ARaw(
    const AscendC::LocalTensor<DstT> &l0aTensor,
    const AscendC::LocalTensor<SrcT> &l1Tensor,
    uint32_t repeat,
    uint32_t srcStride)
{
    AscendC::LoadData(
        l0aTensor,
        l1Tensor.template ReinterpretCast<DstT>(),
        AscendC::LoadData2dParams(
            /* baseIdx    */ 0,
            /* repeat     */ repeat,
            /* srcStride  */ srcStride,
            /* sid        */ 0,
            /* dstStride  */ 0,
            /* transpose  */ false,
            /* addrCalMode*/ 0));
}

// ----------------------------------------------------------------------------
// LoadKVFromL1ToL0B -KV (L1) -> L0B, pure-AscendC replacement for arch32
//   l1_to_l0_b<..., DataFormat::VECTOR, DataFormat::VECTOR>.
//
// Same VECTOR/VECTOR body as L0A (l1_to_l0_iterator.inc L147-148):
//   AscendC::LoadData(l0, l1,
//       LoadData2dParams(0, kPartCeil, kSrcStride, 0, kDstStride, IsTranspose, 0));
// The QK-stage call (arch32 L928-937) passes
//   (l0, l1, mTileCeil=0, kPartCeil=repeat, mSrcStride=0, kSrcStride=1,
//    mDstStride=0, kDstStride=0), i.e. srcStride is fixed to 1 for KV.
//
// Parameters (all element units; caller pre-offsets both tensors):
//   l0bTensor : destination L0B sub-tensor
//   l1Tensor  : source L1 KV sub-tensor
//   repeat    : fractal repeat count (arch32
//               round_embed_split_size * k_round_n / T_CUBE_MATRIX_SIZE for the
//               main segment, round_embed_split_size * qk_round_n /
//               CUBE_MATRIX_SIZE_A5 for the INT8 RoPE tail)
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadKVL1ToL0BRaw(
    const AscendC::LocalTensor<DstT> &l0bTensor,
    const AscendC::LocalTensor<SrcT> &l1Tensor,
    uint32_t repeat)
{
    // arch32 l1_to_l0_b uses a single IN_DTYPE for both L0B and L1-KV. In arch35
    // the L1-KV buffer is declared as half, so in the INT8 path (DstT=int8) the
    // source must be reinterpreted to DstT to satisfy AscendC::LoadData's
    // same-dtype requirement. The L0B destination type governs the transfer.
    AscendC::LoadData(
        l0bTensor,
        l1Tensor.template ReinterpretCast<DstT>(),
        AscendC::LoadData2dParams(
            /* baseIdx    */ 0,
            /* repeat     */ repeat,
            /* srcStride  */ 1,
            /* sid        */ 0,
            /* dstStride  */ 0,
            /* transpose  */ false,
            /* addrCalMode*/ 0));
}

// ----------------------------------------------------------------------------
// ComputeQKMMad -L0A x L0B -> L0C accumulate, pure-AscendC replacement for
//   arch32 mmad<ArchType::ASCEND_V220, ...>.
//
// arch32 mmad body reduces to AscendC::Mmad(l0c, l0a, l0b,
//   MmadParams(m, n, k, /*unitFlag*/0, /*kDirectionAlign*/false, cmatrixInitVal)).
// The QK stage varies only (n, k, cmatrixInitVal) between the FP16/BF16 main
// path, the INT8 main path (n == qk_round_n_l1) and the INT8 RoPE tail
// (cmatrixInitVal == true to accumulate onto the main QK score).
//
// Parameters:
//   l0cTensor : destination L0C sub-tensor
//   l0aTensor : L0A operand sub-tensor
//   l0bTensor : L0B operand sub-tensor
//   m, n, k   : GEMM dims (element units)
//   initC     : cmatrixInitVal -true resets the L0C accumulator (first split),
//               false accumulates onto it.
// ----------------------------------------------------------------------------
template <typename DstT, typename Src0T, typename Src1T>
__aicore__ __attribute__((always_inline)) inline void ComputeQKMMadRaw(
    const AscendC::LocalTensor<DstT> &l0cTensor,
    const AscendC::LocalTensor<Src0T> &l0aTensor,
    const AscendC::LocalTensor<Src1T> &l0bTensor,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool initC,
    uint8_t unitFlag = 3)
{
    // A5 (DAV_3510) Mmad idiom -mirror the official asc-devkit mmad.asc demo,
    // which calls AscendC::Mmad(co1, a2, b2, params) with the explicit prefix.
    // unitFlag drives the A5 CUBE pipeline unit grouping: 2 = mid-accumulation
    // (continue), 3 = pipeline end (commit). A K-split accumulation loop must set
    // 2 on the non-final split and 3 on the final split; a single-shot Mmad = 3.
    // A stuck unitFlag == 0 leaves the CUBE instruction stream with no terminator,
    // so the PC runs off into an illegal address (error263 CCU addr check 0x10f8).
    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.unitFlag = unitFlag;
    mmadParams.cmatrixSource = false;
    mmadParams.cmatrixInitVal = initC;
    AscendC::Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);
}

// ----------------------------------------------------------------------------
// CopyQKResultToGM -L0C -> GM (S score), pure-AscendC replacement for arch32
//   l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, ...>.
//
// A5 (DAV_3510) uses FixpipeParamsArch3510<CO2Layout::ROW_MAJOR>. quantPre is
// selected at compile time by the destination element type:
//   - bf16   -> QuantMode_t::F322BF16
//   - half   -> QuantMode_t::F322F16
//   - float  -> QuantMode_t::NoQuant
//   - int8/int32 accumulator (INT8 path) -> QuantMode_t::DEQF16 (Scalar Quant;
//     A5 does not support VDEQF16 Vector Quant for this fixpipe). deqScalar
//     wiring is left as a follow-up (arch32 leaves the INT8 dequant TODO too).
//
// Parameters (all element units):
//   gmTensor    : destination GM sub-tensor (already offset)
//   l0cTensor   : source L0C sub-tensor
//   mSize       : row count (arch32 MSize == m)
//   nSize       : column count (arch32 NSize == qk_n / qk_round_n)
//   srcStride   : L0C row stride (arch32 RoundUp<16>(m))
//   dstStride   : GM row stride  (arch32 qk_round_n)
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void CopyQKResultToGMRaw(
    const AscendC::GlobalTensor<DstT> &gmTensor,
    const AscendC::LocalTensor<SrcT> &l0cTensor,
    uint32_t mSize,
    uint32_t nSize,
    uint32_t srcStride,
    uint32_t dstStride)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> params(
        /* nSize     */ nSize,
        /* mSize     */ mSize,
        /* srcStride */ srcStride,
        /* dstStride */ dstStride);
    // A5 FixPipe closes the CUBE pipeline: unitFlag = 3 (pipeline end / commit).
    params.unitFlag = 3;

    if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value) {
        params.quantPre = QuantMode_t::F322BF16;
    } else if constexpr (AscendC::IsSameType<DstT, half>::value) {
        params.quantPre = QuantMode_t::F322F16;
    } else if constexpr (AscendC::IsSameType<DstT, float>::value) {
        params.quantPre = QuantMode_t::NoQuant;
    } else {
        // INT8 path (int32 accumulator -> lower precision). Scalar dequant only
        // on A5; deqScalar wiring pending (matches arch32 TODO).
        params.quantPre = QuantMode_t::DEQF16;
    }

    AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(gmTensor, l0cTensor, params);
}

// ----------------------------------------------------------------------------
// CopyQKResultToUBRaw - L0C -> UB (S score), A5 shared-UB variant of
//   CopyQKResultToGMRaw. Identical FixpipeParamsArch3510 / quantPre selection,
//   but the destination is a UB LocalTensor instead of GM (A5 FixPipe supports
//   both L0C->GM and L0C->UB; only the destination tensor kind differs). This
//   removes the S GM round-trip: AIC writes the QK score straight into the UB
//   region shared with the two AIV cores in the same AI Core.
//
// Parameters mirror CopyQKResultToGMRaw except gmTensor -> ubTensor.
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void CopyQKResultToUBRaw(
    const AscendC::LocalTensor<DstT> &ubTensor,
    const AscendC::LocalTensor<SrcT> &l0cTensor,
    uint32_t mSize,
    uint32_t nSize,
    uint32_t srcStride,
    uint32_t dstStride)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> params(
        /* nSize     */ nSize,
        /* mSize     */ ((mSize + 1) / 2) * 2,  // A5 UB direct-write needs M-dim
                                                // even alignment (== DivCeil(m,2)*2)
        /* srcStride */ srcStride,
        /* dstStride */ dstStride);
    // A5 UB-direct FixPipe: dualDstCtl = 0b01 splits by the M dim across the two
    // paired AIV UB halves; unitFlag = 3 terminates the CUBE pipeline.
    params.dualDstCtl = 0b01;
    params.unitFlag = 3;

    if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value) {
        params.quantPre = QuantMode_t::F322BF16;
    } else if constexpr (AscendC::IsSameType<DstT, half>::value) {
        params.quantPre = QuantMode_t::F322F16;
    } else if constexpr (AscendC::IsSameType<DstT, float>::value) {
        params.quantPre = QuantMode_t::NoQuant;
    } else {
        params.quantPre = QuantMode_t::DEQF16;
    }

    // FixPipe L0C->UB direct write. CFG_ROW_MAJOR_UB is defined at namespace
    // scope (see top of file) because the Fixpipe `config` non-type template
    // parameter is a `const FixpipeConfig&` and requires a variable with linkage.
    AscendC::Fixpipe<DstT, SrcT, CFG_ROW_MAJOR_UB>(ubTensor, l0cTensor, params);
}

// ----------------------------------------------------------------------------
// LoadVTransposeToL0BRaw -V (L1) -> L0B with transpose, pure-AscendC replacement
//   for arch32 AscendC::LoadDataWithTranspose used in PlatformLoadKVTranspose
//   {SmallN,LargeN}. Both arch32 branches wrap a single LoadDataWithTranspose in
//   a loop; this Raw wraps ONE call and leaves loop iteration + the
//   LoadData2dTransposeParams construction to the arch35 platform layer, which
//   picks the SmallN/LargeN field values per split.
//
// Parameters (all element units; caller pre-offsets both tensors):
//   l0bTensor : destination L0B sub-tensor (already offset)
//   l1Tensor  : source L1 KV sub-tensor   (already offset)
//   params    : LoadData2dTransposeParams built by the platform layer
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadVTransposeToL0BRaw(
    const AscendC::LocalTensor<DstT> &l0bTensor,
    const AscendC::LocalTensor<SrcT> &l1Tensor,
    const AscendC::LoadData2dTransposeParams &params)
{
    AscendC::LoadDataWithTranspose(
        l0bTensor,
        l1Tensor.template ReinterpretCast<DstT>(),
        params);
}

// ----------------------------------------------------------------------------
// LoadPL1ToL0AInt8Raw -P (L1) -> L0A for the INT8 path, pure-AscendC replacement
//   for arch32 l1_to_l0_a<..., false, DataFormat::NZ, DataFormat::ZZ>.
//
// The NZ->ZZ specialization (l1_to_l0_iterator.inc L97-126) loops
//   mTileCeil / 16 times issuing
//   AscendC::LoadData(l0[i*16*kPartCeil], l1[i*FRACTAL_SIZE],
//     LoadData2dParams(0, kPartCeil/COL_BLOCK_SIZE, mTileCeil/16, 0, 0, false, 0))
//   with COL_BLOCK_SIZE = 32 / sizeof(DataType) (== 32 for int8).
// arch32 calls it as (l0, l1, RoundUp<16>(row_num), qk_round_n_2_l1, 0,0,0,0),
// i.e. mTileCeil = RoundUp<16>(row_num), kPartCeil = qk_round_n_2_l1.
//
// Parameters (all element units; caller pre-offsets the L0A tensor):
//   l0aTensor : destination L0A sub-tensor (already offset)
//   l1Tensor  : source L1 P sub-tensor (base, iterated internally)
//   mTileCeil : row fractal count  (arch32 RoundUp<16>(row_num))
//   kPartCeil : col element count  (arch32 qk_round_n_2_l1)
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadPL1ToL0AInt8Raw(
    const AscendC::LocalTensor<DstT> &l0aTensor,
    const AscendC::LocalTensor<SrcT> &l1Tensor,
    uint32_t mTileCeil,
    uint32_t kPartCeil)
{
    // L1L0 block/fractal geometry for INT8 (16 x 32 fractal on A5, matching the
    // arch32 NZ->ZZ specialization constants).
    constexpr uint32_t ROW_BLOCK_SIZE = 16;
    constexpr uint32_t COL_BLOCK_SIZE = 32 / sizeof(DstT);
    constexpr uint32_t FRACTAL_SIZE = ROW_BLOCK_SIZE * COL_BLOCK_SIZE;
    const AscendC::LocalTensor<DstT> l1Cast = l1Tensor.template ReinterpretCast<DstT>();
    for (uint32_t i = 0; i < mTileCeil / ROW_BLOCK_SIZE; ++i) {
        AscendC::LoadData(
            l0aTensor[i * ROW_BLOCK_SIZE * kPartCeil],
            l1Cast[i * FRACTAL_SIZE],
            AscendC::LoadData2dParams(
                /* baseIdx    */ 0,
                /* repeat     */ static_cast<uint16_t>(kPartCeil / COL_BLOCK_SIZE),
                /* srcStride  */ mTileCeil / ROW_BLOCK_SIZE,
                /* sid        */ 0,
                /* dstStride  */ 0,
                /* transpose  */ false,
                /* addrCalMode*/ 0));
    }
}

// ----------------------------------------------------------------------------
// LoadPL1ToL0AGeneralRaw -P (L1) -> L0A for the non-INT8 path, pure-AscendC
//   replacement for arch32 l1_to_l0_a<..., false, VECTOR, VECTOR> issued inside
//   PlatformLoadPToL0AGeneral's loop.
//
// The VECTOR/VECTOR body (l1_to_l0_iterator.inc L22-30) is a single
//   AscendC::LoadData(l0, l1,
//     LoadData2dParams(0, kPartCeil, kSrcStride, 0, kDstStride, false, 0)).
// arch32 calls it per loa_load_idx as
//   (l0[..], l1[..], 0, qk_round_n_2/T_BLOCK_SIZE, 0, p_load_coeff/BLOCK_SIZE, 0, 0),
// i.e. kPartCeil = qk_round_n_2/T_BLOCK_SIZE, kSrcStride = p_load_coeff/BLOCK_SIZE,
// kDstStride = 0. This Raw wraps ONE LoadData; the platform layer keeps the loop
// and the sub-tensor offsets.
//
// Parameters (all element units; caller pre-offsets both tensors):
//   l0aTensor : destination L0A sub-tensor (already offset)
//   l1Tensor  : source L1 P sub-tensor (already offset)
//   repeat    : fractal repeat count (arch32 qk_round_n_2 / T_BLOCK_SIZE)
//   srcStride : inter-fractal L1 stride (arch32 p_load_coeff / BLOCK_SIZE)
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void LoadPL1ToL0AGeneralRaw(
    const AscendC::LocalTensor<DstT> &l0aTensor,
    const AscendC::LocalTensor<SrcT> &l1Tensor,
    uint32_t repeat,
    uint32_t srcStride)
{
    AscendC::LoadData(
        l0aTensor,
        l1Tensor.template ReinterpretCast<DstT>(),
        AscendC::LoadData2dParams(
            /* baseIdx    */ 0,
            /* repeat     */ repeat,
            /* srcStride  */ srcStride,
            /* sid        */ 0,
            /* dstStride  */ 0,
            /* transpose  */ false,
            /* addrCalMode*/ 0));
}

} // namespace MlaArch35
} // namespace XllmOps

#endif // MULTI_LATENT_ATTENTION_UTILS_ARCH35_H
