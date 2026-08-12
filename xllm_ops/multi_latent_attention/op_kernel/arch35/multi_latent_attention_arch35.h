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
// multi_latent_attention_arch35.h
//
// A5 (ascend950 / DAV_3510) PLATFORM layer for MLA.
//
// This header holds ONLY architecture-specific primitives (atomic data-move
// wrappers and pipeline-sync helpers) implemented with pure AscendC APIs. The
// business flow in multi_latent_attention_bs.h calls these primitives through
// arch-neutral names so that a future A3 (arch32) port only has to provide a
// header exposing the SAME function names/signatures.
//
// Contract exposed to the business layer (bs.h):
//   - CopyGmToL1Nd2Nz(l1, gm, Nd2NzParams)  : one ND -> NZ GM->L1 DataCopy.
//   - PlatformSetQLoadComplete()            : terminal UB<->L1 producer sync.
//   - PlatformWaitKVLoadReady(flag)         : KV ping-pong slot acquire.
//   - PlatformSetKVMainLoadComplete(flag)   : publish K main, wait rope slot.
//   - PlatformSetKVRopeLoadComplete(flag)   : publish KV block to L0B consumer.
//
// A3 -> A5 event-channel mapping is encapsulated HERE: A3's MTE2 UB<->L1 events
// become A5's MTE3 on-chip UB->L1 channel, so bs.h never names a raw pipe.
// ============================================================================

#ifndef MULTI_LATENT_ATTENTION_ARCH35_PLATFORM_H
#define MULTI_LATENT_ATTENTION_ARCH35_PLATFORM_H

#include "kernel_operator.h"

namespace XllmOps {
namespace MlaArch35 {

// ----------------------------------------------------------------------------
// CopyGmToL1Nd2Nz — atomic ND -> NZ GM->L1 transfer.
//
// A5 pure-AscendC replacement for arch32 gm_to_l1<..., ND, NZ>. The caller
// (bs.h business layer) owns the branch selection (single-token / batch /
// per-token) and fully constructs the Nd2NzParams; this wrapper only issues
// the single DataCopy. Kept arch-neutral in name so an arch32 header can offer
// the same primitive with its own iterator-based body.
//
//   l1Tensor : destination L1 sub-tensor (already offset by the caller)
//   gmTensor : source GM sub-tensor (already offset by the caller)
//   params   : fully-populated ND->NZ transfer descriptor
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void CopyGmToL1Nd2Nz(
    const AscendC::LocalTensor<DstT> &l1Tensor,
    const AscendC::GlobalTensor<SrcT> &gmTensor,
    const AscendC::Nd2NzParams &params)
{
    AscendC::DataCopy(l1Tensor, gmTensor, params);
}

// ----------------------------------------------------------------------------
// CopyUbToL1Nd2Nz — one ND -> NZ UB->L1 DataCopy (A5 shared-UB P source).
//
// A5 shared-UB replacement for the P GM->L1 stage: instead of reading the
// softmax result P back from GM, the paired AIV Vector core has already written
// P into the shared UB (p_ubuf). The Cube core moves it UB->L1 over the MTE3
// producer channel with the SAME ND->NZ semantics as the GM path — only the
// source tensor kind (LocalTensor vs GlobalTensor) differs, so AscendC selects
// the UB->L1 DataCopy overload automatically.
//   l1Tensor : destination L1 sub-tensor (already offset by the caller)
//   ubTensor : source UB sub-tensor (already offset by the caller)
//   params   : fully-populated ND->NZ transfer descriptor
// ----------------------------------------------------------------------------
template <typename DstT, typename SrcT>
__aicore__ __attribute__((always_inline)) inline void CopyUbToL1Nd2Nz(
    const AscendC::LocalTensor<DstT> &l1Tensor,
    const AscendC::LocalTensor<SrcT> &ubTensor,
    const AscendC::Nd2NzParams &params)
{
    AscendC::DataCopy(l1Tensor, ubTensor, params);
}

// ----------------------------------------------------------------------------
// PlatformSetQLoadComplete — terminal sync after the Q / Q_rope L1 load
// (arch32 L796-797). A3 issued SET/WAIT_FLAG(MTE2, MTE1, EVENT_ID0) on its
// UB<->L1 producer channel; on A5 that channel maps to MTE3, so the pair
// becomes SET/WAIT_FLAG(MTE3, MTE1, EVENT_ID0). Encapsulated here so bs.h
// stays arch-neutral (name aligned with the arch32 platform layer).
// ----------------------------------------------------------------------------
__aicore__ __attribute__((always_inline)) inline void PlatformSetQLoadComplete()
{
    SET_FLAG(MTE3, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE1, EVENT_ID0);
}

// ----------------------------------------------------------------------------
// KV ping-pong L1 producer/consumer sync (arch32 L817 / L845 / L896).
//
// The KV loader double-buffers the L1 KV region (slot = n_idx % 2). Three sync
// points guard the producer (GM->L1 load) against the consumer (Cube L1->L0B):
//   1. before loading  : wait the consumer to release this slot.
//   2. after K main     : publish K main, then wait the rope sub-slot (flag+2).
//   3. after K rope     : publish the whole block to the L0B consumer.
// A3 drove these on its MTE2 UB<->L1 producer channel; on A5 that channel maps
// to MTE3, so every SET/WAIT below uses MTE3<->MTE1. Encapsulated here so bs.h
// never names a raw pipe (name aligned with the arch32 platform layer).
// ----------------------------------------------------------------------------
__aicore__ __attribute__((always_inline)) inline void PlatformWaitKVLoadReady(
    uint32_t flag)
{
    WAIT_FLAG(MTE1, MTE3, flag);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetKVMainLoadComplete(
    uint32_t flag)
{
    SET_FLAG(MTE3, MTE1, flag);
    WAIT_FLAG(MTE1, MTE3, flag + 2);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetKVRopeLoadComplete(
    uint32_t flag)
{
    SET_FLAG(MTE3, MTE1, flag + 2);
}

// ============================================================================
// ComputeQK stage sync primitives (arch32 L897-1081, CUBE1 stage1).
//
// The embedding-split QK GEMM pipes data L1->L0A / L1->L0B, runs the mmad into
// mm1 L0C, then flushes L0C->S GM. Each stage guards the shared L0A/L0B/L0C
// buffers with M / MTE1 / FIX event flags (double-buffered by esi % 2 and the
// KV ping-pong flag). A3 drove the L1<->L0 producer channel on MTE2; on A5 that
// channel maps to MTE3. All raw pipe names are encapsulated here so the bs.h
// sub-functions stay arch-neutral (names aligned with the arch32 platform layer).
// ============================================================================

// -- LoadQFromL1ToL0A: guard the L0A slot (esi % 2) around the Q L1->L0A load.
__aicore__ __attribute__((always_inline)) inline void PlatformWaitL0AReady(
    uint32_t l0aFlag)
{
    WAIT_FLAG(M, MTE1, l0aFlag);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetL0ALoadComplete(
    uint32_t l0aFlag)
{
    SET_FLAG(MTE1, M, l0aFlag);
}

// -- LoadKVFromL1ToL0B: KV L1->L0B publish/consume handshakes. esi==0/esi==4
//    carry the extra KV ping-pong (flag / flag+2) producer sync (A3 MTE2->A5
//    MTE3), the common path guards the L0B slot (esi%2+2) and, at esi==0, waits
//    the FIX->M release of the mm1 L0C block.
__aicore__ __attribute__((always_inline)) inline void PlatformWaitKVMainSlot(
    uint32_t kvFlag)
{
    WAIT_FLAG(MTE3, MTE1, kvFlag);
}

__aicore__ __attribute__((always_inline)) inline void PlatformWaitKVRopeSlot(
    uint32_t kvFlag)
{
    WAIT_FLAG(MTE3, MTE1, kvFlag + 2);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetKVRopeSlotFree(
    uint32_t kvFlag)
{
    SET_FLAG(MTE1, MTE3, kvFlag + 2);
}

__aicore__ __attribute__((always_inline)) inline void PlatformWaitL0BReady(
    uint32_t l0bFlag)
{
    WAIT_FLAG(M, MTE1, l0bFlag + 2);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetL0BLoadComplete(
    uint32_t l0bFlag)
{
    SET_FLAG(MTE1, M, l0bFlag + 2);
}

// -- Fence consumed by the mmad after both L0A and L0B are loaded (esi % 2 /
//    +2), plus (esi==0) the first-iteration FIX->M L0C release.
__aicore__ __attribute__((always_inline)) inline void PlatformWaitL0AB(
    uint32_t flag)
{
    WAIT_FLAG(MTE1, M, flag);
    WAIT_FLAG(MTE1, M, flag + 2);
}

__aicore__ __attribute__((always_inline)) inline void PlatformWaitL0cReleased(
    uint32_t l1KvFlag)
{
    WAIT_FLAG(FIX, M, l1KvFlag);
}

// -- ComputeQKMMad: barrier + release both L0A/L0B slots after the mmad.
__aicore__ __attribute__((always_inline)) inline void PlatformSetMmadComplete(
    uint32_t flag)
{
    PIPE_BARRIER(M);
    SET_FLAG(M, MTE1, flag);
    SET_FLAG(M, MTE1, flag + 2);
}

// -- CopyQKResultToGM / RoPE S write-back: fence the mm1 L0C block against the
//    FIX (L0C->GM) unit, then release it back to the mmad.
__aicore__ __attribute__((always_inline)) inline void PlatformWaitL0cForFix(
    uint32_t l1KvFlag)
{
    SET_FLAG(M, FIX, l1KvFlag);
    WAIT_FLAG(M, FIX, l1KvFlag);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetFixComplete(
    uint32_t l1KvFlag)
{
    SET_FLAG(FIX, M, l1KvFlag);
}

// ============================================================================
// ComputePV stage sync primitives (arch32 L1084-1208, CUBE2 stage1).
//
// The PV GEMM double-buffers the L0B (V transposed), L0A (P) and mm2 L0C slots
// across embed_split_loop_v iterations. As in ComputeQK, every A3 UB<->L1
// producer channel (MTE2) maps to MTE3 on A5, so the SET/WAIT pairs below use
// MTE3<->MTE1. Only the FLAG handshakes live here; the data movement (V L1->L0B
// transpose, P GM->L1, P L1->L0A, mmad, L0C->GM) is issued via the utils Raw
// helpers, driven by the bs.h sub-functions.
// ============================================================================

// -- LoadKVTransposeToL0B: guard the L0B slot (l0b_flag+2) before the V
//    transpose load (arch32 PlatformLoadKVTranspose{SmallN,LargeN} head).
__aicore__ __attribute__((always_inline)) inline void PlatformWaitPVL0BReady(
    uint32_t l0bFlag)
{
    WAIT_FLAG(M, MTE1, l0bFlag + 2);
}

// -- SetKVLoadComplete: publish the L1 KV slot to the L0B consumer on the last
//    embed split (arch32 L256-262, A3 SET_FLAG(MTE1,MTE2,flag) -> A5 MTE3).
__aicore__ __attribute__((always_inline)) inline void PlatformSetPVKVLoadComplete(
    bool is_last_split, uint32_t l1KvFlag)
{
    if (is_last_split) {
        SET_FLAG(MTE1, MTE3, l1KvFlag);
    }
}

// -- LoadPFromGMToL1: cross-core wait on SOFTMAX_READY, then guard the L1 P slot
//    around the GM->L1 load (arch32 L269-281, A3 MTE2<->MTE1 -> A5 MTE3<->MTE1).
__aicore__ __attribute__((always_inline)) inline void PlatformWaitPBeforeLoad()
{
    WaitFlagDev(SOFTMAX_READY_DECODER);
    WAIT_FLAG(MTE1, MTE3, EVENT_ID7);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetPL1LoadComplete()
{
    SET_FLAG(MTE3, MTE1, EVENT_ID7);
    WAIT_FLAG(MTE3, MTE1, EVENT_ID7);
}

// -- LoadPToL0A{Int8,General}: guard the L0A P slot (l0_p_flag) before the
//    L1->L0A load (arch32 L289 / L300).
__aicore__ __attribute__((always_inline)) inline void PlatformWaitPVL0AReady(
    uint32_t l0pFlag)
{
    WAIT_FLAG(M, MTE1, l0pFlag);
}

// -- SetPLoadComplete: publish the L1 P slot back to the loader (arch32 L311-314,
//    A3 SET_FLAG(MTE1,MTE2,ID7) -> A5 MTE3).
__aicore__ __attribute__((always_inline)) inline void PlatformSetPVPLoadComplete()
{
    SET_FLAG(MTE1, MTE3, EVENT_ID7);
}

// -- ComputePVMmad: fence L0B into the mmad, wait the mm2 L0C release, then
//    release the L0B slot (and, on the last split, the L0A P slot) after the
//    mmad (arch32 L322-333).
__aicore__ __attribute__((always_inline)) inline void PlatformWaitPVMmad(
    uint32_t l0bFlag, uint32_t l0cFlag)
{
    SET_FLAG(MTE1, M, l0bFlag);
    WAIT_FLAG(MTE1, M, l0bFlag);
    WAIT_FLAG(FIX, M, l0cFlag);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetPVMmadComplete(
    uint32_t l0bFlag, uint32_t l0pFlag, bool is_last_split)
{
    SET_FLAG(M, MTE1, l0bFlag + 2);
    if (is_last_split) {
        SET_FLAG(M, MTE1, l0pFlag);
    }
}

// -- CopyPVResultToGM: fence the mm2 L0C block against the FIX (L0C->GM) unit,
//    then release it back to the mmad (arch32 L341-347).
__aicore__ __attribute__((always_inline)) inline void PlatformWaitPVL0cForFix(
    uint32_t l0cFlag)
{
    SET_FLAG(M, FIX, l0cFlag);
    WAIT_FLAG(M, FIX, l0cFlag);
}

__aicore__ __attribute__((always_inline)) inline void PlatformSetPVFixComplete(
    uint32_t l0cFlag)
{
    SET_FLAG(FIX, M, l0cFlag);
}

}  // namespace MlaArch35
}  // namespace XllmOps

#endif  // MULTI_LATENT_ATTENTION_ARCH35_PLATFORM_H