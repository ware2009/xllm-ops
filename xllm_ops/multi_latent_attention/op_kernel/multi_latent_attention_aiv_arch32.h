// Copyright 2025 The xLLM Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
//
// ==================== AIV 平台处理层（第三层）====================
// 本文件仅被 MLADecoderAiv 类 include，包含 Vector 侧平台同步函数。
// AIC 侧平台函数在 multi_latent_attention_arch32.h 中。

#pragma once

// 平台函数：非TP1 SoftmaxStage1 前置管道同步
// 在 SoftmaxStage1 调用之前执行：等待 QK 就绪 + 等待 MTE3→MTE2 管道同步
__aicore__ __attribute__((always_inline)) inline void PlatformSoftmaxStage1PreSync()
{
    WaitFlagDev(QK_READY_DECODER);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
}

// 平台函数：非TP1 SoftmaxStage1 后置管道同步
// 在 SoftmaxStage1 调用之后执行：核间同步 + 设置 MTE3→MTE2 管道 flag
__aicore__ __attribute__((always_inline)) inline void PlatformSoftmaxStage1PostSync()
{
    FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);
    SET_FLAG(MTE3, MTE2, EVENT_ID3);
}

// 平台函数：Vector 非TP1 管道同步初始化（Run 方法头部 9 条 SET_FLAG）
__aicore__ __attribute__((always_inline)) inline void PlatformInitVectorPipeSync()
{
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID2);
    SET_FLAG(MTE3, MTE2, EVENT_ID3);
    SET_FLAG(MTE3, MTE2, EVENT_ID4);
    SET_FLAG(V, MTE2, EVENT_ID4);
    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID2);
    SET_FLAG(V, MTE2, EVENT_ID2);
}

// 平台函数：等待 Vector 非TP1 管道同步完成（Run 方法尾部 9 条 WAIT_FLAG）
__aicore__ __attribute__((always_inline)) inline void PlatformWaitVectorPipeSync()
{
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE3, V, EVENT_ID2);
    WAIT_FLAG(V, MTE2, EVENT_ID2);
}

// 平台函数：MTE2→V EVENT_ID0 同步（SET+WAIT）
// 用于 DeQuantPerHeadImpl：src 搬入后通知 V 管道可读
__aicore__ __attribute__((always_inline)) inline void PlatformMte2ToVSyncEvent0()
{
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
}

// 平台函数：MTE2→V EVENT_ID2 同步（SET+WAIT）
// 用于 DeQuantPerHeadImpl：deScale 搬入后通知 V 管道可做 online 乘法
__aicore__ __attribute__((always_inline)) inline void PlatformMte2ToVSyncEvent2()
{
    SET_FLAG(MTE2, V, EVENT_ID2);
    WAIT_FLAG(MTE2, V, EVENT_ID2);
}

// 平台函数：V→MTE2 管道等待（等待上一轮 V→MTE2 通知）
// 用于 SoftmaxStage1 LoadQKData 段开头：等待上一轮 QuantizeAndOutput 发出的 MTE2 可搬入通知
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte2Wait()
{
    WAIT_FLAG(V, MTE2, EVENT_ID2);
}

// 平台函数：V 管道同步屏障
// 用于 SoftmaxStage1 LoadQKData 段 INT8 路径：Add 之后确保数据可见
__aicore__ __attribute__((always_inline)) inline void PlatformVPipeBarrier()
{
    PIPE_BARRIER(V);
}

// 平台函数：Vector TP1 管道同步初始化（RunTP1 方法头部 10 条 SET_FLAG）
// 与非TP1 相比多出 SET_FLAG(MTE3, MTE2, EVENT_ID1)
__aicore__ __attribute__((always_inline)) inline void PlatformInitVectorPipeSyncTP1()
{
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);
    SET_FLAG(MTE3, MTE2, EVENT_ID2);
    SET_FLAG(MTE3, MTE2, EVENT_ID3);
    SET_FLAG(MTE3, MTE2, EVENT_ID4);
    SET_FLAG(V, MTE2, EVENT_ID4);
    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID2);
    SET_FLAG(V, MTE2, EVENT_ID2);
}

// 平台函数：等待 Vector TP1 管道同步完成（RunTP1 方法尾部 10 条 WAIT_FLAG）
// 与非TP1 相比多出 WAIT_FLAG(MTE3, MTE2, EVENT_ID1)
__aicore__ __attribute__((always_inline)) inline void PlatformWaitVectorPipeSyncTP1()
{
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE3, V, EVENT_ID2);
    WAIT_FLAG(V, MTE2, EVENT_ID2);
}

// 平台函数：向量广播乘法 go = go * dm_block
// 封装 brcb dm → mul 循环 → mask 尾部 → PIPE_BARRIER
// 用于 Stage2 段落B：go = go * dm_block（三函数完全一致）
__aicore__ __attribute__((always_inline)) inline void PlatformMulVectorByBroadcast(
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    AscendC::LocalTensor<uint32_t> tv32_ubuf_tensor,
    AscendC::LocalTensor<float> dm32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_sub_m,
    uint32_t round_v,
    uint32_t __v)
{
    // expand_to_block: dm → tv
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor,
        dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
        1,               // dstBlockStride
        8,               // dstRepeatStride
        (uint8_t)(round_sub_m / FLOAT_BLOCK_SIZE)  // repeat
    );
    PIPE_BARRIER(V);

    // go = go * dm_block (full vector iterations)
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
        mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
            go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
            tv32_ubuf_tensor.template ReinterpretCast<float>(),
            (uint8_t)sub_m,        // repeat
            1,            // dstBlockStride
            1,            // src0BlockStride
            0,            // src1BlockStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // dstRepeatStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // src0RepeatStride
            1             // src1RepeatStride
        );
    }
    // tail mask iteration
    if (__v % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(__v % FLOAT_VECTOR_SIZE);
        mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tv32_ubuf_tensor.template ReinterpretCast<float>(),
            (uint8_t)sub_m,        // repeat
            1,            // dstBlockStride
            1,            // src0BlockStride
            0,            // src1BlockStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // dstRepeatStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // src0RepeatStride
            1             // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// 平台函数：向量广播除法 go = go / gl_block
// 封装 brcb gl → div 循环 → mask 尾部 → PIPE_BARRIER
// 用于 Stage2 段落C：go = go / gl_block（非TP1与TP1完全一致）
__aicore__ __attribute__((always_inline)) inline void PlatformDivVectorByBroadcast(
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    AscendC::LocalTensor<uint32_t> tv32_ubuf_tensor,
    AscendC::LocalTensor<float> gl32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_sub_m,
    uint32_t round_v,
    uint32_t __v,
    uint32_t head_loop_idx)
{
    // gl_block = expand_to_block(gl), stored in tv
    brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor,
        gl32_ubuf_tensor.ReinterpretCast<uint32_t>()[head_loop_idx * 16],
        1,               // dstBlockStride
        8,               // dstRepeatStride
        (uint8_t)(round_sub_m / FLOAT_BLOCK_SIZE)  // repeat
    );
    PIPE_BARRIER(V);

    // go = go / gl_block (full vector iterations)
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
        div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
            go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
            tv32_ubuf_tensor.template ReinterpretCast<float>(),
            (uint8_t)sub_m,                 // repeat
            1,                     // dstBlockStride
            1,                     // src0BlockStride
            0,                     // src1BlockStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // dstRepeatStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // src0RepeatStride
            1                      // src1RepeatStride
        );
    }
    // tail mask iteration
    if (__v % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(__v % FLOAT_VECTOR_SIZE);
        div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tv32_ubuf_tensor.template ReinterpretCast<float>(),
            (uint8_t)sub_m,                 // repeat
            1,                     // dstBlockStride
            1,                     // src0BlockStride
            0,                     // src1BlockStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // dstRepeatStride
            (uint8_t)(round_v / FLOAT_BLOCK_SIZE),  // src0RepeatStride
            1                      // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // fix hidden_size=96
    }
    PIPE_BARRIER(V);
}

// 平台函数：中间结果保存（段落D）
// 封装 SET_FLAG(V,MTE3,E5) → WAIT_FLAG → ub_to_gm go32
// 用于 Stage2 head_loop>1 且非最后一轮时保存 go32 到 GM
// 三函数（非TP1/TP1/Tail）完全一致
__aicore__ __attribute__((always_inline)) inline void PlatformIntermediateSave(
    AscendC::GlobalTensor<float> go_gm_tensor,
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_v)
{
    SET_FLAG(V, MTE3, EVENT_ID5);
    WAIT_FLAG(V, MTE3, EVENT_ID5);
    ub_to_gm<ArchType::ASCEND_V220, float>(
        go_gm_tensor,
        go32_ubuf_tensor,
        0,
        1,
        sub_m * round_v / FLOAT_BLOCK_SIZE,
        0,
        0
    );
}

// 平台函数：从 GM 搬入 go 到 UB（段落B 使用）
// 封装 gm_to_ub + SET_FLAG/WAIT_FLAG(MTE2, V, EVENT_ID0)
// 用于 Stage2MergeAccumulate head_loop>1 时加载上一轮 go
__aicore__ __attribute__((always_inline)) inline void PlatformLoadGoFromGM(
    AscendC::GlobalTensor<float> go_gm_tensor,
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_v)
{
    gm_to_ub<ArchType::ASCEND_V220, float>(
        go32_ubuf_tensor, go_gm_tensor,
        0, 1, sub_m * round_v / FLOAT_BLOCK_SIZE, 0, 0);
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
}

// 平台函数：向量加法 go = lo + go（段落B 使用）
// 封装 add_v + PIPE_BARRIER(V)
// 用于 Stage2MergeAccumulate 累加 lo 到 go
__aicore__ __attribute__((always_inline)) inline void PlatformAddLoToGo(
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    AscendC::LocalTensor<float> lo_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_v)
{
    add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
        go32_ubuf_tensor, lo_ubuf_tensor,
        (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
        1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
}

// 平台函数：FP32→FP16/BF16 类型转换 + V→MTE3 同步（段落C 使用）
// 封装 conv_v + SET_FLAG/WAIT_FLAG(V, MTE3, EVENT_ID0)
// 用于 Stage2FinalizeAndOutput go=castfp32to16(go)
__aicore__ __attribute__((always_inline)) inline void PlatformConvGoToOutput(
    AscendC::LocalTensor<OUT_DTYPE> go_ubuf_tensor,
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_v)
{
    conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(go_ubuf_tensor,
        go32_ubuf_tensor,
        (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
        1, 1, 4, 8);
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
}

// 平台函数：DataCopyPad 输出到 GM（段落C 使用）
// 封装 head_res_row_num / numhead_per_process / tail_res_row_num 三段 DataCopyPad
// 用于 Stage2FinalizeAndOutput 输出 attention 结果到 o_gm
__aicore__ __attribute__((always_inline)) inline void PlatformDataCopyPadOutput(
    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
    AscendC::LocalTensor<OUT_DTYPE> go_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_v,
    uint32_t q_seq_len,
    uint32_t numhead_per_process,
    uint32_t head_res_row_num,
    uint32_t head_start_sblock_idx,
    uint32_t tail_res_row_num)
{
    uint32_t inner_o_gm_offset = 0;
    uint32_t inner_go_ubuf_offset = 0;

    if (head_res_row_num != 0) {
        AscendC::DataCopyPad(
            o_gm_tensor[inner_o_gm_offset + q_heads * __v * head_start_sblock_idx],
            go_ubuf_tensor[inner_go_ubuf_offset],
            AscendC::DataCopyExtParams(head_res_row_num, __v * 2, 0,
                __v * (q_heads - 1) * 2, 0));
        inner_o_gm_offset += __v;
        inner_go_ubuf_offset += head_res_row_num * __v;
    }

    for (uint32_t i = 0; i < numhead_per_process; i++) {
        AscendC::DataCopyPad(
            o_gm_tensor[inner_o_gm_offset],
            go_ubuf_tensor[inner_go_ubuf_offset],
            AscendC::DataCopyExtParams(q_seq_len, __v * 2, 0,
                __v * (q_heads - 1) * 2, 0));
        inner_o_gm_offset += __v;
        inner_go_ubuf_offset += q_seq_len * __v;
    }

    if (tail_res_row_num != 0) {
        AscendC::DataCopyPad(
            o_gm_tensor[inner_o_gm_offset],
            go_ubuf_tensor[inner_go_ubuf_offset],
            AscendC::DataCopyExtParams(tail_res_row_num, __v * 2, 0,
                __v * (q_heads - 1) * 2, 0));
    }
}

// 平台函数：Ring LSE copyout（段落C 使用）
// 封装 ln_v + PIPE_BARRIER + add_v + PIPE_BARRIER + conv_v + SET/WAIT_FLAG + ub_to_gm_align + SET/WAIT_FLAG
// 用于 Stage2FinalizeAndOutput IS_RING 路径的 LSE 输出
__aicore__ __attribute__((always_inline)) inline void PlatformRingLSECopyout(
    AscendC::LocalTensor<float> lse32_ubuf_tensor,
    AscendC::LocalTensor<float> gl32_ubuf_tensor,
    AscendC::LocalTensor<float> gm32_ubuf_tensor,
    AscendC::LocalTensor<OUT_DTYPE> lse_conv_ubuf_tensor,
    AscendC::GlobalTensor<OUT_DTYPE> lse_gm_tensor,
    uint32_t sub_m,
    uint32_t head_loop,
    uint64_t o_offset)
{
    uint32_t sub_m_d64 = (sub_m + 63) / 64;
    uint32_t lenBurst = sizeof(OUT_DTYPE);
    ln_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor, gl32_ubuf_tensor,
        sub_m_d64, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    add_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor, lse32_ubuf_tensor,
        gm32_ubuf_tensor, sub_m_d64, 1, 1, 1, 8, 8, 8);
    PIPE_BARRIER(V);
    conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lse_conv_ubuf_tensor,
        lse32_ubuf_tensor, sub_m_d64, 1, 1, 4, 8);
    SET_FLAG(V, MTE3, EVENT_ID1);
    WAIT_FLAG(V, MTE3, EVENT_ID1);
    ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
        lse_gm_tensor[(int64_t)(o_offset / __k)],
        lse_conv_ubuf_tensor,
        0, 1, lenBurst * sub_m * head_loop, 0, 0, 0, 0);
    SET_FLAG(MTE3, V, EVENT_ID1);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
}

// 平台函数：Tail copyout gl/gm/go（段落C Tail 使用）
// 封装 ub_to_gm(gl) + ub_to_gm(gm) + SET_FLAG/WAIT_FLAG(V,MTE3,E5) + ub_to_gm(go)
// 用于 TailStage2FinalizeAndOutput 输出 gl/gm/go 三个 GM buffer
__aicore__ __attribute__((always_inline)) inline void PlatformTailCopyout(
    AscendC::GlobalTensor<float> gl_gm_tensor,
    AscendC::GlobalTensor<float> gm_gm_tensor,
    AscendC::GlobalTensor<float> go_gm_tensor,
    AscendC::LocalTensor<float> gl32_ubuf_tensor,
    AscendC::LocalTensor<float> gm32_ubuf_tensor,
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t round_v)
{
    ub_to_gm<ArchType::ASCEND_V220, float>(
        gl_gm_tensor, gl32_ubuf_tensor,
        0, 1, sub_m / FLOAT_BLOCK_SIZE, 0, 0);

    ub_to_gm<ArchType::ASCEND_V220, float>(
        gm_gm_tensor, gm32_ubuf_tensor,
        0, 1, sub_m / FLOAT_BLOCK_SIZE, 0, 0);

    SET_FLAG(V, MTE3, EVENT_ID5);
    WAIT_FLAG(V, MTE3, EVENT_ID5);
    ub_to_gm<ArchType::ASCEND_V220, float>(
        go_gm_tensor, go32_ubuf_tensor,
        0, 1, sub_m * round_v / FLOAT_BLOCK_SIZE, 0, 0);
}

// ==================== Stage1 / 通用数据搬运平台函数 ====================

// 平台函数：MTE2→V 通用同步（SET_FLAG + WAIT_FLAG, EVENT_ID0）
// 用于 LoadQKDataFP16: gm_to_ub 后通知 V 管道可读
__aicore__ __attribute__((always_inline)) inline void PlatformMte2ToVSync()
{
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
}

// 平台函数：V→MTE3 同步（SET_FLAG + WAIT_FLAG, EVENT_ID0）
// 用于 QuantizeAndOutput: conv_v 后通知 MTE3 可输出
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte3Sync()
{
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
}

// 平台函数：V→MTE2 通知（SET_FLAG + PIPE_BARRIER）
// 用于 QuantizeAndOutput: ReduceSum 后通知 MTE2 可搬入下一轮
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte2Notify()
{
    SET_FLAG(V, MTE2, EVENT_ID0);
    PIPE_BARRIER(V);
}

// 平台函数：V→MTE2 等待（WAIT_FLAG, EVENT_ID0）
// 用于 HeadLoop: 等待上一轮 V→MTE2 通知
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte2WaitEvent0()
{
    WAIT_FLAG(V, MTE2, EVENT_ID0);
}

// 平台函数：MTE3→MTE2 等待（WAIT_FLAG, EVENT_ID4）
// 用于 HeadLoop: 等待 MTE3→MTE2 管道就绪
__aicore__ __attribute__((always_inline)) inline void PlatformMte3ToMte2WaitEvent4()
{
    WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
}

// 平台函数：V→MTE2 通知（SET_FLAG, EVENT_ID0）— 仅 SET
// 用于 HeadLoop: 通知 MTE2 可读 go
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte2SetEvent0()
{
    SET_FLAG(V, MTE2, EVENT_ID0);
}

// 平台函数：MTE3→MTE2 通知（SET_FLAG, EVENT_ID4）— 仅 SET
// 用于 HeadLoop 尾部: 通知 MTE2 下一轮可搬入
__aicore__ __attribute__((always_inline)) inline void PlatformMte3ToMte2SetEvent4()
{
    SET_FLAG(MTE3, MTE2, EVENT_ID4);
}

// 平台函数：MTE3→MTE2 等待（WAIT_FLAG, EVENT_ID1）— TP1 专用
// 用于 TP1 HeadLoop: 等待 MTE3→MTE2 EVENT_ID1
__aicore__ __attribute__((always_inline)) inline void PlatformMte3ToMte2WaitEvent1()
{
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// 平台函数：MTE3→MTE2 通知（SET_FLAG, EVENT_ID1）— TP1 专用
// 用于 TP1 HeadLoop 尾部: 通知 MTE2 EVENT_ID1
__aicore__ __attribute__((always_inline)) inline void PlatformMte3ToMte2SetEvent1()
{
    SET_FLAG(MTE3, MTE2, EVENT_ID1);
}

// 平台函数：PIPE_BARRIER(ALL)
// 用于 Tail HeadLoop 尾部全管道屏障
__aicore__ __attribute__((always_inline)) inline void PlatformPipeBarrierAll()
{
    PIPE_BARRIER(ALL);
}

// ==================== 通用数据搬运平台函数 ====================

// 平台函数：gm_to_ub（通用 GM→UB 搬入，float 类型）
__aicore__ __attribute__((always_inline)) inline void PlatformGmToUbFloat(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    uint32_t lenBurst)
{
    gm_to_ub<ArchType::ASCEND_V220, float>(dst, src, 0, 1, lenBurst, 0, 0);
}

// 平台函数：gm_to_ub（通用 GM→UB 搬入，模板类型）
template <typename T>
__aicore__ __attribute__((always_inline)) inline void PlatformGmToUb(
    AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src,
    uint32_t lenBurst)
{
    gm_to_ub<ArchType::ASCEND_V220, T>(dst, src, 0, 1, lenBurst, 0, 0);
}

// 平台函数：gm_to_ub（INT32 类型，CeilDiv 对齐）
__aicore__ __attribute__((always_inline)) inline void PlatformGmToUbInt32(
    AscendC::LocalTensor<int32_t> dst,
    AscendC::GlobalTensor<int32_t> src,
    uint32_t sub_m, uint32_t qk_round_n)
{
    gm_to_ub<ArchType::ASCEND_V220, int32_t>(
        dst, src, 0, 1, CeilDiv<FLOAT_BLOCK_SIZE>(sub_m * qk_round_n), 0, 0);
}

// 平台函数：gm_to_ub_align（对齐 GM→UB，mmScaleType）
__aicore__ __attribute__((always_inline)) inline void PlatformGmToUbAlignMmScale(
    AscendC::LocalTensor<mmScaleType> dst,
    AscendC::GlobalTensor<mmScaleType> src,
    uint32_t sub_m)
{
    gm_to_ub_align<ArchType::ASCEND_V220, mmScaleType>(
        dst, src, 0, 1, sub_m * sizeof(mmScaleType), 0, 0, 0, 0);
}

// 平台函数：ub_to_gm（通用 UB→GM 搬出，模板类型）
template <typename T>
__aicore__ __attribute__((always_inline)) inline void PlatformUbToGm(
    AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> src,
    uint32_t lenBurst)
{
    ub_to_gm<ArchType::ASCEND_V220, T>(dst, src, 0, 1, lenBurst, 0, 0);
}

// 平台函数：ub_to_ub（通用 UB→UB 搬移，float 类型）
__aicore__ __attribute__((always_inline)) inline void PlatformUbToUb(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t lenBurst)
{
    ub_to_ub<ArchType::ASCEND_V220, float>(dst, src, 0, 1, lenBurst, 0, 0);
}

// 平台函数：ub_to_ub（UB→UB，多 nBurst）
__aicore__ __attribute__((always_inline)) inline void PlatformUbToUbMulti(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t nBurst,
    uint32_t lenBurst,
    uint32_t srcGap,
    uint32_t dstGap)
{
    ub_to_ub<ArchType::ASCEND_V220, float>(dst, src, 0, nBurst, lenBurst, srcGap, dstGap);
}

// 平台函数：DataCopyPad（不规则 mask 加载）
__aicore__ __attribute__((always_inline)) inline void PlatformDataCopyPadMask(
    AscendC::LocalTensor<OUT_DTYPE> dst,
    AscendC::GlobalTensor<OUT_DTYPE> src,
    uint32_t cur_q_seqlen,
    uint32_t qk_n,
    uint32_t maxKVSeqLen,
    uint32_t qk_round_n)
{
    uint32_t aligned_mask_copy_len = RoundUp<BLOCK_SIZE>(qk_n);
    uint32_t mask_dst_stride = (qk_round_n - aligned_mask_copy_len) / BLOCK_SIZE;
    AscendC::DataCopyPad(
        dst, src,
        AscendC::DataCopyExtParams(cur_q_seqlen, qk_n * 2, maxKVSeqLen * 2 - qk_n * 2, mask_dst_stride, 0),
        AscendC::DataCopyPadExtParams<OUT_DTYPE>(false, 0, 0, 0)
    );
}

// 平台函数：DataCopy（规则 mask 加载）
__aicore__ __attribute__((always_inline)) inline void PlatformDataCopyMask(
    AscendC::LocalTensor<OUT_DTYPE> dst,
    AscendC::GlobalTensor<OUT_DTYPE> src,
    uint32_t cur_q_seqlen,
    uint32_t qk_round_n,
    uint32_t maxKVSeqLen)
{
    AscendC::DataCopy(
        dst, src,
        AscendC::DataCopyParams(
            cur_q_seqlen,
            qk_round_n * 2 / 32,
            MASK_COLUMNS * 2 / 32 - qk_round_n * 2 / 32,
            0)
    );
}

// 平台函数：Cast mask→float
__aicore__ __attribute__((always_inline)) inline void PlatformCastMaskToFloat(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<OUT_DTYPE> src,
    uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}

// 平台函数：SetVectorMask 全置 -1
__aicore__ __attribute__((always_inline)) inline void PlatformSetVectorMaskFull()
{
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
}

// 平台函数：__set_mask 设置部分 mask
__aicore__ __attribute__((always_inline)) inline void PlatformSetMask(uint32_t mask)
{
    __set_mask(mask);
}

// ==================== 向量计算平台函数 ====================

// 平台函数：muls_v（标量乘法，循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformMulsVLoop(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    float scalar,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n)
{
    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
        muls_v<ArchType::ASCEND_V220, float>(
            dst[vadd_idx * FLOAT_VECTOR_SIZE],
            src[vadd_idx * FLOAT_VECTOR_SIZE],
            scalar, sub_m, 1, 1,
            qk_round_n / FLOAT_BLOCK_SIZE,
            qk_round_n / FLOAT_BLOCK_SIZE);
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        muls_v<ArchType::ASCEND_V220, float>(
            dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            scalar, sub_m, 1, 1,
            qk_round_n / FLOAT_BLOCK_SIZE,
            qk_round_n / FLOAT_BLOCK_SIZE);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// 平台函数：Add（AscendC::Add，FP32 逐元素加）
__aicore__ __attribute__((always_inline)) inline void PlatformAddFloat(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    uint32_t count)
{
    AscendC::Add(dst, src0, src1, count);
}

// 平台函数：max_v（逐元素取最大值）
__aicore__ __attribute__((always_inline)) inline void PlatformMaxV(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    uint32_t repeat)
{
    max_v<ArchType::ASCEND_V220, float>(
        dst, src0, src1, (uint8_t)repeat, 1, 1, 1, 8, 8, 8);
}

// 平台函数：sub_v（逐元素减法）
__aicore__ __attribute__((always_inline)) inline void PlatformSubV(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    uint32_t repeat)
{
    sub_v<ArchType::ASCEND_V220, float>(
        dst, src0, src1, (uint8_t)repeat, 1, 1, 1, 8, 8, 8);
}

// 平台函数：exp_v（指数运算）
__aicore__ __attribute__((always_inline)) inline void PlatformExpV(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V220, float>(
        dst, src, (uint8_t)repeat, 1, 1, 8, 8);
}

// 平台函数：mul_v（逐元素乘法）
__aicore__ __attribute__((always_inline)) inline void PlatformMulV(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    uint32_t repeat)
{
    mul_v<ArchType::ASCEND_V220, float>(
        dst, src0, src1, (uint8_t)repeat, 1, 1, 1, 8, 8, 8);
}

// 平台函数：add_v（逐元素加法）
__aicore__ __attribute__((always_inline)) inline void PlatformAddV(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    uint32_t repeat)
{
    add_v<ArchType::ASCEND_V220, float>(
        dst, src0, src1, (uint8_t)repeat, 1, 1, 1, 8, 8, 8);
}

// 平台函数：muls_v（标量乘法，单次调用）
__aicore__ __attribute__((always_inline)) inline void PlatformMulsV(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    float scalar,
    uint32_t repeat)
{
    muls_v<ArchType::ASCEND_V220, float>(
        dst, src, scalar, (uint8_t)repeat, 1, 1, 8, 8);
}

// 平台函数：brcb_v（广播，uint32_t）
__aicore__ __attribute__((always_inline)) inline void PlatformBrcbV(
    AscendC::LocalTensor<uint32_t> dst,
    AscendC::LocalTensor<uint32_t> src,
    uint32_t repeat)
{
    brcb_v<ArchType::ASCEND_V220, uint32_t>(dst, src, 1, 8, (uint8_t)repeat);
}

// 平台函数：brcb_v（广播，float）
__aicore__ __attribute__((always_inline)) inline void PlatformBrcbVFloat(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t round_sub_m)
{
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        dst.ReinterpretCast<uint32_t>(),
        src.ReinterpretCast<uint32_t>(),
        1, 8, (uint8_t)(round_sub_m / FLOAT_BLOCK_SIZE));
}

// 平台函数：conv_v（float→OUT_DTYPE 转换）
__aicore__ __attribute__((always_inline)) inline void PlatformConvVToOutput(
    AscendC::LocalTensor<OUT_DTYPE> dst,
    AscendC::LocalTensor<float> src,
    uint32_t repeat)
{
    conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(
        dst, src, (uint8_t)repeat, 1, 1, 4, 8);
}

// 平台函数：conv_v（int32_t→float 转换，含 repeat_times<255 分支）
__aicore__ __attribute__((always_inline)) inline void PlatformConvInt32ToFloat(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<int32_t> temp,
    uint32_t count)
{
    uint32_t repeat_times = (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
    if (repeat_times < 255) {
        conv_v<ArchType::ASCEND_V220, int32_t, float>(
            dst, temp, (uint8_t)repeat_times, 1, 1, 8, 8);
    } else {
        for (uint64_t vconv_idx = 0; vconv_idx < 2; ++vconv_idx) {
            conv_v<ArchType::ASCEND_V220, int32_t, float>(
                dst[vconv_idx * count / 2], temp[vconv_idx * count / 2],
                (uint8_t)((count / 2 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE),
                1, 1, 8, 8);
        }
    }
}

// 平台函数：conv_v（float→half 转换，含 repeat_times<255 分支）
__aicore__ __attribute__((always_inline)) inline void PlatformConvFloatToHalf(
    AscendC::LocalTensor<IN_DTYPE> dst,
    uint32_t count)
{
    uint32_t repeat_times = (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
    if (repeat_times < 255) {
        conv_v<ArchType::ASCEND_V220, float, half>(
            dst.template ReinterpretCast<half>(),
            dst.template ReinterpretCast<float>(),
            (uint8_t)repeat_times, 1, 1, 4, 8);
    } else {
        for (uint64_t vconv_idx = 0; vconv_idx < 2; ++vconv_idx) {
            conv_v<ArchType::ASCEND_V220, float, half>(
                dst.template ReinterpretCast<half>()[vconv_idx * count / 2],
                dst.template ReinterpretCast<float>()[vconv_idx * count / 2],
                (uint8_t)((count / 2 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE),
                1, 1, 4, 8);
        }
    }
    PIPE_BARRIER(V);
}

// 平台函数：Cast<half, int8_t> 逐行转换（循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformCastHalfToInt8(
    AscendC::LocalTensor<IN_DTYPE> dst,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m)
{
    for (uint32_t row_idx = 0; row_idx < qk_n / HALF_VECTOR_SIZE; ++row_idx) {
        AscendC::Cast<int8_t, half, false>(
            dst.template ReinterpretCast<int8_t>()[row_idx * HALF_VECTOR_SIZE],
            dst.template ReinterpretCast<half>()[row_idx * HALF_VECTOR_SIZE],
            AscendC::RoundMode::CAST_RINT,
            (uint64_t)0, sub_m,
            {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE), (uint8_t)(qk_round_n / BLOCK_SIZE)});
    }
    if (qk_n % HALF_VECTOR_SIZE > 0) {
        __set_mask(qk_n % HALF_VECTOR_SIZE);
        AscendC::Cast<int8_t, half, false>(
            dst.template ReinterpretCast<int8_t>()[qk_n / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
            dst.template ReinterpretCast<half>()[qk_n / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
            AscendC::RoundMode::CAST_RINT,
            (uint64_t)0, sub_m,
            {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE), (uint8_t)(qk_round_n / BLOCK_SIZE)});
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// ==================== Tensor/Reduce 系列平台函数 ====================
// 以下函数从 bs 层整体搬入，它们是平台原语+循环的封装，属于平台操作

// 平台函数：Tensor 逐行减法（brcb + sub_v 循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformTensorSubValueRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& MaxTensor,
    const AscendC::LocalTensor<float>& tempMaxTensor,
    uint32_t sub_m,
    uint32_t round_sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n)
{
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        tempMaxTensor.ReinterpretCast<uint32_t>(),
        MaxTensor.ReinterpretCast<uint32_t>(),
        1, 8, (uint8_t)(round_sub_m / FLOAT_BLOCK_SIZE));
    PIPE_BARRIER(V);
    for (uint32_t sub_v_idx = 0; sub_v_idx < qk_n / FLOAT_VECTOR_SIZE; ++sub_v_idx) {
        sub_v<ArchType::ASCEND_V220, float>(
            dst[sub_v_idx * FLOAT_VECTOR_SIZE],
            src[sub_v_idx * FLOAT_VECTOR_SIZE],
            tempMaxTensor, (uint8_t)sub_m, 1, 1, 0,
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE), 1);
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        sub_v<ArchType::ASCEND_V220, float>(
            dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tempMaxTensor, (uint8_t)sub_m, 1, 1, 0,
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE), 1);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// 平台函数：Tensor 逐行除法（div_v 循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformTensorDivRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& src1,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    PIPE_BARRIER(V);
    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
        div_v<ArchType::ASCEND_V220, float>(
            dst[vadd_idx * FLOAT_VECTOR_SIZE],
            src[vadd_idx * FLOAT_VECTOR_SIZE],
            src1, (uint8_t)sub_m, 1, 1, 0,
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE), 1);
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        div_v<ArchType::ASCEND_V220, float>(
            dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src1, (uint8_t)sub_m, 1, 1, 0,
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE), 1);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// 平台函数：Tensor 行归约求最大值（cmax_v + max_v 循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformReduceMaxRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& tempTensor,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n)
{
    if (qk_n <= FLOAT_VECTOR_SIZE) {
        __set_mask(qk_n);
        cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            dst, src, (uint8_t)sub_m, 1, 1, (uint16_t)(qk_round_n / FLOAT_BLOCK_SIZE));
    } else {
        ub_to_ub<ArchType::ASCEND_V220, float>(
            tempTensor, src, 0, sub_m,
            HALF_VECTOR_SIZE / BLOCK_SIZE,
            (qk_round_n - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE, 0);
        PIPE_BARRIER(V);
        for (uint32_t rowmax_idx = 1; rowmax_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
            max_v<ArchType::ASCEND_V220, float>(
                tempTensor, tempTensor, src[rowmax_idx * FLOAT_VECTOR_SIZE],
                (uint8_t)sub_m, 1, 1, 1, 8, 8, (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE));
            PIPE_BARRIER(V);
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            max_v<ArchType::ASCEND_V220, float>(
                tempTensor, tempTensor,
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                (uint8_t)sub_m, 1, 1, 1, 8, 8, (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE));
        }
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            dst, tempTensor, (uint8_t)sub_m, 1, 1, 8);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);
}

// 平台函数：Tensor 行归约求和（cadd_v + add_v 循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformReduceSumRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n)
{
    if (qk_n <= FLOAT_VECTOR_SIZE) {
        __set_mask(qk_n);
        cadd_v<ArchType::ASCEND_V220, float>(
            dst, src, (uint8_t)sub_m, 1, 1, (uint16_t)(qk_round_n / FLOAT_BLOCK_SIZE));
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else {
        for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
            add_v<ArchType::ASCEND_V220, float>(
                src, src, src[rowsum_idx * FLOAT_VECTOR_SIZE],
                (uint8_t)sub_m, 1, 1, 1,
                (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
                (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
                (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE));
            PIPE_BARRIER(V);
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            add_v<ArchType::ASCEND_V220, float>(
                src, src,
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                (uint8_t)sub_m, 1, 1, 1,
                (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
                (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
                (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE));
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
        cadd_v<ArchType::ASCEND_V220, float>(
            dst, src, (uint8_t)sub_m, 1, 1, (uint16_t)(qk_round_n / FLOAT_BLOCK_SIZE));
    }
}

// 平台函数：Tensor 逐行乘法（mul_v 循环 + tail mask）
__aicore__ __attribute__((always_inline)) inline void PlatformTensorMulRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& src1,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t src1BlockStride)
{
    PIPE_BARRIER(V);
    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
        mul_v<ArchType::ASCEND_V220, float>(
            dst[vadd_idx * FLOAT_VECTOR_SIZE],
            src[vadd_idx * FLOAT_VECTOR_SIZE],
            src1, (uint8_t)sub_m, 1, 1, (uint8_t)src1BlockStride,
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE), 1);
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        mul_v<ArchType::ASCEND_V220, float>(
            dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src1, (uint8_t)sub_m, 1, 1, (uint8_t)src1BlockStride,
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE),
            (uint8_t)(qk_round_n / FLOAT_BLOCK_SIZE), 1);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// ==================== DeQuant/Quant 系列平台函数 ====================

// 平台函数：加载 deScale + online 乘 quantScale
__aicore__ __attribute__((always_inline)) inline void PlatformLoadDeScaleAndOnlineMul(
    const AscendC::GlobalTensor<mmScaleType>& deScaleGm,
    AscendC::LocalTensor<mmScaleType> deScaleUb,
    AscendC::LocalTensor<float> quantScale,
    uint32_t sub_m, bool online)
{
    gm_to_ub_align<ArchType::ASCEND_V220, mmScaleType>(
        deScaleUb, deScaleGm, 0, 1, sub_m * sizeof(mmScaleType), 0, 0, 0, 0);
    if (online) {
        PlatformMte2ToVSyncEvent2();
        PlatformTensorMulRepeatM(deScaleUb, deScaleUb, quantScale, 1, sub_m, RoundUp<16>(sub_m), 1);
    }
}

// 平台函数：加载 src(int32) + brcb 广播 deScale → tempScale
__aicore__ __attribute__((always_inline)) inline void PlatformLoadSrcAndBrcbScale(
    const AscendC::GlobalTensor<int32_t>& src,
    AscendC::LocalTensor<int32_t> temp,
    AscendC::LocalTensor<mmScaleType> deScaleUb,
    AscendC::LocalTensor<mmScaleType> tempScale,
    uint32_t sub_m, uint32_t qk_round_n, bool move_tensor)
{
    if (move_tensor) {
        gm_to_ub<ArchType::ASCEND_V220, int32_t>(
            temp, src, 0, 1, CeilDiv<FLOAT_BLOCK_SIZE>(sub_m * qk_round_n), 0, 0);
    }
    PlatformMte2ToVSyncEvent0();
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        tempScale.template ReinterpretCast<uint32_t>(),
        deScaleUb.template ReinterpretCast<uint32_t>(),
        1, 8, (uint8_t)(RoundUp<16>(sub_m) / FLOAT_BLOCK_SIZE));
    PlatformVPipeBarrier();
}

// 平台函数：INT32→FP32 转换(conv_v) + 乘 tempScale
__aicore__ __attribute__((always_inline)) inline void PlatformConvInt32ToFP32AndMul(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<int32_t> temp,
    AscendC::LocalTensor<mmScaleType> tempScale,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    uint32_t count = sub_m * qk_round_n;
    PlatformConvInt32ToFloat(dst, temp, count);
    PlatformTensorMulRepeatM(dst, dst, tempScale, sub_m, qk_n, qk_round_n, 0);
    PlatformVPipeBarrier();
}

// 平台函数：逐 Head 反量化（编排：加载+转换+乘法）
__aicore__ __attribute__((always_inline)) inline void PlatformDeQuantPerHead(
    const AscendC::GlobalTensor<mmScaleType>& deScaleGm,
    const AscendC::GlobalTensor<int32_t>& src,
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<int32_t> temp,
    AscendC::LocalTensor<mmScaleType> deScaleUb,
    AscendC::LocalTensor<mmScaleType> tempScale,
    AscendC::LocalTensor<float> quantScale,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n,
    bool online,
    bool move_tensor)
{
    PlatformLoadDeScaleAndOnlineMul(deScaleGm, deScaleUb, quantScale, sub_m, online);
    PlatformLoadSrcAndBrcbScale(src, temp, deScaleUb, tempScale, sub_m, qk_round_n, move_tensor);
    PlatformConvInt32ToFP32AndMul(dst, temp, tempScale, sub_m, qk_n, qk_round_n);
}

// 平台函数：逐 Token 量化（FP32 → FP16 → INT8）
__aicore__ __attribute__((always_inline)) inline void PlatformQuantPerToken(
    const AscendC::LocalTensor<IN_DTYPE>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& scale,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t pQuantOnline)
{
    if (pQuantOnline) {
        PlatformTensorDivRepeatM(dst.template ReinterpretCast<float>(), src, scale, sub_m, qk_n, qk_round_n);
    } else {
        PlatformTensorMulRepeatM(dst.template ReinterpretCast<float>(), src, scale, sub_m, qk_n, qk_round_n, 0);
    }
    uint32_t count = sub_m * qk_round_n;
    PlatformConvFloatToHalf(dst, count);
    PlatformCastHalfToInt8(dst, qk_n, qk_round_n, sub_m);
}