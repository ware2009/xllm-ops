// Copyright 2025 The xLLM Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
//
// ==================== AIV 平台处理层（第三层, A5 / arch35）====================
// 本文件仅被 MLADecoderAiv 类 include，包含 Vector 侧平台同步函数。
// AIC 侧平台函数在 multi_latent_attention_arch35.h 中。
// flag 事件协议与 A3(arch32) 完全一致（9 条 MTE3/V/MTE2 事件）。

// 平台函数：非TP1 SoftmaxStage1 前置管道同步
// 在 SoftmaxStage1 调用之前执行：等待 QK 就绪 + 等待 MTE3→MTE2 管道同步
// flag 协议与 A3(arch32) 一致，A5(arch35) 兼容。
__aicore__ __attribute__((always_inline)) inline void PlatformSoftmaxStage1PreSync()
{
    WaitFlagDev(QK_READY_DECODER);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID3);
}

// 平台函数：非TP1 SoftmaxStage1 后置管道同步
// 在 SoftmaxStage1 调用之后执行：核间同步 + 设置 MTE3→MTE2 管道 flag
// flag 协议与 A3(arch32) 一致，A5(arch35) 兼容。
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

// 平台函数：MTE2→V 管道同步（数据搬入后 V 侧可见）
// 用于 SoftmaxStage1 LoadQKData 段：DataCopy 完成后，通知 V 侧可读取。
// flag 协议与 A3(arch32) 一致，A5(arch35) 兼容。
__aicore__ __attribute__((always_inline)) inline void PlatformMte2ToVSync()
{
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
}

// 平台函数：V→MTE3 管道同步（V 计算完成后 MTE3 可搬出）
// 用于 SoftmaxStage1 QuantizeAndOutput 段：转换完成后，通知 MTE3 可写入 GM。
// flag 协议与 A3(arch32) 一致，A5(arch35) 兼容。
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte3Sync()
{
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
}

// 平台函数：V→MTE2 管道通知（V 完成本轮计算，通知 MTE2 可搬入下轮数据）
// 用于 SoftmaxStage1 QuantizeAndOutput 段末尾：ping-pong 流水衔接。
// flag 协议与 A3(arch32) 一致，A5(arch35) 兼容。
__aicore__ __attribute__((always_inline)) inline void PlatformVToMte2Notify()
{
    SET_FLAG(V, MTE2, EVENT_ID2);
}