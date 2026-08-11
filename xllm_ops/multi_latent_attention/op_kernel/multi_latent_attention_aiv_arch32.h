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