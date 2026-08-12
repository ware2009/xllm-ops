// Copyright 2025 The xLLM Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
//
// ==================== AIV 业务逻辑层（第二层）====================
// 本文件仅被 MLADecoderAiv 类 include，包含 Vector 侧业务调度函数。
// AIC 侧业务函数在 multi_latent_attention_bs.h 中。

// ====== AIV refactor: context struct + init functions ======

// 非 TP1 路径上下文
struct VectorContext {
    // input params
    uint32_t cur_batch;
    uint32_t start_head;
    uint32_t cur_nIndx;
    uint32_t cur_q_seqlen;
    uint32_t cur_kv_seqlen;
    uint32_t cur_head_num;
    uint32_t offset_tiling;

    // addresses
    uint64_t addr_o_scalar;
    uint64_t addr_mask_scalar;
    uint32_t mask_offset;

    // loop & size
    uint32_t pp_n_scalar;
    uint32_t sub_n_loop;
    uint32_t real_n_loop;
    uint32_t n_loop;

    // QK dims
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t qk_n_2;
    uint32_t qk_round_n_2;

    // head split
    uint32_t sub_head_num;
    uint32_t sub_m;
    uint32_t head_idx;
    uint64_t o_offset;

    // tail info
    uint32_t tail_len;
    bool prev_tail_mask;
};

__aicore__ __attribute__((always_inline)) inline void InitVectorContext(
    VectorContext &ctx, uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling)
{
    ctx.cur_batch = cur_batch;
    ctx.start_head = start_head;
    ctx.cur_nIndx = cur_nIndx;
    ctx.cur_q_seqlen = cur_q_seqlen;
    ctx.cur_kv_seqlen = cur_kv_seqlen;
    ctx.cur_head_num = cur_head_num;
    ctx.offset_tiling = offset_tiling;

    // addr
    uint32_t addr_o_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 4 + offset_tiling));
    uint32_t addr_o_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 5 + offset_tiling));
    ctx.addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);

    uint32_t addr_mask_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 6 + offset_tiling));
    uint32_t addr_mask_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 7 + offset_tiling));
    ctx.addr_mask_scalar = (uint64_t)(((uint64_t)addr_mask_high32) << 32 | addr_mask_loww32);
    ctx.mask_offset = ctx.addr_mask_scalar;

    // loop & size
    ctx.pp_n_scalar = block_size;
    ctx.sub_n_loop = ctx.pp_n_scalar / block_size;
    ctx.real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;
    ctx.n_loop = (cur_kv_seqlen + ctx.pp_n_scalar - 1) / ctx.pp_n_scalar;

    // QK dims
    ctx.qk_n = ctx.pp_n_scalar;
    ctx.qk_round_n = RoundUp<BLOCK_SIZE>(ctx.qk_n);
    ctx.qk_n_2 = ctx.pp_n_scalar;
    ctx.qk_round_n_2 = RoundUp<BLOCK_SIZE>(ctx.qk_n_2);

    // head split
    ctx.sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2;
    ctx.sub_m = ctx.sub_head_num * cur_q_seqlen;
    ctx.head_idx = (sub_block_idx == 0) ? start_head : start_head + cur_head_num / 2 * cur_q_seqlen;
    ctx.o_offset = ctx.addr_o_scalar + start_head * embedding_size + sub_block_idx * cur_head_num / 2 * embedding_size;

    // tail info
    ctx.tail_len = cur_kv_seqlen - (ctx.n_loop - 1) * ctx.pp_n_scalar;
    ctx.prev_tail_mask = (ctx.n_loop > 1 && ctx.tail_len < cur_q_seqlen - 1);
}

// TP1 路径上下文
struct VectorTP1Context {
    // input params
    uint32_t cur_batch;
    uint32_t start_head;
    uint32_t cur_nIndx;
    uint32_t cur_q_seqlen;
    uint32_t cur_kv_seqlen;
    uint32_t cur_head_num;
    uint32_t offset_tiling;

    // addresses
    uint64_t addr_o_scalar;
    uint64_t addr_mask_scalar;
    uint32_t mask_offset;

    // loop & size
    uint32_t pp_n_scalar;
    uint32_t sub_n_loop;
    uint32_t real_n_loop;
    uint32_t n_loop;

    // QK dims
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t qk_n_2;
    uint32_t qk_round_n_2;

    // head split
    uint32_t sub_head_num;
    uint32_t sub_m;
    uint32_t head_idx;
    uint64_t o_offset;

    // TP1 specific
    uint32_t s_block_stack;
    uint32_t m_slice;
    uint32_t m_end;
};

__aicore__ __attribute__((always_inline)) inline void InitVectorTP1Context(
    VectorTP1Context &ctx, uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling)
{
    ctx.cur_batch = cur_batch;
    ctx.start_head = start_head;
    ctx.cur_nIndx = cur_nIndx;
    ctx.cur_q_seqlen = cur_q_seqlen;
    ctx.cur_kv_seqlen = cur_kv_seqlen;
    ctx.cur_head_num = cur_head_num;
    ctx.offset_tiling = offset_tiling;

    // addr
    uint32_t prev_task = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
    ctx.addr_o_scalar = prev_task * q_heads * embedding_size;
    ctx.addr_mask_scalar = 0;
    ctx.mask_offset = 0;

    // loop & size
    ctx.pp_n_scalar = block_size;
    ctx.sub_n_loop = ctx.pp_n_scalar / block_size;
    ctx.real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;
    ctx.n_loop = (cur_kv_seqlen + ctx.pp_n_scalar - 1) / ctx.pp_n_scalar;

    // QK dims
    ctx.qk_n = ctx.pp_n_scalar;
    ctx.qk_round_n = RoundUp<BLOCK_SIZE>(ctx.qk_n);
    ctx.qk_n_2 = ctx.pp_n_scalar;
    ctx.qk_round_n_2 = RoundUp<BLOCK_SIZE>(ctx.qk_n_2);

    // head split
    ctx.sub_head_num = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2;
    ctx.sub_m = ctx.sub_head_num * cur_q_seqlen;
    ctx.head_idx = (sub_block_idx == 0) ? start_head : start_head + cur_head_num / 2 * cur_q_seqlen;
    ctx.o_offset = ctx.addr_o_scalar + start_head * embedding_size + sub_block_idx * cur_head_num / 2 * embedding_size;

    // TP1 specific
    ctx.s_block_stack = 4;
    ctx.m_slice = FLOAT_VECTOR_SIZE / ctx.s_block_stack;
    ctx.m_end = (ctx.sub_m + ctx.m_slice - 1) / ctx.m_slice;
}

// ====== 非 TP1 路径：Phase 1 / Phase 2 业务调度 ======

// 业务子函数1a：QK 数据加载（INT8 路径）
// INT8 量化 QK 反量化 + RoPE 残差加载 + Add 融合
// 平台调用：PlatformVToMte2Wait / PlatformMte2ToVSync / PlatformVPipeBarrier
__aicore__ __attribute__((always_inline)) inline void LoadQKDataInt8(
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor,
    AscendC::GlobalTensor<float> s_rope_gm_tensor,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    const uint32_t head_idx)
{
    PlatformVToMte2Wait();
    DeQuantPerHeadImpl(
        deq_scale_gm_tensor_q1[head_idx],
        s_gm_tensor,
        ls32_quant_ubuf_tensor, ls32_quant_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
        descale_q1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor, sub_m, qk_n, qk_round_n, 0, 1);
    gm_to_ub<ArchType::ASCEND_V220, float>(
        ls32_ubuf_tensor.template ReinterpretCast<float>(),
        s_rope_gm_tensor,
        0,                        // sid
        1,                        // nBurst
        sub_m * qk_round_n / FLOAT_BLOCK_SIZE,
        0,                        // srcGap
        0                         // dstGap
    );
    PlatformMte2ToVSync();
    AscendC::Add(ls32_ubuf_tensor, ls32_ubuf_tensor, ls32_quant_ubuf_tensor, sub_m * qk_round_n); // float
    PlatformVPipeBarrier();
}

// 业务子函数1b：QK 数据加载（非INT8 路径）
// FP16/BF16 QK 直接加载 + 可选 mask 加载(DataCopyPad/DataCopy) + Cast
// 平台调用：PlatformVToMte2Wait / PlatformMte2ToVSync
__aicore__ __attribute__((always_inline)) inline void LoadQKDataFP16(
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor,
    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    uint32_t cur_q_seqlen,
    bool need_mask)
{
    PlatformVToMte2Wait();
    gm_to_ub<ArchType::ASCEND_V220, mm1CopyType>(
        ls32_ubuf_tensor.template ReinterpretCast<mm1CopyType>(),
        s_gm_tensor,
        0,                        // sid
        1,                        // nBurst
        sub_m * qk_round_n / FLOAT_BLOCK_SIZE,  // lenBurst
        0,                        // srcGap
        0                         // dstGap
    );

    // mask 加载：mask_type==3 不规则 padding, mask_type==4 规则, 默认不加载
    if (mask_type == 3) {
        uint32_t aligned_mask_copy_len = RoundUp<BLOCK_SIZE>(qk_n); // 16
        uint32_t mask_dst_stride = (qk_round_n -  aligned_mask_copy_len) / BLOCK_SIZE; // 0

        AscendC::DataCopyPad(
            mask_ubuf_tensor,
            mask_gm_tensor,
            AscendC::DataCopyExtParams(
                cur_q_seqlen,
                qk_n * 2,
                maxKVSeqLen * 2 - qk_n * 2,
                mask_dst_stride,
            0),
            AscendC::DataCopyPadExtParams<OUT_DTYPE>(false, 0, 0, 0)
        );
    } else if (need_mask && mask_type == 4) {
        AscendC::DataCopy(
            mask_ubuf_tensor,
            mask_gm_tensor,
            AscendC::DataCopyParams(
                cur_q_seqlen,   // blockCount
                qk_round_n * 2 / 32, // blockLen, 2 is sizeof(half)
                MASK_COLUMNS * 2 / 32 - qk_round_n * 2 / 32, // srcStride
                0 // dstStride
                )
        );
    }

    PlatformMte2ToVSync();

    if (mask_type == 3 || (need_mask && mask_type == 4)) {
        AscendC::Cast(
            mask32_ubuf_tensor,
            mask_ubuf_tensor,
            AscendC::RoundMode::CAST_NONE,
            cur_q_seqlen * qk_round_n);
    }
}

// 业务子函数2：QK 缩放 + 非INT8 mask Add
// ls = ls × scale(tor)，非INT8 路径再叠加 mask
__aicore__ __attribute__((always_inline)) inline void ScaleAndMask(
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    uint32_t cur_q_seqlen,
    bool need_mask
)
{
    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
        muls_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor[vadd_idx * FLOAT_VECTOR_SIZE],
            ls32_ubuf_tensor[vadd_idx * FLOAT_VECTOR_SIZE],
            tor,
            sub_m,                          // repeat
            1,                              // dstBlockStride
            1,                              // srcBlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
        );
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        muls_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            ls32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tor,
            sub_m,                          // repeat
            1,                              // dstBlockStride
            1,                              // srcBlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE  // srcRepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);

    if constexpr (tilingKeyType != TilingKeyType::TILING_INT8_DATA) {
        if (mask_type == 3 || (need_mask && mask_type == 4)) {
            uint32_t cur_compute_head_num = sub_m / cur_q_seqlen;
            for (uint32_t i = 0; i < cur_compute_head_num; i++) {
                Add(
                    ls32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                    ls32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                    mask32_ubuf_tensor,
                    cur_q_seqlen * qk_round_n
                );
            }
            PlatformVPipeBarrier();
        }
    }
}

// 业务子函数3：Online Softmax 状态更新
// lm = rowmax(ls) → hm = max(lm, gm) → dm = gm - hm → gm = hm
__aicore__ __attribute__((always_inline)) inline void UpdateSoftmaxState(
    AscendC::LocalTensor<float> dm32_ubuf_tensor,
    uint32_t n_idx,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n
)
{
    uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

    // *** lm = rowmax(ls)
    ReduceMaxRepeatM(lm32_ubuf_tensor, ls32_ubuf_tensor, lp32_ubuf_tensor, sub_m, qk_n, qk_round_n);
    if (n_idx != 0) {
        // *** hm = vmax(lm, gm)
        max_v<ArchType::ASCEND_V220, float>(hm32_ubuf_tensor,
            lm32_ubuf_tensor,
            gm32_ubuf_tensor,
            sub_m_d64,  // repeat
            1,           // dstBlockStride
            1,           // src0BlockStride
            1,           // src1BlockStride
            8,           // dstRepeatStride
            8,           // src0RepeatStride
            8            // src1RepeatStride
        );
        PlatformVPipeBarrier();
        // *** dm = gm - hm
        sub_v<ArchType::ASCEND_V220, float>(dm32_ubuf_tensor,
            gm32_ubuf_tensor,
            hm32_ubuf_tensor,
            sub_m_d64,  // repeat
            1,           // dstBlockStride
            1,           // src0BlockStride
            1,           // src1BlockStride
            8,           // dstRepeatStride
            8,           // src0RepeatStride
            8            // src1RepeatStride
        );
        PlatformVPipeBarrier();
    } else {
        // *** hm = lm
        ub_to_ub<ArchType::ASCEND_V220, float>(
            hm32_ubuf_tensor,
            lm32_ubuf_tensor,
            0,                         // sid
            1,                         // nBurst
            round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                         // srcGap
            0                          // dstGap
        );
        PlatformVPipeBarrier();
    }
    // *** gm = hm
    ub_to_ub<ArchType::ASCEND_V220, float>(
        gm32_ubuf_tensor,
        hm32_ubuf_tensor,
        0,                         // sid
        1,                         // nBurst
        round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
        0,                         // srcGap
        0                          // dstGap
    );
    PIPE_BARRIER(V);
}

// 业务子函数4：减法 + 指数
// ls = ls - hm_block → ls = exp(ls)
__aicore__ __attribute__((always_inline)) inline void SubAndExp(
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n
)
{
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

    // *** ls = ls - hm_block
    TensorSubValueRepeatM(ls32_ubuf_tensor, ls32_ubuf_tensor,
                       hm32_ubuf_tensor, tv32_ubuf_tensor,
                       sub_m, round_sub_m, qk_n, qk_round_n);
    // *** ls = exp(ls)
    exp_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor,
        ls32_ubuf_tensor,
        (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
        1,                               // dstBlockStride
        1,                               // srcBlockStride
        8,                               // dstRepeatStride
        8                                // srcRepeatStride
    );
    PIPE_BARRIER(V);
}

// 业务子函数5：量化/转换输出
// INT8: sub/exp/muls/brcb + QuantPerTokenImpl (FP32→INT8)
// 非INT8: conv_v (FP32→FP16)
// 共通: ub_to_gm(p_gm ← lp) + ReduceSum
__aicore__ __attribute__((always_inline)) inline void QuantizeAndOutput(
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor,
    AscendC::LocalTensor<float> ll_ubuf_tensor,
    AscendC::LocalTensor<float> pm32_ubuf_tensor,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n
)
{
    uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
    float quantMax = (float)1 / (float)127;

    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        sub_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
            lm32_ubuf_tensor,
            hm32_ubuf_tensor,
            sub_m_d64,   // repeat
            1,           // dstBlockStride
            1,           // src0BlockStride
            1,           // src1BlockStride
            8,           // dstRepeatStride
            8,           // src0RepeatStride
            8            // src1RepeatStride
        );
        PlatformVPipeBarrier();
        exp_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
            pm32_ubuf_tensor,
            sub_m_d64,  // repeat
            1,                               // dstBlockStride
            1,                               // srcBlockStride
            8,                               // dstRepeatStride
            8                                // srcRepeatStride
        );
        PlatformVPipeBarrier();
        muls_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
            pm32_ubuf_tensor,
            quantMax,
            sub_m_d64,              // repeat
            1,                      // dstBlockStride
            1,                      // srcBlockStride
            8,                      // dstRepeatStride
            8                        // srcRepeatStride
        );
        PlatformVPipeBarrier();
        brcb_v<ArchType::ASCEND_V220, uint32_t>(
            tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            pm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        QuantPerTokenImpl(lp_ubuf_tensor, ls32_ubuf_tensor, tv32_ubuf_tensor, sub_m, qk_n, qk_round_n, 1);
    } else {
        conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lp_ubuf_tensor,
            ls32_ubuf_tensor,
            (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                               // dstBlockStride
            1,                               // srcBlockStride
            4,                               // dstRepeatStride
            8                                // srcRepeatStride
        );
        PlatformVPipeBarrier();
    }
    PlatformVToMte3Sync();
    ub_to_gm<ArchType::ASCEND_V220, IN_DTYPE>(
        p_gm_tensor,
        lp_ubuf_tensor,
        0,                        // sid
        1,                        // nBurst
        sub_m * qk_round_n * T_BLOCK_OFFSET / T_BLOCK_SIZE,  // lenBurst
        0,                        // srcGap
        0                         // dstGap
    );

    // *** ll = rowsum(ls32)
    ReduceSumRepeatM(ll_ubuf_tensor, ls32_ubuf_tensor, sub_m, qk_n, qk_round_n);
    PlatformVToMte2Notify();
    PIPE_BARRIER(V);
}

// 业务函数：非TP1 Phase 1 — SoftmaxStage1 编排（调用5个业务子函数）
__aicore__ __attribute__((always_inline)) inline void SoftmaxStage1(
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor,
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor,
    AscendC::GlobalTensor<float> s_rope_gm_tensor,
    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor,
    AscendC::LocalTensor<float> dm32_ubuf_tensor,
    AscendC::LocalTensor<float> ll_ubuf_tensor,
    AscendC::LocalTensor<float> pm32_ubuf_tensor,
    uint32_t n_idx,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    uint32_t mask_offset,
    const uint32_t sub_n_loop,
    const uint32_t cur_batch,
    const uint32_t start_kv,
    const uint32_t real_n_loop,
	    const uint32_t head_idx,
    const uint32_t pm_flag_scalar,
    uint32_t cur_q_seqlen,
    uint32_t cur_kv_seqlen,
    bool need_mask
)
{
    // 段1：QK 数据加载（INT8:反量化+融合 / 非INT8:FP16+mask+Cast）
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        LoadQKDataInt8(s_gm_tensor, s_rope_gm_tensor, qk_n, qk_round_n, sub_m, head_idx);
    } else {
        LoadQKDataFP16(s_gm_tensor, mask_gm_tensor, qk_n, qk_round_n, sub_m, cur_q_seqlen, need_mask);
    }

    // 段2：QK 缩放 + 非INT8 mask Add
    ScaleAndMask(qk_n, qk_round_n, sub_m, cur_q_seqlen, need_mask);

    // 段3：Online Softmax 状态更新（ReduceMax + max/sub/ub_to_ub）
    UpdateSoftmaxState(dm32_ubuf_tensor, n_idx, sub_m, qk_n, qk_round_n);

    // 段4：减法 + 指数（ls = ls - hm → ls = exp(ls)）
    SubAndExp(sub_m, qk_n, qk_round_n);

    // 段5：量化/转换输出 + ub_to_gm + ReduceSum
    QuantizeAndOutput(p_gm_tensor, ll_ubuf_tensor, pm32_ubuf_tensor, sub_m, qk_n, qk_round_n);
}

// 业务函数：非TP1 Phase 1 — Softmax Stage1 调度
// 包含 mask 计算、平台同步调用、SoftmaxStage1 调用（ping-pong）
__aicore__ __attribute__((always_inline)) inline void ScheduleSoftmaxStage1(
    VectorContext &ctx, uint32_t n_idx, uint32_t start_kv)
{
    uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    uint32_t cur_kv_seqlen = ctx.cur_kv_seqlen;
    uint32_t cur_head_num = ctx.cur_head_num;
    uint32_t cur_batch = ctx.cur_batch;
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t sub_n_loop = ctx.sub_n_loop;
    uint32_t real_n_loop = ctx.real_n_loop;
    uint32_t n_loop = ctx.n_loop;
    uint32_t sub_m = ctx.sub_m;
    uint32_t head_idx = ctx.head_idx;
    uint32_t mask_offset = ctx.mask_offset;
    uint32_t tail_len = ctx.tail_len;
    bool prev_tail_mask = ctx.prev_tail_mask;

    uint32_t qk_n = ctx.qk_n;
    uint32_t qk_round_n = ctx.qk_round_n;

    // mask 边界判断
    bool need_mask = false;
    uint32_t mask_start_offset = 0;
    if (n_idx == (n_loop - 2)) {
        need_mask = prev_tail_mask;
        mask_start_offset = need_mask ? (tail_len + MASK_COLUMNS - 1) * MASK_COLUMNS : 0;
    }
    if (n_idx == (n_loop - 1)) {
        qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
        qk_round_n = RoundUp<16>(qk_n);
        need_mask = true;
        mask_start_offset = (qk_n - 1) * MASK_COLUMNS;
    }

    // 平台同步：SoftmaxStage1 前置同步（WaitFlagDev + WAIT_FLAG）
    PlatformSoftmaxStage1PreSync();

    // SoftmaxStage1 调用（ping-pong）
    if (sub_m > 0) {
        if (mask_type == 3) {
            mask_start_offset = mask_offset + n_idx * pp_n_scalar;
        }
        // input QK shape (sub_m, qk_round_n)
        if (n_idx % 2 == 0){
            SoftmaxStage1(
                p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET +
                    (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n * T_BLOCK_OFFSET + (uint64_t)(n_idx % 2) * TMP_SIZE * T_BLOCK_OFFSET / 2],
                s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                s_rope_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                mask_gm_tensor[mask_start_offset],
                dm32_ubuf_tensor, ll_ubuf_tensor, pm32_ubuf_tensor,
                n_idx, qk_n, qk_round_n, sub_m, 0, sub_n_loop, cur_batch, start_kv, real_n_loop, head_idx, pm_flag_scalar1, cur_q_seqlen, cur_kv_seqlen, need_mask
            );
        } else {
            SoftmaxStage1(
                p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET  +
                    (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n * T_BLOCK_OFFSET +
                    TMP_SIZE * T_BLOCK_OFFSET / 2],
                s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                    TMP_SIZE_DECODER / 2],
                s_rope_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                    TMP_SIZE_DECODER / 2],
                mask_gm_tensor[mask_start_offset],
                dm32_stage2_ubuf_tensor, ll_stage2_ubuf_tensor, pm32_ubuf_stage2_tensor,
                n_idx, qk_n, qk_round_n, sub_m, 0, sub_n_loop, cur_batch, start_kv, real_n_loop, head_idx, pm_flag_scalar2, cur_q_seqlen, cur_kv_seqlen, need_mask
            );
        }
    }

    // 平台同步：SoftmaxStage1 后置同步（FftsCrossCoreSync + SET_FLAG）
    PlatformSoftmaxStage1PostSync();
}

// 业务函数：非TP1 Phase 2 — SoftmaxStage2MLAHeadLoop 核心 Head Loop 计算
// 包含 load lo → DeQuant → exp(dm) → gl=dm*gl → gl=ll+gl → brcb dm → go=go*dm → go=lo+go
// → brcb gl → go=go/gl → conv_v → DataCopyPad输出 → Ring LSE copyout
__aicore__ __attribute__((always_inline)) inline void SoftmaxStage2MLAHeadLoop(
    AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
    AscendC::GlobalTensor<float> go_gm_tensor,
    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
    AscendC::LocalTensor<float> dm32_ubuf_tensor,
    AscendC::LocalTensor<float> ll_ubuf_tensor,
    AscendC::LocalTensor<float> pm32_ubuf_tensor,
    uint32_t n_idx,
    uint32_t n_loop,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    uint64_t o_offset,
    uint32_t head_idx,
    uint32_t pm_flag_scalar,
    uint32_t head_loop,
    uint32_t head_loop_idx,
    uint32_t q_seq_len,
    uint32_t sub_head_num,
    uint32_t cur_head_num,
    uint32_t numhead_per_process,
    uint32_t head_res_row_num,
    uint32_t head_start_sblock_idx,
    uint32_t tail_res_row_num
    )
{
    uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    if (n_idx != 0) {
        gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
            lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
            o_tmp_gm_tensor,
            0,                    // sid
            1,                    // nBurst
            sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                    // srcGap
            0                     // dstGap
        );
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
           DeQuantPerHeadImpl(
                deq_scale_gm_tensor_k1[head_idx],
                o_tmp_gm_tensor,
                lo_ubuf_tensor, lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),// lo_ubuf_tensor use the same ptr
                descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor, sub_m, round_v, round_v, 1, 0);
        }
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
    if (n_idx != 0) {
        // *** dm = exp(dm)
        if (head_loop_idx == 0) {
            exp_v<ArchType::ASCEND_V220, float>(dm32_ubuf_tensor,
                dm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // srcBlockStride
                8,          // dstRepeatStride
                8           // srcRepeatStride
            );
            PIPE_BARRIER(V);
            // *** gl = dm * gl
            mul_v<ArchType::ASCEND_V220, float>(gl32_ubuf_tensor,
                dm32_ubuf_tensor,
                gl32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // src0BlockStride
                1,          // src1BlockStride
                8,          // dstRepeatStride
                8,          // src0RepeatStride
                8           // src1RepeatStride
            );
            PIPE_BARRIER(V);
            // *** gl = ll + gl
            add_v<ArchType::ASCEND_V220, float>(gl32_ubuf_tensor,
                gl32_ubuf_tensor,
                ll_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // src0BlockStride
                1,          // src1BlockStride
                8,          // dstRepeatStride
                8,          // src0RepeatStride
                8           // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        if (head_loop > 1) {
            gm_to_ub<ArchType::ASCEND_V220, float>(
                go32_ubuf_tensor,
                go_gm_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }

        // *** go = go * dm_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
            mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,        // repeat
                1,            // dstBlockStride
                1,            // src0BlockStride
                0,            // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1             // src1RepeatStride
            );
        }
        if (__v % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(__v % FLOAT_VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,        // repeat
                1,            // dstBlockStride
                1,            // src0BlockStride
                0,            // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1             // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
        // *** go = lo + go
        add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
            go32_ubuf_tensor,
            lo_ubuf_tensor,
            (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                            // dstBlockStride
            1,                            // src0BlockStride
            1,                            // src1BlockStride
            8,                            // dstRepeatStride
            8,                            // src0RepeatStride
            8                             // src1RepeatStride
        );
        PIPE_BARRIER(V);
    } else {
        // *** gl = ll
        if (head_loop_idx == 0) {
            ub_to_ub<ArchType::ASCEND_V220, float>(
                gl32_ubuf_tensor,
                ll_ubuf_tensor,
                0,                // sid
                1,                // nBurst
                64 / FLOAT_BLOCK_SIZE,  // lenBurst
                // round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                // srcGap
                0                 // dstGap
                );
            PIPE_BARRIER(V);
        }

        gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
            go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
            o_tmp_gm_tensor,
            0,                    // sid
            1,                    // nBurst
            sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                    // srcGap
            0                     // dstGap
        );
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            DeQuantPerHeadImpl(
                deq_scale_gm_tensor_k1[head_idx],
                o_tmp_gm_tensor,
                go32_ubuf_tensor, go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
                descale_k1_ubuf_tensor, tv32_ubuf_tensor, pm32_ubuf_tensor, sub_m, round_v, round_v, 1, 0);
        } else {
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
    }
    SET_FLAG(V, MTE2, EVENT_ID0);

    if (n_idx == n_loop - 1) {
        // *** gl_block = expand_to_block(gl)
        brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            gl32_ubuf_tensor.ReinterpretCast<uint32_t>()[head_loop_idx * 16],
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        // *** go = go / gl_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
            div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,                 // repeat
                1,                     // dstBlockStride
                1,                     // src0BlockStride
                0,                     // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                      // src1RepeatStride
            );
        }
        if (__v % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(__v % FLOAT_VECTOR_SIZE);
            div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,                 // repeat
                1,                     // dstBlockStride
                1,                     // src0BlockStride
                0,                     // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                      // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // fix hidden_size=96
        }
        PIPE_BARRIER(V);

        // *** go = castfp32to16(go)
        conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(go_ubuf_tensor,
            go32_ubuf_tensor,
            (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                            // dstBlockStride
            1,                            // srcBlockStride
            4,                            // dstRepeatStride
            8                             // srcRepeatStride
        );
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        uint32_t inner_o_gm_offset = 0;
        uint32_t inner_go_ubuf_offset = 0;

        if (head_res_row_num != 0) {
            AscendC::DataCopyPad(
                o_gm_tensor[inner_o_gm_offset + q_heads * __v * head_start_sblock_idx],
                go_ubuf_tensor[inner_go_ubuf_offset],
                AscendC::DataCopyExtParams(
                    head_res_row_num,  // blockCount
                    __v * 2,    // blockLen
                    0,          // srcStride
                    __v * (q_heads - 1) * 2,  // dstStride
                    0           // rsv
                )
            );
            inner_o_gm_offset += __v;
            inner_go_ubuf_offset += head_res_row_num * __v;
        }

        for (uint32_t i = 0; i < numhead_per_process; i++) {
            AscendC::DataCopyPad(
                o_gm_tensor[inner_o_gm_offset],
                go_ubuf_tensor[inner_go_ubuf_offset],
                AscendC::DataCopyExtParams(
                    q_seq_len,  // blockCount
                    __v * 2,    // blockLen
                    0,          // srcStride
                    __v * (q_heads - 1) * 2,  // dstStride
                    0           // rsv
                )
            );
            inner_o_gm_offset += __v;
            inner_go_ubuf_offset += q_seq_len * __v;
        }

        if (tail_res_row_num != 0) {
            AscendC::DataCopyPad(
                o_gm_tensor[inner_o_gm_offset],
                go_ubuf_tensor[inner_go_ubuf_offset],
                AscendC::DataCopyExtParams(
                    tail_res_row_num,  // blockCount
                    __v * 2,    // blockLen
                    0,          // srcStride
                    __v * (q_heads - 1) * 2,  // dstStride
                    0           // rsv
                )
            );
        }
        // ********************* move O to GM ************************
        if constexpr (IS_RING) {
            uint32_t lenBurst = sizeof(OUT_DTYPE);
            ln_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                gl32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // srcBlockStride
                8,          // dstRepeatStride
                8           // srcRepeatStride
            );
            PIPE_BARRIER(V);
            add_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                lse32_ubuf_tensor,
                gm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // src0BlockStride
                1,          // src1BlockStride
                8,          // dstRepeatStride
                8,          // src0RepeatStride
                8           // src1RepeatStride
            );
            PIPE_BARRIER(V);
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lse_conv_ubuf_tensor,
                lse32_ubuf_tensor,
                sub_m_d64,                    // repeat
                1,                            // dstBlockStride
                1,                            // srcBlockStride
                4,                            // dstRepeatStride
                8                             // srcRepeatStride
            );
            SET_FLAG(V, MTE3, EVENT_ID1);
            WAIT_FLAG(V, MTE3, EVENT_ID1);
            // copyout lse
            ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                lse_gm_tensor[(int64_t)(o_offset / __k)],
                lse_conv_ubuf_tensor,
                0,                 // sid
                1,                 // nBurst
                lenBurst * sub_m * head_loop,  // lenBurst
                0,                 // leftPaddingNum
                0,                 // rightPaddingNum
                0,                 // srcGap
                0                  // dstGap
            );
            SET_FLAG(MTE3, V, EVENT_ID1);
            WAIT_FLAG(MTE3, V, EVENT_ID1);
        }

    } else if (head_loop > 1) {
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
    SET_FLAG(MTE3, MTE2, EVENT_ID4);
}

// 业务函数：非TP1 Phase 2 — Softmax Stage2 调度
// 包含 head 循环计算、WaitFlag、SoftmaxStage2MLAHeadLoop 调用
__aicore__ __attribute__((always_inline)) inline void ScheduleSoftmaxStage2(
    VectorContext &ctx, uint32_t n_idx)
{
    uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    uint32_t cur_kv_seqlen = ctx.cur_kv_seqlen;
    uint32_t cur_head_num = ctx.cur_head_num;
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t sub_m = ctx.sub_m;
    uint32_t sub_head_num = ctx.sub_head_num;
    uint32_t head_idx = ctx.head_idx;

    uint32_t qk_n_2 = ctx.qk_n_2;
    uint32_t qk_round_n_2 = ctx.qk_round_n_2;

    uint32_t process_row_num = 16;
    uint32_t numhead_per_process = process_row_num / cur_q_seqlen;

    if (n_idx == n_loop) {
        qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
        qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
    }
    WaitFlagDev(UPDATE_READY_DECODER);
    if (sub_m > 0) {
        uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

        uint32_t head_res_row_num = 0;
        uint32_t head_start_sblock_idx = 0;
        uint32_t tail_res_row_num = 0;

        for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
            uint32_t head_offset = head_loop_idx * process_row_num * round_v;
            uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num; // 15 or 3

            // complete head num
            head_start_sblock_idx = tail_res_row_num;
            head_res_row_num = (cur_q_seqlen - tail_res_row_num) % cur_q_seqlen;
            uint32_t cur_numhead_per_process = (cur_sub_m - head_res_row_num) / cur_q_seqlen;
            tail_res_row_num = cur_sub_m - cur_numhead_per_process * cur_q_seqlen - head_res_row_num;

            uint32_t out_o_offset = head_loop_idx * numhead_per_process * round_v; // modified, round_v = 512

            SoftmaxStage2MLAHeadLoop(
                o_tmp_gm_tensor[(uint64_t)(block_idx * TMP_SIZE * 2 + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset + ((n_idx - 1) % 2) * TMP_SIZE)],
                go_gm_tensor[(uint64_t)(block_idx * TMP_SIZE + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset)],
                o_gm_tensor[(uint64_t)(o_offset + out_o_offset)],
                dm32_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 128 + head_loop_idx * process_row_num)],
                ll_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 256 + head_loop_idx * process_row_num)],
                pm32_ubuf_tensor[(uint64_t)((n_idx - 1) % 2 * 128 + head_loop_idx * process_row_num)],
                n_idx - 1, n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m, o_offset,
                head_idx + head_loop_idx * process_row_num,
                pm_flag_scalar1, head_loop, head_loop_idx, cur_q_seqlen, sub_head_num, cur_head_num,
                cur_numhead_per_process,
                head_res_row_num, head_start_sblock_idx, tail_res_row_num);
        }
    }
}

// 业务函数：非 TP1 — InnerRunVectorChange（内层业务编排）
// 初始化 VectorContext + n_loop 软流水循环（Stage1↔Stage2 交替）
__aicore__ __attribute__((always_inline)) inline void InnerRunVectorChange(
    uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
{
    VectorContext ctx;
    InitVectorContext(ctx, cur_batch, start_head, cur_nIndx,
        cur_q_seqlen, cur_kv_seqlen, cur_head_num, offset_tiling);

    o_offset = ctx.o_offset;
    uint32_t start_kv = 0;

    for (uint32_t n_idx = 0; n_idx < ctx.n_loop + 1; n_idx++) {
        if (n_idx != ctx.n_loop) {
            ScheduleSoftmaxStage1(ctx, n_idx, start_kv);
        }
        if (n_idx != 0) {
            ScheduleSoftmaxStage2(ctx, n_idx);
        }
    }
}

// 业务函数：调度 Vector 非 TP1 任务（Run 方法的中段业务逻辑）
// 包含 batch/head 循环调度，读取 tiling 参数后调用 InnerRunVectorChange
__aicore__ __attribute__((always_inline)) inline void ScheduleVectorTasks()
{
    uint64_t cur_batch = 0;
    uint32_t q_block_num_per_batch = (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
    uint32_t process_num = q_block_num_per_batch * num_batches;
    for (uint32_t process = block_idx; process < process_num; process += (uint32_t)block_num) {
        cur_batch = process / q_block_num_per_batch;
        if (cur_batch >= num_batches) break;
        uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
        uint32_t q_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
        uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
        if (kv_seqlen == 0) {
            continue;
        }
        uint32_t start_head = (process % q_block_num_per_batch) * cur_qn_blk_size;
        uint32_t cur_q_seq_len = q_seqlen;
        uint32_t cur_kv_seqlen = kv_seqlen;
        uint32_t cur_head_num = cur_qn_blk_size;
        uint32_t cur_nIndx = 0;
        InnerRunVectorChange(cur_batch, start_head, cur_nIndx, cur_q_seq_len, cur_kv_seqlen, cur_head_num,
            offset_tiling, 512, embed_split_loop_v_former);
    }
}

// ====== TP1 路径：Phase 1 / Phase 2 业务调度 ======

// 业务函数：TP1 Phase 1 — OnlineSoftmax Stage1 调度（InnerRunVectorChangeTP1 / TailInnerRunVectorChangeTP1 共用）
// 包含 tail qk_n 计算、WaitFlag、OnlineSoftmaxStage1 调用（m_ind 循环 ping-pong）、FftsCrossCoreSync
__aicore__ __attribute__((always_inline)) inline void ScheduleOnlineSoftmaxStage1(
    VectorTP1Context &ctx, uint32_t n_idx)
{
    uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    uint32_t cur_head_num = ctx.cur_head_num;
    uint32_t cur_kv_seqlen = ctx.cur_kv_seqlen;
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t sub_m = ctx.sub_m;
    uint32_t n_loop = ctx.n_loop;
    uint32_t s_block_stack = ctx.s_block_stack;
    uint32_t m_slice = ctx.m_slice;
    uint32_t m_end = ctx.m_end;

    uint32_t qk_n = ctx.qk_n;
    uint32_t qk_round_n = ctx.qk_round_n;

    if (n_idx + s_block_stack > n_loop - 1) {
        qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
    } else {
        qk_n = pp_n_scalar * s_block_stack;
    }
    qk_round_n = RoundUp<16>(qk_n);
    if (sub_m == 0) {
        WaitFlagDev(QK_READY_DECODER);
    }
    uint32_t pingpong_flag = 0;
    for (uint32_t m_ind = 0; m_ind < m_end; m_ind++) {
        uint32_t row_offset = m_ind * m_slice;
        uint32_t curr_m = m_ind == m_end - 1 ? sub_m - row_offset : m_slice;
        uint32_t s_ub_offset = pingpong_flag * 8192;
        uint32_t p_gm_offset = (uint64_t)block_idx * TMP_SIZE * 2 +
                                (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + row_offset * qk_round_n + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE;
        uint32_t s_gm_offset = (int64_t)block_idx * TMP_SIZE_DECODER * 4 +
                                (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n + row_offset * qk_round_n + (uint64_t)((n_idx / s_block_stack) % 2) * TMP_SIZE_DECODER * 2;
        if (m_ind == 0) {
            WaitFlagDev(QK_READY_DECODER);
        }
        if (curr_m == 0) {
            continue;
        }
        OnlineSoftmaxStage1<float, float, IN_DTYPE, IN_DTYPE, MaskType::MASK_TYPE_NONE> (
            ls32_ubuf_tensor[s_ub_offset],
            mask_ubuf_tensor,
            mask_ubuf_tensor.template ReinterpretCast<float>(),
            lm32_ubuf_tensor[row_offset],
            hm32_ubuf_tensor[row_offset],
            gm32_ubuf_tensor[row_offset],
            dm32_ubuf_tensor[((n_idx / s_block_stack) % 2) * UB_FLOAT_LINE_SIZE + row_offset],
            ls32_ubuf_tensor[s_ub_offset],
            ll_ubuf_tensor[row_offset],
            gl32_ubuf_tensor[row_offset],
            lp_ubuf_tensor[s_ub_offset * 2],
            tv32_ubuf_tensor,
            s_gm_tensor[s_gm_offset],
            p_gm_tensor[p_gm_offset],
            n_idx == 0, this->tor,
            curr_m, qk_n, qk_round_n, pingpong_flag
        );
        pingpong_flag = 1 - pingpong_flag;
    }
    FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);
}

// 业务函数：TP1 Phase 2 — SoftmaxStage2MLAHeadLoopTP1 核心 Head Loop 计算
// 与非TP1相似，差异：n_idx!=4 vs n_idx!=0，无 DeQuant/exp(dm)/gl=dm*gl/gl=ll+gl 路径
__aicore__ __attribute__((always_inline)) inline void SoftmaxStage2MLAHeadLoopTP1(
    AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
    AscendC::GlobalTensor<float> go_gm_tensor,
    AscendC::GlobalTensor<OUT_DTYPE> o_gm_tensor,
    AscendC::LocalTensor<float> dm32_ubuf_tensor,
    AscendC::LocalTensor<float> ll_ubuf_tensor,
    AscendC::LocalTensor<float> pm32_ubuf_tensor,
    uint32_t n_idx,
    uint32_t n_loop,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    uint64_t o_offset,
    uint32_t head_idx,
    uint32_t pm_flag_scalar,
    uint32_t head_loop,
    uint32_t head_loop_idx,
    uint32_t q_seq_len,
    uint32_t sub_head_num,
    uint32_t cur_head_num,
    uint32_t numhead_per_process,
    uint32_t head_res_row_num,
    uint32_t head_start_sblock_idx,
    uint32_t tail_res_row_num
    )
{
    uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    if (n_idx != 4) {
        gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
            lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
            o_tmp_gm_tensor,
            0,                    // sid
            1,                    // nBurst
            sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                    // srcGap
            0                     // dstGap
        );
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
    if (n_idx != 4) {
        // expand_to_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        if (head_loop > 1) {
            gm_to_ub<ArchType::ASCEND_V220, float>(
                go32_ubuf_tensor,
                go_gm_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }

        // *** go = go * dm_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
            mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,        // repeat
                1,            // dstBlockStride
                1,            // src0BlockStride
                0,            // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1             // src1RepeatStride
            );
        }
        if (__v % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(__v % FLOAT_VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,        // repeat
                1,            // dstBlockStride
                1,            // src0BlockStride
                0,            // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1             // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
        // *** go = lo + go
        add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
            go32_ubuf_tensor,
            lo_ubuf_tensor,
            (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                            // dstBlockStride
            1,                            // src0BlockStride
            1,                            // src1BlockStride
            8,                            // dstRepeatStride
            8,                            // src0RepeatStride
            8                             // src1RepeatStride
        );
        PIPE_BARRIER(V);
    } else {
        // *** go = lo

        gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
            go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
            o_tmp_gm_tensor,
            0,                    // sid
            1,                    // nBurst
            sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                    // srcGap
            0                     // dstGap
        );
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
    }
    SET_FLAG(V, MTE2, EVENT_ID0);

    if (n_idx + 4 > n_loop + 4 - 1) {
        // *** gl_block = expand_to_block(gl), 存放于 tv
        brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            gl32_ubuf_tensor.ReinterpretCast<uint32_t>()[head_loop_idx * 16],
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        // *** go = go / gl_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vdiv_idx = 0; vdiv_idx < __v / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
            div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[vdiv_idx * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,                 // repeat
                1,                     // dstBlockStride
                1,                     // src0BlockStride
                0,                     // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                      // src1RepeatStride
            );
        }
        if (__v % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(__v % FLOAT_VECTOR_SIZE);
            div_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,                 // repeat
                1,                     // dstBlockStride
                1,                     // src0BlockStride
                0,                     // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1                      // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);  // fix hidden_size=96
        }
        PIPE_BARRIER(V);

        // *** go = castfp32to16(go)
        conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(go_ubuf_tensor,
            go32_ubuf_tensor,
            (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                            // dstBlockStride
            1,                            // srcBlockStride
            4,                            // dstRepeatStride
            8                             // srcRepeatStride
        );
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        uint32_t inner_o_gm_offset = 0;
        uint32_t inner_go_ubuf_offset = 0;

        if (head_res_row_num != 0) {
            AscendC::DataCopyPad(
                o_gm_tensor[inner_o_gm_offset + q_heads * __v * head_start_sblock_idx],
                go_ubuf_tensor[inner_go_ubuf_offset],
                AscendC::DataCopyExtParams(
                    head_res_row_num,  // blockCount
                    __v * 2,    // blockLen
                    0,          // srcStride
                    __v * (q_heads - 1) * 2,  // dstStride
                    0           // rsv
                )
            );
            inner_o_gm_offset += __v;
            inner_go_ubuf_offset += head_res_row_num * __v;
        }

        for (uint32_t i = 0; i < numhead_per_process; i++) {
            AscendC::DataCopyPad(
                o_gm_tensor[inner_o_gm_offset],
                go_ubuf_tensor[inner_go_ubuf_offset],
                AscendC::DataCopyExtParams(
                    q_seq_len,  // blockCount
                    __v * 2,    // blockLen
                    0,          // srcStride
                    __v * (q_heads - 1) * 2,  // dstStride
                    0           // rsv
                )
            );
            inner_o_gm_offset += __v;
            inner_go_ubuf_offset += q_seq_len * __v;
        }

        if (tail_res_row_num != 0) {
            AscendC::DataCopyPad(
                o_gm_tensor[inner_o_gm_offset],
                go_ubuf_tensor[inner_go_ubuf_offset],
                AscendC::DataCopyExtParams(
                    tail_res_row_num,  // blockCount
                    __v * 2,    // blockLen
                    0,          // srcStride
                    __v * (q_heads - 1) * 2,  // dstStride
                    0           // rsv
                )
            );
        }
        // ********************* move O to GM ************************
        if constexpr (IS_RING) {
            uint32_t lenBurst = sizeof(OUT_DTYPE);
            ln_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                gl32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // srcBlockStride
                8,          // dstRepeatStride
                8           // srcRepeatStride
            );
            PIPE_BARRIER(V);
            add_v<ArchType::ASCEND_V220, float>(lse32_ubuf_tensor,
                lse32_ubuf_tensor,
                gm32_ubuf_tensor,
                sub_m_d64,  // repeat
                1,          // dstBlockStride
                1,          // src0BlockStride
                1,          // src1BlockStride
                8,          // dstRepeatStride
                8,          // src0RepeatStride
                8           // src1RepeatStride
            );
            PIPE_BARRIER(V);
            conv_v<ArchType::ASCEND_V220, float, OUT_DTYPE>(lse_conv_ubuf_tensor,
                lse32_ubuf_tensor,
                sub_m_d64,                    // repeat
                1,                            // dstBlockStride
                1,                            // srcBlockStride
                4,                            // dstRepeatStride
                8                             // srcRepeatStride
            );
            SET_FLAG(V, MTE3, EVENT_ID1);
            WAIT_FLAG(V, MTE3, EVENT_ID1);
            // copyout lse
            ub_to_gm_align<ArchType::ASCEND_V220, OUT_DTYPE>(
                lse_gm_tensor[(int64_t)(o_offset / __k)],
                lse_conv_ubuf_tensor,
                0,                 // sid
                1,                 // nBurst
                lenBurst * sub_m * head_loop,  // lenBurst
                0,                 // leftPaddingNum
                0,                 // rightPaddingNum
                0,                 // srcGap
                0                  // dstGap
            );
            SET_FLAG(MTE3, V, EVENT_ID1);
            WAIT_FLAG(MTE3, V, EVENT_ID1);
        }
    }
    else if (head_loop > 1) {
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
    SET_FLAG(MTE3, MTE2, EVENT_ID4);
}

// 业务函数：TP1 Phase 2 — SoftmaxStage2MLAHeadLoopTP1 调度（非 Tail 路径）
// 包含 head 循环计算、WaitFlag、SoftmaxStage2MLAHeadLoopTP1 调用
__aicore__ __attribute__((always_inline)) inline void ScheduleSoftmaxStage2TP1(
    VectorTP1Context &ctx, uint32_t n_idx)
{
    uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    uint32_t cur_head_num = ctx.cur_head_num;
    uint32_t cur_kv_seqlen = ctx.cur_kv_seqlen;
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t sub_m = ctx.sub_m;
    uint32_t sub_head_num = ctx.sub_head_num;
    uint32_t n_loop = ctx.n_loop;
    uint32_t head_idx = ctx.head_idx;
    uint32_t s_block_stack = ctx.s_block_stack;

    uint32_t qk_n_2 = ctx.qk_n_2;
    uint32_t qk_round_n_2 = ctx.qk_round_n_2;

    uint32_t process_row_num = 16;
    uint32_t numhead_per_process = process_row_num / cur_q_seqlen;

    if (n_idx == n_loop) {
        qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
        qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
    }
    WaitFlagDev(UPDATE_READY_DECODER);
    if (sub_m > 0) {
        uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

        uint32_t head_res_row_num = 0;
        uint32_t head_start_sblock_idx = 0;
        uint32_t tail_res_row_num = 0;

        for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
            uint32_t head_offset = head_loop_idx * process_row_num * round_v;
            uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num;

            // complete head num
            head_start_sblock_idx = tail_res_row_num;
            head_res_row_num = (cur_q_seqlen - tail_res_row_num) % cur_q_seqlen;
            uint32_t cur_numhead_per_process = (cur_sub_m - head_res_row_num) / cur_q_seqlen;
            tail_res_row_num = cur_sub_m - cur_numhead_per_process * cur_q_seqlen - head_res_row_num;

            uint32_t out_o_offset = head_loop_idx * numhead_per_process * round_v;

            SoftmaxStage2MLAHeadLoopTP1(
                o_tmp_gm_tensor[(uint64_t)(block_idx * TMP_SIZE * 2 + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset + ((n_idx / s_block_stack - 1) % 2) * TMP_SIZE)],
                go_gm_tensor[(uint64_t)(block_idx * TMP_SIZE + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset)],
                o_gm_tensor[(uint64_t)(o_offset + out_o_offset)],
                dm32_ubuf_tensor[(uint64_t)((n_idx / s_block_stack - 1) % 2 * UB_FLOAT_LINE_SIZE + head_loop_idx * process_row_num)],
                ll_ubuf_tensor[(uint64_t)((n_idx / s_block_stack - 1) % 2 * 256 + head_loop_idx * process_row_num)],
                pm32_ubuf_tensor,
                n_idx, n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m, o_offset, head_idx,
                pm_flag_scalar1, head_loop, head_loop_idx, cur_q_seqlen, sub_head_num, cur_head_num, cur_numhead_per_process,
                head_res_row_num, head_start_sblock_idx, tail_res_row_num);
        }
    }
}

// 业务函数：TP1 Phase 2 (Tail) — TailSoftmaxStage2MLAHeadLoopTP1 核心 Head Loop 计算
// 与 TP1 版本相似，但增加额外的 gl/gm/go copyout 逻辑
__aicore__ __attribute__((always_inline)) inline void TailSoftmaxStage2MLAHeadLoopTP1(
    AscendC::GlobalTensor<mm2CopyType> o_tmp_gm_tensor,
    AscendC::GlobalTensor<float> go_gm_tensor,
    AscendC::GlobalTensor<float> gl_gm_tensor,
    AscendC::GlobalTensor<float> gm_gm_tensor,
    AscendC::LocalTensor<float> dm32_ubuf_tensor,
    AscendC::LocalTensor<float> go32_ubuf_tensor,
    AscendC::LocalTensor<float> gl32_ubuf_tensor,
    AscendC::LocalTensor<float> gm32_ubuf_tensor,
    uint32_t n_idx,
    uint32_t n_loop,
    uint32_t qk_n,
    uint32_t qk_round_n,
    uint32_t sub_m,
    uint64_t o_offset,
    uint32_t head_idx,
    uint32_t head_loop,
    uint32_t head_loop_idx,
    uint32_t q_seq_len,
    uint32_t sub_head_num,
    uint32_t cur_head_num,
    uint32_t numhead_per_process
    )
{
    uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
    WAIT_FLAG(V, MTE2, EVENT_ID0);
    if (n_idx != 4) {
        gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
            lo_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
            o_tmp_gm_tensor,
            0,                    // sid
            1,                    // nBurst
            sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                    // srcGap
            0                     // dstGap
        );
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID4);
    if (n_idx != 4) {
        // expand_to_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        brcb_v<ArchType::ASCEND_V220, uint32_t>(tv32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            dm32_ubuf_tensor.ReinterpretCast<uint32_t>(),
            1,               // dstBlockStride
            8,               // dstRepeatStride
            round_sub_m / FLOAT_BLOCK_SIZE  // repeat
        );
        PIPE_BARRIER(V);
        if (head_loop > 1) {
            gm_to_ub<ArchType::ASCEND_V220, float>(
                go32_ubuf_tensor,
                go_gm_tensor,
                0,
                1,
                sub_m * round_v / FLOAT_BLOCK_SIZE,
                0,
                0
            );
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }

        // *** go = go * dm_block
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t vmul_idx = 0; vmul_idx < __v / FLOAT_VECTOR_SIZE; ++vmul_idx) {
            mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,        // repeat
                1,            // dstBlockStride
                1,            // src0BlockStride
                0,            // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1             // src1RepeatStride
            );
        }
        if (__v % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(__v % FLOAT_VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                go32_ubuf_tensor[__v / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tv32_ubuf_tensor,
                sub_m,        // repeat
                1,            // dstBlockStride
                1,            // src0BlockStride
                0,            // src1BlockStride
                round_v / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                round_v / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                1             // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);
        // *** go = lo + go
        add_v<ArchType::ASCEND_V220, float>(go32_ubuf_tensor,
            go32_ubuf_tensor,
            lo_ubuf_tensor,
            (sub_m * round_v + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                            // dstBlockStride
            1,                            // src0BlockStride
            1,                            // src1BlockStride
            8,                            // dstRepeatStride
            8,                            // src0RepeatStride
            8                             // src1RepeatStride
        );
        PIPE_BARRIER(V);
    } else {
        // *** go = lo

        gm_to_ub<ArchType::ASCEND_V220, mm2CopyType>(
            go32_ubuf_tensor.template ReinterpretCast<mm2CopyType>(),
            o_tmp_gm_tensor,
            0,                    // sid
            1,                    // nBurst
            sub_m * round_v / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                    // srcGap
            0                     // dstGap
        );
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
    }
    SET_FLAG(V, MTE2, EVENT_ID0);

    if (n_idx + 4 > n_loop + 4 - 1) {
        // The last step to process the o with dividing and copyout
        // TODO: Maybe the following two don't need waiting
        // Copyout gl32_ubuf_tensor to gl_gm_tensor
        ub_to_gm<ArchType::ASCEND_V220, float>(
            gl_gm_tensor,
            gl32_ubuf_tensor,
            0,
            1,
            sub_m / FLOAT_BLOCK_SIZE,
            0,
            0
        );
        
        // Copyout rowmax to global gm32_ubuf_tensor
        ub_to_gm<ArchType::ASCEND_V220, float>(
            gm_gm_tensor,
            gm32_ubuf_tensor,
            0,
            1,
            sub_m / FLOAT_BLOCK_SIZE,
            0,
            0
        );
        // Copyout go32_ubuf_tensor to go_gm_tensor
        // This is needed to wait for former calculation
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
    else if (head_loop > 1) {
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
    SET_FLAG(MTE3, MTE2, EVENT_ID4);
    PIPE_BARRIER(ALL);
}

// 业务函数：TP1 Phase 2 (Tail) — TailSoftmaxStage2MLAHeadLoopTP1 调度（Tail 路径）
// 包含 head 循环计算、WaitFlag、TailSoftmaxStage2MLAHeadLoopTP1 调用
__aicore__ __attribute__((always_inline)) inline void ScheduleTailSoftmaxStage2TP1(
    VectorTP1Context &ctx, uint32_t n_idx)
{
    uint32_t cur_q_seqlen = ctx.cur_q_seqlen;
    uint32_t cur_head_num = ctx.cur_head_num;
    uint32_t cur_kv_seqlen = ctx.cur_kv_seqlen;
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t sub_m = ctx.sub_m;
    uint32_t sub_head_num = ctx.sub_head_num;
    uint32_t n_loop = ctx.n_loop;
    uint32_t head_idx = ctx.head_idx;
    uint32_t start_head = ctx.start_head;
    uint32_t s_block_stack = ctx.s_block_stack;

    uint32_t qk_n_2 = ctx.qk_n_2;
    uint32_t qk_round_n_2 = ctx.qk_round_n_2;

    uint32_t process_row_num = 16;
    uint32_t numhead_per_process = process_row_num / cur_q_seqlen;

    if (n_idx == n_loop) {
        qk_n_2 = (cur_kv_seqlen - (n_idx - 1) * pp_n_scalar);
        qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
    }
    WaitFlagDev(UPDATE_READY_DECODER);
    if (sub_m > 0) {
        uint32_t head_loop = (sub_m + process_row_num - 1) / process_row_num;

        uint32_t head_res_row_num = 0;
        uint32_t head_start_sblock_idx = 0;
        uint32_t tail_res_row_num = 0;

        for (uint32_t head_loop_idx = 0; head_loop_idx < head_loop; ++head_loop_idx) {
            uint32_t head_offset = head_loop_idx * process_row_num * round_v;
            uint32_t cur_sub_m = head_loop_idx == (head_loop - 1) ? sub_m - head_loop_idx * process_row_num : process_row_num;

            // complete head num
            head_start_sblock_idx = tail_res_row_num;
            head_res_row_num = (cur_q_seqlen - tail_res_row_num) % cur_q_seqlen;
            uint32_t cur_numhead_per_process = (cur_sub_m - head_res_row_num) / cur_q_seqlen;
            tail_res_row_num = cur_sub_m - cur_numhead_per_process * cur_q_seqlen - head_res_row_num;

            uint32_t out_o_offset = head_loop_idx * numhead_per_process * round_v;

            TailSoftmaxStage2MLAHeadLoopTP1(
                o_tmp_gm_tensor[(uint64_t)(block_idx * TMP_SIZE * 2 + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset + ((n_idx / s_block_stack - 1) % 2) * TMP_SIZE)],
                go_gm_tensor[(uint64_t)(block_idx * TMP_SIZE + sub_block_idx * cur_head_num * cur_q_seqlen / 2 * round_v + head_offset)],
                tmp_gm_tensor[(uint64_t)(block_idx * q_heads + start_head + sub_block_idx * cur_head_num * cur_q_seqlen / 2 + head_loop_idx * process_row_num)],
                tmp_gm_tensor[(uint64_t)(block_num * q_heads + block_idx * q_heads + start_head + sub_block_idx * cur_head_num * cur_q_seqlen / 2 + head_loop_idx * process_row_num)],
                dm32_ubuf_tensor[(uint64_t)((n_idx / s_block_stack - 1) % 2 * UB_FLOAT_LINE_SIZE + head_loop_idx * process_row_num)],
                go32_ubuf_tensor,
                gl32_ubuf_tensor[(uint64_t)(head_loop_idx * process_row_num)],
                gm32_ubuf_tensor[(uint64_t)(head_loop_idx * process_row_num)],
                n_idx, n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), cur_sub_m, o_offset, head_idx,
                head_loop, head_loop_idx, cur_q_seqlen, sub_head_num, cur_head_num, cur_numhead_per_process
            );
        }
    }
}

// 业务函数：TP1 — InnerRunVectorChangeTP1（内层业务编排）
// 初始化 VectorTP1Context + n_loop 软流水循环（Stage1↔Stage2TP1 交替）
__aicore__ __attribute__((always_inline)) inline void InnerRunVectorChangeTP1(
    uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
{
    VectorTP1Context ctx;
    InitVectorTP1Context(ctx, cur_batch, start_head, cur_nIndx,
        cur_q_seqlen, cur_kv_seqlen, cur_head_num, offset_tiling);

    o_offset = ctx.o_offset;

    for (uint32_t n_idx = 0; n_idx < ctx.n_loop + ctx.s_block_stack; n_idx += ctx.s_block_stack) {
        if (n_idx < ctx.n_loop) {
            ScheduleOnlineSoftmaxStage1(ctx, n_idx);
        }
        if (n_idx >= ctx.s_block_stack) {
            ScheduleSoftmaxStage2TP1(ctx, n_idx);
        }
    }
}

// 业务函数：TP1 Tail — TailInnerRunVectorChangeTP1（内层业务编排）
// 初始化 VectorTP1Context + n_loop 软流水循环（Stage1↔TailStage2TP1 交替）
__aicore__ __attribute__((always_inline)) inline void TailInnerRunVectorChangeTP1(
    uint32_t start_head,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
{
    VectorTP1Context ctx;
    InitVectorTP1Context(ctx, 0, start_head, 0,
        cur_q_seqlen, cur_kv_seqlen, cur_head_num, offset_tiling);

    o_offset = ctx.o_offset;

    for (uint32_t n_idx = 0; n_idx < ctx.n_loop + ctx.s_block_stack; n_idx += ctx.s_block_stack) {
        if (n_idx < ctx.n_loop) {
            ScheduleOnlineSoftmaxStage1(ctx, n_idx);
        }
        if (n_idx >= ctx.s_block_stack) {
            ScheduleTailSoftmaxStage2TP1(ctx, n_idx);
        }
    }
}

// ====== Tensor 逐行重复计算（Vector 业务基础函数）======

// Tensor 逐行减法（brcb 广播 Max + sub_v 循环 + tail 处理）
__aicore__ __attribute__((always_inline)) inline void TensorSubValueRepeatM(
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
        1,               // dstBlockStride
        8,               // dstRepeatStride
        round_sub_m / FLOAT_BLOCK_SIZE  // repeat
    );
    PIPE_BARRIER(V);
    for (uint32_t sub_v_idx = 0; sub_v_idx < qk_n / FLOAT_VECTOR_SIZE; ++sub_v_idx) {
        sub_v<ArchType::ASCEND_V220, float>(dst[sub_v_idx * FLOAT_VECTOR_SIZE],
            src[sub_v_idx * FLOAT_VECTOR_SIZE],
            tempMaxTensor,
            sub_m,                    // repeat
            1,                        // dstBlockStride
            1,                        // src0BlockStride
            0,                        // src1BlockStride
            qk_round_n / FLOAT_BLOCK_SIZE, // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE, // src0RepeatStride
            1                         // src1RepeatStride
        );
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        sub_v<ArchType::ASCEND_V220, float>(dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            tempMaxTensor,
            sub_m,                    // repeat
            1,                        // dstBlockStride
            1,                        // src0BlockStride
            0,                        // src1BlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
            1                         // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// Tensor 逐行除法（div_v 循环 + tail 处理）
__aicore__ __attribute__((always_inline)) inline void TensorDivRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& src1,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    PIPE_BARRIER(V);
    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
        div_v<ArchType::ASCEND_V220, float>(dst[vadd_idx * FLOAT_VECTOR_SIZE],
            src[vadd_idx * FLOAT_VECTOR_SIZE],
            src1,
            sub_m,                                  // repeat
            1,                                      // dstBlockStride
            1,                                      // src0BlockStride
            0,                                      // src1BlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // src0RepeatStride
            1                                       // src1RepeatStride
        );
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        div_v<ArchType::ASCEND_V220, float>(dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src1,
            sub_m,                                   // repeat
            1,                                      // dstBlockStride
            1,                                      // src0BlockStride
            0,                                      // src1BlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // src0RepeatStride
            1                                       // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// Tensor 行归约求最大值（cmax_v + max_v 循环 + tail 处理）
__aicore__ __attribute__((always_inline)) inline void ReduceMaxRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& tempTensor,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n)
{
    if (qk_n <= FLOAT_VECTOR_SIZE) {
        __set_mask(qk_n);
        cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(dst,
            src,
            sub_m,                    // repeat
            1,                        // dstRepeatStride
            1,                        // srcBlockStride
            qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
        );
    } else {
        ub_to_ub<ArchType::ASCEND_V220, float>(
            tempTensor,
            src,
            0,                                             // sid
            sub_m,                                         // nBurst
            HALF_VECTOR_SIZE / BLOCK_SIZE,                 // lenBurst
            (qk_round_n - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE,  // srcGap
            0                                              // dstGap
        );
        PIPE_BARRIER(V);
        for (uint32_t rowmax_idx = 1; rowmax_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
            max_v<ArchType::ASCEND_V220, float>(
                tempTensor,
                tempTensor,
                src[rowmax_idx * FLOAT_VECTOR_SIZE],
                sub_m,                         // repeat
                1,                             // dstBlockStride
                1,                             // src0BlockStride
                1,                             // src1BlockStride
                8,                             // dstRepeatStride
                8,                             // src0RepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE  // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            max_v<ArchType::ASCEND_V220, float>(
                tempTensor,
                tempTensor,
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sub_m,                         // repeat
                1,                             // dstBlockStride
                1,                             // src0BlockStride
                1,                             // src1BlockStride
                8,                             // dstRepeatStride
                8,                             // src0RepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE  // src1RepeatStride
            );
        }
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        cmax_v<ArchType::ASCEND_V220, float, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            dst,
            tempTensor,
            sub_m,      // repeat
            1,          // dstRepeatStride
            1,          // srcBlockStride
            8           // srcRepeatStride
        );
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);
}

// Tensor 行归约求和（cadd_v + add_v 循环 + tail 处理）
__aicore__ __attribute__((always_inline)) inline void ReduceSumRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    uint32_t sub_m,
    uint32_t qk_n,
    uint32_t qk_round_n)
{
    if (qk_n <= FLOAT_VECTOR_SIZE) {
        __set_mask(qk_n);
        cadd_v<ArchType::ASCEND_V220, float>(
            dst,
            src,
            sub_m,           // repeat
            1,               // dstRepeatStride
            1,               // srcBlockStride
            qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else {
        for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
            add_v<ArchType::ASCEND_V220, float>(
                src,
                src,
                src[rowsum_idx * FLOAT_VECTOR_SIZE],
                sub_m,           // repeat
                1,               // dstBlockStride
                1,               // src0BlockStride
                1,               // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE   // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            add_v<ArchType::ASCEND_V220, float>(
                src,
                src,
                src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sub_m,           // repeat
                1,               // dstBlockStride
                1,               // src0BlockStride
                1,               // src1BlockStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                qk_round_n / FLOAT_BLOCK_SIZE   // src1RepeatStride
            );
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        PIPE_BARRIER(V);

        cadd_v<ArchType::ASCEND_V220, float>(
            dst,
            src,
            sub_m,           // repeat
            1,               // dstRepeatStride
            1,               // srcBlockStride
            qk_round_n / FLOAT_BLOCK_SIZE   // srcRepeatStride
        );
    }
}

// Tensor 逐行乘法（mul_v 循环 + tail 处理）
__aicore__ __attribute__((always_inline)) inline void TensorMulRepeatM(
    const AscendC::LocalTensor<float>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& src1,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t src1BlockStride
) {
    PIPE_BARRIER(V);
    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
        mul_v<ArchType::ASCEND_V220, float>(dst[vadd_idx * FLOAT_VECTOR_SIZE],
            src[vadd_idx * FLOAT_VECTOR_SIZE],
            src1,
            sub_m,                                  // repeat
            1,                                      // dstBlockStride
            1,                                      // src0BlockStride
            src1BlockStride,                        // src1BlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // src0RepeatStride
            1                                       // src1RepeatStride
        );
    }
    if (qk_n % FLOAT_VECTOR_SIZE > 0) {
        __set_mask(qk_n % FLOAT_VECTOR_SIZE);
        mul_v<ArchType::ASCEND_V220, float>(dst[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src[qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
            src1,
            sub_m,                                   // repeat
            1,                                      // dstBlockStride
            1,                                      // src0BlockStride
            src1BlockStride,                        // src1BlockStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // dstRepeatStride
            qk_round_n / FLOAT_BLOCK_SIZE,          // src0RepeatStride
            1                                       // src1RepeatStride
        );
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}

// ====== 量化/反量化业务函数 ======

// ---- DeQuantPerHeadImpl 业务子函数 ----

// 业务子函数1：加载 deScale + online 乘 quantScale
// 平台调用：PlatformMte2ToVSyncEvent2 / TensorMulRepeatM
__aicore__ __attribute__((always_inline)) inline void LoadDeScaleAndOnlineMul(
    const AscendC::GlobalTensor<mmScaleType>& deScaleGm,
    AscendC::LocalTensor<mmScaleType> deScaleUb,
    AscendC::LocalTensor<float> quantScale,
    uint32_t sub_m, bool online)
{
    gm_to_ub_align<ArchType::ASCEND_V220, mmScaleType>(deScaleUb,
                                                    deScaleGm,
                                                    0,                                      // sid
                                                    1,                                      // nBurst
                                                    sub_m * sizeof(mmScaleType),             // lenBurst
                                                    0,                                      // leftPaddingNum
                                                    0,                                      // rightPaddingNum
                                                    0,                                      // srcGap
                                                    0                                       // dstGap
    );
    if (online) {
        PlatformMte2ToVSyncEvent2();
        TensorMulRepeatM(deScaleUb, deScaleUb, quantScale, 1, sub_m, RoundUp<16>(sub_m), 1);
    }
}

// 业务子函数2：加载 src(int32) + brcb 广播 deScale → tempScale
// 平台调用：PlatformMte2ToVSyncEvent0 / brcb_v / PlatformVPipeBarrier
__aicore__ __attribute__((always_inline)) inline void LoadSrcAndBrcbScale(
    const AscendC::GlobalTensor<int32_t>& src,
    AscendC::LocalTensor<int32_t> temp,
    AscendC::LocalTensor<mmScaleType> deScaleUb,
    AscendC::LocalTensor<mmScaleType> tempScale,
    uint32_t sub_m, uint32_t qk_round_n, bool move_tensor)
{
    if (move_tensor) {
        gm_to_ub<ArchType::ASCEND_V220, int32_t>(
            temp,
            src,
            0,                        // sid
            1,                        // nBurst
            CeilDiv<FLOAT_BLOCK_SIZE>(sub_m * qk_round_n),  // lenBurst
            0,                        // srcGap
            0                         // dstGap
        );
    }
    PlatformMte2ToVSyncEvent0();
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        tempScale.template ReinterpretCast<uint32_t>(),
        deScaleUb.template ReinterpretCast<uint32_t>(),
        1,               // dstBlockStrides
        8,               // dstRepeatStride
        RoundUp<16>(sub_m) / FLOAT_BLOCK_SIZE  // repeat
    );
    PlatformVPipeBarrier();
}

// 业务子函数3：INT32→FP32 转换(conv_v) + 乘 tempScale
// 平台调用：conv_v / TensorMulRepeatM / PlatformVPipeBarrier
__aicore__ __attribute__((always_inline)) inline void ConvInt32ToFP32AndMul(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<int32_t> temp,
    AscendC::LocalTensor<mmScaleType> tempScale,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    uint32_t count = sub_m * qk_round_n;
    uint32_t repeat_times = (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
    if (repeat_times < 255) {
        conv_v<ArchType::ASCEND_V220, int32_t, float>(
            dst, // dst
            temp, // src
            repeat_times,                  // repeat_times
            1,                            // dstBlockStride
            1,                            // srcBlockStride
            8,                            // dstRepeatStride
            8                             // srcRepeatStride
        );
    } else {
        for (uint64_t vconv_idx = 0; vconv_idx < 2; ++vconv_idx) {
            conv_v<ArchType::ASCEND_V220, int32_t, float>(
                dst[vconv_idx * count / 2], // dst
                temp[vconv_idx * count / 2], // src
                (count / 2 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,             // repeat_times
                1,                                                                   // dstBlockStride
                1,                                                                   // srcBlockStride
                8,                                                                   // dstRepeatStride
                8                                                                    // srcRepeatStride
            );
        }
    }
    TensorMulRepeatM(dst, dst, tempScale, sub_m, qk_n, qk_round_n, 0);
    PlatformVPipeBarrier();
}

// 逐 Head 反量化：INT32 → FP32 + deScale 乘法（编排层）
// 依赖：LoadDeScaleAndOnlineMul / LoadSrcAndBrcbScale / ConvInt32ToFP32AndMul
__aicore__ __attribute__((always_inline)) inline void DeQuantPerHeadImpl(
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
    bool move_tensor
){
    LoadDeScaleAndOnlineMul(deScaleGm, deScaleUb, quantScale, sub_m, online);
    LoadSrcAndBrcbScale(src, temp, deScaleUb, tempScale, sub_m, qk_round_n, move_tensor);
    ConvInt32ToFP32AndMul(dst, temp, tempScale, sub_m, qk_n, qk_round_n);
}

// 逐 Token 量化：FP32 → FP16 → INT8
// 依赖：TensorDivRepeatM / TensorMulRepeatM / conv_v / Cast
__aicore__ __attribute__((always_inline)) inline void QuantPerTokenImpl(
    const AscendC::LocalTensor<IN_DTYPE>& dst,
    const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& scale,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t pQuantOnline)
{
    if (pQuantOnline) {
        // scr / scale 
        TensorDivRepeatM(dst.template ReinterpretCast<float>(), src, scale, sub_m, qk_n, qk_round_n);
    } else {
        // scr * scale
        TensorMulRepeatM(dst.template ReinterpretCast<float>(), src, scale, sub_m, qk_n, qk_round_n, 0);
    }
    // src fp32 -> casttofp16 -> casttoint8
    uint32_t count = sub_m * qk_round_n;
    uint32_t repeat_times = (count + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
    if (repeat_times < 255) {
        conv_v<ArchType::ASCEND_V220, float, half>(
            dst.template ReinterpretCast<half>(), // dst
            dst.template ReinterpretCast<float>(), // src
            repeat_times,                  // repeat_times
            1,                            // dstBlockStride
            1,                            // srcBlockStride
            4,                            // dstRepeatStride
            8                             // srcRepeatStride
        );
    } else {
        for (uint64_t vconv_idx = 0; vconv_idx < 2; ++vconv_idx) {
            conv_v<ArchType::ASCEND_V220, float, half>(
                dst.template ReinterpretCast<half>()[vconv_idx * count / 2], // dst
                dst.template ReinterpretCast<float>()[vconv_idx * count / 2], // src
                (count / 2 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,             // repeat_times
                1,                                                                   // dstBlockStride
                1,                                                                   // srcBlockStride
                4,                                                                   // dstRepeatStride
                8                                                                    // srcRepeatStride
            );
        }
    }
    PIPE_BARRIER(V);
    for (uint32_t row_idx = 0; row_idx < qk_n / HALF_VECTOR_SIZE; ++row_idx) {
        AscendC::Cast<int8_t, half, false>(dst.template ReinterpretCast<int8_t>()[row_idx * HALF_VECTOR_SIZE],
                                           dst.template ReinterpretCast<half>()[row_idx * HALF_VECTOR_SIZE], AscendC::RoundMode::CAST_RINT,
                                           (uint64_t)0, sub_m, {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE), (uint8_t)(qk_round_n / BLOCK_SIZE)});
    }
    if (qk_n % HALF_VECTOR_SIZE > 0) {
        __set_mask(qk_n % HALF_VECTOR_SIZE);
        AscendC::Cast<int8_t, half, false>(dst.template ReinterpretCast<int8_t>()[qk_n / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE],
                                           dst.template ReinterpretCast<half>()[qk_n / HALF_VECTOR_SIZE * HALF_VECTOR_SIZE], AscendC::RoundMode::CAST_RINT,
                                           (uint64_t)0, sub_m, {1, 1, (uint8_t)((qk_round_n) / BLOCK_SIZE), (uint8_t)(qk_round_n / BLOCK_SIZE)});
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(V);
}