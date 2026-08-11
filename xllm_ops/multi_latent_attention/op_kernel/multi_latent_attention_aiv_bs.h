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

// ====== 非 TP1 路径：Phase 1 / Phase 2 业务调度 ======

// 业务函数：非TP1 Phase 1 — SoftmaxStage1 核心计算（Vector 业务实现）
// 包含 DeQuant/gm_to_ub、mask 处理、muls、ReduceMax、exp、QuantPerToken/conv、ub_to_gm、ReduceSum
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
    uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
    uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
    float quantMax = (float)1 / (float)127;
    WAIT_FLAG(V, MTE2, EVENT_ID2);
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
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
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        AscendC::Add(ls32_ubuf_tensor, ls32_ubuf_tensor, ls32_quant_ubuf_tensor, sub_m * qk_round_n); // float
        PIPE_BARRIER(V);
    } else {
        gm_to_ub<ArchType::ASCEND_V220, mm1CopyType>(
            ls32_ubuf_tensor.template ReinterpretCast<mm1CopyType>(),
            s_gm_tensor,
            0,                        // sid
            1,                        // nBurst
            sub_m * qk_round_n / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                        // srcGap
            0                         // dstGap
        );

        // TODO add mask type condition
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

        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        if (mask_type == 3 || (need_mask && mask_type == 4)) {
            AscendC::Cast(
                mask32_ubuf_tensor,
                mask_ubuf_tensor,
                AscendC::RoundMode::CAST_NONE,
                cur_q_seqlen * qk_round_n);
        }
    }

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
            PIPE_BARRIER(V);
        }
    }

    // *** lm = rowmax(ls)
    ReduceMaxRepeatM(lm32_ubuf_tensor, ls32_ubuf_tensor, lp32_ubuf_tensor, sub_m, qk_n, qk_round_n);
    // ReduceMaxChange(lm32_ubuf_tensor, ls32_ubuf_tensor, tv32_ubuf_tensor, round_sub_m, qk_n, qk_round_n);
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
        PIPE_BARRIER(V);
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
        PIPE_BARRIER(V);
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
        PIPE_BARRIER(V);
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
    // *** hm_block = expand_to_block(hm)

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
    // *** lp = castfp32to16(ls)
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
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
            pm32_ubuf_tensor,
            sub_m_d64,  // repeat
            1,                               // dstBlockStride
            1,                               // srcBlockStride
            8,                               // dstRepeatStride
            8                                // srcRepeatStride
        );
        PIPE_BARRIER(V);
        muls_v<ArchType::ASCEND_V220, float>(pm32_ubuf_tensor,
            pm32_ubuf_tensor,
            quantMax,
            sub_m_d64,              // repeat
            1,                      // dstBlockStride
            1,                      // srcBlockStride
            8,                      // dstRepeatStride
            8                        // srcRepeatStride
        );
        PIPE_BARRIER(V);
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
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
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
    SET_FLAG(V, MTE2, EVENT_ID2);
    PIPE_BARRIER(V);
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
        uint32_t start_core_idx = (cur_batch * q_block_num_per_batch) % block_num;
        uint32_t q_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));
        uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
        if (kv_seqlen == 0) {
            continue;
        }
        uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;
        uint32_t start_head = (process % q_block_num_per_batch) * cur_qn_blk_size;
        uint32_t start_kv = 0;
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