// Copyright 2025 The xLLM Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
//
// ==================== AIV 业务逻辑层（第二层, A5 / arch35）====================
// 本文件仅被 MLADecoderAiv 类 include，包含 Vector 侧业务调度函数。
// AIC 侧业务函数在 multi_latent_attention_bs.h 中。

// ==================== SoftmaxStage1 5 段计算体（A5 标准 AscendC 重写）====================
// 参考 A3(arch32) aiv_bs.h L18-373 的 5 段结构，但向量原语全部改用标准 AscendC API
// （Muls/Max/Sub/Exp/Cast/WholeReduceMax/ReduceSum/DataCopy），因 A5 未 include
// mixkernels/simd.h（A3 *_v 原语来源）。UB tensor 直接引用类成员（主文件已切分）。
// 本批仅实现 FP16(非INT8) 主路径；INT8(TILING_INT8_DATA) 路径先留 TODO。

// 业务子函数1：QK 数据加载 + mask 加载（FP16 非INT8 路径）
// 非INT8: DataCopy(s_gm→ls32) + mask(DataCopyPad/DataCopy 按 mask_type 3/4) + Cast(mask→mask32)
__aicore__ __attribute__((always_inline)) inline void LoadQKData(
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor,
    AscendC::GlobalTensor<float> s_rope_gm_tensor,
    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor,
    uint32_t n_idx, uint32_t qk_n, uint32_t qk_round_n, uint32_t sub_m,
    const uint32_t head_idx, uint32_t cur_q_seqlen, bool need_mask)
{
    WAIT_FLAG(V, MTE2, EVENT_ID2);
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        // TODO(INT8): 反量化 DeQuantPerHeadImpl + s_rope Add 融合，本批先不支持。
        (void)s_rope_gm_tensor; (void)head_idx;
    } else {
        // *** ls32 ← s_gm（QK 结果，FP16/BF16 搬入 UB，按 float 视图对齐）
        // A3: gm_to_ub<...,mm1CopyType>(ls32.ReinterpretCast<mm1CopyType>(), s_gm,
        //     lenBurst = sub_m*qk_round_n/FLOAT_BLOCK_SIZE)。A5 用 DataCopy 等价。
        AscendC::DataCopy(
            ls32_ubuf_tensor.template ReinterpretCast<mm1CopyType>(),
            s_gm_tensor,
            sub_m * qk_round_n);

        // *** mask 加载（mask_type==3 用 DataCopyPad；need_mask&&mask_type==4 用 DataCopy）
        if (mask_type == 3) {
            uint32_t aligned_mask_copy_len = RoundUp<BLOCK_SIZE_16>(qk_n);
            uint32_t mask_dst_stride = (qk_round_n - aligned_mask_copy_len) / BLOCK_SIZE_16;
            AscendC::DataCopyPad(
                mask_ubuf_tensor,
                mask_gm_tensor,
                AscendC::DataCopyExtParams(
                    cur_q_seqlen,
                    qk_n * 2,
                    maxKVSeqLen * 2 - qk_n * 2,
                    mask_dst_stride,
                    0),
                AscendC::DataCopyPadExtParams<OUT_DTYPE>(false, 0, 0, 0));
        } else if (need_mask && mask_type == 4) {
            AscendC::DataCopy(
                mask_ubuf_tensor,
             mask_gm_tensor,
                AscendC::DataCopyParams(
                    cur_q_seqlen,                                    // blockCount
                    qk_round_n * 2 / 32,                             // blockLen (2=sizeof(half))
                    MASK_COLUMNS * 2 / 32 - qk_round_n * 2 / 32,     // srcStride
                    0));                                             // dstStride
        }

        // *** MTE2→V 同步：等待 DataCopy 完成，V 侧可读
        PlatformMte2ToVSync();

        // *** mask FP16→FP32
        if (mask_type == 3 || (need_mask && mask_type == 4)) {
            AscendC::Cast(
                mask32_ubuf_tensor,
                mask_ubuf_tensor,
                AscendC::RoundMode::CAST_NONE,
                cur_q_seqlen * qk_round_n);
        }
        (void)s_rope_gm_tensor; (void)head_idx; (void)n_idx;
    }
}

// 业务子函数2：QK 缩放 + 非INT8 mask Add
// ls = ls × scale(tor)，非INT8 路径再叠加 mask
__aicore__ __attribute__((always_inline)) inline void ScaleAndMask(
    uint32_t qk_n, uint32_t qk_round_n, uint32_t sub_m,
    uint32_t cur_q_seqlen, bool need_mask)
{
    // *** ls = ls × tor（逐行缩放）
    // A3 用 muls_v 按 FLOAT_VECTOR_SIZE 分块 + 尾块 __set_mask；A5 用标准 Muls
    // 逐行处理：每行 qk_n 个元素，行间步长 qk_round_n。
    for (uint32_t m = 0; m < sub_m; ++m) {
        AscendC::Muls(
            ls32_ubuf_tensor[m * qk_round_n],
            ls32_ubuf_tensor[m * qk_round_n],
            tor,
            qk_n);
    }
    PIPE_BARRIER(V);

    // *** 非INT8：叠加 mask（每个 head 加同一份 mask）
    if constexpr (tilingKeyType != TilingKeyType::TILING_INT8_DATA) {
        if (mask_type == 3 || (need_mask && mask_type == 4)) {
            uint32_t cur_compute_head_num = sub_m / cur_q_seqlen;
            for (uint32_t i = 0; i < cur_compute_head_num; i++) {
                AscendC::Add(
                    ls32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                    ls32_ubuf_tensor[cur_q_seqlen * qk_round_n * i],
                    mask32_ubuf_tensor,
                    cur_q_seqlen * qk_round_n);
            }
            PIPE_BARRIER(V);
        }
    }
}

// 业务子函数3：Online Softmax 状态更新
// lm = rowmax(ls) → hm = max(lm, gm) → dm = gm - hm → gm = hm
// A3 用 ReduceMaxRepeatM/max_v/sub_v/ub_to_ub；A5 用逐行 WholeReduceMax + 标准 Max/Sub/DataCopy。
__aicore__ __attribute__((always_inline)) inline void UpdateSoftmaxState(
    uint32_t n_idx, uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

    // *** lm = rowmax(ls)：对每行 qk_n 个元素求最大值，存 lm32[m]
    for (uint32_t m = 0; m < sub_m; ++m) {
        AscendC::WholeReduceMax<float>(
            lm32_ubuf_tensor[m],
            ls32_ubuf_tensor[m * qk_round_n],
            qk_n,           // mask（本行有效元素数）
            1,              // repeatTimes
            1,              // dstRepStride
            1,              // srcBlkStride
            8,              // srcRepStride
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    }
    PIPE_BARRIER(V);

    if (n_idx != 0) {
        // *** hm = max(lm, gm)
        AscendC::Max(hm32_ubuf_tensor, lm32_ubuf_tensor, gm32_ubuf_tensor, sub_m);
        PIPE_BARRIER(V);
        // *** dm = gm - hm
        AscendC::Sub(dm32_ubuf_tensor, gm32_ubuf_tensor, hm32_ubuf_tensor, sub_m);
        PIPE_BARRIER(V);
    } else {
        // *** hm = lm
        AscendC::DataCopy(hm32_ubuf_tensor, lm32_ubuf_tensor, round_sub_m);
        PIPE_BARRIER(V);
    }
    // *** gm = hm
    AscendC::DataCopy(gm32_ubuf_tensor, hm32_ubuf_tensor, round_sub_m);
    PIPE_BARRIER(V);
}

// 业务子函数4：减法 + 指数
// ls = ls - hm_block（每行减该行 max）→ ls = exp(ls)
// A3 用 TensorSubValueRepeatM + exp_v；A5 逐行广播减法 + 标准 Exp。
__aicore__ __attribute__((always_inline)) inline void SubAndExp(
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    // *** ls[m,:] = ls[m,:] - hm[m]（逐行减去该行最大值，标量广播）
    for (uint32_t m = 0; m < sub_m; ++m) {
        float row_max = hm32_ubuf_tensor.GetValue(m);
        AscendC::Adds(
            ls32_ubuf_tensor[m * qk_round_n],
            ls32_ubuf_tensor[m * qk_round_n],
            -row_max,
            qk_n);
    }
    PIPE_BARRIER(V);
    // *** ls = exp(ls)
    AscendC::Exp(ls32_ubuf_tensor, ls32_ubuf_tensor, sub_m * qk_round_n);
    PIPE_BARRIER(V);
    (void)qk_n;
}

// 业务子函数5：量化/转换输出（FP16 非INT8 路径）
// 非INT8: Cast(lp←ls32 FP32→OUT_DTYPE) + DataCopy(p_gm←lp) + ReduceSum(ll=rowsum(ls))
__aicore__ __attribute__((always_inline)) inline void QuantizeAndOutput(
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor,
    uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
{
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        // TODO(INT8): sub/exp/muls/brcb + QuantPerTokenImpl（FP32→INT8），本批先不支持。
        (void)qk_n;
    } else {
        // *** lp ← ls32（FP32 → OUT_DTYPE，概率转半精度）
        AscendC::Cast(
            lp_ubuf_tensor.template ReinterpretCast<OUT_DTYPE>(),
            ls32_ubuf_tensor,
            AscendC::RoundMode::CAST_RINT,
            sub_m * qk_round_n);
        PIPE_BARRIER(V);
    }

    // *** V→MTE3 同步：Cast 完成后 MTE3 可搬出
    PlatformVToMte3Sync();

    // *** p_gm ← lp（概率写回 GM）
    AscendC::DataCopy(
        p_gm_tensor,
        lp_ubuf_tensor,
        sub_m * qk_round_n * T_BLOCK_OFFSET / T_BLOCK_SIZE);

    // *** ll = rowsum(ls)：对每行 qk_n 个元素求和，存 ll32[m]
    for (uint32_t m = 0; m < sub_m; ++m) {
        AscendC::WholeReduceSum<float>(
            ll_ubuf_tensor[m],
            ls32_ubuf_tensor[m * qk_round_n],
            qk_n,           // mask
            1,              // repeatTimes
            1,              // dstRepStride
            1,              // srcBlkStride
            8);             // srcRepStride
    }

    // *** V→MTE2 通知：本轮完成，MTE2 可搬入下轮
    PlatformVToMte2Notify();
    PIPE_BARRIER(V);
}

// 业务函数：非TP1 Phase 1 — SoftmaxStage1 计算本体（串联 5 段）
// 签名与 ScheduleSoftmaxStage1 的调用一致（4 GM tensor + 14 标量）。
// UB tensor 直接引用类成员，故不像 A3 那样传 dm32/ll/pm32 形参。
__aicore__ __attribute__((always_inline)) inline void SoftmaxStage1(
    AscendC::GlobalTensor<IN_DTYPE> p_gm_tensor_off,
    AscendC::GlobalTensor<mm1CopyType> s_gm_tensor_off,
    AscendC::GlobalTensor<float> s_rope_gm_tensor_off,
    AscendC::GlobalTensor<OUT_DTYPE> mask_gm_tensor_off,
    uint32_t n_idx, uint32_t qk_n, uint32_t qk_round_n, uint32_t sub_m,
    const uint32_t sub_n_loop, const uint32_t cur_batch, const uint32_t start_kv,
    const uint32_t real_n_loop, const uint32_t head_idx, const uint32_t pm_flag_scalar,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, bool need_mask)
{
    (void)sub_n_loop; (void)cur_batch; (void)start_kv;
    (void)real_n_loop; (void)pm_flag_scalar; (void)cur_kv_seqlen;

    // 段1：QK 数据加载（非INT8:FP16+mask+Cast；INT8 留 TODO）
    LoadQKData(s_gm_tensor_off, s_rope_gm_tensor_off, mask_gm_tensor_off,
               n_idx, qk_n, qk_round_n, sub_m, head_idx, cur_q_seqlen, need_mask);
    // 段2：QK 缩放 + 非INT8 mask Add
    ScaleAndMask(qk_n, qk_round_n, sub_m, cur_q_seqlen, need_mask);
    // 段3：Online Softmax 状态更新（rowmax + max/sub/copy）
    UpdateSoftmaxState(n_idx, sub_m, qk_n, qk_round_n);
    // 段4：减法 + 指数（ls = exp(ls - hm)）
    SubAndExp(sub_m, qk_n, qk_round_n);
    // 段5：转换输出 + p_gm 写回 + rowsum
    QuantizeAndOutput(p_gm_tensor_off, sub_m, qk_n, qk_round_n);
}

// 业务函数：非TP1 Phase 1 — Softmax Stage1 调度
// 包含 mask 边界计算、平台前/后置同步、SoftmaxStage1 ping-pong 调用。
// 与 A3(arch32) aiv_bs.h ScheduleSoftmaxStage1 调度体一致；GM 偏移为纯标量，
// 依赖 A5 已就绪的 block_idx/sub_block_idx/TMP_SIZE/TMP_SIZE_DECODER_A5/
// T_BLOCK_OFFSET/MASK_COLUMNS/mask_type/pm_flag_scalar1/2/RoundUp。
__aicore__ __attribute__((always_inline)) inline void ScheduleSoftmaxStage1(
    VectorContext &ctx, uint32_t n_idx, uint32_t &start_kv)
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
    uint32_t qk_round_n= ctx.qk_round_n;

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
        if (n_idx % 2 == 0) {
            SoftmaxStage1(
                p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET +
                    (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n * T_BLOCK_OFFSET +
                    (uint64_t)(n_idx % 2) * TMP_SIZE * T_BLOCK_OFFSET / 2],
                s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER_A5 +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                    (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER_A5 / 2],
                s_rope_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER_A5 +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                    (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER_A5 / 2],
                mask_gm_tensor[mask_start_offset],
                n_idx, qk_n, qk_round_n, sub_m, sub_n_loop, cur_batch, start_kv,
                real_n_loop, head_idx, pm_flag_scalar1, cur_q_seqlen, cur_kv_seqlen, need_mask);
        } else {
            SoftmaxStage1(
                p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET +
                    (uint64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n * T_BLOCK_OFFSET +
                    TMP_SIZE * T_BLOCK_OFFSET / 2],
                s_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER_A5 +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                    TMP_SIZE_DECODER_A5 / 2],
                s_rope_gm_tensor[(int64_t)block_idx * TMP_SIZE_DECODER_A5 +
                    (int64_t)sub_block_idx * cur_head_num * cur_q_seqlen / 2 * qk_round_n +
                    TMP_SIZE_DECODER_A5 / 2],
                mask_gm_tensor[mask_start_offset],
                n_idx, qk_n, qk_round_n, sub_m, sub_n_loop, cur_batch, start_kv,
                real_n_loop, head_idx, pm_flag_scalar2, cur_q_seqlen, cur_kv_seqlen, need_mask);
        }
    }

    // 平台同步：SoftmaxStage1 后置同步（FftsCrossCoreSync + SET_FLAG）
    PlatformSoftmaxStage1PostSync();
}

// 业务函数：softmax 阶段二（PV + rescale + O 写回）——本批空桩。
// 与 A3(arch32) aiv_bs.h ScheduleSoftmaxStage2(VectorContext&,n_idx) 签名一致。
// 本体依赖 SoftmaxStage2MLAHeadLoop 本体、round_v、o_offset、WaitFlagDev 等，
// 留待后续分批移植。
__aicore__ __attribute__((always_inline)) inline void ScheduleSoftmaxStage2(
    VectorContext &ctx, uint32_t n_idx)
{
    (void)ctx; (void)n_idx;
    // TODO(next-batch): 移植 arch32 ScheduleSoftmaxStage2 业务体。
}

// 业务函数：单个 Vector 任务的计算体（非 TP1 路径）。
// 与 A3(arch32) InnerRunVectorChange 极简调度版一致（主文件 L2802-2822）：
// 构造 VectorContext → InitVectorContext（纯标量）→ 取 o_offset →
// 按 n_loop+1 双缓冲驱动 ScheduleSoftmaxStage1/2。
__aicore__ __attribute__((always_inline)) inline void InnerRunVectorChange(
    uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
{
    (void)embed_split_size_v; (void)embed_split_loop_v;

    VectorContext ctx;
    InitVectorContext(ctx, cur_batch, start_head, cur_nIndx, cur_q_seqlen,
        cur_kv_seqlen, cur_head_num, offset_tiling);
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
// 包含 batch/head 循环调度，读取 tiling 参数后调用 InnerRunVectorChange。
// 与 A3(arch32) aiv_bs.h 任务划分一致：process = block_idx .. process_num，步长 block_num。
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
        InnerRunVectorChange(cur_batch, start_head, cur_nIndx, cur_q_seq_len,
            cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
    }
}