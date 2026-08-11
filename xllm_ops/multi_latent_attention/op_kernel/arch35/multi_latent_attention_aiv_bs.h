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

// 业务函数：单个 Vector 任务的计算体（非 TP1 路径）。
// 与 A3(arch32) InnerRunVectorChange 签名一致（主文件 L2931）。
// 本步骤仅完成方法定义与主流程接线，业务实现体待 step2 移植
// （arch32 主文件 L2931-3347：tiling 地址解算 / QK / softmax / PV / rescale）。
__aicore__ __attribute__((always_inline)) inline void InnerRunVectorChange(
    uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx,
    uint32_t cur_q_seqlen, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
    uint32_t offset_tiling, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
{
    // 消警：参数在 step2 业务体移植后使用
    (void)cur_batch; (void)start_head; (void)cur_nIndx;
    (void)cur_q_seqlen; (void)cur_kv_seqlen; (void)cur_head_num;
    (void)offset_tiling; (void)embed_split_size_v; (void)embed_split_loop_v;
    // TODO(step2): 移植 arch32 InnerRunVectorChange 业务实现体（L2931-3347）
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
        (void)start_core_idx; (void)kv_seqlen_align; (void)start_kv;
        InnerRunVectorChange(cur_batch, start_head, cur_nIndx, cur_q_seq_len,
            cur_kv_seqlen, cur_head_num, offset_tiling, 512, embed_split_loop_v_former);
    }
}