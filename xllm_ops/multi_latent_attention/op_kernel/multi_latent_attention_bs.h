// Copyright 2025 The xLLM Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//
// 文件：multi_latent_attention_bs.h
// 层级：第二层 - 业务逻辑函数（Load*/Init*/Compute*/Copy* 系列）
// 说明：本文件被 include 到 MLAttentionDecoderAic 类的 private 区域内，
//       包含所有纯流程控制函数，这些函数负责调用第三层 Platform* 系列函数
//       完成具体的业务编排，不直接调用平台 API。
//       依赖：第三层函数（Platform* 系列）来自 multi_latent_attention_arch32.h

// === 第二层：业务逻辑函数 ===

// 业务函数：将 Q 主体数据从 GM 搬运到 L1（ND→NZ 格式转换）
// 覆盖三种场景：
//   1. cur_q_seqlen == 1 → gm_to_l1 单矩阵搬运
//   2. cur_q_seqlen > 1 && q_heads < 128 → DataCopy 批量多矩阵搬运
//   3. cur_q_seqlen > 1 && q_heads >= 128 → for循环逐token搬运（规避 stride 位宽限制）
__aicore__ __attribute__((always_inline)) inline void LoadQMainFromGMToL1(
    AscendC::LocalTensor<IN_DTYPE> &l1_dst,
    AscendC::GlobalTensor<IN_DTYPE> &gm_src,
    uint64_t src_offset,
    uint32_t cur_q_seqlen,
    uint32_t cur_head_num)
{
    if (cur_q_seqlen == 1) {
        gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::NZ>(
            l1_dst,
            gm_src[src_offset],
            cur_head_num,                     // nValue
            RoundUp<16>(cur_head_num),        // dstNzC0Stride
            0,                                // dstNzMatrixStride, unused
            512,                              // dValue
            0,                                // dstNzMatrixStride, unused
            512                               // srcDValue
        );
    } else {
        if (q_heads < 128) {
            AscendC::DataCopy(
                l1_dst,
                gm_src[src_offset],
                AscendC::Nd2NzParams(
                    cur_q_seqlen,                            // ndNum
                    cur_head_num,                            // nValue
                    512,                                     // dValue
                    512 * q_heads,                           // srcNdMatrixStride
                    512,                                     // srcDValue
                    RoundUp<16>(cur_head_num * cur_q_seqlen), // dstNzC0Stride
                    cur_q_seqlen,                            // dstNzNStride
                    16                                       // dstNzMatrixStride
                )
            );
        } else {
            for (uint32_t ii = 0; ii < cur_q_seqlen; ii++) {
                AscendC::DataCopy(
                    l1_dst[ii * 16], // offset one datablock
                    gm_src[src_offset + ii * q_heads * 512],
                    AscendC::Nd2NzParams(
                        1,                                      // ndNum
                        cur_head_num,                           // nValue
                        512,                                    // dValue
                        0,                                      // srcNdMatrixStride
                        512,                                    // srcDValue
                        RoundUp<16>(cur_q_seqlen * cur_head_num), // dstNzC0Stride
                        cur_q_seqlen,                            // dstNzNStride
                        16                                       // dstNzMatrixStride
                    )
                );
            }
        }
    }
}

// ==================== TP1 QK 业务函数 ====================

// TP1 QK 参数结构体
struct TP1QKParams {
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t l1_kv_pingpong_flag;
    uint32_t embed_split_size;
    uint32_t round_embed_split_size;
    int64_t now_l1_offset;
    int64_t kv_offset;
    int64_t kv_offset_rope;
    uint32_t sv_round_n;
    uint32_t gm_split_idx;  // split_idx 用于 GM 偏移计算
    uint32_t pp_n_scalar;   // block_size，用于 GM 偏移计算
};

// 业务函数：初始化 TP1 QK 参数
// 对应原始 CUBE1 stage1 中 split_idx 循环体内的参数计算
__aicore__ __attribute__((always_inline)) inline void InitTP1QKParams(
    TP1Context &ctx, uint32_t split_idx, uint32_t n_idx, TP1QKParams &params)
{
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t now_idx = n_idx + split_idx;

    params.l1_kv_pingpong_flag = now_idx % 2;

    // 动态计算 qk_n / qk_round_n
    if (now_idx == (n_loop - 1)) {
        params.qk_n = (ctx.cur_kv_seqlen - now_idx * pp_n_scalar);
    } else {
        params.qk_n = pp_n_scalar;
    }
    params.qk_round_n = RoundUp<BLOCK_SIZE>(params.qk_n);

    // embed_split_size 默认 128，idx==4 时为 64（在编排层设置）
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);

    // now_l1_offset 默认 0，在 LoadTP1KVMain/KVRope 中更新
    params.now_l1_offset = 0;

    // block_table_id / kv_offset / kv_offset_rope 地址计算
    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                    ctx.cur_batch * max_num_blocks_per_query + ctx.start_kv / block_size + now_idx));
    params.kv_offset = (int64_t)block_table_id * block_size * stride_kv;
    params.kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;

    // sv_round_n 计算（用于 S→GM 的 dstStride）
    uint32_t sv_n_triu = n_loop * pp_n_scalar;
    uint32_t sv_n;
    if (n_idx + ctx.s_block_stack > n_loop - 1) {
        sv_n = ctx.cur_kv_seqlen - n_idx * pp_n_scalar;
    } else {
        sv_n = pp_n_scalar * ctx.s_block_stack;
    }
    params.sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    params.gm_split_idx = split_idx;
    params.pp_n_scalar = pp_n_scalar;
}

// 业务函数：TP1 Q 数据从 L1 加载到 L0A
// 复用非 TP1 的 PlatformWaitQLoadComplete / PlatformLoadQToL0A / PlatformSetQLoadComplete
// TP1 中 q_load_coeff = m，与非 TP1 一致
__aicore__ __attribute__((always_inline)) inline void LoadTP1QDataToL0A(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size, m, false);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：TP1 KV Main 数据从 GM 搬运到 L1（idx 0,2）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVMainFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 256;
    PlatformLoadTP1KVMainToL1(embed_split_idx, params.qk_n, params.qk_round_n,
                              params.l1_kv_pingpong_flag, params.kv_offset,
                              embed_split_idx * 128);
}

// 业务函数：TP1 KV Rope 数据从 GM 搬运到 L1（idx 4）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVRopeFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
    PlatformLoadTP1KVRopeToL1(params.qk_n, params.qk_round_n,
                               params.l1_kv_pingpong_flag, params.kv_offset_rope);
}

// 业务函数：TP1 KV 数据从 L1 加载到 L0B（所有 embed_split_idx）
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVDataToL0B(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformLoadTP1KVToL0B(embed_split_idx, params.round_embed_split_size,
                           params.qk_round_n, params.now_l1_offset,
                           params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeTP1QKMMad(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformComputeTP1QKMMad(embed_split_idx, params.embed_split_size,
                             m, params.qk_n, params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK 结果拷贝到 GM（idx 4 时）
// GM 偏移: block_idx * TMP_SIZE_DECODER * 4 + ((n_idx/s_block_stack)%2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar
__aicore__ __attribute__((always_inline)) inline void CopyTP1QKResultToGM(
    const TP1QKParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    uint64_t gm_dst_offset = (uint64_t)block_idx * TMP_SIZE_DECODER * 4 +
                             (uint64_t)((n_idx / 4) % 2) * TMP_SIZE_DECODER * 2 +
                             params.gm_split_idx * params.pp_n_scalar;
    PlatformCopyTP1QKResultToGM(m, params.qk_round_n,
                                 params.l1_kv_pingpong_flag,
                                 gm_dst_offset, params.sv_round_n);
}

// 业务函数：将 Q Rope 数据从 GM 搬运到 L1（ND→NZ 格式转换）
// INT8 场景：用 gm_to_l1 搬到独立的 l1q_rope_buf_addr_tensor
// 非INT8 场景：用 DataCopy 搬到 l1q_buf_addr_tensor 的 Q 主体之后
__aicore__ __attribute__((always_inline)) inline void LoadQRopeFromGMToL1(
    AscendC::LocalTensor<IN_DTYPE> &l1_q,
    AscendC::LocalTensor<IN_ROPE_DTYPE> &l1_q_rope,
    AscendC::GlobalTensor<IN_ROPE_DTYPE> &gm_src,
    uint64_t src_offset,
    uint32_t cur_q_seqlen,
    uint32_t cur_head_num)
{
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        gm_to_l1<ArchType::ASCEND_V220, IN_ROPE_DTYPE, DataFormat::ND, DataFormat::NZ>(
                l1_q_rope,
                gm_src[src_offset],
                cur_head_num,                     // nValue
                RoundUp<16>(cur_head_num),        // dstNzC0Stride
                0,                                // dstNzMatrixStride, unused
                64,                               // dValue
                0,                                // dstNzMatrixStride, unused
                64                                // srcDValue
            );
    } else {
        AscendC::DataCopy(
                l1_q[RoundUp<16>(cur_head_num * cur_q_seqlen) * 512],
                gm_src[src_offset],
                AscendC::Nd2NzParams(
                    cur_head_num,                              // ndNum
                    cur_q_seqlen,                              // nValue
                    64,                                        // dValue
                    64,                                        // srcNdMatrixStride
                    64 * q_heads,                              // srcDValue
                    RoundUp<16>(cur_head_num * cur_q_seqlen),  // dstNzC0Stride
                    1,                                         // dstNzNStride
                    16 * cur_q_seqlen                          // dstNzMatrixStride
                )
            );
    }
}

// ==================== TP1 QK 业务函数 ====================

// TP1 QK 参数结构体
struct TP1QKParams {
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t l1_kv_pingpong_flag;
    uint32_t embed_split_size;
    uint32_t round_embed_split_size;
    int64_t now_l1_offset;
    int64_t kv_offset;
    int64_t kv_offset_rope;
    uint32_t sv_round_n;
    uint32_t gm_split_idx;  // split_idx 用于 GM 偏移计算
    uint32_t pp_n_scalar;   // block_size，用于 GM 偏移计算
};

// 业务函数：初始化 TP1 QK 参数
// 对应原始 CUBE1 stage1 中 split_idx 循环体内的参数计算
__aicore__ __attribute__((always_inline)) inline void InitTP1QKParams(
    TP1Context &ctx, uint32_t split_idx, uint32_t n_idx, TP1QKParams &params)
{
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t now_idx = n_idx + split_idx;

    params.l1_kv_pingpong_flag = now_idx % 2;

    // 动态计算 qk_n / qk_round_n
    if (now_idx == (n_loop - 1)) {
        params.qk_n = (ctx.cur_kv_seqlen - now_idx * pp_n_scalar);
    } else {
        params.qk_n = pp_n_scalar;
    }
    params.qk_round_n = RoundUp<BLOCK_SIZE>(params.qk_n);

    // embed_split_size 默认 128，idx==4 时为 64（在编排层设置）
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);

    // now_l1_offset 默认 0，在 LoadTP1KVMain/KVRope 中更新
    params.now_l1_offset = 0;

    // block_table_id / kv_offset / kv_offset_rope 地址计算
    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                    ctx.cur_batch * max_num_blocks_per_query + ctx.start_kv / block_size + now_idx));
    params.kv_offset = (int64_t)block_table_id * block_size * stride_kv;
    params.kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;

    // sv_round_n 计算（用于 S→GM 的 dstStride）
    uint32_t sv_n_triu = n_loop * pp_n_scalar;
    uint32_t sv_n;
    if (n_idx + ctx.s_block_stack > n_loop - 1) {
        sv_n = ctx.cur_kv_seqlen - n_idx * pp_n_scalar;
    } else {
        sv_n = pp_n_scalar * ctx.s_block_stack;
    }
    params.sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    params.gm_split_idx = split_idx;
    params.pp_n_scalar = pp_n_scalar;
}

// 业务函数：TP1 Q 数据从 L1 加载到 L0A
// 复用非 TP1 的 PlatformWaitQLoadComplete / PlatformLoadQToL0A / PlatformSetQLoadComplete
// TP1 中 q_load_coeff = m，与非 TP1 一致
__aicore__ __attribute__((always_inline)) inline void LoadTP1QDataToL0A(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size, m, false);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：TP1 KV Main 数据从 GM 搬运到 L1（idx 0,2）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVMainFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 256;
    PlatformLoadTP1KVMainToL1(embed_split_idx, params.qk_n, params.qk_round_n,
                              params.l1_kv_pingpong_flag, params.kv_offset,
                              embed_split_idx * 128);
}

// 业务函数：TP1 KV Rope 数据从 GM 搬运到 L1（idx 4）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVRopeFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
    PlatformLoadTP1KVRopeToL1(params.qk_n, params.qk_round_n,
                               params.l1_kv_pingpong_flag, params.kv_offset_rope);
}

// 业务函数：TP1 KV 数据从 L1 加载到 L0B（所有 embed_split_idx）
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVDataToL0B(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformLoadTP1KVToL0B(embed_split_idx, params.round_embed_split_size,
                           params.qk_round_n, params.now_l1_offset,
                           params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeTP1QKMMad(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformComputeTP1QKMMad(embed_split_idx, params.embed_split_size,
                             m, params.qk_n, params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK 结果拷贝到 GM（idx 4 时）
// GM 偏移: block_idx * TMP_SIZE_DECODER * 4 + ((n_idx/s_block_stack)%2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar
__aicore__ __attribute__((always_inline)) inline void CopyTP1QKResultToGM(
    const TP1QKParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    uint64_t gm_dst_offset = (uint64_t)block_idx * TMP_SIZE_DECODER * 4 +
                             (uint64_t)((n_idx / 4) % 2) * TMP_SIZE_DECODER * 2 +
                             params.gm_split_idx * params.pp_n_scalar;
    PlatformCopyTP1QKResultToGM(m, params.qk_round_n,
                                 params.l1_kv_pingpong_flag,
                                 gm_dst_offset, params.sv_round_n);
}

// 业务函数：将 KV 主体数据从 GM 搬运到 L1
// ND_FORMAT 场景：ND→NZ 格式转换（gm_to_l1<ND, NZ>）
// INT8 / 非INT8 NZ 场景：NZ→NZ 格式搬运（gm_to_l1<NZ, NZ>）
template <typename DST_DTYPE, typename SRC_DTYPE = DST_DTYPE>
__aicore__ __attribute__((always_inline)) inline void LoadKVMainFromGMToL1(
    AscendC::LocalTensor<DST_DTYPE> l1_dst,
    AscendC::GlobalTensor<SRC_DTYPE> gm_src,
    uint32_t n_value,           // 实际行数
    uint32_t dst_nz_c0_stride,  // L1 对齐行数（dstNzC0Stride）
    uint32_t d_value,           // 列数（512）
    uint32_t src_d_value,       // GM 行 stride（ND场景为stride_kv，NZ场景为0）
    bool is_nd_to_nz)           // true=ND→NZ, false=NZ→NZ
{
    if (is_nd_to_nz) {
        gm_to_l1<ArchType::ASCEND_V220, DST_DTYPE, DataFormat::ND, DataFormat::NZ>(
            l1_dst,
            gm_src,
            n_value,                // nValue
            dst_nz_c0_stride,       // dstNzC0Stride
            0,                      // dstNzMatrixStride, unused
            d_value,                // dValue
            0,                      // dstNzMatrixStride, unused
            src_d_value             // srcDValue
        );
    } else {
        gm_to_l1<ArchType::ASCEND_V220, DST_DTYPE, DataFormat::NZ, DataFormat::NZ>(
            l1_dst,
            gm_src,
            n_value,                // nValue
            dst_nz_c0_stride,       // dstNzC0Stride
            0,                      // dstNzMatrixStride, unused
            d_value,                // dValue
            0,                      // dstNzMatrixStride, unused
            0                       // srcDValue (NZ→NZ always 0)
        );
    }
}

// ==================== TP1 QK 业务函数 ====================

// TP1 QK 参数结构体
struct TP1QKParams {
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t l1_kv_pingpong_flag;
    uint32_t embed_split_size;
    uint32_t round_embed_split_size;
    int64_t now_l1_offset;
    int64_t kv_offset;
    int64_t kv_offset_rope;
    uint32_t sv_round_n;
    uint32_t gm_split_idx;  // split_idx 用于 GM 偏移计算
    uint32_t pp_n_scalar;   // block_size，用于 GM 偏移计算
};

// 业务函数：初始化 TP1 QK 参数
// 对应原始 CUBE1 stage1 中 split_idx 循环体内的参数计算
__aicore__ __attribute__((always_inline)) inline void InitTP1QKParams(
    TP1Context &ctx, uint32_t split_idx, uint32_t n_idx, TP1QKParams &params)
{
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t now_idx = n_idx + split_idx;

    params.l1_kv_pingpong_flag = now_idx % 2;

    // 动态计算 qk_n / qk_round_n
    if (now_idx == (n_loop - 1)) {
        params.qk_n = (ctx.cur_kv_seqlen - now_idx * pp_n_scalar);
    } else {
        params.qk_n = pp_n_scalar;
    }
    params.qk_round_n = RoundUp<BLOCK_SIZE>(params.qk_n);

    // embed_split_size 默认 128，idx==4 时为 64（在编排层设置）
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);

    // now_l1_offset 默认 0，在 LoadTP1KVMain/KVRope 中更新
    params.now_l1_offset = 0;

    // block_table_id / kv_offset / kv_offset_rope 地址计算
    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                    ctx.cur_batch * max_num_blocks_per_query + ctx.start_kv / block_size + now_idx));
    params.kv_offset = (int64_t)block_table_id * block_size * stride_kv;
    params.kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;

    // sv_round_n 计算（用于 S→GM 的 dstStride）
    uint32_t sv_n_triu = n_loop * pp_n_scalar;
    uint32_t sv_n;
    if (n_idx + ctx.s_block_stack > n_loop - 1) {
        sv_n = ctx.cur_kv_seqlen - n_idx * pp_n_scalar;
    } else {
        sv_n = pp_n_scalar * ctx.s_block_stack;
    }
    params.sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    params.gm_split_idx = split_idx;
    params.pp_n_scalar = pp_n_scalar;
}

// 业务函数：TP1 Q 数据从 L1 加载到 L0A
// 复用非 TP1 的 PlatformWaitQLoadComplete / PlatformLoadQToL0A / PlatformSetQLoadComplete
// TP1 中 q_load_coeff = m，与非 TP1 一致
__aicore__ __attribute__((always_inline)) inline void LoadTP1QDataToL0A(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size, m, false);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：TP1 KV Main 数据从 GM 搬运到 L1（idx 0,2）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVMainFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 256;
    PlatformLoadTP1KVMainToL1(embed_split_idx, params.qk_n, params.qk_round_n,
                              params.l1_kv_pingpong_flag, params.kv_offset,
                              embed_split_idx * 128);
}

// 业务函数：TP1 KV Rope 数据从 GM 搬运到 L1（idx 4）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVRopeFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
    PlatformLoadTP1KVRopeToL1(params.qk_n, params.qk_round_n,
                               params.l1_kv_pingpong_flag, params.kv_offset_rope);
}

// 业务函数：TP1 KV 数据从 L1 加载到 L0B（所有 embed_split_idx）
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVDataToL0B(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformLoadTP1KVToL0B(embed_split_idx, params.round_embed_split_size,
                           params.qk_round_n, params.now_l1_offset,
                           params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeTP1QKMMad(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformComputeTP1QKMMad(embed_split_idx, params.embed_split_size,
                             m, params.qk_n, params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK 结果拷贝到 GM（idx 4 时）
// GM 偏移: block_idx * TMP_SIZE_DECODER * 4 + ((n_idx/s_block_stack)%2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar
__aicore__ __attribute__((always_inline)) inline void CopyTP1QKResultToGM(
    const TP1QKParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    uint64_t gm_dst_offset = (uint64_t)block_idx * TMP_SIZE_DECODER * 4 +
                             (uint64_t)((n_idx / 4) % 2) * TMP_SIZE_DECODER * 2 +
                             params.gm_split_idx * params.pp_n_scalar;
    PlatformCopyTP1QKResultToGM(m, params.qk_round_n,
                                 params.l1_kv_pingpong_flag,
                                 gm_dst_offset, params.sv_round_n);
}

// 业务函数：将 KV Rope 数据从 GM 搬运到 L1
// ND_FORMAT 场景：ND→NZ 格式转换（gm_to_l1<ND, NZ>），d_value=64, src_d_value=stride_kv_rope
// INT8 / 非INT8 NZ 场景：NZ→NZ 格式搬运（gm_to_l1<NZ, NZ>），d_value=64, src_d_value=0
template <typename DST_DTYPE, typename SRC_DTYPE = DST_DTYPE>
__aicore__ __attribute__((always_inline)) inline void LoadKVRopeFromGMToL1(
    AscendC::LocalTensor<DST_DTYPE> l1_dst,
    AscendC::GlobalTensor<SRC_DTYPE> gm_src,
    uint32_t n_value,           // 实际行数
    uint32_t dst_nz_c0_stride,  // L1 对齐行数（dstNzC0Stride）
    uint32_t d_value,           // 列数（64）
    uint32_t src_d_value,       // GM 行 stride（ND场景为stride_kv_rope，NZ场景为0）
    bool is_nd_to_nz)           // true=ND→NZ, false=NZ→NZ
{
    // 逻辑与 LoadKVMainFromGMToL1 完全一致，仅参数不同（d_value=64）
    LoadKVMainFromGMToL1<DST_DTYPE, SRC_DTYPE>(l1_dst, gm_src, n_value, dst_nz_c0_stride, d_value, src_d_value, is_nd_to_nz);
}

// 业务函数：初始化 QK 参数
__aicore__ __attribute__((always_inline)) inline void InitQKParams(
    MLAContext &ctx, uint32_t l1_kv_pingpong_flag, QKParams &params)
{
    params.qk_n = ctx.qk_n;
    params.qk_round_n = ctx.qk_round_n;
    params.qk_round_n_l1 = ctx.qk_round_n_l1;
    params.hidden_size = ctx.hidden_size;
    params.k_round_n = ctx.k_round_n;
    params.row_num = ctx.row_num;
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);
    params.q_load_coeff = m;
    params.hidden_split_time = (params.hidden_size + 128 - 1) / 128;
    params.l1_kv_pingpong_flag = l1_kv_pingpong_flag;
}

// 业务函数：Q 数据从 L1 加载到 L0A
__aicore__ __attribute__((always_inline)) inline void LoadQDataToL0A(
    const QKParams &params, uint32_t embed_split_idx, bool is_rope)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size,
                       params.q_load_coeff, is_rope);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：KV 数据从 L1 加载到 L0B
__aicore__ __attribute__((always_inline)) inline void LoadKVDataToL0B(
    const QKParams &params, uint32_t embed_split_idx, bool is_rope)
{
    PlatformLoadKVToL0B(embed_split_idx, params.round_embed_split_size,
                        params.k_round_n, params.qk_round_n,
                        params.l1_kv_pingpong_flag, params.hidden_size, is_rope);
}

// 业务函数：QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeQKMMad(
    const QKParams &params, uint32_t embed_split_idx, bool is_rope)
{
    PlatformComputeQKMMad(embed_split_idx, params.embed_split_size,
                          m, params.qk_n, params.qk_round_n_l1,
                          params.l1_kv_pingpong_flag, is_rope);
}

// 业务函数：QK 结果拷贝到 GM
__aicore__ __attribute__((always_inline)) inline void CopyQKResultToGM(
    const QKParams &params, uint32_t embed_split_idx,
    uint32_t n_idx, bool is_rope)
{
    PlatformCopyQKResultToGM(embed_split_idx, m, params.qk_n,
                             params.qk_round_n, params.l1_kv_pingpong_flag,
                             n_idx, is_rope);
}

// 业务函数：初始化 PV 参数
__aicore__ __attribute__((always_inline)) inline void InitPVParams(
    MLAContext &ctx, uint32_t n_idx, PVParams &params)
{
    params.qk_n_2 = ctx.qk_n_2;
    params.qk_round_n_2 = ctx.qk_round_n_2;
    params.qk_round_n_2_l1 = ctx.qk_round_n_2_l1;
    params.k_round_n = ctx.k_round_n;
    params.row_num = ctx.row_num;
    params.hidden_size = ctx.hidden_size;

    if (n_idx == ctx.n_loop) {
        params.qk_n_2 = (ctx.cur_kv_seqlen - (n_idx - 1) * ctx.pp_n_scalar);
        params.qk_round_n_2 = RoundUp<BLOCK_SIZE>(params.qk_n_2);
        params.qk_round_n_2_l1 = RoundUp<T_BLOCK_SIZE>(params.qk_n_2);
    }
    params.k_round_n = params.qk_round_n_2_l1;
    params.l1_kv_pingpong_flag = (n_idx - 1) % 2;
    params.l0_p_pingpong_flag = (n_idx - 1) % 2;
    params.embed_split_size = 128;
    embed_split_loop_v = 4;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);
}

// 业务函数：KV L1→L0B 转置加载
__aicore__ __attribute__((always_inline)) inline void LoadKVTransposeToL0B(
    const PVParams &params, uint32_t embed_split_idx)
{
    bool is_last_split = (embed_split_idx == embed_split_loop_v - 1);
    if (params.k_round_n <= params.round_embed_split_size) {
        PlatformLoadKVTransposeSmallN(
            params.l0b_pingpong_flag, params.l1_kv_pingpong_flag,
            params.l1kv_offset, params.k_round_n, params.hidden_size,
            params.embed_split_size, params.round_embed_split_size);
    } else {
        PlatformLoadKVTransposeLargeN(
            params.l0b_pingpong_flag, params.l1_kv_pingpong_flag,
            params.l1kv_offset, params.hidden_size,
            params.qk_round_n_2, params.round_embed_split_size);
    }
    PlatformSetKVLoadComplete(is_last_split, params.l1_kv_pingpong_flag);
}

// 业务函数：P 数据加载（GM→L1→L0A）
__aicore__ __attribute__((always_inline)) inline void LoadPDataToL0A(
    const PVParams &params, uint32_t n_idx)
{
    PlatformLoadPFromGMToL1(
        n_idx, params.row_num, params.k_round_n, params.qk_round_n_2);
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        PlatformLoadPToL0AInt8(
            params.l0_p_pingpong_flag, params.row_num,
            params.qk_round_n_2_l1);
    } else {
        PlatformLoadPToL0AGeneral(
            params.l0_p_pingpong_flag, params.row_num,
            params.qk_round_n_2);
    }
    PlatformSetPLoadComplete();
}

// 业务函数：PV MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputePVMmad(
    const PVParams &params, uint32_t embed_split_idx)
{
    bool is_last_split = (embed_split_idx == embed_split_loop_v - 1);
    PlatformComputePVMmad(
        params.l0b_pingpong_flag, params.l0c_pingpong_flag,
        params.l0_p_pingpong_flag, params.embed_split_size,
        params.qk_n_2, is_last_split);
}

// 业务函数：PV 结果拷贝到 GM
__aicore__ __attribute__((always_inline)) inline void CopyPVResultToGM(
    const PVParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    PlatformCopyPVResultToGM(
        params.l0c_pingpong_flag, embed_split_idx,
        n_idx, params.round_embed_split_size);
}

// ==================== Rope 业务函数（INT8 专有）====================

// 业务函数：Rope 计算（Q_Rope × KV_Rope）
__aicore__ __attribute__((always_inline)) inline void ComputeQRope(
    const QKParams &params, uint32_t n_idx)
{
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        // Rope 部分固定使用 embed_split_idx=4, embed_split_size=64
        uint32_t rope_embed_split_idx = 4;
        uint32_t rope_embed_split_size = 64;
        uint32_t rope_round_embed_split_size = 64;

        // 1. 加载 Q_Rope 到 L0A
        PlatformLoadQRopeToL0A(
            rope_embed_split_idx, params.q_load_coeff,
            rope_round_embed_split_size);

        // 2. 加载 KV_Rope 到 L0B
        PlatformLoadKVRopeToL0B(
            rope_embed_split_idx, params.l1_kv_pingpong_flag,
            rope_round_embed_split_size, params.qk_round_n);

        // 3. mmad 矩阵乘
        PlatformComputeQRopeMMad(
            rope_embed_split_idx, params.l1_kv_pingpong_flag,
            m, params.qk_n, rope_embed_split_size);

        // 4. 结果搬回 GM
        PlatformCopyQRopeResultToGM(
            params.l1_kv_pingpong_flag, n_idx,
            m, params.qk_round_n);
    }
}

// ==================== TP1 QK 业务函数 ====================

// TP1 QK 参数结构体
struct TP1QKParams {
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t l1_kv_pingpong_flag;
    uint32_t embed_split_size;
    uint32_t round_embed_split_size;
    int64_t now_l1_offset;
    int64_t kv_offset;
    int64_t kv_offset_rope;
    uint32_t sv_round_n;
    uint32_t gm_split_idx;  // split_idx 用于 GM 偏移计算
    uint32_t pp_n_scalar;   // block_size，用于 GM 偏移计算
};

// 业务函数：初始化 TP1 QK 参数
// 对应原始 CUBE1 stage1 中 split_idx 循环体内的参数计算
__aicore__ __attribute__((always_inline)) inline void InitTP1QKParams(
    TP1Context &ctx, uint32_t split_idx, uint32_t n_idx, TP1QKParams &params)
{
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t now_idx = n_idx + split_idx;

    params.l1_kv_pingpong_flag = now_idx % 2;

    // 动态计算 qk_n / qk_round_n
    if (now_idx == (n_loop - 1)) {
        params.qk_n = (ctx.cur_kv_seqlen - now_idx * pp_n_scalar);
    } else {
        params.qk_n = pp_n_scalar;
    }
    params.qk_round_n = RoundUp<BLOCK_SIZE>(params.qk_n);

    // embed_split_size 默认 128，idx==4 时为 64（在编排层设置）
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);

    // now_l1_offset 默认 0，在 LoadTP1KVMain/KVRope 中更新
    params.now_l1_offset = 0;

    // block_table_id / kv_offset / kv_offset_rope 地址计算
    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                    ctx.cur_batch * max_num_blocks_per_query + ctx.start_kv / block_size + now_idx));
    params.kv_offset = (int64_t)block_table_id * block_size * stride_kv;
    params.kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;

    // sv_round_n 计算（用于 S→GM 的 dstStride）
    uint32_t sv_n_triu = n_loop * pp_n_scalar;
    uint32_t sv_n;
    if (n_idx + ctx.s_block_stack > n_loop - 1) {
        sv_n = ctx.cur_kv_seqlen - n_idx * pp_n_scalar;
    } else {
        sv_n = pp_n_scalar * ctx.s_block_stack;
    }
    params.sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    params.gm_split_idx = split_idx;
    params.pp_n_scalar = pp_n_scalar;
}

// 业务函数：TP1 Q 数据从 L1 加载到 L0A
// 复用非 TP1 的 PlatformWaitQLoadComplete / PlatformLoadQToL0A / PlatformSetQLoadComplete
// TP1 中 q_load_coeff = m，与非 TP1 一致
__aicore__ __attribute__((always_inline)) inline void LoadTP1QDataToL0A(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size, m, false);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：TP1 KV Main 数据从 GM 搬运到 L1（idx 0,2）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVMainFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 256;
    PlatformLoadTP1KVMainToL1(embed_split_idx, params.qk_n, params.qk_round_n,
                              params.l1_kv_pingpong_flag, params.kv_offset,
                              embed_split_idx * 128);
}

// 业务函数：TP1 KV Rope 数据从 GM 搬运到 L1（idx 4）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVRopeFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
    PlatformLoadTP1KVRopeToL1(params.qk_n, params.qk_round_n,
                               params.l1_kv_pingpong_flag, params.kv_offset_rope);
}

// 业务函数：TP1 KV 数据从 L1 加载到 L0B（所有 embed_split_idx）
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVDataToL0B(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformLoadTP1KVToL0B(embed_split_idx, params.round_embed_split_size,
                           params.qk_round_n, params.now_l1_offset,
                           params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeTP1QKMMad(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformComputeTP1QKMMad(embed_split_idx, params.embed_split_size,
                             m, params.qk_n, params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK 结果拷贝到 GM（idx 4 时）
// GM 偏移: block_idx * TMP_SIZE_DECODER * 4 + ((n_idx/s_block_stack)%2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar
__aicore__ __attribute__((always_inline)) inline void CopyTP1QKResultToGM(
    const TP1QKParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    uint64_t gm_dst_offset = (uint64_t)block_idx * TMP_SIZE_DECODER * 4 +
                             (uint64_t)((n_idx / 4) % 2) * TMP_SIZE_DECODER * 2 +
                             params.gm_split_idx * params.pp_n_scalar;
    PlatformCopyTP1QKResultToGM(m, params.qk_round_n,
                                 params.l1_kv_pingpong_flag,
                                 gm_dst_offset, params.sv_round_n);
}

// 业务函数：调度 Cube MLA 任务（Run 方法的 for 循环部分）
// 遍历所有 batch，解析 tiling 参数，调用 InnerRunCubeMLA
__aicore__ __attribute__((always_inline)) inline void ScheduleCubeTasks()
{
    uint32_t q_block_num_per_batch = (q_heads + cur_qn_blk_size - 1) / cur_qn_blk_size;
    uint32_t process_num = q_block_num_per_batch * num_batches;

    for (uint32_t process = block_idx; process < process_num; process += (uint32_t)block_num) {
        uint32_t cur_batch = process / q_block_num_per_batch;
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

        InnerRunCubeMLA(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen,
                        offset_tiling);
    }
}

// ==================== TP1 QK 业务函数 ====================

// TP1 QK 参数结构体
struct TP1QKParams {
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t l1_kv_pingpong_flag;
    uint32_t embed_split_size;
    uint32_t round_embed_split_size;
    int64_t now_l1_offset;
    int64_t kv_offset;
    int64_t kv_offset_rope;
    uint32_t sv_round_n;
    uint32_t gm_split_idx;  // split_idx 用于 GM 偏移计算
    uint32_t pp_n_scalar;   // block_size，用于 GM 偏移计算
};

// 业务函数：初始化 TP1 QK 参数
// 对应原始 CUBE1 stage1 中 split_idx 循环体内的参数计算
__aicore__ __attribute__((always_inline)) inline void InitTP1QKParams(
    TP1Context &ctx, uint32_t split_idx, uint32_t n_idx, TP1QKParams &params)
{
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t now_idx = n_idx + split_idx;

    params.l1_kv_pingpong_flag = now_idx % 2;

    // 动态计算 qk_n / qk_round_n
    if (now_idx == (n_loop - 1)) {
        params.qk_n = (ctx.cur_kv_seqlen - now_idx * pp_n_scalar);
    } else {
        params.qk_n = pp_n_scalar;
    }
    params.qk_round_n = RoundUp<BLOCK_SIZE>(params.qk_n);

    // embed_split_size 默认 128，idx==4 时为 64（在编排层设置）
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);

    // now_l1_offset 默认 0，在 LoadTP1KVMain/KVRope 中更新
    params.now_l1_offset = 0;

    // block_table_id / kv_offset / kv_offset_rope 地址计算
    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                    ctx.cur_batch * max_num_blocks_per_query + ctx.start_kv / block_size + now_idx));
    params.kv_offset = (int64_t)block_table_id * block_size * stride_kv;
    params.kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;

    // sv_round_n 计算（用于 S→GM 的 dstStride）
    uint32_t sv_n_triu = n_loop * pp_n_scalar;
    uint32_t sv_n;
    if (n_idx + ctx.s_block_stack > n_loop - 1) {
        sv_n = ctx.cur_kv_seqlen - n_idx * pp_n_scalar;
    } else {
        sv_n = pp_n_scalar * ctx.s_block_stack;
    }
    params.sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    params.gm_split_idx = split_idx;
    params.pp_n_scalar = pp_n_scalar;
}

// 业务函数：TP1 Q 数据从 L1 加载到 L0A
// 复用非 TP1 的 PlatformWaitQLoadComplete / PlatformLoadQToL0A / PlatformSetQLoadComplete
// TP1 中 q_load_coeff = m，与非 TP1 一致
__aicore__ __attribute__((always_inline)) inline void LoadTP1QDataToL0A(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size, m, false);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：TP1 KV Main 数据从 GM 搬运到 L1（idx 0,2）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVMainFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 256;
    PlatformLoadTP1KVMainToL1(embed_split_idx, params.qk_n, params.qk_round_n,
                              params.l1_kv_pingpong_flag, params.kv_offset,
                              embed_split_idx * 128);
}

// 业务函数：TP1 KV Rope 数据从 GM 搬运到 L1（idx 4）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVRopeFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
    PlatformLoadTP1KVRopeToL1(params.qk_n, params.qk_round_n,
                               params.l1_kv_pingpong_flag, params.kv_offset_rope);
}

// 业务函数：TP1 KV 数据从 L1 加载到 L0B（所有 embed_split_idx）
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVDataToL0B(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformLoadTP1KVToL0B(embed_split_idx, params.round_embed_split_size,
                           params.qk_round_n, params.now_l1_offset,
                           params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeTP1QKMMad(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformComputeTP1QKMMad(embed_split_idx, params.embed_split_size,
                             m, params.qk_n, params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK 结果拷贝到 GM（idx 4 时）
// GM 偏移: block_idx * TMP_SIZE_DECODER * 4 + ((n_idx/s_block_stack)%2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar
__aicore__ __attribute__((always_inline)) inline void CopyTP1QKResultToGM(
    const TP1QKParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    uint64_t gm_dst_offset = (uint64_t)block_idx * TMP_SIZE_DECODER * 4 +
                             (uint64_t)((n_idx / 4) % 2) * TMP_SIZE_DECODER * 2 +
                             params.gm_split_idx * params.pp_n_scalar;
    PlatformCopyTP1QKResultToGM(m, params.qk_round_n,
                                 params.l1_kv_pingpong_flag,
                                 gm_dst_offset, params.sv_round_n);
}

// 业务函数：调度 Cube MLA TP1 任务（RunTP1 方法的中段业务逻辑）
// 包含主循环调度 + tail 优化三分支（cores_per_seq 动态调整）
__aicore__ __attribute__((always_inline)) inline void ScheduleCubeTasksTP1()
{
    uint32_t tail = totalTaskNum % block_num;
    if constexpr (EnableOptimization) {

    } else{
        tail = 0; // control whether to run tail optimization
    }
    uint32_t totalTaskNumRound = totalTaskNum - tail;


    for (uint32_t process = block_idx; process < totalTaskNumRound; process += (uint32_t)block_num) {  // for task
        uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
        uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

        uint32_t q_seqlen = 1;
        uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
        if (kv_seqlen == 0) {
            continue;
        }
        uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;

        uint32_t start_head = 0;
        uint32_t start_kv = 0;
        uint32_t cur_q_seq_len = q_seqlen;
        uint32_t cur_kv_seqlen = kv_seqlen;
        uint32_t cur_head_num = q_heads;

        InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
    }

    // suppose all seqs have same length
    if (tail > 0){
        uint32_t sample_kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + tiling_head_size + 2));
        bool enableExtraOptimization = true;
        if (block_num % 4 == 3) {
            // cannot optimize this situation due to math problem
            enableExtraOptimization = false;
        }
        if (sample_kv_seqlen <= 2048){
            // Too Short to benefit from optimization
            enableExtraOptimization = false;
        }
        if (!enableExtraOptimization || tail <= block_num / 2) {
            // collect all metadata
            uint32_t cores_per_seq = 1;
            if (0 < tail && tail <= block_num / 4) {// 6 tasks left, each works with 4 cores
                cores_per_seq = 4;
                if (tail == 1){
                    cores_per_seq = block_num;
                }
                else if (tail == 2){
                    cores_per_seq = block_num / 2;
                }
                else if(tail == 3){
                    cores_per_seq = block_num / 3;
                }
                else if(tail == 4){
                    cores_per_seq = block_num / 4;
                }
            }
            else if(block_num / 4 < tail && tail <= block_num / 3) { // 8 tasks left, each works with 3 cores
                cores_per_seq = 3;

            }
            else if(block_num / 3 < tail && tail <= block_num / 2) { // 12 tasks left, each works with 2 cores
                cores_per_seq = 2;
            }
            else {
                // no extra optimization for tail > 12
                cores_per_seq = 1;
            }

            if(!enableExtraOptimization){
                cores_per_seq = 1;
            }

            uint32_t process = totalTaskNumRound + block_idx / cores_per_seq;
            if (process < totalTaskNum) {
                uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

                uint32_t q_seqlen = 1;
                uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                uint32_t actual_work_cores = kv_seqlen / kv_seqlen_align + (kv_seqlen % kv_seqlen_align != 0);
                // cores_per_seq = actual_work_cores;
                uint32_t kv_seqlen_process = 0;
                if (block_idx < block_idx / cores_per_seq * cores_per_seq + actual_work_cores){
                    kv_seqlen_process = (block_idx % cores_per_seq == actual_work_cores - 1) ?
                        (kv_seqlen - kv_seqlen_align * (actual_work_cores - 1)) : kv_seqlen_align;
                }

                if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                    uint32_t start_head = 0;
                    uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                    uint32_t cur_q_seq_len = q_seqlen;
                    uint32_t cur_kv_seqlen = kv_seqlen_process;
                    uint32_t cur_head_num = q_heads;

                    // no need to modify anything in cube kernel, just call the same kernel
                    InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                }
            }
        }
        else if (tail > 3 * block_num / 4){
            // no benefit for optimizing this situation
            uint32_t process = totalTaskNumRound + block_idx;
            if (process < totalTaskNum) {
                uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

                uint32_t q_seqlen = 1;
                uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                if (kv_seqlen > 0) {
                    uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;

                    uint32_t start_head = 0;
                    uint32_t start_kv = 0;
                    uint32_t cur_q_seq_len = q_seqlen;
                    uint32_t cur_kv_seqlen = kv_seqlen;
                    uint32_t cur_head_num = q_heads;

                    InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                }
            }
        }
        else {
            // 18 >= tail >= 12
            // first 12 tasks, two cores per task
            {
                uint32_t cores_per_seq = 2;
                uint32_t process = totalTaskNumRound + block_idx / cores_per_seq;
                uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

                uint32_t q_seqlen = 1;
                uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                uint32_t kv_seqlen_process = (block_idx % cores_per_seq == cores_per_seq - 1) ?
                (kv_seqlen - kv_seqlen_align * (cores_per_seq - 1)) : kv_seqlen_align;

                if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                    uint32_t start_head = 0;
                    uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                    uint32_t cur_q_seq_len = q_seqlen;
                    uint32_t cur_kv_seqlen = kv_seqlen_process;
                    uint32_t cur_head_num = q_heads;

                    // no need to modify anything in cube kernel, just call the same kernel
                    InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                }
            }
            {
                uint32_t cores_per_seq = 4;
                uint32_t process = totalTaskNumRound + block_num / 2 + block_idx / cores_per_seq;
                if (process < totalTaskNum) {
                    uint32_t offset_tiling = tiling_head_size + tiling_para_size * process;
                    uint32_t cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + offset_tiling));

                    uint32_t q_seqlen = 1;
                    uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 2 + offset_tiling));
                    uint32_t kv_seqlen_each = kv_seqlen / cores_per_seq;
                    uint32_t kv_seqlen_align = (kv_seqlen_each + block_size - 1) / block_size * block_size;
                    uint32_t kv_seqlen_process = (block_idx % cores_per_seq == cores_per_seq - 1) ?
                    (kv_seqlen - kv_seqlen_align * (cores_per_seq - 1)) : kv_seqlen_align;

                    if (kv_seqlen > 0 && kv_seqlen_process > 0) {
                        uint32_t start_head = 0;
                        uint32_t start_kv = (block_idx % cores_per_seq) * kv_seqlen_align;
                        uint32_t cur_q_seq_len = q_seqlen;
                        uint32_t cur_kv_seqlen = kv_seqlen_process;
                        uint32_t cur_head_num = q_heads;

                        // no need to modify anything in cube kernel, just call the same kernel
                        InnerRunCubeMLATP1(cur_batch, start_head, cur_head_num, start_kv, cur_q_seq_len, cur_kv_seqlen, offset_tiling);
                    }
                }
            }
        }
    }
}

// ==================== TP1 QK 业务函数 ====================

// TP1 QK 参数结构体
struct TP1QKParams {
    uint32_t qk_n;
    uint32_t qk_round_n;
    uint32_t l1_kv_pingpong_flag;
    uint32_t embed_split_size;
    uint32_t round_embed_split_size;
    int64_t now_l1_offset;
    int64_t kv_offset;
    int64_t kv_offset_rope;
    uint32_t sv_round_n;
    uint32_t gm_split_idx;  // split_idx 用于 GM 偏移计算
    uint32_t pp_n_scalar;   // block_size，用于 GM 偏移计算
};

// 业务函数：初始化 TP1 QK 参数
// 对应原始 CUBE1 stage1 中 split_idx 循环体内的参数计算
__aicore__ __attribute__((always_inline)) inline void InitTP1QKParams(
    TP1Context &ctx, uint32_t split_idx, uint32_t n_idx, TP1QKParams &params)
{
    uint32_t pp_n_scalar = ctx.pp_n_scalar;
    uint32_t n_loop = ctx.n_loop;
    uint32_t now_idx = n_idx + split_idx;

    params.l1_kv_pingpong_flag = now_idx % 2;

    // 动态计算 qk_n / qk_round_n
    if (now_idx == (n_loop - 1)) {
        params.qk_n = (ctx.cur_kv_seqlen - now_idx * pp_n_scalar);
    } else {
        params.qk_n = pp_n_scalar;
    }
    params.qk_round_n = RoundUp<BLOCK_SIZE>(params.qk_n);

    // embed_split_size 默认 128，idx==4 时为 64（在编排层设置）
    params.embed_split_size = 128;
    params.round_embed_split_size = RoundUp<T_BLOCK_SIZE>(params.embed_split_size);

    // now_l1_offset 默认 0，在 LoadTP1KVMain/KVRope 中更新
    params.now_l1_offset = 0;

    // block_table_id / kv_offset / kv_offset_rope 地址计算
    uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                    ctx.cur_batch * max_num_blocks_per_query + ctx.start_kv / block_size + now_idx));
    params.kv_offset = (int64_t)block_table_id * block_size * stride_kv;
    params.kv_offset_rope = (int64_t)block_table_id * block_size * stride_kv_rope;

    // sv_round_n 计算（用于 S→GM 的 dstStride）
    uint32_t sv_n_triu = n_loop * pp_n_scalar;
    uint32_t sv_n;
    if (n_idx + ctx.s_block_stack > n_loop - 1) {
        sv_n = ctx.cur_kv_seqlen - n_idx * pp_n_scalar;
    } else {
        sv_n = pp_n_scalar * ctx.s_block_stack;
    }
    params.sv_round_n = (sv_n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    params.gm_split_idx = split_idx;
    params.pp_n_scalar = pp_n_scalar;
}

// 业务函数：TP1 Q 数据从 L1 加载到 L0A
// 复用非 TP1 的 PlatformWaitQLoadComplete / PlatformLoadQToL0A / PlatformSetQLoadComplete
// TP1 中 q_load_coeff = m，与非 TP1 一致
__aicore__ __attribute__((always_inline)) inline void LoadTP1QDataToL0A(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformWaitQLoadComplete(embed_split_idx);
    PlatformLoadQToL0A(embed_split_idx, params.round_embed_split_size, m, false);
    PlatformSetQLoadComplete(embed_split_idx);
}

// 业务函数：TP1 KV Main 数据从 GM 搬运到 L1（idx 0,2）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVMainFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 256;
    PlatformLoadTP1KVMainToL1(embed_split_idx, params.qk_n, params.qk_round_n,
                              params.l1_kv_pingpong_flag, params.kv_offset,
                              embed_split_idx * 128);
}

// 业务函数：TP1 KV Rope 数据从 GM 搬运到 L1（idx 4）
// 更新 now_l1_offset 供后续 L1→L0B 使用
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVRopeFromGM(
    TP1QKParams &params, uint32_t embed_split_idx)
{
    params.now_l1_offset = params.l1_kv_pingpong_flag * 128 * 64 + 2 * 256 * 128;
    PlatformLoadTP1KVRopeToL1(params.qk_n, params.qk_round_n,
                               params.l1_kv_pingpong_flag, params.kv_offset_rope);
}

// 业务函数：TP1 KV 数据从 L1 加载到 L0B（所有 embed_split_idx）
__aicore__ __attribute__((always_inline)) inline void LoadTP1KVDataToL0B(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformLoadTP1KVToL0B(embed_split_idx, params.round_embed_split_size,
                           params.qk_round_n, params.now_l1_offset,
                           params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK MMAD 计算
__aicore__ __attribute__((always_inline)) inline void ComputeTP1QKMMad(
    const TP1QKParams &params, uint32_t embed_split_idx)
{
    PlatformComputeTP1QKMMad(embed_split_idx, params.embed_split_size,
                             m, params.qk_n, params.l1_kv_pingpong_flag);
}

// 业务函数：TP1 QK 结果拷贝到 GM（idx 4 时）
// GM 偏移: block_idx * TMP_SIZE_DECODER * 4 + ((n_idx/s_block_stack)%2) * TMP_SIZE_DECODER * 2 + split_idx * pp_n_scalar
__aicore__ __attribute__((always_inline)) inline void CopyTP1QKResultToGM(
    const TP1QKParams &params, uint32_t embed_split_idx, uint32_t n_idx)
{
    uint64_t gm_dst_offset = (uint64_t)block_idx * TMP_SIZE_DECODER * 4 +
                             (uint64_t)((n_idx / 4) % 2) * TMP_SIZE_DECODER * 2 +
                             params.gm_split_idx * params.pp_n_scalar;
    PlatformCopyTP1QKResultToGM(m, params.qk_round_n,
                                 params.l1_kv_pingpong_flag,
                                 gm_dst_offset, params.sv_round_n);
}