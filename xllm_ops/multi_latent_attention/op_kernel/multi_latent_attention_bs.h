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