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
// 文件：multi_latent_attention_arch32.h
// 层级：第三层 - 平台处理函数（Platform* 系列）
// 说明：本文件被 include 到 MLAttentionDecoderAic 类的 private 区域内，
//       包含所有与 Ascend C 平台 API 直接交互的函数（gm_to_l1 / l1_to_l0_a /
//       l1_to_l0_b / l0c_to_gm / mmad / SET_FLAG / WAIT_FLAG / PIPE_BARRIER 等）。
//       这些函数不包含业务逻辑，仅负责平台相关的数据搬运与同步操作。

// === 第三层：平台处理函数 ===

// 平台函数：设置 Q 数据搬运完成同步（MTE2→MTE1）
__aicore__ __attribute__((always_inline)) inline void PlatformSetQLoadComplete()
{
    SET_FLAG(MTE2, MTE1, EVENT_ID0);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
}

// 平台函数：等待 KV 数据搬运就绪（MTE1→MTE2，等待 V→L0B 完成）
__aicore__ __attribute__((always_inline)) inline void PlatformWaitKVLoadReady(uint32_t l1_kv_pingpong_flag)
{
    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);
}

// 平台函数：设置 KV 主体搬运完成 + 等待 V→L0B 完成（MTE2→MTE1 set, MTE1→MTE2 wait）
__aicore__ __attribute__((always_inline)) inline void PlatformSetKVMainLoadComplete(uint32_t l1_kv_pingpong_flag)
{
    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
    WAIT_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
}

// 平台函数：设置 KV Rope 搬运完成（MTE2→MTE1）
__aicore__ __attribute__((always_inline)) inline void PlatformSetKVRopeLoadComplete(uint32_t l1_kv_pingpong_flag)
{
    SET_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
}

// 平台函数：Q 数据从 L1 加载到 L0A
__aicore__ __attribute__((always_inline)) inline void PlatformLoadQToL0A(
    uint32_t embed_split_idx, uint32_t round_embed_split_size,
    uint32_t q_load_coeff, bool is_rope)
{
    uint32_t loa_repeat = is_rope ? round_embed_split_size / BLOCK_SIZE
                                   : round_embed_split_size / T_BLOCK_SIZE;
    for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
        if (is_rope) {
            l1_to_l0_a<ArchType::ASCEND_V220, IN_ROPE_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2 + loa_load_idx * round_embed_split_size * BLOCK_SIZE],
                l1q_rope_buf_addr_tensor[loa_load_idx * CUBE_MATRIX_SIZE],
                0, loa_repeat, 0, q_load_coeff / BLOCK_SIZE, 0, 0);
        } else {
            l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                l0a_buf_tensor[embed_split_idx % 2 * 16384 + loa_load_idx * round_embed_split_size * BLOCK_SIZE],
                l1q_buf_addr_tensor[embed_split_idx * q_load_coeff * 128 + loa_load_idx * T_CUBE_MATRIX_SIZE],
                0, loa_repeat, 0, q_load_coeff / BLOCK_SIZE, 0, 0);
        }
    }
}

// 平台函数：设置 Q 加载完成（MTE1→M）
__aicore__ __attribute__((always_inline)) inline void PlatformSetQLoadComplete(
    uint32_t embed_split_idx)
{
    SET_FLAG(MTE1, M, embed_split_idx % 2);
}

// 平台函数：等待 Q 加载完成（M→MTE1）
__aicore__ __attribute__((always_inline)) inline void PlatformWaitQLoadComplete(
    uint32_t embed_split_idx)
{
    WAIT_FLAG(M, MTE1, embed_split_idx % 2);
}

// 平台函数：KV 数据从 L1 加载到 L0B
__aicore__ __attribute__((always_inline)) inline void PlatformLoadKVToL0B(
    uint32_t embed_split_idx, uint32_t round_embed_split_size,
    uint32_t k_round_n, uint32_t qk_round_n,
    uint32_t l1_kv_pingpong_flag, uint64_t hidden_size, bool is_rope)
{
    uint32_t l0b_repeat;
    uint32_t l1_src_offset;
    if (is_rope) {
        l0b_repeat = round_embed_split_size * qk_round_n / CUBE_MATRIX_SIZE;
        l1_src_offset = l1_kv_pingpong_flag * 128 * 64;
    } else {
        l0b_repeat = round_embed_split_size * k_round_n / T_CUBE_MATRIX_SIZE;
        l1_src_offset = l1_kv_pingpong_flag * 128 * hidden_size + embed_split_idx * k_round_n * 128;
    }

    if (embed_split_idx == 0) {
        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag);
    }
    if (embed_split_idx == 4) {
        WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
    }
    WAIT_FLAG(M, MTE1, embed_split_idx % 2 + 2);

    if (is_rope) {
        l1_to_l0_b<ArchType::ASCEND_V220, IN_ROPE_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
            l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
            l1kv_rope_buf_addr_tensor[l1_src_offset],
            0, l0b_repeat, 0, 1, 0, 0);
    } else {
        l1_to_l0_b<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
            l0b_buf_tensor[embed_split_idx % 2 * 16384],
            l1kv_buf_addr_tensor[l1_src_offset],
            0, l0b_repeat, 0, 1, 0, 0);
    }

    if (embed_split_idx == 4) {
        SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
    }
    SET_FLAG(MTE1, M, embed_split_idx % 2 + 2);
}

// 平台函数：QK MMAD 计算（含同步）
__aicore__ __attribute__((always_inline)) inline void PlatformComputeQKMMad(
    uint32_t embed_split_idx, uint32_t embed_split_size,
    uint32_t m_value, uint32_t qk_n, uint32_t qk_round_n_l1,
    uint32_t l1_kv_pingpong_flag, bool is_rope)
{
    WAIT_FLAG(MTE1, M, embed_split_idx % 2);
    WAIT_FLAG(MTE1, M, embed_split_idx % 2 + 2);
    if (embed_split_idx == 0) {
        WAIT_FLAG(FIX, M, l1_kv_pingpong_flag);
    }

    if (is_rope) {
        mmad<ArchType::ASCEND_V220, IN_ROPE_DTYPE, IN_ROPE_DTYPE, float, false>(
            mm1_l0c_buf_tensor.template ReinterpretCast<float>()[l1_kv_pingpong_flag * 16384],
            l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
            l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
            m_value, qk_n, embed_split_size, 1);
    } else {
        uint32_t n_value = qk_n;
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            n_value = qk_round_n_l1;
        }
        mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm1OutputType, false>(
            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
            l0a_buf_tensor[embed_split_idx % 2 * 16384],
            l0b_buf_tensor[embed_split_idx % 2 * 16384],
            m_value, n_value, embed_split_size, embed_split_idx == 0);
    }

    PIPE_BARRIER(M);
    SET_FLAG(M, MTE1, embed_split_idx % 2);
    SET_FLAG(M, MTE1, embed_split_idx % 2 + 2);
}

// 平台函数：L0C 结果拷贝到 GM（含同步）
__aicore__ __attribute__((always_inline)) inline void PlatformCopyQKResultToGM(
    uint32_t embed_split_idx, uint32_t m_value, uint32_t qk_n,
    uint32_t qk_round_n, uint32_t l1_kv_pingpong_flag,
    uint32_t n_idx, bool is_rope)
{
    if (is_rope) {
        SET_FLAG(M, FIX, l1_kv_pingpong_flag);
        WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);
        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float>(
                s_rope_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                mm1_l0c_buf_tensor.template ReinterpretCast<float>()[l1_kv_pingpong_flag * 16384],
                m_value, qk_round_n, RoundUp<16>(m_value), qk_round_n);
        } else {
            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
                s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                m_value, qk_round_n, RoundUp<16>(m_value), qk_round_n);
        }
        SET_FLAG(FIX, M, l1_kv_pingpong_flag);
        return;
    }

    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        if (embed_split_idx == 3) {
            SET_FLAG(M, FIX, l1_kv_pingpong_flag);
            WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);
            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
                s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
                mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
                m_value, qk_n, RoundUp<16>(m_value), qk_round_n);
            SET_FLAG(FIX, M, l1_kv_pingpong_flag);
        }
    }
    if (embed_split_idx == 4) {
        SET_FLAG(M, FIX, l1_kv_pingpong_flag);
        WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);
        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
            m_value, qk_round_n, RoundUp<16>(m_value), qk_round_n);
        SET_FLAG(FIX, M, l1_kv_pingpong_flag);
    }
}

// 平台函数：KV L1→L0B 转置加载（k_round_n <= round_embed_split_size 分支）
__aicore__ __attribute__((always_inline)) inline void PlatformLoadKVTransposeSmallN(
    uint32_t l0b_pingpong_flag, uint32_t l1_kv_pingpong_flag,
    uint64_t l1kv_offset, uint64_t k_round_n, uint64_t hidden_size,
    uint32_t embed_split_size, uint32_t round_embed_split_size)
{
    WAIT_FLAG(M, MTE1, l0b_pingpong_flag + 2);
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.dstGap = 0;
    loadDataParams.startIndex = 0;
    loadDataParams.dstFracGap = 0;
    loadDataParams.repeatTimes = round_embed_split_size / T_BLOCK_SIZE;
    loadDataParams.srcStride = k_round_n / T_BLOCK_SIZE;
    uint16_t dstGap = sizeof(IN_DTYPE) == 1 ? 1 : 0;
    loadDataParams.dstGap = dstGap;
    for (uint32_t l0b_load_idx = 0; l0b_load_idx < k_round_n / T_BLOCK_SIZE; ++l0b_load_idx) {
        AscendC::LoadDataWithTranspose(
            l0b_buf_tensor[l0b_pingpong_flag * 16384 + l0b_load_idx * RoundUp<16>(embed_split_size) * T_BLOCK_SIZE],
            l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + l1kv_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
            loadDataParams);
    }
}

// 平台函数：KV L1→L0B 转置加载（k_round_n > round_embed_split_size 分支）
__aicore__ __attribute__((always_inline)) inline void PlatformLoadKVTransposeLargeN(
    uint32_t l0b_pingpong_flag, uint32_t l1_kv_pingpong_flag,
    uint64_t l1kv_offset, uint64_t hidden_size,
    uint32_t qk_round_n_2, uint32_t round_embed_split_size)
{
    WAIT_FLAG(M, MTE1, l0b_pingpong_flag + 2);
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.dstGap = 0;
    loadDataParams.startIndex = 0;
    loadDataParams.dstFracGap = 0;
    for (uint32_t l0b_load_idx = 0; l0b_load_idx < round_embed_split_size / T_BLOCK_SIZE; ++l0b_load_idx) {
        loadDataParams.repeatTimes = qk_round_n_2 / T_BLOCK_SIZE;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = round_embed_split_size / BLOCK_SIZE - 1;
        AscendC::LoadDataWithTranspose(
            l0b_buf_tensor[l0b_pingpong_flag * 16384 + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE],
            l1kv_buf_addr_tensor[l1_kv_pingpong_flag * 128 * hidden_size + l1kv_offset + l0b_load_idx * qk_round_n_2 * T_BLOCK_SIZE],
            loadDataParams);
    }
}

// 平台函数：KV 加载完成后设置 MTE1→MTE2 同步
__aicore__ __attribute__((always_inline)) inline void PlatformSetKVLoadComplete(
    bool is_last_split, uint32_t l1_kv_pingpong_flag)
{
    if (is_last_split) {
        SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag);
    }
}

// 平台函数：P 数据从 GM 加载到 L1
__aicore__ __attribute__((always_inline)) inline void PlatformLoadPFromGMToL1(
    uint32_t n_idx, uint32_t row_num, uint64_t k_round_n,
    uint32_t qk_round_n_2)
{
    WaitFlagDev(SOFTMAX_READY_DECODER);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
    gm_to_l1<ArchType::ASCEND_V220, IN_DTYPE, DataFormat::ND, DataFormat::NZ>(
        l1p_buf_addr_tensor,
        p_gm_tensor[(uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET + ((n_idx - 1) % 2) * TMP_SIZE * T_BLOCK_OFFSET / 2],
        row_num,
        RoundUp<BLOCK_SIZE>(row_num),
        0,
        k_round_n,
        0,
        qk_round_n_2 * 2 / sizeof(IN_DTYPE));
    SET_FLAG(MTE2, MTE1, EVENT_ID7);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID7);
}

// 平台函数：P 数据从 L1 加载到 L0A（INT8 路径）
__aicore__ __attribute__((always_inline)) inline void PlatformLoadPToL0AInt8(
    uint32_t l0_p_pingpong_flag, uint32_t row_num,
    uint32_t qk_round_n_2_l1)
{
    WAIT_FLAG(M, MTE1, l0_p_pingpong_flag);
    l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::NZ, DataFormat::ZZ>(
        l0a_buf_tensor[l0_p_pingpong_flag * 16384], l1p_buf_addr_tensor,
        RoundUp<BLOCK_SIZE>(row_num), qk_round_n_2_l1, 0, 0, 0, 0);
}

// 平台函数：P 数据从 L1 加载到 L0A（非 INT8 路径）
__aicore__ __attribute__((always_inline)) inline void PlatformLoadPToL0AGeneral(
    uint32_t l0_p_pingpong_flag, uint32_t row_num,
    uint32_t qk_round_n_2)
{
    WAIT_FLAG(M, MTE1, l0_p_pingpong_flag);
    uint32_t p_load_coeff = RoundUp<16>(row_num);
    for (uint64_t loa_load_idx = 0; loa_load_idx < p_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
        l1_to_l0_a<ArchType::ASCEND_V220, IN_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
            l0a_buf_tensor[l0_p_pingpong_flag * 16384 + loa_load_idx * qk_round_n_2 * BLOCK_SIZE],
            l1p_buf_addr_tensor[loa_load_idx * T_CUBE_MATRIX_SIZE],
            0, qk_round_n_2 / T_BLOCK_SIZE, 0, p_load_coeff / BLOCK_SIZE, 0, 0);
    }
}

// 平台函数：P 加载完成后设置 MTE1→MTE2 同步
__aicore__ __attribute__((always_inline)) inline void PlatformSetPLoadComplete()
{
    SET_FLAG(MTE1, MTE2, EVENT_ID7);
}

// 平台函数：PV MMAD 计算（含同步）
__aicore__ __attribute__((always_inline)) inline void PlatformComputePVMmad(
    uint32_t l0b_pingpong_flag, uint32_t l0c_pingpong_flag,
    uint32_t l0_p_pingpong_flag, uint32_t embed_split_size,
    uint32_t qk_n_2, bool is_last_split)
{
    SET_FLAG(MTE1, M, l0b_pingpong_flag);
    WAIT_FLAG(MTE1, M, l0b_pingpong_flag);
    WAIT_FLAG(FIX, M, l0c_pingpong_flag);
    mmad<ArchType::ASCEND_V220, IN_DTYPE, IN_DTYPE, mm2OutputType, false>(
        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
        l0a_buf_tensor[l0_p_pingpong_flag * 16384],
        l0b_buf_tensor[l0b_pingpong_flag * 16384],
        m, embed_split_size, qk_n_2, 1);
    SET_FLAG(M, MTE1, l0b_pingpong_flag + 2);
    if (is_last_split) {
        SET_FLAG(M, MTE1, l0_p_pingpong_flag);
    }
}

// 平台函数：L0C 结果拷贝到 GM（含同步）
__aicore__ __attribute__((always_inline)) inline void PlatformCopyPVResultToGM(
    uint32_t l0c_pingpong_flag, uint32_t embed_split_idx,
    uint32_t n_idx, uint32_t round_embed_split_size)
{
    SET_FLAG(M, FIX, l0c_pingpong_flag);
    WAIT_FLAG(M, FIX, l0c_pingpong_flag);
    l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm2CopyType, mm2OutputType>(
        o_tmp_gm_tensor[(uint64_t)block_idx * TMP_SIZE * 2 + embed_split_idx * round_embed_split_size + ((n_idx - 1) % 2) * TMP_SIZE],
        mm2_l0c_buf_tensor[l0c_pingpong_flag * 16384],
        m, RoundUp<16>(round_embed_split_size), RoundUp<16>(m), round_v);
    SET_FLAG(FIX, M, l0c_pingpong_flag);
}

// ==================== Rope 平台函数（INT8 专有）====================

// 平台函数：Q_Rope 从 L1 加载到 L0A（for 循环 l1_to_l0_a）
__aicore__ __attribute__((always_inline)) inline void PlatformLoadQRopeToL0A(
    uint32_t embed_split_idx, uint32_t q_load_coeff,
    uint32_t round_embed_split_size)
{
    WAIT_FLAG(M, MTE1, embed_split_idx % 2);
    for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
        l1_to_l0_a<ArchType::ASCEND_V220, IN_ROPE_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
            l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2 + loa_load_idx * round_embed_split_size * BLOCK_SIZE],
            l1q_rope_buf_addr_tensor[loa_load_idx * CUBE_MATRIX_SIZE],
            0,
            round_embed_split_size / BLOCK_SIZE,
            0,
            q_load_coeff / BLOCK_SIZE,
            0,
            0
        );
    }
    SET_FLAG(MTE1, M, embed_split_idx % 2);
}

// 平台函数：KV_Rope 从 L1 加载到 L0B（单次 l1_to_l0_b + 同步）
__aicore__ __attribute__((always_inline)) inline void PlatformLoadKVRopeToL0B(
    uint32_t embed_split_idx, uint32_t l1_kv_pingpong_flag,
    uint32_t round_embed_split_size, uint32_t qk_round_n)
{
    WAIT_FLAG(MTE2, MTE1, l1_kv_pingpong_flag + 2);
    WAIT_FLAG(M, MTE1, embed_split_idx % 2 + 2);
    l1_to_l0_b<ArchType::ASCEND_V220, IN_ROPE_DTYPE, false, DataFormat::VECTOR, DataFormat::VECTOR>(
        l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
        l1kv_rope_buf_addr_tensor[l1_kv_pingpong_flag * 128 * 64],
        0,
        round_embed_split_size * qk_round_n / CUBE_MATRIX_SIZE,
        0,
        1,
        0,
        0
    );
    SET_FLAG(MTE1, MTE2, l1_kv_pingpong_flag + 2);
    SET_FLAG(MTE1, M, embed_split_idx % 2 + 2);
}

// 平台函数：Rope MMAD 计算（含同步）
__aicore__ __attribute__((always_inline)) inline void PlatformComputeQRopeMMad(
    uint32_t embed_split_idx, uint32_t l1_kv_pingpong_flag,
    uint32_t m_value, uint32_t qk_n, uint32_t embed_split_size)
{
    WAIT_FLAG(MTE1, M, embed_split_idx % 2);
    WAIT_FLAG(MTE1, M, embed_split_idx % 2 + 2);
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        WAIT_FLAG(FIX, M, l1_kv_pingpong_flag);
    }
    mmad<ArchType::ASCEND_V220, IN_ROPE_DTYPE, IN_ROPE_DTYPE, float, false>(
        mm1_l0c_buf_tensor.template ReinterpretCast<float>()[l1_kv_pingpong_flag * 16384],
        l0a_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
        l0b_buf_tensor.template ReinterpretCast<IN_ROPE_DTYPE>()[embed_split_idx % 2 * 16384 * 2],
        m_value,
        qk_n,
        embed_split_size,
        1
    );
    PIPE_BARRIER(M);
    SET_FLAG(M, MTE1, embed_split_idx % 2);
    SET_FLAG(M, MTE1, embed_split_idx % 2 + 2);
}

// 平台函数：Rope 结果 L0C 拷贝到 GM（含同步）
__aicore__ __attribute__((always_inline)) inline void PlatformCopyQRopeResultToGM(
    uint32_t l1_kv_pingpong_flag, uint32_t n_idx,
    uint32_t m_value, uint32_t qk_round_n)
{
    SET_FLAG(M, FIX, l1_kv_pingpong_flag);
    WAIT_FLAG(M, FIX, l1_kv_pingpong_flag);
    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, float, float>(
            s_rope_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
            mm1_l0c_buf_tensor.template ReinterpretCast<float>()[l1_kv_pingpong_flag * 16384],
            m_value,
            qk_round_n,
            RoundUp<16>(m_value),
            qk_round_n
        );
    } else {
        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, mm1CopyType, mm1OutputType>(
            s_gm_tensor[(uint64_t)block_idx * TMP_SIZE_DECODER + (uint64_t)(n_idx % 2) * TMP_SIZE_DECODER / 2],
            mm1_l0c_buf_tensor[l1_kv_pingpong_flag * 16384],
            m_value,
            qk_round_n,
            RoundUp<16>(m_value),
            qk_round_n
        );
    }
    SET_FLAG(FIX, M, l1_kv_pingpong_flag);
}