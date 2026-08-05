/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <numeric>
#include <algorithm>
#include "multi_latent_attention_tiling_impl_a5.h"
#include "multi_latent_attention_tiling_impl.h"
#include "mla.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common.h"
#include "math.h"
#include "tiling/platform/platform_ascendc.h"

#if defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510)

namespace AtbOps {

// ============================================================================
// A5 常量定义
// ============================================================================
const int32_t A5_NUM0 = 0;
const int32_t A5_NUM1 = 1;
const int32_t A5_NUM2 = 2;
const int32_t A5_NUM3 = 3;
const int32_t A5_NUM4 = 4;
const int32_t A5_NUM5 = 5;
const int32_t A5_NUM6 = 6;
const int32_t A5_NUM7 = 7;
const int32_t A5_NUM16 = 16;
const int32_t A5_NUM32 = 32;
const int32_t A5_NUM64 = 64;
const int32_t A5_NUM128 = 128;
const int32_t A5_NUM256 = 256;
const int32_t A5_NUM512 = 512;
const int32_t A5_NUM576 = 576;

// A5 硬件规格常量
// A5 (ascend950/DAV_3510): L0A/B=64KB, L0C=256KB, UB=248KB, AIC=28核
// A5 tile size 提升到 256（A3 为 128），利用更大的 L0A/B/L0C
const int32_t A5_TILE_SIZE_QK = 256;   // GEMM1 (Q×K) embed_split_size_qk
const int32_t A5_TILE_SIZE_V  = 256;   // GEMM2 (P×V) embed_split_size_v

// A5 workspace block size: tile 256 → 256 * 256 * 2(half) = 131072
// A3 为 128 * 256 * 2 = 65536
const int32_t A5_WORKSPACE_BLOCK_SIZE_DB = 131072;

// Tiling 字段索引（与 kernel 头文件保持一致）
const int32_t A5_TILING_BATCH = 0;
const int32_t A5_TILING_NUMHEADS = 1;
const int32_t A5_TILING_HEADDIM = 2;
const int32_t A5_TILING_NUMBLOKS = 3;
const int32_t A5_TILING_BLOCKSIZE = 4;
const int32_t A5_TILING_MAXBLOCKS = 5;
const int32_t A5_TILING_TOR = 6;
const int32_t A5_TILING_KVHEADS = 7;
const int32_t A5_TILING_HEADSIZE = 8;
const int32_t A5_TILING_PARASIZE = 9;
const int32_t A5_TILING_MTP_HEAD_SPLIT_SIZE = 10;
const int32_t A5_TILING_TOTAL_BLOCK_NUM = 11;
const int32_t A5_TILING_MASK_TYPE_ND = 12;
const int32_t A5_TILING_TASK_NUM = 13;
const int32_t A5_TILING_MAX_KV_SEQ_LEN = 14;
// A5 新增字段：索引 25 = embed_split_size（tile size for GEMM1/GEMM2）
const int32_t A5_TILING_BLOCKSIZE_CALC = 25;

const int32_t A5_HIGH_32BIT = 32;

constexpr std::array<int32_t, A5_NUM6> A5_QN_TILE_LIST = { 128, 64, 32, 16, 8, 1 };

using A5AddrOffsets = struct {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrOSeqOffset = 0;
    uint64_t addrOFdSeqOffset = 0;
    uint64_t addrLSeqOffset = 0;
    uint64_t addrMaskOffset = 0;
};

// ============================================================================
// 辅助函数
// ============================================================================

inline uint32_t A5GetHigh32Bit(uint64_t v) {
    return static_cast<uint32_t>(v >> A5_HIGH_32BIT);
}

inline uint32_t A5GetLow32Bit(uint64_t v) {
    return static_cast<uint32_t>(v);
}

void A5GetAddrOffsetMLA(uint32_t *tilingParam, const A5AddrOffsets &addrOffsets,
                        const int32_t tilingOffset)
{
    tilingParam[tilingOffset + A5_NUM2] = A5GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + A5_NUM3] = A5GetLow32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + A5_NUM4] = A5GetHigh32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[tilingOffset + A5_NUM5] = A5GetLow32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[tilingOffset + A5_NUM6] = A5GetHigh32Bit(addrOffsets.addrMaskOffset);
    tilingParam[tilingOffset + A5_NUM7] = A5GetLow32Bit(addrOffsets.addrMaskOffset);
}

int32_t A5GetQNBlockTile(const MLAInfo &mmInfo, int32_t qSeqLen)
{
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(qSeqLen)));
    tileListIdx = (tileListIdx > A5_NUM5) ? A5_NUM5 : tileListIdx;
    int32_t qNBlockTile = A5_QN_TILE_LIST[tileListIdx];
    int32_t group = mmInfo.numHeads / mmInfo.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;
    return qNBlockTile;
}

int32_t A5GetMaxQseqlen(const OpParam::MLA &param)
{
    auto qSeqLen = param.qSeqLen;
    auto maxQSeqlenIter = std::max_element(qSeqLen.begin(), qSeqLen.end());
    auto maxQseqlen = maxQSeqlenIter != qSeqLen.end() ? *maxQSeqlenIter : 1;
    return maxQseqlen;
}

int32_t A5GetMaxKVseqlen(const OpParam::MLA &param)
{
    auto kvSeqLen = param.kvSeqLen;
    auto maxKVSeqlenIter = std::max_element(kvSeqLen.begin(), kvSeqLen.end());
    auto maxKVseqlen = maxKVSeqlenIter != kvSeqLen.end() ? *maxKVSeqlenIter : 1;
    return maxKVseqlen;
}

// ============================================================================
// Non-TP1 per-batch tiling 参数填充
// 与 A3 逻辑相同，per-batch 8 字段：qSeqLen, kvSeqlen, addrQHi, addrQLo,
// addrOHi, addrOLo, addrMaskHi, addrMaskLo
// ============================================================================
ge::graphStatus A5GetNdMLATiling(const MLAInfo &mmInfo, uint32_t &blockDim,
                                 uint32_t *tilingParam, const OpParam::MLA &param)
{
    A5AddrOffsets addrOffsets{};

    int32_t maxQseqlen = A5GetMaxQseqlen(param);
    if (maxQseqlen <= 0) {
        printf("A5: qSeqlen max value(%d) invalid\n", maxQseqlen);
        return ge::GRAPH_FAILED;
    }

    int32_t maxKVseqlen = A5GetMaxKVseqlen(param);
    if (maxKVseqlen <= 0) {
        printf("A5: kvSeqlen max value(%d) invalid\n", maxKVseqlen);
        return ge::GRAPH_FAILED;
    }

    int32_t curQNBlockTile = A5GetQNBlockTile(mmInfo, maxQseqlen);

    uint32_t emptySeq = (mmInfo.qSeqLen == nullptr) ? 1 : 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = (emptySeq == 1) ? 1 : *(mmInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);

        int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE * seqIdx;
        tilingParam[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingParam[tilingOffset + 1] = static_cast<uint32_t>(kvSeqlen);

        A5GetAddrOffsetMLA(tilingParam, addrOffsets, tilingOffset);

        uint64_t addressQffset = static_cast<uint64_t>(mmInfo.numHeads * qSeqLen);
        uint64_t addressOffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSize * qSeqLen);
        uint64_t addressMaskOffset = static_cast<uint64_t>(qSeqLen * maxKVseqlen);
        addrOffsets.addrQSeqOffset += addressQffset;
        addrOffsets.addrOSeqOffset += addressOffset;
        addrOffsets.addrMaskOffset += addressMaskOffset;
    }

    tilingParam[A5_TILING_MTP_HEAD_SPLIT_SIZE] = static_cast<uint32_t>(curQNBlockTile);
    tilingParam[A5_TILING_MAX_KV_SEQ_LEN] = static_cast<uint32_t>(maxKVseqlen);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// TP1 per-task tiling 参数填充
// 与 A3 逻辑相同，per-task 4 字段：seqIdx, prevTaskNum, effectiveKVLen
// ============================================================================
void A5GetNdMLAMtpTilingTP1(const MLAInfo &mmInfo, uint32_t &blockDim,
                            uint32_t *tilingParam, const OpParam::MLA &param)
{
    int32_t prevTaskNum = 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = mmInfo.qSeqLen == nullptr ? 1 : *(mmInfo.qSeqLen + seqIdx);
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);
        for (int32_t qSeq = 0; qSeq < qSeqLen; qSeq++) {
            int32_t tilingOffset = TILING_HEAD_SIZE + TILING_PARA_SIZE_TP1 * prevTaskNum;
            tilingParam[tilingOffset] = seqIdx;
            tilingParam[tilingOffset + A5_NUM1] = prevTaskNum;
            tilingParam[tilingOffset + A5_NUM2] = kvSeqlen - qSeqLen + qSeq + 1;
            prevTaskNum++;
        }
    }
}

// ============================================================================
// Tiling head 填充（15 字段 + A5 新增 BLOCKSIZE_CALC 字段）
// ============================================================================
void A5GetTilingHead(const MLAInfo &mmInfo, const OpParam::MLA &param,
                     uint32_t *tilingParam, const uint32_t *torPtr)
{
    tilingParam[A5_TILING_BATCH] = static_cast<uint32_t>(mmInfo.batch);
    tilingParam[A5_TILING_HEADSIZE] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    tilingParam[A5_TILING_PARASIZE] = mmInfo.mtpTp1Flag ?
        static_cast<uint32_t>(TILING_PARA_SIZE_TP1) :
        static_cast<uint32_t>(TILING_PARA_SIZE);
    tilingParam[A5_TILING_NUMHEADS] = static_cast<uint32_t>(mmInfo.numHeads);
    tilingParam[A5_TILING_HEADDIM] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[A5_TILING_NUMBLOKS] = static_cast<uint32_t>(mmInfo.numBlocks);
    tilingParam[A5_TILING_BLOCKSIZE] = static_cast<uint32_t>(mmInfo.blockSize);
    tilingParam[A5_TILING_MAXBLOCKS] = static_cast<uint32_t>(mmInfo.maxNumBlocksPerQuery);
    tilingParam[A5_TILING_TOR] = *torPtr;
    tilingParam[A5_TILING_KVHEADS] = (mmInfo.kvHeads == 0) ? mmInfo.numHeads : mmInfo.kvHeads;
    tilingParam[A5_TILING_MASK_TYPE_ND] = static_cast<uint32_t>(mmInfo.maskType);
    tilingParam[A5_TILING_TASK_NUM] = static_cast<uint32_t>(mmInfo.totalTaskNum);

    // A5 新增：填充 tile size 字段（索引 25）
    // kernel 通过 TILING_BLOCKSIZE_CALC 读取 GEMM1/GEMM2 的 embed_split_size
    // A5 tile 提升到 256（A3 为 128），利用 A5 更大的 L0A/B(64KB) 和 L0C(256KB)
    tilingParam[A5_TILING_BLOCKSIZE_CALC] = static_cast<uint32_t>(A5_TILE_SIZE_QK);
}

// ============================================================================
// A5 GetMLATilingParam 主分发
// 与 A3 差异：
// 1. 去除 BATCH_MLA==32 ? 20 硬编码，保持动态 blockDim
// 2. 调用 A5 版本的 tiling head（填充 BLOCKSIZE_CALC=256）
// ============================================================================
ge::graphStatus GetMLATilingParamA5(OpParam::MLA param, const MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize)
{
    float tor = mmInfo.tor;
    uint32_t *torPtr = reinterpret_cast<uint32_t *>(&tor);

    if (mmInfo.mtpTp1Flag) {
        A5GetNdMLAMtpTilingTP1(mmInfo, blockDim, tilingParam, param);
    } else {
        ge::graphStatus ret = A5GetNdMLATiling(mmInfo, blockDim, tilingParam, param);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        // A5: 去除 A3 的硬编码 blockDim = (batch==32) ? 20 : blockDim
        // A5 使用 GetCoreNumAic() 返回的实际核数（28），支持动态分配
    }
    A5GetTilingHead(mmInfo, param, tilingParam, torPtr);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// A5 Tiling 主入口
// 与 A3 (MLATiling) 差异：
// 1. 动态 blockDim（28核），无硬编码
// 2. A5_WORKSPACE_BLOCK_SIZE_DB=131072（A3 为 65536），tile 256 后 workspace 增大
// 3. 填充 TILING_BLOCKSIZE_CALC=256 字段
// 4. TilingKey 编码：A5 仅 3 种 dataType，无 kNz/ring 位
// ============================================================================
ge::graphStatus MLATilingA5(gert::TilingContext *context)
{
    // 复用 A3 的参数提取逻辑
    OpParam::MLA param = GetParamFromTilingContext(context);
    auto qTensor = context->GetInputTensor(DIM_0);
    auto qRopeTensor = context->GetInputTensor(DIM_1);

    MLAInfo mmInfo = {};
    GetTilingKeyTypeBase(mmInfo, qTensor, qRopeTensor);
    GetMLAInfo(context, mmInfo, param);

    // A5: 动态获取 AIC 核数（28），不硬编码
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto blockDim = ascendcPlatform.GetCoreNumAic();

    uint64_t tilingSizeWithoutWorkspaceParam = GetTilingSize(param);
    uint64_t tilingSizeWithWorkSpace = sizeof(uint64_t) * 6 + tilingSizeWithoutWorkspaceParam;
    uint32_t *tilingParam = static_cast<uint32_t *>(context->GetRawTilingData()->GetData());
    context->GetRawTilingData()->SetDataSize(tilingSizeWithWorkSpace);

    ge::graphStatus ret = GetMLATilingParamA5(param, mmInfo, blockDim,
                                               tilingParam + 6 * 2, tilingSizeWithWorkSpace);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("A5: GetMLATilingParamA5 failed: %d\n", ret);
        return ret;
    }

    // ========================================================================
    // A5 Workspace 计算
    // A5 tile 256 → A5_WORKSPACE_BLOCK_SIZE_DB = 131072 (256 * 256 * 2)
    // A3 tile 128 → WORKSPACE_BLOCK_SIZE_DB     = 65536  (128 * 256 * 2)
    // ========================================================================
    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t dataLenInt = sizeof(int32_t);
    uint64_t basicWorkSpaceHalf = blockDim * A5_WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat = blockDim * A5_WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    uint64_t basicWorkSpaceInt8 = blockDim * A5_WORKSPACE_BLOCK_SIZE_DB * dataLenInt;

    bool isQuant = (static_cast<int32_t>(mmInfo.type) < A5_NUM2) ? 0 : 1;
    uint64_t pWorkSpaceSize = isQuant ? basicWorkSpaceInt8 : basicWorkSpaceHalf * 2;
    uint64_t oTempWorkSpcaceSize = isQuant ? basicWorkSpaceInt8 * 2 : basicWorkSpaceFloat * 2;
    uint64_t tailWorkSpaceFloat = blockDim * 128 * 2 * dataLenFloat;

    uint64_t *workspaceParam = reinterpret_cast<uint64_t *>(tilingParam);
    if (isQuant) {
        workspaceParam[0] = basicWorkSpaceFloat;
        workspaceParam[1] = basicWorkSpaceFloat;
        workspaceParam[2] = pWorkSpaceSize;
        workspaceParam[3] = oTempWorkSpcaceSize;
        workspaceParam[4] = basicWorkSpaceFloat;
        workspaceParam[5] = tailWorkSpaceFloat;
    } else {
        workspaceParam[0] = basicWorkSpaceFloat * 2;
        workspaceParam[1] = A5_NUM512;
        workspaceParam[2] = pWorkSpaceSize;
        workspaceParam[3] = oTempWorkSpcaceSize;
        workspaceParam[4] = basicWorkSpaceFloat;
        workspaceParam[5] = tailWorkSpaceFloat;
    }

    uint64_t usrSize = workspaceParam[0] + workspaceParam[1] + workspaceParam[2] +
                       workspaceParam[3] + workspaceParam[4] + workspaceParam[5];
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    // ========================================================================
    // A5 TilingKey 编码
    // A5 简化：仅 dataType(0-2)，无 kNz/ring 位
    // A3: dataType + (kNz << 4) + (mtpTp1Flag << 2) + (isRing << 5)
    // ============================================================================
    uint32_t dataType = static_cast<uint32_t>(mmInfo.type);
    // A5: 确保 dataType 映射到 0-2（A5 不支持 INT8_BF16=3）
    if (dataType > A5_NUM2) {
        dataType = A5_NUM2; // 降级为 INT8
    }

    uint32_t tilingKey = dataType + (mmInfo.mtpTp1Flag << A5_NUM2);
    context->SetTilingKey(tilingKey);

    context->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

} // namespace AtbOps

#endif // CATLASS_ARCH == 3510