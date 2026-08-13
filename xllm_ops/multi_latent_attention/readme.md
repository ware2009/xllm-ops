# 关键类说明

## A5关键类实现说明

### MLAttentionDecoderAic 类属性说明

> 源文件：`op_kernel/arch35/multi_latent_attention.h`
> 适用平台：A5 (ascend950 / DAV_3510)
> 命名空间：`XllmOps::MlaArch35`

#### 类模板参数

```cpp
template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA,
          typename IN_DTYPE = half,
          typename IN_ROPE_DTYPE = half,
          typename OUT_DTYPE = half,
          typename IN_KVDTYPE = half,
          InputFormat KInputType = InputFormat::ND_FORMAT,
          bool EnableOptimization = false,
          BlockStack BlockFlow = BlockStack::ONE_FLOW>
class MLAttentionDecoderAic
```

| 模板参数 | 默认值 | 说明 |
|---|---|---|
| `tilingKeyType` | `TILING_HALF_DATA` | 数据类型枚举，决定 GEMM 中间精度。HALF/BF16 → float，INT8 → int32_t |
| `IN_DTYPE` | `half` | Q/P 的输入数据类型 |
| `IN_ROPE_DTYPE` | `half` | RoPE 解耦维度的数据类型 |
| `OUT_DTYPE` | `half` | 输出数据类型 |
| `IN_KVDTYPE` | `half` | KV 的输入数据类型（BF16 路径下与 IN_DTYPE 不同） |
| `KInputType` | `ND_FORMAT` | KV 输入格式（ND 或 NZ） |
| `EnableOptimization` | `false` | 优化开关（预留） |
| `BlockFlow` | `ONE_FLOW` | **A5 shared-UB 布局选择器**，必须与配对的 `MLADecoderAiv` 的 `BlockFlow` 一致，以保证 `UbufAllocA5<BlockFlow>` 计算的 S/P/lo/go UB 偏移在 AIC 与 AIV 两侧完全相同（Normal/Ring 用 `ONE_FLOW`，TP1 用 `FOUR_FLOW`）。偏移不一致是 507015 的潜在诱因。 |

#### 类型别名（Type Aliases）

通过 `AttentionTypeA5<tilingKeyType>` 特化获取，用于 GEMM 中间计算的数据类型：

| 别名 | HALF/BF16 | INT8 | 说明 |
|---|---|---|---|
| `mm1OutputType` | `float` | `int32_t` | GEMM1 (Q×K) 的 L0C 输出类型 |
| `mm1CopyType` | `float` | `int32_t` | GEMM1 结果拷贝到 GM/UB 的类型 |
| `mm2OutputType` | `float` | `int32_t` | GEMM2 (P×V) 的 L0C 输出类型 |
| `mm2CopyType` | `float` | `int32_t` | GEMM2 结果拷贝到 GM/UB 的类型 |

#### 编译期常量（static constexpr）

| 常量 | 计算方式 | 说明 |
|---|---|---|
| `T_CUBE_MATRIX_SIZE` | `CUBE_MATRIX_SIZE_512_A5 / sizeof(IN_DTYPE)` | Cube 矩阵单元大小（元素数），A5 为 512 / sizeof(dtype) |
| `T_BLOCK_SIZE` | `BLOCK_SIZE_32 / sizeof(IN_DTYPE)` | 块大小（元素数），用于对齐计算 |
| `T_BLOCK_OFFSET` | `2 / sizeof(IN_DTYPE)` | 块偏移量，用于地址步进 |
| `L1_KV_HALF_SIZE` | `L1_KV_HALF_BUF_SIZE_A5` (= 73728) | L1 中 KV 半缓冲区大小（元素数），即 2×128×288 |

#### 1. 原始 GM 指针（Raw GM Pointers）

直接指向 Global Memory 的裸指针，用于 `reinterpret_cast` 绑定和地址运算。

| 属性 | 类型 | 说明 |
|---|---|---|
| `q_gm` | `__gm__ IN_DTYPE*` | Query 压缩向量 GM 地址（MLA 压缩维 d=512） |
| `q_rope_gm` | `__gm__ IN_ROPE_DTYPE*` | Query RoPE 解耦向量 GM 地址（d=64） |
| `k_gm` | `__gm__ IN_KVDTYPE*` | Key 压缩向量 GM 地址（KV Cache，d=512） |
| `k_rope_gm` | `__gm__ IN_ROPE_DTYPE*` | Key RoPE 解耦向量 GM 地址（d=64） |
| `o_tmp_gm` | `__gm__ mm2CopyType*` | GEMM2 PV 输出临时 GM 地址（Attention output） |
| `block_tables_gm` | `__gm__ int32_t*` | Block table GM 地址（PagedAttention 的块索引表） |
| `tiling_gm` | `__gm__ uint8_t*` | Tiling 参数 GM 地址（Host 下发的切分参数） |

> **A5 shared-UB 变更**：A3 中的 `s_gm`（QK score）/`p_gm`（softmax P）两个 GM 中转指针在 A5 已**移除**。A5 让 AIC 与 AIV 共享同一物理 UB（同一 AI Core 内 1 Cube + 2 Vector），QK score 经 FixPipe 从 L0C 直达 UB（`s_ubuf_tensor`），softmax P 经 MTE3 从 UB 直达 L1（`p_ubuf_tensor`），消除了 A3 的 S/P GM 往返，不再需要 `s_gm`/`p_gm`。

#### 2. GlobalTensor 包装器

封装了 GM 地址的 AscendC GlobalTensor 对象，支持 Nd2Nz/DataCopy 等高级搬运操作。

| 属性 | 类型 | 说明 |
|---|---|---|
| `q_gm_tensor` | `GlobalTensor<IN_DTYPE>` | Query 压缩向量 GlobalTensor |
| `q_rope_gm_tensor` | `GlobalTensor<IN_ROPE_DTYPE>` | Query RoPE GlobalTensor |
| `k_gm_tensor` | `GlobalTensor<IN_KVDTYPE>` | Key 压缩向量 GlobalTensor |
| `k_rope_gm_tensor` | `GlobalTensor<IN_ROPE_DTYPE>` | Key RoPE GlobalTensor |
| `o_tmp_gm_tensor` | `GlobalTensor<mm2CopyType>` | PV 输出 GlobalTensor（GEMM2 输出） |
| `block_tables_gm_tensor` | `GlobalTensor<int32_t>` | Block table GlobalTensor |
| `s_rope_gm_tensor` | `GlobalTensor<float>` | INT8 路径专用：QK RoPE score 输出 GlobalTensor |

> **A5 shared-UB 变更**：随 `s_gm`/`p_gm` 裸指针移除，A3 中的 `s_gm_tensor`（QK score）/`p_gm_tensor`（softmax P）两个 GlobalTensor 在 A5 也已**移除**，改由共享 UB 的 `s_ubuf_tensor`/`p_ubuf_tensor` 承担（详见「共享 UB 基础设施」章节）。

#### 3. L1 Buffer 偏移量常量

定义 L1 (512KB) 缓冲区中各数据区域的起始偏移（字节）。BF16/HALF 路径使用 3 个区域，INT8 路径使用 5 个区域。

| 属性 | 值 | L1 区域 | 说明 |
|---|---|---|---|
| `l1q_buf_addr_offset` | `0` | Q 区域 (0~147456) | Q 压缩(128×512×2B=128KB) + Q RoPE(128×64×2B=16KB) = 144KB，BF16/HALF 路径 |
| `l1q_rope_buf_addr_offset` | `65536` | Q RoPE 区域 | INT8 路径专用：Q RoPE 独立缓冲区起始偏移 (128×512×1B=64KB) |
| `l1kv_buf_addr_offset` | `147456` | KV 区域 (147456~442368) | KV PingPong 双缓冲：2×128×576×2B = 288KB，BF16/HALF 路径 |
| `l1kv_rope_buf_addr_offset` | `278528` | KV RoPE 区域 | INT8 路径专用：KV RoPE 独立缓冲区起始偏移 |
| `l1p_buf_addr_offset` | `442368` | P 区域 (442368~524288) | Softmax P 结果存储区，剩余 80KB |

##### L1 512KB 布局图

```
L1 (512KB = 524288 bytes)
┌─────────────────────────────────────────────────────────────────┐
│ Q 区域          │ Q RoPE(INT8) │ KV 区域(双缓冲)        │KV RoPE│ P 区域     │
│ 0 ~ 147456      │ 65536~       │ 147456 ~ 442368        │278528~│ 442368~    │
│ Q压缩+Q RoPE    │  (INT8 only) │ 2×128×576 BF16         │(INT8) │ 剩余80KB   │
│ BF16/HALF 路径  │              │                        │       │            │
└─────────────────────────────────────────────────────────────────┘
  BF16/HALF: 3区域 (Q/KV/P)
  INT8:      5区域 (Q/Q_rope/KV/KV_rope/P)
```

#### 4. L1 Buffer 管理器与 L1 Buffer Tensor

A5 使用 AscendC 的 `TPipe` + `TBuf<TPosition>` 管理片上各层级存储（**不再使用 A3 的 `AsdopsBuffer<ArchType::ASCEND_V220>`**）。`TPipe` 统一管理 buffer 生命周期，`TBuf<TPosition>` 按硬件位置声明各层级缓冲区，再由其上分配 `LocalTensor`。

| 属性 | 类型 | 说明 |
|---|---|---|
| `pipe` | `AscendC::TPipe` | 统一 buffer 管理器，负责各 `TBuf` 的初始化与地址分配 |
| `l1TBuf` | `AscendC::TBuf<AscendC::TPosition::A1>` | L1 缓冲区句柄（TPosition::A1 = L1） |
| `l0aTBuf` | `AscendC::TBuf<AscendC::TPosition::A2>` | L0A 缓冲区句柄（TPosition::A2 = L0A） |
| `l0bTBuf` | `AscendC::TBuf<AscendC::TPosition::B2>` | L0B 缓冲区句柄（TPosition::B2 = L0B） |
| `l0cTBuf` | `AscendC::TBuf<AscendC::TPosition::CO1>` | L0C 缓冲区句柄（TPosition::CO1 = L0C） |
| `l1q_buf_addr_tensor` | `LocalTensor<IN_DTYPE>` | L1 中 Q 数据缓冲区 |
| `l1q_rope_buf_addr_tensor` | `LocalTensor<IN_ROPE_DTYPE>` | L1 中 Q RoPE 数据缓冲区（INT8 路径） |
| `l1kv_buf_addr_tensor` | `LocalTensor<IN_KVDTYPE>` | L1 中 KV 数据缓冲区（PingPong 双缓冲） |
| `l1kv_rope_buf_addr_tensor` | `LocalTensor<IN_ROPE_DTYPE>` | L1 中 KV RoPE 数据缓冲区（INT8 路径） |
| `l1p_buf_addr_tensor` | `LocalTensor<IN_DTYPE>` | L1 中 P (softmax 结果) 缓冲区 |

##### INT8 vs BF16/HALF 的 SetArgs 分配逻辑

```cpp
if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_DATA) {
    // INT8: 5 个独立 buffer，Q RoPE 和 KV RoPE 分开存储
    l1q_buf_addr_tensor         = buf.GetBuffer<..., IN_DTYPE>(0);
    l1q_rope_buf_addr_tensor    = buf.GetBuffer<..., IN_ROPE_DTYPE>(128*512*2);
    l1kv_buf_addr_tensor        = buf.GetBuffer<..., IN_DTYPE>(128*576*2);
    l1kv_rope_buf_addr_tensor   = buf.GetBuffer<..., IN_ROPE_DTYPE>(128*576*2 + 128*512*2);
    l1p_buf_addr_tensor         = buf.GetBuffer<..., IN_DTYPE>(128*576*6);
} else {
    // BF16/HALF: 3 个 buffer，Q+RoPE 连续存储，KV+RoPE 连续存储
    l1q_buf_addr_tensor  = buf.GetBuffer<..., IN_DTYPE>(l1q_buf_addr_offset);   // 0
    l1kv_buf_addr_tensor = buf.GetBuffer<..., IN_KVDTYPE>(l1kv_buf_addr_offset); // 147456
}
```

> **A3→A5 变更**：A3 通过 `AsdopsBuffer<ArchType::ASCEND_V220>::GetBuffer` 按字节偏移直接分配各层级 LocalTensor；A5 改为标准 `TPipe` + `TBuf<TPosition>` 范式，由 `pipe.InitBuffer` 初始化各 `TBuf` 后再切分子 tensor，L1 偏移布局（0/65536/147456/278528/442368）保持不变。

#### 4b. 共享 UB 基础设施（A5 新增）

A5 让 **同一 AI Core 内的 1 个 Cube（AIC）与 2 个 Vector（AIV）共享同一块物理 UB**（248KB，`MAX_UB_SIZE_A5 = 253952`）。AIC 通过 `TBuf<TPosition::VECCALC>` 声明 UB 句柄，并用 `UbufAllocA5<BlockFlow>` 统一计算各子区域偏移，使 AIC 与配对 AIV 上的 S/P/lo/go 偏移**完全一致**。这是 A5 消除 A3 的 S/P GM 中转的核心基础设施。

| 属性 | 类型 | 说明 |
|---|---|---|
| `ubTBuf` | `AscendC::TBuf<AscendC::TPosition::VECCALC>` | 共享 UB 缓冲区句柄（VECCALC = UB） |
| `ubufAlloc` | `UbufAllocA5<BlockFlow>` | UB 子区域偏移计算器，`BlockFlow` 必须与 AIV 侧一致，保证两核偏移对齐 |
| `s_ubuf_tensor` | `LocalTensor<float>` | QK score，经 FixPipe 从 L0C→UB（对应 AIV 侧 `ls32`） |
| `p_ubuf_tensor` | `LocalTensor<IN_DTYPE>` | softmax P，作为 MTE3 从 UB→L1 的搬运源（对应 AIV 侧 `lp`） |
| `go_ubuf_tensor` | `LocalTensor<OUT_DTYPE>` | PV 输出 O，经 FixPipe 从 L0C→UB（对应 AIV 侧 `go`） |
| `lo_ubuf_tensor` | `LocalTensor<float>` | PV mmad 中间结果，L0C→UB，长度 4×BLK（对应 AIV 侧 `lo`） |

> **对齐约束**：AIC 与 AIV 必须使用相同的 `BlockFlow` 模板实参（Normal/Ring 用 `ONE_FLOW`，TP1 用 `FOUR_FLOW`），否则 `UbufAllocA5<BlockFlow>` 在两核算出的 `s/p/lo/go` UB 偏移不一致，会导致读写错位——此为 507015 的潜在诱因之一。

#### 5. L0A/L0B/L0C Tensor

Cube 矩阵计算单元的输入/输出缓冲区。L0A/L0B 各 64KB，L0C 256KB（A5 较 A3 的 64KB 扩大 4 倍）。

| 属性 | 类型 | 说明 |
|---|---|---|
| `l0a_buf_tensor` | `LocalTensor<IN_DTYPE>` | L0A 缓冲区：GEMM 左矩阵输入（Q 或 P） |
| `l0b_buf_tensor` | `LocalTensor<IN_DTYPE>` | L0B 缓冲区：GEMM 右矩阵输入（K 或 V） |
| `mm1_l0c_buf_tensor` | `LocalTensor<mm1OutputType>` | L0C 缓冲区：GEMM1 (Q×K) 输出 |
| `mm2_l0c_buf_tensor` | `LocalTensor<mm2OutputType>` | L0C 缓冲区：GEMM2 (P×V) 输出 |

#### 6. Tiling 参数

在 `SetArgs` 中从 GM 读取的 Host 下发切分参数。

| 属性 | 类型 | Tiling索引 | 说明 |
|---|---|---|---|
| `num_batches` | `uint32_t` | [0] TILING_BATCH | batch 大小 |
| `q_heads` | `uint32_t` | [1] TILING_NUMHEADS | Query 头数 |
| `kv_heads` | `uint32_t` | [7] TILING_KVHEADS | KV 头数（MLA 中通常为 1） |
| `embedding_size` | `uint32_t` | [2] TILING_HEADDIM | 嵌入维度（MLA 压缩维 d=512） |
| `block_size` | `uint32_t` | [4] TILING_BLOCKSIZE | KV Cache block 大小 |
| `max_num_blocks_per_query` | `uint32_t` | [5] TILING_MAXBLOCKS | 每个 query 最大 block 数 |
| `stride_kv` | `uint32_t` | 计算值 | KV 步长 = kv_heads × 512（MLA 压缩维） |
| `stride_kv_rope` | `uint32_t` | 计算值 | KV RoPE 步长 = kv_heads × 64（MLA 解耦维） |
| `stride_vo` | `uint32_t` | 计算值 | V/O 步长 = kv_heads × embedding_size |
| `m` | `uint32_t` | - | M 维度（运行时计算） |
| `__k` | `uint32_t` | = embedding_size | K 维度（GEMM1 的内维） |
| `__v` | `uint32_t` | = embedding_size | V 维度（GEMM2 的内维） |
| `round_k` | `uint32_t` | 计算值 | K 对齐到 T_BLOCK_SIZE 的值 |
| `round_v` | `uint32_t` | 计算值 | V 对齐到 BLOCK_SIZE_16 的值 |
| `tiling_head_size` | `uint32_t` | [8] TILING_HEADSIZE | Tiling head 大小 |
| `tiling_para_size` | `uint32_t` | [9] TILING_PARASIZE | 每个 batch 的 tiling 参数大小 |
| `block_size_calc` | `uint32_t` | [25] TILING_BLOCKSIZE_CALC | A5 新增：GEMM tile 大小（A5=256，A3=128） |
| `mask_type` | `uint32_t` | [12] TILING_MASK_TYPE_ND | Mask 类型 |
| `totalTaskNum` | `uint32_t` | [13] TILING_TASK_NUM | 总任务数（TP1 模式） |
| `maxKVSeqLen` | `uint32_t` | [14] TILING_MAX_KV_SEQ_LEN | 最大 KV 序列长度 |
| `cur_qn_blk_size` | `uint32_t` | [10] TILING_MTP_HEAD_SPLIT_SIZE | MTP head 分片大小 |
| `num_batches_pad` | `uint32_t` | 计算值 | batch 对齐到 16 的值 |
| `embed_split_size_qk` | `uint32_t` | 计算值 | GEMM1 QK 的 embed 切分大小（从 block_size_calc 读取，默认 128） |
| `embed_split_loop_qk` | `uint32_t` | 计算值 | GEMM1 QK 的 embed 切分循环数 = ceil(embedding_size / embed_split_size_qk) |

#### 7. Ping-Pong 标志位

控制多级缓冲区双缓冲交替的标志位（0/1 交替），实现计算与搬运流水重叠。

| 属性 | 初始值 | 说明 |
|---|---|---|
| `l1_pingpong_flag` | `0` | L1 Q 缓冲区双缓冲标志 |
| `l1b_pingpong_flag` | `0` | L1 KV 缓冲区双缓冲标志 |
| `l0_pingpong_flag` | `0` | L0A 缓冲区双缓冲标志 |
| `l0b_pingpong_flag` | `0` | L0B 缓冲区双缓冲标志 |
| `l0c_pingpong_flag` | `0` | L0C 缓冲区双缓冲标志 |
| `l1p_pingpong_flag` | `0` | L1 P 缓冲区双缓冲标志 |

#### 8. Ping-Pong 偏移量

根据 Ping-Pong 标志位计算的各缓冲区地址偏移量，实现双缓冲交替访问。

| 属性 | 计算方式 | 说明 |
|---|---|---|
| `l1_offset` | `l1_pingpong_flag × L1_HALF_BUF_SIZE_DECODER_A5 / sizeof(IN_DTYPE)` | L1 Q 缓冲区偏移（128×128 元素） |
| `l1b_offset` | `l1b_pingpong_flag × L1_KV_HALF_BUF_SIZE_A5 / sizeof(IN_DTYPE)` | L1 KV 缓冲区偏移（73728 元素） |
| `l0_offset` | `l0_pingpong_flag × L0AB_UINT8_BUF_SIZE_A5 / sizeof(IN_DTYPE)` | L0A 缓冲区偏移 |
| `l0c_offset` | `l0c_pingpong_flag × L0C_FLOAT_BUF_SIZE_A5` | L0C 缓冲区偏移（65536 元素，256KB） |
| `l0b_offset` | `l0b_pingpong_flag × L0AB_UINT8_BUF_SIZE_A5 / sizeof(IN_DTYPE)` | L0B 缓冲区偏移 |
| `l1p_start_offset` | `l1p_pingpong_flag × L1_HALF_BUF_SIZE_A5 / sizeof(IN_DTYPE)` | L1 P 缓冲区偏移（65536 元素） |

> **注意**：以上偏移量在类定义时根据初始 flag=0 计算为初始值。在 `Run()`/`RunTP1()` 执行过程中，flag 交替翻转后需重新计算对应偏移。

## A5关键类方法实现说明

### 事件功能说明

> 源文件：`op_kernel/arch35/multi_latent_attention.h`
> 方法：`MLAttentionDecoderAic::Run()` / `MLAttentionDecoderAic::RunTP1()`
> 适用平台：A5 (ascend950 / DAV_3510)

`Run()` 和 `RunTP1()` 方法使用 AscendC 异步同步事件（`SET_FLAG` / `WAIT_FLAG`）构建 **8 事件乒乓流水线**，实现 Cube 矩阵计算与片上数据搬运的流水重叠。两个方法的事件定义完全一致，共 25 个事件，分为 Init 阶段（`SET_FLAG` ×25）和 Cleanup 阶段（`WAIT_FLAG` ×25 + `PIPE_BARRIER(ALL)`）。

#### 涉及的硬件管道

| 管道标识 | 全称 | 功能 |
|---|---|---|
| `M` | Scalar | 标量计算，负责地址计算、GEMM 参数设置等控制流 |
| `MTE1` | Move Engine 1 | GM → L1 数据加载（Q/KV 从全局内存搬到 L1） |
| `MTE3` | Move Engine 3 | UB → L1 数据搬运（A5 新增片上通道，替代 A3 的 UB→GM→L1） |
| `FIX` | FixPipe | L0C → UB 数据搬运（GEMM 结果从 Cube 输出缓冲搬到 UB 消费） |

#### 5 类同步事件分组

##### 第 1 组：M → MTE1（×8，EVENT_ID0-7）

```
SET_FLAG(M, MTE1, EVENT_ID0~7)   // Init
WAIT_FLAG(M, MTE1, EVENT_ID0~7)  // Cleanup
```

| 属性 | 值 |
|---|---|
| **事件源** | M (Scalar) |
| **事件目标** | MTE1 (GM→L1 加载) |
| **EVENT_ID 数量** | 8 (ID0~ID7) |
| **功能** | Scalar 完成地址计算和 GEMM 参数设置后，通知 MTE1 可以开始从 GM 加载数据到 L1。8 个事件 ID 对应 8 级乒乓流水线的不同流水级，使得连续迭代的 L1 加载可以重叠执行。 |

##### 第 2 组：FIX → M（×2，EVENT_ID0-1）

```
SET_FLAG(FIX, M, EVENT_ID0~1)    // Init
WAIT_FLAG(FIX, M, EVENT_ID0~1)   // Cleanup
```

| 属性 | 值 |
|---|---|
| **事件源** | FIX (FixPipe, L0C→UB) |
| **事件目标** | M (Scalar) |
| **EVENT_ID 数量** | 2 (ID0~ID1) |
| **功能** | FixPipe 完成 L0C→UB 的 GEMM 结果搬运后，释放 Scalar 进入下一轮 GEMM 参数设置。2 个事件 ID 支持 2 级深度的 GEMM 设置流水。 |

##### 第 3 组：MTE1 → MTE3（×8，EVENT_ID0-7）

```
SET_FLAG(MTE1, MTE3, EVENT_ID0~7)   // Init
WAIT_FLAG(MTE1, MTE3, EVENT_ID0~7)  // Cleanup
```

| 属性 | 值 |
|---|---|
| **事件源** | MTE1 (GM→L1 加载) |
| **事件目标** | MTE3 (UB→L1 搬运) |
| **EVENT_ID 数量** | 8 (ID0~ID7) |
| **功能** | MTE1 完成数据从 GM 加载到 L1 后，通知 MTE3 可以开始 UB→L1 的片上搬运。8 个事件 ID 与第 1 组对应，构成完整的 8 级乒乓流水线。 |
| **A3→A5 差异** | A3 使用 `(MTE1, MTE2)` 事件对，A5 改为 `(MTE1, MTE3)`。原因：A5 新增 MTE3 硬件通道实现 UB→L1 直达搬运，替代了 A3 的 UB→GM→L1 绕行路径。 |

##### 第 4 组：FIX → MTE1（×6，EVENT_ID0-5）

```
SET_FLAG(FIX, MTE1, EVENT_ID0~5)   // Init
WAIT_FLAG(FIX, MTE1, EVENT_ID0~5)  // Cleanup
```

| 属性 | 值 |
|---|---|
| **事件源** | FIX (FixPipe, L0C→UB) |
| **事件目标** | MTE1 (GM→L1 加载) |
| **EVENT_ID 数量** | 6 (ID0~ID5) |
| **功能** | FixPipe 完成当前迭代的 L0C→UB 搬运后，释放 MTE1 加载下一轮迭代的 L1 数据。6 个事件 ID 支持计算与搬运的深度流水重叠，使得下一轮的 GM→L1 加载可以在当前轮 GEMM 计算期间启动。 |

##### 第 5 组：M → FIX（×1，EVENT_ID0）

```
SET_FLAG(M, FIX, EVENT_ID0)    // Init
WAIT_FLAG(M, FIX, EVENT_ID0)   // Cleanup
```

| 属性 | 值 |
|---|---|
| **事件源** | M (Scalar) |
| **事件目标** | FIX (FixPipe, L0C→UB) |
| **EVENT_ID 数量** | 1 (ID0) |
| **功能** | Scalar 完成 FixPipe 参数/地址设置后，释放 FixPipe 进行 L0C→UB 的结果消费。1 个事件 ID 用于单级同步。 |
| **A3→A5 差异** | A3 使用 `(MTE2, FIX)` 事件对，A5 **无 MTE3→FIX 通道**，改用 `(M, FIX)`。代码注释明确：`A3 used MTE2->FIX; A5 has no MTE3->FIX, use M->FIX instead`。 |

#### 事件总数汇总

| 编号 | 事件组 | 源→目标 | EVENT_ID 范围 | 数量 |
|---|---|---|---|---|
| 1 | GM→L1 加载同步 | M → MTE1 | ID0~ID7 | 8 |
| 2 | FixPipe→Scalar 释放 | FIX → M | ID0~ID1 | 2 |
| 3 | L1 加载→UB→L1 搬运同步 | MTE1 → MTE3 | ID0~ID7 | 8 |
| 4 | FixPipe→下一轮 L1 重载 | FIX → MTE1 | ID0~ID5 | 6 |
| 5 | Scalar→FixPipe 消费 | M → FIX | ID0 | 1 |
| | **合计** | | | **25** |

#### A3 → A5 同步事件映射

A5 硬件架构新增了 MTE3（UB→L1）片上直达通道，替代了 A3 中 UB→GM→L1 的绕行路径。因此，涉及 UB↔L1 搬运的同步事件需从 MTE2 映射为 MTE3，其余事件保持不变。

| 事件组 | A3 (V220) | A5 (DAV_3510) | 变更原因 |
|---|---|---|---|
| 第 1 组 | `(M, MTE1)` | `(M, MTE1)` | 不变（GM→L1 路径未变） |
| 第 2 组 | `(FIX, M)` | `(FIX, M)` | 不变（L0C→UB 路径未变） |
| 第 3 组 | `(MTE1, MTE2)` | `(MTE1, MTE3)` | MTE2 是 UB↔L1 搬运管道，A5 改用 MTE3 |
| 第 4 组 | `(FIX, MTE1)` | `(FIX, MTE1)` | 不变（GM→L1 路径未变） |
| 第 5 组 | `(MTE2, FIX)` | `(M, FIX)` | A5 无 MTE3→FIX 通道，改用 Scalar 释放 FixPipe（`A5 has no MTE3->FIX, use M->FIX instead`） |

> **关键变化**：第 3 组源/目标管道从 MTE2 变为 MTE3（8 个事件）；第 5 组因 A5 无 MTE3→FIX 通道，由 `(MTE2, FIX)` 改为 `(M, FIX)`（1 个事件）。合计 9 个事件变更，其余 16 个保持不变。

#### 8 事件乒乓流水线工作原理

Init 阶段一次性 `SET_FLAG` 所有 25 个事件，预置 8 级流水线的初始令牌。任务循环中，各管道通过 `WAIT_FLAG` 消费令牌后执行工作，再通过 `SET_FLAG` 补充令牌，形成乒乓交替：

```
流水级:  0    1    2    3    4    5    6    7
         ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
MTE1:   加载 加载 加载 加载 加载 加载 加载 加载  (GM→L1, 事件ID0~7)
MTE3:   搬运 搬运 搬运 搬运 搬运 搬运 搬运 搬运  (UB→L1, 事件ID0~7)
FIX:    消费 消费 消费 消费 消费 消费           (L0C→UB, 事件ID0~5)
M:      设置 设置                               (GEMM参数, 事件ID0~1)
```

- **第 1 组（M→MTE1 ×8）**：8 级 GM→L1 加载流水
- **第 3 组（MTE1→MTE3 ×8）**：8 级 UB→L1 搬运流水，与第 1 组一一对应
- **第 4 组（FIX→MTE1 ×6）**：6 级 FixPipe→MTE1 回流，支持 6 深度的计算-加载重叠
- **第 2 组（FIX→M ×2）**：2 级 FixPipe→Scalar 回流，支持 2 深度的 GEMM 设置流水
- **第 5 组（M→FIX ×1）**：1 级 Scalar→FixPipe 消费同步（A5 无 MTE3→FIX 通道）

Cleanup 阶段通过 `WAIT_FLAG` 等待所有 25 个事件完成，最后 `PIPE_BARRIER(ALL)` 确保全管道同步后退出。