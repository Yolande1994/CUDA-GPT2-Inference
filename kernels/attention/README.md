# Multi-Head Causal Self-Attention — CUDA 算子优化

从朴素三 Kernel、手写 Flash Attention，到 cuBLAS Tensor Core、FP16/BF16 混合精度与 cuDNN 融合 SDPA，记录 GPT-2 多头因果自注意力前向传播的 11 个版本迭代。

本项目沿 Attention 的优化过程，逐步分析：

- 朴素实现为什么慢，瓶颈在计算还是显存；
- Kernel Fusion、Warp 协作、Shared Memory、在线 Softmax 如何减少全局内存往返；
- Flash Attention 为什么能把注意力矩阵的显存复杂度从 `O(T²)` 降到 `O(T)`；
- FP16/BF16 混合精度、Stride 零拷贝和官方 SDPA 融合还能继续带来多少收益。

---

## 算子简介

多头因果自注意力位于 GPT 类模型的每个 Transformer 块中。

输入 `inp` = Q、K、V：

```text
inp = [ Q | K | V ]
```

每个 Attention Head 独立执行：

```text
S = Q @ Kᵀ / √HS
S = causal_mask(S)
P = softmax(S)
O = P @ V
```

其中：

- `Q`：Query，表示当前 token 正在“查询”什么信息；
- `K`：Key，表示每个历史 token 可以提供什么信息；
- `V`：Value，表示真正被聚合的内容；
- `HS`：单头维度，本项目中 `HS = 768/12 = 64`；
- `1/√HS`：缩放因子，防止点积值随维度增大而过大，导致 Softmax 饱和；
- `causal_mask`：第 `t` 个 token 只能关注 `0..t`，不能关注未来 token。

完整计算过程包含两次矩阵乘和一次行级 Softmax：

```text
第一次GEMM：Q @ Kᵀ
逐行归一化：Softmax
第二次GEMM：P @ V
```

这是 Attention 要解决的核心问题：

1. `Q @ Kᵀ` 和 `P @ V` 是规则矩阵乘，适合 Tensor Core；
2. Softmax 带行最大值、指数、求和和因果掩码，不完全等同于 GEMM；
3. 中间注意力矩阵形状为 `[T, T]`，序列变长时显存和 HBM 流量快速膨胀；
4. Flash Attention 必须依赖在线 Softmax，才能把完整 `[T,T]` 矩阵拆成 Tile 并在片上完成计算。

---

## 张量布局与显存开销

### 张量形状

| 张量 | 形状 | 说明 |
|---|---|---|
| `inp` | `[B, T, 3C]` | Q/K/V 拼接输入，逻辑布局为 `[B,T,3,NH,HS]` |
| `out` | `[B, T, C]` | Attention 输出，逻辑布局为 `[B,T,NH,HS]` |
| `Q/K/V` | `[B, NH, T, HS]` | permute 后头优先布局，便于批量 GEMM |
| `preatt` | `[B, NH, T, T]` | Softmax 前的注意力分数 |
| `att` | `[B, NH, T, T]` | Softmax 后的注意力权重 |
| `L` | `[B, NH, T]` | Flash Attention 写出的 logsumexp，主要供反向传播使用 |

### 为什么注意力矩阵很贵

在本项目配置下：

```text
B * NH * T * T
= 8 * 12 * 1024 * 1024
= 100,663,296 个元素
```

FP32 下，单个注意力矩阵约为：`100,663,296 * 4 bytes ≈ 384 MiB`

朴素实现和 cuBLAS 分离式实现通常同时需要 `preatt` 和 `att` 两个矩阵：`384 MiB + 384 MiB = 768 MiB`

这还不包括 Q、K、V、输出和 cuBLAS Workspace。

> 当代开源模型的上下文长度已从 GPT-2 的 `T=1024` 提升到 Llama 2 的 `T=4096`、Llama 3 的 `T=8192`，闭源模型如 Claude 3 与 Gemini 1.5 更是扩展到 `100k~2M`。以 64 头、`T=4096` 的 Llama 2 70B 为例，单序列下 FP32 单个注意力矩阵就要约 **4 GiB**，`preatt + att` 两个矩阵合计约 **8 GiB**；当 `T` 进入 128k，即使降到 BF16，单个注意力矩阵也会达到 **1~2 TiB**（取决于头数），HBM 物理上根本放不下完整 `[T,T]` 矩阵。

因此 Attention 的主要优化目标不只是提高 FLOPS，更是：

- 避免完整 `[T,T]` 中间矩阵落回 HBM；
- 减少 Q/K/V 和中间矩阵的重复读取；
- 让 GEMM、Softmax、Mask、Scale 在片上流水起来；
- 使用低精度 IO 减少带宽压力；
- 避免不必要的物理 permute。

---

## 优化主线

本项目的 11 个版本可分成三条技术路线。

### 第一阶段：手写 CUDA Kernel，理解硬件映射

- **v1**：QK、Softmax、PV 分成三个独立 Kernel；
- **v2**：融合 QK、Softmax、PV，一个 Warp 处理一行；
- **v3**：朴素 Flash Attention，列优先，复现原始论文，重点演示在线 Softmax；
- **v4**：FlashAttention-2 行优先调度，Q Tile 在外、KV Tile 在内；
- **v5**：Warp 级 split-Q，每个 Warp 负责连续多行 Q。

### 第二阶段：使用成熟 GEMM 库 + 自定义融合 Softmax

- **v6**：cuBLAS TF32 Strided Batched GEMM + 独立 Scale/Mask + Softmax；
- **v7**：cuBLAS GEMM + 融合 Scale/Mask/Online Softmax；
- **v8**：FP16/BF16 IO，FP32 内部累加；
- **v9**：跳过 permute/unpermute，单独测量低精度核心算子下界。

### 第三阶段：官方融合 SDPA

- **v10**：cuDNN Flash Attention，包含 FP32↔低精度 转换；
- **v11**：全半精度 cuDNN SDPA，假设输入输出已经是 FP16/BF16。

---

## 性能总览

> 所有性能数据均采集自 Nsight Compute，命令口径为 `ncu --set full --cache-control all`。  
> 默认测试配置：GPT-2，`B=8`，`T=1024`，`C=768`，`NH=12`，`HS=C/NH=64`，TF32 开启，Profile Block Size = 128。

![GPT-2 Attention NCU 性能总览](images/Attention.png)

### 耗时总览

| 版本 | 核心实现 | Kernel 数 | Kernel 耗时分解 [ms] | 总计 [ms] |
|:---:|---|:---:|---|---:|
| v1 | 朴素三 Kernel：QK/Softmax/PV | 3 | QK `84.47` + Softmax `12.23` + PV `324.45` | **421.15** |
| v2 | Warp 级融合 Kernel | 1 | Fused Attention `37.61` | **37.61** |
| v3 | 列优先朴素 Flash Attention | 3 | permute `0.46` + 列优先 Flash `643.99` + unpermute `0.23` | **644.68** |
| v4 | FlashAttention-2 行优先调度 | 3 | permute `0.46` + 行优先 FA2 `412.48` + unpermute `0.23` | **413.17** |
| v5 | FA2 Warp 级 split-Q | 3 | permute `0.46` + Warp FA2 `198.11` + unpermute `0.23` | **198.80** |
| v6 | cuBLAS TF32 + 独立 Softmax | 6 | permute `0.46` + QK `1.83` + scale `2.11` + softmax `3.19` + PV `1.32` + unpermute `0.23` | **9.14** |
| v7 | cuBLAS + 融合在线 Softmax | 5 | permute `0.46` + QK `1.83` + fused softmax `1.22` + PV `1.32` + unpermute `0.23` | **5.06** |
| v8 | FP16/BF16 IO + FP32 累加 | 5 | lowp permute `0.39` + QK `0.93` + lowp softmax `0.67` + PV `0.67` + lowp unpermute `0.23` | **2.89** |
| v9 | 跳过 permute 的低精度核心 | 3 | QK `0.93` + lowp softmax `0.67` + PV `0.67` | **2.27** |
| v10 | cuDNN SDPA + 精度转换 | 3 | FP32→lowp `0.57` + cuDNN SDPA `0.97` + lowp→FP32 `0.19` | **1.73** |
| v11 | 全低精度 cuDNN SDPA | 1 | cuDNN SDPA `0.97` | **0.97** |

---

## 版本演进

## v1 — 朴素三 Kernel：最直观的 CPU 移植

v1 将注意力拆成三个独立 Kernel：

```cuda
/* ----------------------------------------------------------------------------
   核 1/3：QK 点积核 —— 计算注意力分数 S = Q @ Kᵀ / √hs
   ----------------------------------------------------------------------------
   定位：版本1 由三个独立 kernel 组成，本核为第一个。
   并行策略：一个线程计算输出张量的一个元素（分数矩阵的一个标量）。
   索引映射：输出为行优先 [B, NH, T, T] 布局，需从一维 idx 反解出 (b, h, t, t2) 四个坐标，反解公式为标准行优先布局的除法/取模。
   因果掩码：列号 t2 > 行号 t 的上三角位置直接写 -INFINITY 并返回，不参与点积计算。
   ---------------------------------------------------------------------------- */
__global__ void attention_query_key_kernel1(float* preatt, const float* inp,
                                            int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;  // 一个线程算一个元素
    if (idx < total_threads) {
        // 从全局索引反解出四个维度坐标「批、头、行、列」，对应输出张量的内存布局。
        int b = idx / (NH * T * T);      // 最高维：批索引
        int h = (idx / (T * T)) % NH;    // 第三维：头索引
        int t = (idx / T) % T;           // 第二维：行索引（Query 的位置）
        int t2 = idx % T;                // 最低维：列索引（Key 的位置）
        if (t2 > t) {  // 因果掩码
            preatt[idx] = -INFINITY;
            return;    // 上三角区域（t2 > t）置负无穷后提前返回，无需计算点积
        }
        int C3 = C * 3;
        int hs = C / NH;
        const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
        // 计算 q 向量与 k 向量的点积
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= 1.0 / sqrtf(hs);
        preatt[idx] = val;
    }
}
/* ----------------------------------------------------------------------------
   核 2/3：行 Softmax 核 —— 注意力分数 → 注意力权重
   ----------------------------------------------------------------------------
   定位：版本1 三核中的第二个，输入输出形状均为 [B, NH, T, T]。
   并行策略：一个线程处理权重矩阵的一整行（共 B*NH*T 行），
            行内串行执行「找最大值 → 求指数和 → 归一化」。
   数值稳定性：先减行内最大值再求指数，避免 exp 上溢。
   注意：上三角（未来位置）显式置 0 仅用于调试与 PyTorch 结果对齐，严格来说并非必需。
   ---------------------------------------------------------------------------- */
__global__ void attention_softmax_kernel1(float* att, const float* preatt,
                                          int B, int T, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T;   // 一个线程算一行
    if (idx < total_threads) {
        int b = idx / (NH * T);  // 最高维：批索引
        int h = (idx / T) % NH;  // 第二维：头索引
        int t = idx % T;         // 最低维：行索引（Query 位置）
        // preatt_bth、att_bth 均为长度 T 的一维行向量，即当前线程要处理的行
        const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_bth = att + b*NH*T*T + h*T*T + t*T;
        // 因果范围内寻找最大值
        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++) {
            if (preatt_bth[t2] > maxval) {
                maxval = preatt_bth[t2];
            }
        }
        // 计算指数值并累加求和
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = expf(preatt_bth[t2] - maxval);
            att_bth[t2] = expv;
            expsum += expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;  // 总和的倒数（带除零保护）
        // 归一化得到最终权重
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att_bth[t2] *= expsum_inv;
            } else {
                att_bth[t2] = 0.0f;  // 本无需置零，显式置零（覆盖脏数据）仅用于调试与 PyTorch 对比
            }
        }
    }
}
/* ----------------------------------------------------------------------------
   核 3/3：加权求和核 —— P @ V
   ----------------------------------------------------------------------------
   定位：版本1 三核中的最后一个，输出最终注意力结果。
   并行策略：一个线程负责输出的一行（一个 (b,h,t) 对应的 hs 维向量）。
   计算方式：遍历 0~t 的所有 token，按权重对 V 向量逐元素加权累加；
            累加器必须先清零，避免脏数据污染结果。
   ---------------------------------------------------------------------------- */
__global__ void attention_value_kernel1(float* out, const float* att, const float* inp,
                                        int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T;
    if (idx < total_threads) {
        int b = idx / (NH * T);
        int h = (idx / T) % NH;
        int t = idx % T;
        int C3 = C * 3;
        int hs = C / NH;
        const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
        float* out_bth = out + b * T * C + t * C + h * hs;   // 乘 C 而非 C3
        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }  // 后续使用 += 累加，初始值必须为 0

        for (int t2 = 0; t2 <= t; t2++) {
            const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // V 矩阵内 t2 行向下遍历
            float att_btht2 = att_bth[t2];                                     // 权重行内 t2 标量向右遍历
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}
```

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| QK | 84.47 | 12.49 | 97.54 | 38 |
| Softmax | 12.23 | 6.43 | 74.14 | 40 |
| PV | 324.45 | 7.02 | 73.71 | 36 |
| **合计** | **421.15** | — | — | — |

**瓶颈**

1. QK 中每个点积都重新读取 Q/K，线程之间缺少向量级复用；
2. Softmax 一个线程串行处理一行，并行度不足；
3. PV Kernel 最慢：每个线程对长度为 HS 的输出向量串行遍历所有历史 token；
4. `preatt` 和 `att` 都完整写回 HBM，再被下一个 Kernel 读回来；

大部分 Kernel 的 Memory 指标显著高于 Compute，属于典型访存受限。

---

<br>

## v2 — Warp 级融合 Kernel

v2 将 QK、Softmax、PV 融合到一个 Kernel 中：

- 一个 Warp 固定 32 个线程；
- 一个 Warp 处理一个 `(b,h,t)` 行；
- Warp 内线程协作计算 Q/K 点积；
- 使用 Cooperative Groups 的 `cg::reduce` 做 Warp 规约；
- Grid 为 `(ceil(T / warps_per_block), NH, B)`。

```cuda
__global__ void attention_forward_kernel2(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int C, int NH) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();  // 当前 warp 负责的行索引（query 行 / 权重矩阵行）
    int h = blockIdx.y;
    int b = blockIdx.z;
    if(t >= T) return;
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);
    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // 第一步：Q @ Kᵀ，同时记录行最大值
    float maxval = -INFINITY;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
        // warp 内 32 线程协作计算点积
        float val = 0.0f;
        for (int i = warp.thread_rank(); i < hs; i += warp.size()) {  // 32 线程分工处理同一行 hs 个元素
            val += query_t[i] * key_t2[i];  // 线程局部和
        }
        val = cg::reduce(warp, val, cg::plus<float>{});  // warp 归约求和：向量点积结果
        val *= scale;
        // 更新行最大值（边算边记，相比朴素版少一次全局内存遍历）
        maxval = max(maxval, val);
        if(warp.thread_rank() == 0) {
            preatt_bth[t2] = val;
        }
    }

    // 第二步：计算指数值并累加求和
    float expsum = 0.0f;
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        float expv = expf(preatt_bth[t2] - maxval);
        expsum += expv;
    }
    expsum = cg::reduce(warp, expsum, cg::plus<float>{});
    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // 第三步：归一化得到权重矩阵，与下一步循环合并以减少内存往返
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        att_bth[t2] = expf(preatt_bth[t2] - maxval) * expsum_inv;
    }

    // 第四步：权重矩阵 @ V 加权求和得到最终输出
    float* out_bth = out + b * T * C + t * C + h * hs;
    for (int i = warp.thread_rank(); i < hs; i += warp.size()) {  // 一个 warp 负责一行，每个线程计算自己的输出元素
        float o = 0.f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // V 矩阵内 t2 行向下遍历
            float att_btht2 = att_bth[t2];                                       // 权重行内 t2 标量向右遍历
            o += att_btht2 * value_t2[i];  // 该线程负责的输出位置的最终值
        }
        out_bth[i] = o;
    }
}
```

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers | Block Size |
|---|---:|---:|---:|---:|---:|
| attention_forward_kernel2 | 37.61 | 93.25 | 93.25 | 40 | 128 |

**收益**

- 三个 Kernel 合并为一个，减少 Kernel Launch 开销；
- 点积由 Warp 内 32 个线程共同完成；
- Softmax 的指数和可以在 Warp 内并行规约；
- 输出向量的不同维度由不同 Lane 负责。

**问题**

v2 仍然会把 `preatt` 和 `att` 写入全局内存：

```text
QK 分数写 preatt → 读 preatt 算 Softmax → 写 att → 读 att 算 PV
```

因此它只是做了 Kernel Fusion，还没有像 Flash Attention 那样让注意力分数完全在片上流动。

---

<br>

## v3 — Flash Attention 朴素实现

> 改编自 flash-attention-minimal (https://github.com/tspeterkim/flash-attention-minimal)

> 该版本因列优先遍历、片上矩阵乘未优化，性能很慢，仅作原理演示。

v3 开始引入 Tile 分块：

- `Bc = Br = 32`；
- Q、K、V、S Tile 放入 Shared Memory；
- 外层循环遍历 KV Block，内层循环遍历 Q Block；
- 每行维护 `m` 和 `l`，即当前最大值和指数和；
- 每处理一个 KV Tile，就按新的最大值修正历史结果。

![FA1](images/FA1.png)

```cuda
__global__ void attention_forward_kernel3(const float* Q, const float* K, const float* V, const int T,
                                          const int d, const int Tc, const int Tr, const int Bc, const int Br,
                                          const float softmax_scale, float* l, float* m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;  // batch 索引
    int by = blockIdx.y;  // 注意力头索引
    // 定位当前批、当前头的起始位置
    int qkv_offset = (bx * gridDim.y * T * d) + (by * T * d);  // 跳过 bx 个完整批次 + by 个完整注意力头
    int lm_offset  = (bx * gridDim.y * T)     + (by * T);      // l、m 的偏移量
    // 为 Q、K、V、S 划分共享内存（一维连续 float 数组）
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // Qi、Kj、Vj 的大小（Br = Bc = 32）
    float* Qi = sram;                 // 逻辑形状 [Br, d]
    float* Kj = &sram[tile_size];     // 逻辑形状 [Bc, d]
    float* Vj = &sram[tile_size * 2]; // 逻辑形状 [Bc, d]
    float* S  = &sram[tile_size * 3]; // 逻辑形状 [Br, Bc]

    // Flash Attention 分块双循环：本版外层 j 遍历 KV 块，内层 i 遍历 Q 块
    // 外层 j：每次加载一块 KV 到共享内存，该块被所有 Q 块复用
    for (int j = 0; j < Tc; j++) {
        // 加载 Kj、Vj 到共享内存
        for (int x = 0; x < d; x++) {  // 32 线程并行，每线程负责 K/V 的一行，循环 d 次加载完整行
            Kj[(tx * d) + x] = K[qkv_offset + (j * tile_size) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (j * tile_size) + (tx * d) + x];
        }
        __syncthreads();  // 等待共享内存加载完成，保证内层使用正确的 Kj、Vj

        // 内层 i：遍历 Q 块（共 Tr 次）
        for (int i = 0; i < Tr; i++)  {
            if (i * Br + tx >= T) { break; }  // T 不是 Br 整数倍时，末尾 Q 块的无效 token 跳过
            // 加载 Qi 到共享内存
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (i * tile_size) + (tx * d) + x];
            }
            // 从全局内存加载历史累计状态（一行 Q token 在全局只有唯一一份 m、l）
            float row_m_prev = m[lm_offset + (i * Br) + tx];  // 已处理 KV 块的累计最大值，j=0 时为初始值 -10000.0f
            float row_l_prev = l[lm_offset + (i * Br) + tx];  // 已处理 KV 块的累计指数和，j=0 时为初始值 0.0f

            // 在共享内存上计算 S = Q @ Kᵀ，同时记录块内行最大值
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {  // 遍历 Kj 块内每一行 token
                if (j * Bc + y >= T) { break; }
                // 沿 head 维度做点积
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                // 因果掩码：未来位置置负无穷（也可前置到加载阶段）
                if (i * Br + tx < j * Bc + y) { // 全局 Q 序号 < 全局 K 序号
                    sum = -INFINITY;
                }
                // 分数存入共享内存 S[tx][y]
                S[(tx * Bc) + y] = sum;
                // 更新当前块行最大值
                if (sum > row_m) {
                    row_m = sum;
                }
            }

            // 对 S 做指数化（分子）并累加块内指数和
            float row_l = 0.0f;  // 当前块的行指数和（分母的块内部分）
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= T) { break; }
                // 因果掩码二次处理：未来位置的指数值直接强制置 0
                if (i * Br + tx < j * Bc + y)
                    S[(tx * Bc) + y] = 0;
                else
                    S[(tx * Bc) + y] = __expf(S[(Bc * tx) + y] - row_m);  // 原地覆盖：分数 → 未归一化指数（分子）
                row_l += S[(tx * Bc) + y];  // 块内局部和，尚非全局总和
            }

            // 在线递推合并全局 m、l
            float row_m_new = max(row_m_prev, row_m);  // 历史最大值与当前块最大值取较大者
            // 历史/当前块的指数和都折算到新基准 row_m_new 后相加，得到新基准下的全局指数和
            float row_l_new = (row_l_prev * __expf(row_m_prev - row_m_new)) + (row_l * __expf(row_m - row_m_new));
            // 计算 pv = S @ V，并更新 O 后写回 HBM
            for (int x = 0; x < d; x++) {
                /* 为什么不先做完整 Softmax 再乘 V：
                   1. 完整 Softmax 依赖全局最大值与全局指数和，必须遍历完所有 K 块才能确定；
                   2. 若先存全部分数再归一化，需存储完整 [T,T] 矩阵，显存退回 O(T²)；
                   3. 数学等价：先对未归一化指数加权累加、最后统一归一化，与先归一化再乘 V 的结果完全一致。 */
                float pv = 0.0f;  // 未归一化的加权和（分子）
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= T) { break; }  // V token 全局索引不越界
                    pv += S[(tx * Bc) + y] * Vj[(y * d) + x];  // S[tx][y] * V[y][x]
                }
                /* pv 的基准和当前块的 S 一致，是以 row_m 为基准的。最大值更新了，所有基于旧最大值的量都要先缩放对齐到新基准，才能合并。
                   O_new = (1/row_l_new) * [ O_old * row_l_prev * exp(m_prev-m_new) + pv * exp(m-m_new) ]
                   即「还原 → 对齐 → 归一」三步：还原历史乘 row_l_prev，对齐基准乘 exp 系数，归一除 row_l_new。 */
                O[qkv_offset + (i * tile_size) + (tx * d) + x] = (1.0f / row_l_new) *
                    ( O[qkv_offset + (i * tile_size) + (tx * d) + x] * row_l_prev * __expf(row_m_prev - row_m_new) + (pv * __expf(row_m - row_m_new)) );
            }

            m[lm_offset + (i * Br) + tx] = row_m_new;
            l[lm_offset + (i * Br) + tx] = row_l_new;
        }
        __syncthreads();  // 确保所有线程完成当前 KV 块计算、不再读取 Kj/Vj（下一轮将加载新块覆盖共享内存）
    }
}
```

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.12 | 85.70 | 18 |
| 列优先 Flash Kernel | 643.99 | 6.44 | 55.03 | 40 |
| unpermute | 0.23 | 83.20 | 47.40 | 26 |
| **合计** | **644.68** | — | — | — |

### 为什么实现了 Flash 思想还很慢

v3 是教学演示版本，性能慢有多个原因，以下列出几点：

- 列优先调度：固定 KV Tile，遍历所有 Q Tile，导致每个 Q 行的 `m/l/O` 每处理一个 KV Block 都要写回 HBM（下版用行优先）；
- 序列维度没有充分并行：Grid 只有 `(B, NH)`，没有对 Q Tile 开 Block；
- 矩阵乘是手写标量循环，没有使用 Tensor Core；
- Block 只有 32 个线程，硬件并行度和延迟隐藏能力不足。

这个版本的价值在于清楚展示 Flash Attention 的数学推导，尤其是在线 Softmax 如何合并历史 Tile。

---

<br>

## v4 — FlashAttention-2

**相对版本3 的核心改进：**
1. 交换循环顺序：Q 块在外、KV 块在内，每个 block 固定处理一个 Q 块，Q 仅从 HBM 读取一次，行状态（m、l、输出累加器）全程驻留片上
2. 序列维度并行：Grid 排布 (Tr, NH, B)，Q 分块放 x 轴，block 间无依赖、无通信；连续调度的 block 访问同一段 KV，提升 L2 缓存命中率
3. 延迟归一化：内层循环只做乘法/累加、不做除法，遍历完所有 KV 块后统一除以指数和，每行仅 1 次除法

![FA2](images/FA2.png)

```cuda
__global__ void attention_forward_kernel3_row_major_fa2(const float* Q, const float* K, const float* V, const int T,
                                                        const int d, const int Tc, const int Bc, const int Br,
                                                        const float softmax_scale, float* L, float* O) {
    // Grid 排布：(Tr, NH, B)
    int tx = threadIdx.x;
    int bx = blockIdx.x;  // Q 分块索引（序列维度并行；放 x 轴提升 L2 命中）
    int by = blockIdx.y;  // 头索引
    int bz = blockIdx.z;  // batch 索引
    // 全局基地址偏移
    int qkv_offset = (bz * gridDim.y * T * d) + (by * T * d);
    int l_offset   = (bz * gridDim.y * T)     + (by * T);
    // ---------- 定位当前 block 负责的 Q 块 ----------
    int q_row = bx * Br + tx; // 当前线程对应的全局 Q 行号
    if (q_row >= T) return;   // 序列边界保护，越界直接退出
    // 共享内存划分
    extern __shared__ float sram[];
    int qi_size = Br * d;  // Q 分块大小 [Br, d]
    int kv_size = Bc * d;  // K/V 分块大小 [Bc, d]
    int s_size  = Br * Bc; // 注意力分数块大小 [Br, Bc]
    float* Qi = sram;                       // Q 分块，常驻共享内存
    float* Kj = sram + qi_size;             // K 分块，随 j 循环加载
    float* Vj = sram + qi_size + kv_size;   // V 分块，随 j 循环加载
    float* S  = sram + qi_size + 2*kv_size; // 注意力分数缓冲区
    float* Out_buf = sram + qi_size + 2*kv_size + s_size; // 未归一化输出缓冲区
    // Flash Attention 2 分块双循环：Q 块在外，内层 j 遍历 KV 块
    // 1. 加载当前 Q 块到共享内存（Q 仅从 HBM 读取一次）
    for (int x = 0; x < d; x++) {
        Qi[tx * d + x] = Q[qkv_offset + bx * qi_size + tx * d + x];
    }
    // Qi 为线程私有读写，无需同步

    // 2. 初始化行状态：全程常驻寄存器/共享内存，不写全局内存
    float row_m = -INFINITY; // 全局累计最大值
    float row_l = 0.0f;      // 全局累计指数和
    // 未归一化输出累加器清零（起点必须为 0）
    for (int x = 0; x < d; x++) { Out_buf[tx * d + x] = 0.0f; }

    // ---------- 内层：遍历所有 KV 分块 ----------
    for (int j = 0; j < Tc; j++) {
        int k_col = j * Bc; // 当前 KV 块的起始列号
        if (k_col >= T) break;

        // 3. 加载当前 KV 块到共享内存
        for (int i = tx; i < Bc * d; i += blockDim.x) {
            Kj[i] = K[qkv_offset + j * kv_size + i];
            Vj[i] = V[qkv_offset + j * kv_size + i];
        }
        __syncthreads(); // 确保整块 KV 加载完成再开始计算

        // 4. 计算 QKᵀ 注意力分数
        float cur_m = -INFINITY; // 当前 KV 块内的行最大值
        for (int y = 0; y < Bc; y++) {
            if (k_col + y >= T) break;
            float sum = 0.0f;
            for (int x = 0; x < d; x++) {
                sum += Qi[tx * d + x] * Kj[y * d + x];
            }
            sum *= softmax_scale;
            // 因果掩码：未来位置置负无穷
            if (k_col + y > q_row) {
                sum = -INFINITY;
            }
            S[tx * Bc + y] = sum;
            // 更新当前块行最大值
            if (sum > cur_m) cur_m = sum;
        }

        // 5. 指数化 + 块内求和
        float cur_l = 0.0f; // 当前 KV 块内的行指数和
        for (int y = 0; y < Bc; y++) {
            if (k_col + y >= T) break;
            if (k_col + y > q_row) {
                S[tx * Bc + y] = 0.0f;
            } else {
                S[tx * Bc + y] = __expf(S[tx * Bc + y] - cur_m);
            }
            cur_l += S[tx * Bc + y];
        }

        // 6. 在线递推合并全局统计量
        float new_m    = max(row_m, cur_m);
        float exp_prev = __expf(row_m - new_m);
        float exp_cur  = __expf(cur_m - new_m);
        float new_l    = row_l * exp_prev + cur_l * exp_cur;

        // 7. 递推更新未归一化输出
        // Flash Attention 2 优化：内层不做除法，只累加分子，最终统一归一化
        for (int x = 0; x < d; x++) {
            float pv = 0.0f;
            for (int y = 0; y < Bc; y++) {
                if (k_col + y >= T) break;
                pv += S[tx * Bc + y] * Vj[y * d + x];
            }
            // 历史输出对齐新基准 + 当前块贡献，全程在共享内存完成
            Out_buf[tx * d + x] = Out_buf[tx * d + x] * exp_prev + pv * exp_cur;
        }
        // 更新状态，供下一个 KV 块使用
        row_m = new_m;
        row_l = new_l;
        __syncthreads(); // 等待所有线程用完 KV，下一轮可安全覆盖
    }

    // 8. 最终归一化 + 一次性写回全局内存
    // 除法移到外循环，每行只做一次最终除法，减少了高频除法开销
    float inv_l = 1.0f / row_l;
    for (int x = 0; x < d; x++) {
        float val = Out_buf[tx * d + x] * inv_l;
        O[qkv_offset + bx * qi_size + tx * d + x] = val;
    }
    // 写回 logsumexp L = m + log(l)，替代分开存 m、l，减少反向传播 HBM 读取（纯推理可不写）
    L[l_offset + q_row] = row_m + logf(row_l);
}
```

**相对 v3 的关键变化**

| 对比项 | v3：列优先 | v4：行优先 |
|---|---|---|
| 外层循环 | KV Tile | Q Tile |
| 内层循环 | Q Tile | KV Tile |
| KV 复用方式 | 一个 KV Tile 被多个 Q Tile 共享 | 一个 Q Tile 流式遍历 KV |
| Q 行状态 | 每个 KV Tile 后写回 HBM | 驻留片上直到结束 |
| 输出 O | 多次 HBM 往返 | 最后统一写回 |
| Grid 并行 | `(B,H)` | `(Tr,H,B)` |
| L2 命中 | 调度不友好 | 连续 Q Block 更容易共享 KV |

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.12 | 85.41 | 18 |
| 行优先 FA2 Kernel | 412.48 | 10.20 | 83.68 | 40 |
| unpermute | 0.23 | 83.13 | 47.23 | 26 |
| **合计** | **413.17** | — | — | — |

调度方向虽然正确了，但最耗时的矩阵乘仍然是每个线程串行遍历 HS 和 KV Tile，没有 Tensor Core，也没有成熟 GEMM Kernel 中的软件流水线和寄存器分块。

---

<br>

## v5 — FlashAttention-2 Warp 级优化

![FA-warp](images/FA-warp.png)

**核心特性：**
- 一个 Block 有 4 个 Warp；每个 Warp 独立负责连续 8 行 Q；
- Lane 0–31 分别对应 KV Tile 中的 32 个列；
- 每个 Warp 内部完成 Max、Sum 规约；
- Warp 间零通信、零同步，消除跨 warp 规约开销；
- Grid 仍为 `(Tr, NH, B)`：Q 分块放 x 轴优先调度，优化 KV 的 L2 缓存命中。

```cuda
__global__ void attention_forward_kernel3_row_major_fa2_warp(const float* Q, const float* K, const float* V, const int T,
                                                             const int d, const int Tc, const int Bc, const int Br,
                                                             const float softmax_scale, float* L, float* O) {
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());  // 获取当前 warp 协作组
    // 每 warp 负责 8 行 Q，Br = 4*8 = 32
    const int ROWS_PER_WARP = 8;
    int tx = threadIdx.x;
    int warp_id = tx / 32;  // warp 在 block 内的编号
    int lane_id = tx % 32;  // warp 内线程编号 0~31
    // Grid 三维索引
    int bx = blockIdx.x;  // Q 分块索引（序列维度）
    int by = blockIdx.y;  // 注意力头索引
    int bz = blockIdx.z;  // batch 索引
    // 全局显存基地址偏移
    int qkv_offset = (bz * gridDim.y * T * d) + (by * T * d);
    int l_offset   = (bz * gridDim.y * T)     + (by * T);
    // 共享内存分区
    extern __shared__ float sram[];
    int qi_size = Br * d;  // Q 分块大小 [Br, d]
    int kv_size = Bc * d;  // K/V 分块大小 [Bc, d]
    int s_size  = Br * Bc; // 注意力分数矩阵大小 [Br, Bc]
    float* Qi = sram;                        // Q 分块缓冲区，常驻共享内存
    float* Kj = sram + qi_size;              // K 分块缓冲区，随 j 循环加载
    float* Vj = sram + qi_size + kv_size;    // V 分块缓冲区，随 j 循环加载
    float* S  = sram + qi_size + 2*kv_size;  // 注意力分数临时缓冲区
    float* Out_buf = sram + qi_size + 2*kv_size + s_size; // 未归一化输出累加缓冲区

    // 1. 加载当前 Q 分块到共享内存
    int warp_q_base = warp_id * ROWS_PER_WARP; // 当前 warp 负责的起始行号
    for (int r = 0; r < ROWS_PER_WARP; r++) {  // 遍历本 warp 负责的 8 行
        int row = warp_q_base + r;  // Q 分块内的局部行号（范围 0~Br）
        int q_row = bx * Br + row;  // 全局行号（范围 0~T-1）
        if (q_row >= T) continue; // 序列越界跳过
        // warp 内 32 线程协作加载一行
        for (int x = lane_id; x < d; x += 32) {
            Qi[row * d + x] = Q[qkv_offset + bx * qi_size + row * d + x];
        }
    }

    // 2. 初始化每行的统计量与输出缓冲区
    // 每个 warp 只维护自己负责行的 rowmax、rowsum
    float row_m_arr[ROWS_PER_WARP];  // 寄存器数组：行最大值
    float row_l_arr[ROWS_PER_WARP];  // 寄存器数组：行指数和
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        row_m_arr[r] = -INFINITY;
        row_l_arr[r] = 0.0f;
        int row = warp_q_base + r;
        int q_row = bx * Br + row;
        if (q_row >= T) continue;
        // warp 内 32 线程协作清零一行
        for (int x = lane_id; x < d; x += 32) {
            Out_buf[row * d + x] = 0.0f;
        }
    }

    // ---------- 内层：遍历所有 KV 分块 ----------
    for (int j = 0; j < Tc; j++) {
        int k_col = j * Bc; // 当前 KV 分块的起始列号
        if (k_col >= T) break;

        // 3. 加载当前 KV 分块到共享内存
        // 全 block 线程协同加载，充分利用多线程带宽
        for (int i = tx; i < Bc * d; i += blockDim.x) {
            Kj[i] = K[qkv_offset + j * kv_size + i];
            Vj[i] = V[qkv_offset + j * kv_size + i];
        }
        __syncthreads(); // 确保整块 KV 加载完成后再开始计算

        // 4. split-Q 核心：每个 warp 独立处理自己负责的 Q 行，warp 间无依赖、无通信、无同步
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int row = warp_q_base + r;
            int q_row = bx * Br + row;
            bool row_valid = (q_row < T);

            // 4.1 计算 QKᵀ 注意力分数
            float score = -INFINITY;  // 初始值设为负无穷
            int y = lane_id;  // lane 编号直接作为 KV 分块的列索引（前提：Bc=32）
            if (row_valid && y < Bc && k_col + y < T) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[row * d + x] * Kj[y * d + x];  // S[row][y] += Q[row][x] * K[y][x]
                }
                sum *= softmax_scale;
                // 因果掩码：未来位置置负无穷
                score = (k_col + y > q_row) ? -INFINITY : sum;
            }  // 边界守卫：不满足条件的线程 score 保持 -INFINITY

            // 4.2 warp 内归约求行最大值
            float cur_m = cg::reduce(warp, score, cg::greater<float>());

            // 4.3 数值稳定指数化 + warp 内归约求行指数和
            float exp_val = (score == -INFINITY) ? 0.0f : __expf(score - cur_m);  // 每个线程计算自己那一列的指数值
            if (y < Bc) S[row * Bc + y] = exp_val;
            float cur_l = cg::reduce(warp, exp_val, cg::plus<float>());

            // 4.4 在线递推合并全局统计量
            float new_m = max(row_m_arr[r], cur_m);
            float exp_prev = __expf(row_m_arr[r] - new_m);
            float exp_cur = __expf(cur_m - new_m);
            float new_l = row_l_arr[r] * exp_prev + cur_l * exp_cur;

            // 4.5 递推更新未归一化输出
            if (row_valid) {
                for (int x = lane_id; x < d; x += 32) {
                    float pv = 0.0f;
                    for (int y = 0; y < Bc; y++) {
                        if (k_col + y >= T) break;
                        pv += S[row * Bc + y] * Vj[y * d + x];
                    }
                    // 历史输出对齐新基准 + 当前块贡献，全程在共享内存完成
                    Out_buf[row * d + x] = Out_buf[row * d + x] * exp_prev + pv * exp_cur;
                }
            }
            // 更新状态，供下一个 KV 块使用
            row_m_arr[r] = new_m;
            row_l_arr[r] = new_l;
        }
        __syncthreads(); // 等待所有 warp 计算完成，安全覆盖下一块 KV
    }

    // 5. 最终统一归一化 + 一次性写回全局内存
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int row = warp_q_base + r;
        int q_row = bx * Br + row;
        if (q_row >= T) continue;
        float inv_l = 1.0f / row_l_arr[r];
        for (int x = lane_id; x < d; x += 32) {
            Out_buf[row * d + x] *= inv_l;
            O[qkv_offset + bx * qi_size + row * d + x] = Out_buf[row * d + x];
        }
        // 存储 logsumexp = m + log(l)，供反向传播使用
        L[l_offset + q_row] = row_m_arr[r] + logf(row_l_arr[r]);
    }
}
```

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.18 | 85.49 | 18 |
| Warp FA2 Kernel | 198.11 | 22.21 | 88.81 | 110 |
| unpermute | 0.23 | 83.15 | 47.15 | 26 |
| **合计** | **198.80** | — | — | — |

> 在 Attention 这种包含大矩阵乘的算子中，手写 FMA 循环很难替代 cuBLAS、CUTLASS 或 cuDNN 中的 Tensor Core Kernel。

v5 是手写 Flash Attention 的较好理论演示版本，但还有很多可优化方向，比如：

- 接入 Tensor Core MMA（wmma /mma.sync）
- KV tile 进 Shared Memory，跨 Warp 共享
- float4 向量化访存
- `cp.async` 双缓冲软件流水线
- Causal mask 跳过整块（见本仓库 [trimat_forward.cu](trimat_forward.cu)）
- bank conflict 消除（用 padding/swizzle）
- 调 tile 尺寸和 autotune

---

<br>

## v6 — cuBLAS TF32 + 自定义 Softmax（工业框架式基础）

**定位：** 两次计算密集的矩阵乘 `Q @ Kᵀ`、`P @ V` 全部交给 cuBLAS，
    仅 Softmax 这类带因果掩码的非标运算由自定义 kernel 完成，类似 PyTorch 的实现。

**流程：** 维度重排 → cuBLAS QKᵀ → 缩放+因果掩码 → Softmax → cuBLAS P@V → 反重排

> v6、v7 使用的 Softmax 函数在本仓库 Softmax 文件内已有介绍。

```cuda
// 工具函数：Warp 内最大值规约（基于 __shfl_down_sync）
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/* ----------------------------------------------------------------------------
   分数预处理核：缩放 + 因果掩码
   ----------------------------------------------------------------------------
   1. 有效位置（下三角）乘以缩放系数 scale = 1/√d
   2. 未来位置（上三角，列号 > 行号）强制置 -INFINITY，保证 softmax 后对应权重为 0
   ---------------------------------------------------------------------------- */
__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int t2 = (idx/T)% T;
        int t = idx % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}

/* ----------------------------------------------------------------------------
   自定义 Softmax 核：两级规约版
   ----------------------------------------------------------------------------
   定位：版本6 使用的行 Softmax 核，支持任意 32 倍数的 block_size。
   并行粒度：一个 Block 处理一行。
   规约方式：
     第一级（Warp 内）：shuffle 指令在寄存器中完成规约，延迟极低；
     第二级（跨 Warp）：各 Warp 的 0 号线程将结果写入共享内存，由 block 0 号线程串行合并。
   共享内存：仅需 2*warpsPerBlock 个 float（前半存最大值、后半存求和）。
   流程：找最大值 → 指数写回 → 求指数和 → 归一化。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel5(float* out, const float* inp, int N, int C) {
    int warpsPerBlock = blockDim.x / 32; // 每个块包含的 warp 数
    extern __shared__ float shared[];    // 需分配 2*warpsPerBlock 个元素，前半存最大值，后半存求和
    float* maxvals = shared;                  // 各 warp 的局部最大值
    float* sumvals = &shared[warpsPerBlock];  // 各 warp 的局部指数和
    int idx = blockIdx.x;          // 行索引：一个块处理一行
    int tid = threadIdx.x;         // 块内线程编号
    int warpId = threadIdx.x / 32; // 块内 warp 编号
    int laneId = threadIdx.x % 32; // warp 内线程编号
    // 定位输入的一行，形状 (C,)
    const float* x = inp + idx * C;

    // 第一步：求行最大值（线程粗化 + 两级规约）
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {  // 线程粗化：从 tid 开始、步长 block_size，遍历自己负责的元素
        maxval = fmaxf(maxval, x[i]);  // 线程局部最大值
    }
    // 第一级：warp 内 shuffle 规约
    maxval = warpReduceMax(maxval);
    if (laneId == 0) maxvals[warpId] = maxval;  // 各 warp 的 0 号线程将结果写入共享内存
    __syncthreads();
    // 第二级：跨 warp 求最终最大值
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;  // 最终最大值存入第一个位置
    }
    __syncthreads();
    float offset = maxvals[0];  // 广播给所有线程

    // 第二步：计算指数并写回全局内存
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);  // 线程粗化：每个元素减去最大值再求指数
    }  // 此时 out 中临时存放的是 exp(x - max(x))

    // 第三步：求指数和（两级规约）
    x = out + idx * C;  // 指针切到 out，对指数结果求和
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);  // 线程局部和归约为 warp 总和
    if (laneId == 0) sumvals[warpId] = sumval;  // 写入共享内存
    __syncthreads();
    // 跨 warp 求最终总和
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];  // 广播给所有线程

    // 第四步：整行除以总和完成归一化
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}
```

两次 GEMM 使用 `cublasSgemmStridedBatched`，它一次提交 `B * NH` 个小矩阵乘，比在 Host 端循环调用单矩阵 GEMM 开销更低。

代码开启 `CUBLAS_TF32_TENSOR_OP_MATH`，因此 FP32 接口下的 cuBLAS GEMM 可以使用 TF32 Tensor Core。

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.22 | 85.61 | 18 |
| cuBLAS QK | 1.83 | 71.19 | 62.79 | 255 |
| scale + mask | 2.11 | 59.30 | 73.28 | 16 |
| softmax kernel5 | 3.19 | 49.98 | 64.94 | 23 |
| cuBLAS PV | 1.32 | 96.45 | 89.25 | 90 |
| unpermute | 0.23 | 83.11 | 47.11 | 26 |
| **合计** | **9.14** | — | — | — |

**瓶颈转移**

v6 中两个 GEMM 合计：`1.83 + 1.32 = 3.15 ms`  
Scale 和 Softmax 合计：`2.11 + 3.19 = 5.30 ms`  
Scale + Softmax 占 v6 总耗时约 **58%**，成为新的瓶颈。

当最重的矩阵乘被库优化到足够快后，原来‘顺便执行’的逐元素 Scale、Mask、Softmax 和布局转换会变成主要成本。

**v6、v7的限制**: v6、v7 仍然显式分配并写回完整 `[T,T]` 注意力矩阵，显存复杂度仍是 `O(T²)`，长序列下会受限。适用短到中等序列长度的通用场景。

---

<br>

## v7 — cuBLAS + 融合在线 Softmax

v7 保持两次 cuBLAS GEMM 不变，但将 Scale、Causal Mask、Online Softmax、float4、__ldcs/__stcs 流式访存融合进 `softmax_forward_kernel6`

> 除 Softmax 替换为以下的 softmax_forward_kernel6 外，其余流程与 v6 完全一致

```cuda
/* ----------------------------------------------------------------------------
   融合在线 Softmax 核（缩放 + float4 向量化 + 在线递推 + 因果掩码）
   ----------------------------------------------------------------------------
   定位：版本7 使用的 Softmax 核。一个 Warp 处理一行，在 Softmax 计算的同时完成 QK 分数缩放，省掉独立的 scale_kernel。
   输入输出：inp/out 形状 (N, T, T)，其中 N = B*NH；
            inv_temperature 为缩放系数（即 1/√hs，也称逆温度）。
   在线 Softmax 递推（与显式两遍法数学等价）：
     1. 记旧：记录旧最大值 old_maxval
     2. 更新：用新元素更新最大值 maxval
     3. 折旧：旧总和乘以 exp(old_maxval - maxval)，折算到新基准
     4. 加新：累加新元素的指数值
   向量化：主体以 float4（4 个 float）为单位读取，提升访存带宽利用率；
          因果边界处不足 4 个元素的尾部按标量处理。
   归一化阶段直接重算指数（重算仅需几个时钟周期，远快于全局内存往返），
   并使用 __ldcs / __stcs 流式加载/存储，避免污染缓存。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel6(float* out, float inv_temperature, const float* inp, int N, int T) {
    assert(T % 4  == 0);  // 向量化对齐要求，保证主体循环可完整按 float4 遍历
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();  // 当前 warp 负责的行索引
    if(idx >= N * T) { return; }
    int t = idx % T;     // 当前行在单个 T×T 方阵中的行号（因果掩码边界：只算 0~t 列）
    int t_by_4 = t / 4;  // 当前行内完整 float4 块的数量

    // 定位输入的一行，形状 (T,)
    const float* x = inp + idx * T;
    float maxval = -FLT_MAX;  // 不使用 -INFINITY，避免两值相减意外得到 NaN
    float sumval = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x);  // 指针强转为 float4
    // 在线递推：处理所有完整 float4 块（记旧、更新、折旧、加新）
    for (int t2 = warp.thread_rank(); t2 < t_by_4; t2 += warp.size()) {  // 循环到 t_by_4，处理完所有完整 float4 块
        float4 v = x_vec[t2];
        float old_maxval = maxval;  // 1.记旧
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));  // 2.更新：遍历 4 个元素更新最大值
        }
        // 3.折旧：最大值更新后，旧总和 × exp(old_max - new_max)，折算到新基准（无需重算旧元素的指数）
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));  // 4.加新：累加 4 个元素的指数值
        }
    }
    // 尾部处理：因果边界 t 不一定是 4 的倍数，剩余不足 4 个的元素逐个处理，逻辑与块内一致
    if(t_by_4*4 + warp.thread_rank() <= t) {
        float old_maxval = maxval;  // 1.记旧
        maxval = fmaxf(maxval, x[t_by_4*4 + warp.thread_rank()]);  // 2.更新
        sumval *= expf(inv_temperature * (old_maxval - maxval));   // 3.折旧
        sumval += expf(inv_temperature * (x[t_by_4*4 + warp.thread_rank()] - maxval));  // 4.加新
    }

    // warp 归约求全行最大值与指数和
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});  // 全行最大值
    sumval *= expf(inv_temperature * (maxval - global_maxval));  // 各线程的局部总和统一折算到全行最大值基准
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});     // 全行指数总和
    float norm = 1.f / sum;

    // 整行归一化（只算下三角）
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        // 重算指数比走全局内存往返更快：expf 仅几个时钟周期，而全局内存读取需上百周期
        float ev = expf(inv_temperature * (__ldcs(x + t2) - global_maxval));
        __stcs(out + idx * T + t2, ev * norm);
    }
}
```

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| permute | 0.46 | 40.30 | 85.46 | 18 |
| cuBLAS QK | 1.83 | 71.14 | 62.66 | 255 |
| fused online softmax | 1.22 | 30.81 | 84.10 | 36 |
| cuBLAS PV | 1.32 | 96.45 | 89.25 | 90 |
| unpermute | 0.23 | 83.17 | 47.16 | 26 |
| **合计** | **5.06** | — | — | — |

v6 的 Scale + Softmax 是 `5.30 ms`，v7 融合后只有 `1.22 ms`。收益点有：

- 省掉独立 Scale Kernel；不再单独对完整注意力矩阵做一遍缩放读写；
- 在线 Softmax 减少行遍历次数；
- float4 一次处理 4 个 FP32，提高访存效率；
- Streaming Load/Store 减少缓存污染。

**同样的限制**

和 v6 一样，v7 仍然要把 Softmax 后的 `att` 交给第二个 cuBLAS GEMM，因此完整注意力矩阵仍然物化在 HBM 中。它优化了 Softmax，但不是 Flash Attention 式的完整融合。

---

<br>

## v8 — FP16/BF16 混合精度：IO 用低精度，累加用 FP32

v8 使用 `cublasGemmStridedBatchedEx`，显式指定：

- A/B/C 矩阵的数据类型；
- GEMM 内部计算精度；
- 支持 FP16/BF16 输入输出；
- 内部使用 FP32 累加。

代码通过 `floatX` 宏支持 FP16/BF16；走 Tensor Core 路径。

Softmax 的指数运算对精度敏感，FP16 下容易溢出或精度崩坏。v8 做法是：

```text
加载 FP16/BF16
→ 转成 FP32
→ 求 max/exp/sum
→ 归一化
→ 转回 FP16/BF16
```

类型转换主要发生在寄存器中，成本远低于 HBM 访问。且 Softmax 是典型的 IO 绑定（带宽受限）算子，计算量小，计算精度从 16 位升到 32 位几乎不影响性能。

### 除开上述内容外，v8 与上版 v7 功能完全一致，因此这里不贴代码，完整代码见文件：[attention_forward.cu](attention_forward.cu)。

**性能数据**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| lowp permute | 0.39 | 66.48 | 75.39 | 26 |
| 低精度 cuBLAS QK | 0.93 | 69.86 | 65.92 | 255 |
| lowp online softmax | 0.67 | 79.08 | 75.43 | 26 |
| 低精度 cuBLAS PV | 0.67 | 95.65 | 88.13 | 96 |
| lowp unpermute | 0.23 | 82.85 | 30.01 | 26 |
| **合计** | **2.89** | — | — | — |

低精度同时带来两类收益：

1. 显存带宽减半：FP16/BF16 每个元素 2 字节，FP32 为 4 字节；
2. Tensor Core 吞吐更高：低精度矩阵乘更容易跑满专用硬件。

**布局转换成本：** v8 中 permute 和 unpermute 合计：`0.39 + 0.23 = 0.62 ms`，占 v8 总耗时约 21.5%。当核心 GEMM 和 Softmax 已经足够快后，布局转换成为不可忽略的成本。

---

<br>

## v9 — 跳过 permute/unpermute（仅作性能对比基准）

v9 直接调用 v8，并跳过排列操作 —— `QK GEMM → fused softmax → PV GEMM`

**性能数据**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| 低精度 cuBLAS QK | 0.93 | 69.84 | 66.02 | 255 |
| lowp online softmax | 0.67 | 79.01 | 75.82 | 26 |
| 低精度 cuBLAS PV | 0.67 | 94.93 | 95.15 | 96 |
| **合计** | **2.27** | — | — | — |

比 v8 快约 **1.27×**。

v9 用于单独测量 "矩阵乘 + Softmax" 的性能，量化布局转换和精度转换的成本。

---

<br>

## v10 — cuDNN Flash Attention

v10 使用 cuDNN Frontend 构建 SDPA 计算图。cuDNN 在内部融合：  
```text
Q @ Kᵀ
scale
causal mask
softmax
P @ V
```  
中间注意力矩阵不需要作为完整 `[T,T]` 张量写回 HBM。

**cuDNN 可以直接省掉 permute**

输入物理布局为：

```text
[B, T, 3, H, HS]
```

但可以通过 Stride 让 cuDNN 直接看到三个逻辑张量：

```text
Q/K/V logical shape : [B, H, T, HS]
Q/K/V stride        : (3*H*HS*T, HS, 3*H*HS, 1)
```

这意味着 Q、K、V 不需要物理 permute，也不需要额外复制成三份连续数组。输出直接按 Token 优先布局写出：

```text
O logical shape : [B, H, T, HS]
O physical shape: [B, T, H, HS]
O stride        : (H*T*HS, HS, H*HS, 1)
```

完整的图构建流程：

```cuda
// ============================================================================
// cuDNN 官方 SDPA（Flash Attention）实现
// ============================================================================
#ifdef ENABLE_CUDNN
/* 计算图（Graph）构建与执行全流程：
   调用方请求构建 SDPA 图
       ↓
   定义张量、配置算子 → 组装计算图
       ↓
   合法性校验
       ↓
   生成 key 并查询缓存
       ├─ 命中：直接返回已编译的旧图
       └─ 未命中：编译优化 → 写入缓存 → 返回新图
       ↓
   绑定真实输入/输出显存地址（Variant Pack）
       ↓
   execute() 启动 GPU 计算
       ↓
   得到最终输出结果                          */

// 图相关的所有张量打包，便于缓存与传递
using graph_tensors_fwd = std::tuple<              // 元组内部结构：
    std::shared_ptr<fe::graph::Graph>,             // 计算图本体
    std::shared_ptr<fe::graph::Tensor_attributes>, // Q 张量
    std::shared_ptr<fe::graph::Tensor_attributes>, // K 张量
    std::shared_ptr<fe::graph::Tensor_attributes>, // V 张量
    std::shared_ptr<fe::graph::Tensor_attributes>, // 注意力缩放系数
    std::shared_ptr<fe::graph::Tensor_attributes>, // 输出 O
    std::shared_ptr<fe::graph::Tensor_attributes>>; // Softmax 统计量（反向传播用）
// 计算图缓存：key 为图的哈希，value 为完整图 + 张量包
// 图的构建、优化与内核编译非常慢，可能是单次推理的几十倍；但只要输入形状、
// 参数不变，图就完全相同，可缓存重复使用
using cache_type_fwd = std::unordered_map<std::size_t, graph_tensors_fwd>;

template <typename... Args>  // 接收任意数量、类型的参数
// 计算图构建与缓存：相同输入形状直接复用已编译图，否则新建并编译。
auto lookup_cache_or_build_graph_fwd(Args... args) {
    static cache_type_fwd user_maintained_cache_fwd; // 静态全局缓存
    auto [B, H, T, HS, is_inference_only] = std::make_tuple(args...);  // 「拆包」：模板参数包打包成元组后提取最方便

    // 1. 创建计算图，配置混合精度：IO 低精度省带宽，计算高精度保稳定
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)                    // 输入输出：FP16/BF16
         .set_intermediate_data_type(fe::DataType_t::FLOAT)  // 中间张量：FP32
         .set_compute_data_type(fe::DataType_t::FLOAT);      // 内部计算：FP32

    /* 2. 定义 Q/K/V 张量：直接从打包的 QKV 内存切片，无需外部重排。
       打包 QKV 形状 [B,T,3,H,HS]，逻辑张量形状 [B,H,T,HS]，
       通过 stride 控制偏移，同一块连续内存切出三个逻辑视图 —— 全程没有任何物理数据拷贝或重排：
       1. 消除一次全局 IO：省掉单独的 permute 核，无需完整读写一遍数据；
       2. 节省临时显存：无需额外三块显存存重排结果，直接复用输入缓冲区；
       3. 延迟隐藏：重排开销分散到后续 SDPA 的读取过程中，与计算重叠。 */
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                 .set_name("Q")
                 .set_dim({B, H, T, HS})
                 .set_stride({3*H*HS*T,  HS,  3*H*HS,  1}));  // 逻辑形状 [B,H,T,HS] 各维度的物理步长
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                 .set_name("K")
                 .set_dim({B, H, T, HS})
                 .set_stride({3*H*HS*T,  HS,  3*H*HS,  1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                 .set_name("V")
                 .set_dim({B, H, T, HS})
                 .set_stride({3*H*HS*T,  HS,  3*H*HS,  1}));

    // 3. 定义缩放系数张量
    // 1/√HS 本质是标量，但计算图要求所有运算输入均为张量，因此包装为标量张量
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                        .set_name("attn_scale")
                        .set_dim({1, 1, 1, 1})  // 与四维张量保持维度数一致，运算时自动广播
                        .set_stride({1, 1, 1, 1})  // 步长不影响结果，仅为格式对应
                        .set_is_pass_by_value(true)  // 按值传递：常数直接内嵌到算子参数中，不占显存、无需读取
                        .set_data_type(fe::DataType_t::FLOAT));  // FP32

    // 4. 配置 SDPA（Flash Attention）算子
    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only); // 推理模式：不生成反向统计量
    sdpa_options.set_attn_scale(attn_scale);          // 绑定缩放系数 1/√d
    sdpa_options.set_causal_mask(true);               // 启用因果掩码，屏蔽未来 token

    /* 5. 向图中添加 SDPA 算子，获取输出张量（本步只组装图，不执行计算，真正的计算在图编译完成后的 execute() 阶段）。
       cuDNN SDPA 内部自动融合以下步骤：
         1. Q、K 矩阵乘
         2. 乘以缩放系数
         3. 因果掩码（如开启）
         4. Softmax
         5. 与 V 矩阵乘
       中间结果全程不写回全局显存，在片上共享内存中流转。 */
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);  // O [B,H,T,HS]：最终输出；stats：Softmax 中间统计量，推理模式下为空

    /* 6. 配置输出张量属性
       set_output(true)：标记为最终输出，cuDNN 必须保证结果完整写入用户显存，不能被优化掉
       set_dim：逻辑形状与输入 Q 一致
       set_stride：对应 B、H、T、HS 四个逻辑维度的物理步长 */
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H*T*HS, HS, H*HS, 1});
    assert(stats == nullptr || is_inference_only == false);  // 异常情况：训练模式但 stats 为空，或推理模式但 stats 非空
    // 训练模式下配置 Softmax 统计量输出（用于反向传播）
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)  // 标记为输出，强制 FP32
             .set_dim({B, H, T, 1})  // 每个 token 对应一个标量统计值
             .set_stride({H*T, T, 1, 1});
    }

    // 合法性校验，把错误拦在编译之前
    assert(graph->validate().is_good());

    // 7. 查缓存：已构建过则直接返回，跳过编译
    auto key = graph->key();  // 缓存键：图结构与参数完全一致则 key 相同
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // 8. 未命中缓存：编译计算图（极慢，同形状只执行一次）
    assert(graph->build_operation_graph(cudnn_handle).is_good());    // 构建底层运算图
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A}); // 生成执行计划
    assert(graph->check_support(cudnn_handle).is_good());            // 硬件支持性校验
    assert(graph->build_plans(cudnn_handle).is_good());              // 编译内核

    // 打包成品并写入缓存（缓存的是图，不是数据）
    auto tuple = std::make_tuple(graph, Q, K, V, attn_scale, O, stats);
    user_maintained_cache_fwd.insert({key, tuple});
    return tuple;
}
```

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| FP32 → FP16/BF16 | 0.57 | 17.14 | 51.20 | 16 |
| cuDNN SDPA | 0.97 | 83.05 | 13.36 | 255 |
| FP16/BF16 → FP32 | 0.19 | 26.16 | 37.66 | 16 |
| **合计** | **1.73** | — | — | — |

cuDNN SDPA 的 Memory 指标只有 **13.36%**，Compute 指标为 **83.05%**，说明融合后大量中间访存被消除，Kernel 不再像朴素版本那样严重依赖 HBM 搬运。

---

<br>

## v11 — 全半精度 cuDNN SDPA

v11 在 v10 的基础上跳过 FP32↔低精度 转换，直接用 FP16/BF16 —— `FP16/BF16 QKV → cuDNN fused SDPA → FP16/BF16 output`

**性能**

| Kernel | Duration [ms] | Compute [%] | Memory [%] | Registers |
|---|---:|---:|---:|---:|
| cuDNN SDPA | 0.97 | 83.02 | 13.36 | 255 |
| **合计** | **0.97** | — | — | — |

v10 中精度转换合计：`0.57 + 0.19 = 0.76 ms`，占总耗时约 **43.9%**。v11 去掉这部分。

### 适用场景

v11 是现代大模型推理框架里 attention 这一步的真实形态。满足以下条件时，v11 就是可直接上线的路径：

- **上游权重和激活已经是 BF16/FP16**：模型加载后就是半精度，不需要 `FP32 → BF16/FP16` 转换；
- **下游算子也接受半精度输入**：下一层 MLP、RoPE、残差连接都能直接吃 BF16，attention 输出不需要转回 FP32；
- **QKV 是 packed 布局**：`(B,S,3,H,D)` 连续存储，cuDNN 通过 stride 零拷贝切出 Q/K/V，不需要 permute/unpermute kernel；
- **BF16 精度已通过校准**：输出与 FP32 参考的误差在可接受范围内（通常 perplexity 下降 < 1%）。

典型场景：vLLM、TensorRT-LLM、SGLang 等推理引擎跑 LLaMA / Qwen / DeepSeek 等已经半精度化的模型。

---

<br>

## 后续优化方向

- **基于 CUTLASS 实现 Tensor Core 版 Flash Attention**  
   将手写 FMA 矩阵乘替换为 CUTLASS MMA / Tensor Core 路径，对比 cuBLAS 和 cuDNN 实现。

- **支持更多 Attention 形态**  
   包括 GQA、MQA、MHA KV Cache、Variable-Length Batching、PagedAttention 和 Sliding Window Attention。

- **Autotune Tile 和 Block 参数**  
   根据 Head Dim、序列长度、Shared Memory 容量和寄存器限制，自动搜索 `Br`、`Bc`、Warps per Block、Rows per Warp。

- **扩展低精度路径**  
   在硬件支持且精度验证通过的前提下，继续测试 BF16、FP8 等低精度格式，并区分训练与推理的可接受误差。

- **消除布局转换**  
   与上游 QKV Projection 和下游 Output Projection 联合设计 Layout，使 Attention 边界尽量不做物理 permute。

---

<br>

## 参考资料

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Dao et al., 2022
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, Tri Dao, 2023
- NVIDIA cuDNN Frontend Attention 文档：  
  <https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md>
- flash-attention-minimal：  
  <https://github.com/tspeterkim/flash-attention-minimal>
- CUDA C++ Programming Guide：Cooperative Groups、Shared Memory、Warp Shuffle、Tensor Core
- cuBLAS Documentation：`cublasSgemmStridedBatched`、`cublasGemmStridedBatchedEx`