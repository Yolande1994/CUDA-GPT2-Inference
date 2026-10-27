# GPT-2 推理算子：Matmul / Encoder / Permute

<br>

| 算子 | 作用 | 在 GPT-2 中的位置 |
|---|---|---|
| **matmul_forward** | 线性层 `out = inp @ weight.T + bias` | 所有全连接层（QKV 投影、输出投影、MLP 两层） |
| **encoder_forward** | 词嵌入 + 位置编码相加 | 网络最入口，把 token id 变成向量 |
| **permute** | 维度置换，只改内存布局不改数值 | 注意力计算前后，调整多头数据排列 |

<br>

# 一、matmul_forward：线性层

## 算子简介

把 `(B, T)` 这两维合并后就是标准 GEMM：`M = B*T`，`N = OC`，`K = C`。和 GEMM 唯一区别是多加个 `bias`。

#### 我在独立专题 **[CUDA-GEMM 优化教学](https://github.com/Yolande1994/CUDA-GEMM)** 中手写了 GEMM 的优化路线，里面有更详细完整的演进脉络，因此这里不再细讲 matmul_forward，只做代表性的演示。

## 版本简介

| 版本 | 内容 | 目标 |
|---|---|---|
| v1 | 朴素实现，每线程一个输出、沿 C 循环 | 测速基准 |
| v2 | 共享内存分块 + float4 向量化 | 让 Block 内线程共享数据，减少全局访存 |
| v3 | 修正索引（GEMM NT） | K 方向连续访问，合并访存 |
| v4 | cuBLAS + 单独 add_bias | 用专业库做 GEMM |
| v5 | cuBLASLt epilogue 融合 | 把 bias/GELU 融进 GEMM 收尾，省掉额外 kernel |

**epilogue 融合**是本算子的关键：GEMM 算完输出、写回显存前，数据还在寄存器里，此时直接完成 bias 加法 / GELU 激活，就不必"写回→再读出→再写回"。cuBLASLt 还会通过启发式接口自动选择最优算法。

**手写 v1~v3 的价值在学习**：理解数据在显存、共享内存、寄存器之间如何流动

**生产环境可用 v5（cuBLASLt）**：官方库自动利用 tensor core、选最优算法。（当然也可自定义算子，如果能媲美 cuBLASLt 的话）

<br>

# 二、encoder_forward：词嵌入 + 位置编码

## 算子简介

```
out[b, t, :] = wte[inp[b, t], :] + wpe[t, :]
```

- **词嵌入 wte**`(V, C)`：这个 token 是哪个词，查表
- **位置编码 wpe**`(T, C)`：这个 token 在句中排第几，查表
- 两者逐元素相加

这是**纯访存受限型算子**：每个输出只做一次加法，时间全花在查表和搬数据上。

## 版本简介

| 版本 | 做法 | 目标 |
|---|---|---|
| v1 | B、T 并行，C 维串行 | 朴素基线 |
| v2 | 全维度并行，每线程一个元素 | 把 C 维摊到更多线程，提升并行度 |
| v3 | 128bit 向量化，流式读 + 普通写 | 压缩访存指令数；wte/wpe 读完不占缓存，输出留缓存供后续使用 |

"全维度并行 + 向量化 + 合并访存"是所有带宽型算子的通用套路，在 LayerNorm / GELU / Softmax 等文档中同样适用，此处不重复说明。

<br>

# 三、permute：维度重排

## 算子简介

比如把多头数据从 `(B, T, NH, hs)` 重排为 `(B, NH, T, hs)`：原生布局下单个头的数据跨步离散，置换后每个头的 `(T, hs)` 矩阵连续，让后续矩阵乘触发合并访存。

## 核心理论

```
线性索引与多维坐标的互转

已知维度大小：dim1, dim2, dim3, dim4
对于任意 4 维坐标 (i1, i2, i3, i4)，其线性索引（一维位置）为：
    linear_idx = i1 * (dim2 * dim3 * dim4) + i2 * (dim3 * dim4) + i3 * dim4 + i4

反过来，已知线性索引 idx，求各维坐标（从高维到低维依次拆解）：
    i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
    i2 = (idx / (dim3 * dim4)) % dim2;
    i3 = (idx / dim4) % dim3;
    i4 = idx % dim4;
   
规律说明：要求解任意一维的索引，先将线性编号 idx 除以「该维度右侧所有维度的乘积」，再对「当前维度的大小」取模。
```

> 详情见各文件代码内注释