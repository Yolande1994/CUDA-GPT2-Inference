/* ============================================================================
   残差连接 + LayerNorm 融合算子 (Fused Residual + LayerNorm Forward)
   ============================================================================
   
   编译示例：
     nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_forward.cu -o fused_forward

   算子说明：
     将 Transformer 中的两个连续操作「残差连接 (inp1 + inp2)」和「LayerNorm」融合为单个算子。
     数学表达：
       residual = inp1 + inp2
       mean     = mean(residual)
       rstd     = 1 / sqrt(var(residual) + eps)
       normed   = (residual - mean) * rstd * weight + bias

     融合的价值：如果不融合，中间结果 residual（B*T*C 大小）需要写回显存再读回来，额外增加 2 次全局内存往返。
     融合后，residual 可在寄存器/共享内存中直接传递给 LayerNorm，大幅减少显存流量。

   版本说明：
     版本1：朴素非融合实现（两个独立 Kernel 顺序执行）
     版本2：朴素融合（每线程处理 1 个 token，跨步访存，合并失效）
     版本3：Warp 级融合（每 Warp 处理 1 个 token，实现合并访存）
     版本4：向量化 + 单遍统计 + 流式访存 + 锯齿循环
     版本5：全共享内存缓存（权重/偏置/残差全缓存进 Shared Memory）
     版本6：三维 Block + 网格步长循环，生产级自适应设计

   运行命令：
     ./fused_forward 1
     ./fused_forward 2
     ...
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assert.h"
#include <cuda_runtime.h>
#define ENABLE_BF16
#include "../common.h"

// ============================================================================
// CPU 参考实现
// ============================================================================
void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* x = inp + b * T * C + t * C;
            // 计算均值
            float m = 0.0f;
            for (int i = 0; i < C; i++) m += x[i];
            m = m / C;
            // 计算方差（无偏校正）
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            float s = 1.0f / sqrtf(v + eps);
            // 归一化 + 缩放平移
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = (s * (x[i] - m)) * weight[i] + bias[i];
            }
            // 缓存中间结果，供反向传播使用
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   版本1：朴素非融合实现（两个独立 Kernel 顺序执行）
   ----------------------------------------------------------------------------
   定位：直接移植的版本，作为性能基线（Baseline）。
   缺点：
     1. 残差结果 residual 需要先写回全局内存，再被 LayerNorm Kernel 读回来，
        额外产生 2 次全局内存往返（约 2*B*T*C 个元素的读写）。
     2. 两个 Kernel 串行启动，存在启动开销。
   ---------------------------------------------------------------------------- */
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

__global__ void layernorm_forward_kernel1(floatX* out, floatX* mean, floatX* rstd,
                                          const floatX* inp, const floatX* weight, const floatX* bias,
                                          int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;
    if (idx < N) {
        const floatX* x = inp + idx * C;
        // 计算均值
        float m = 0.0f;
        for (int i = 0; i < C; i++) m += (float)x[i];
        m = m / C;
        // 计算方差（无偏校正）
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        float s = 1.0f / sqrtf(v + eps);
        // 归一化 + 缩放平移
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            out_idx[i] = (floatX)((s * ((float)x[i] - m)) * (float)weight[i] + (float)bias[i]);
        }
        mean[idx] = (floatX)m;
        rstd[idx] = (floatX)s;
    }
}

void fused_residual_forward1(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    residual_forward_kernel1<<<grid_size_resid, block_size>>>(residual, inp1, inp2, N * C);
    cudaCheck(cudaGetLastError());
    const int grid_size_ln = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size_ln, block_size>>>(normed, mean, rstd, residual, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   版本2：朴素融合（每线程处理 1 个 token）
   ----------------------------------------------------------------------------
   优化目标：消除 v1 中 residual 中间张量的全局内存读写，将两个算子融合到一个 Kernel。

   问题：每个线程负责一个完整的 token（C 个元素），导致相邻线程之间的访存地址间隔了 C 个元素，访存不合并。
   ---------------------------------------------------------------------------- */
__global__ void fused_forward_kernel2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // 指针偏移到当前 token
    residual += idx * C;
    normed += idx * C;
    inp1 += idx * C;
    inp2 += idx * C;
    float eps = 1e-5f;

    // 残差相加 + 计算均值（一次遍历完成两件事）
    float m = 0.0f;
    for (int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = (floatX)out; // 缓存残差结果，供后续方差和归一化复用
    }
    m = m / C;

    // 计算方差（第二趟遍历，从寄存器外读 residual）
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // 归一化 + 缩放平移（第三趟遍历）
    float s = 1.0f / sqrtf(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m));
        normed[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
    }

    mean[idx] = (floatX)m;
    rstd[idx] = (floatX)s;
}

void fused_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                    const floatX* inp1, const floatX* inp2,
                    const floatX* weight, const floatX* bias,
                    int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size));
    fused_forward_kernel2<<<grid_size, block_size>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   版本3：Warp 级融合（每 Warp 处理 1 个 token）
   ----------------------------------------------------------------------------
   优化目标：解决 v2 中的跨步访存问题，恢复合并访存 (Memory Coalescing)。

   核心改进：
     任务划分从「一个线程处理一个 token」改为「一个 Warp 处理一个 token」。
     Warp 内 32 个线程协同处理同一行的 C 个元素，采用 Warp 内跨步循环：
       for (int c = threadIdx.x; c < C; c += 32)
     这样，Warp 内相邻线程访问的地址是连续的，完美触发合并访存。

   规约方式：
     使用 warpReduceSum (基于 __shfl_xor_sync / __shfl_down_sync) 做 Warp 内规约，
     完全依赖寄存器间的硬件通信，无需共享内存，也无需 __syncthreads()，延迟极低。

   现存问题：
     1. 采用标量访存（每线程 2 或 4 字节），访存指令数较多。
     2. 存在重复读取：第一次遍历写 residual，第二次遍历又从 residual 读回来。
     3. weight / bias 每个 Token 都要重新读一次，未做缓存复用。
   ---------------------------------------------------------------------------- */
__global__ void fused_forward_kernel3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    // 全局 Token 编号 = Block 序号 × 每个 Block 的 Token 数 + Block 内的 Token 序号
    int idx = blockIdx.x * blockDim.y + threadIdx.y;    // 二维线程块 <<<grid_size, dim3(32, token数)>>>
    if (idx >= N) return;
    // 指针偏移到当前 token
    residual += idx * C;
    normed += idx * C;
    inp1 += idx * C;
    inp2 += idx * C;
    float eps = 1e-5f;

    // 残差相加 + 线程局部求和
    float m = 0.0f;
    for (int c = threadIdx.x; c < C; c += WarpSize) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = (floatX)out;
    }
    m = warpReduceSum(m); // Warp 内 shuffle 规约，纯寄存器通信
    m = m / C;

    // 计算方差
    float v = 0.0f;
    for (int c = threadIdx.x; c < C; c += WarpSize) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = warpReduceSum(v);
    v = v / C;

    // 归一化 + 缩放平移
    float s = 1.0f / sqrtf(v + eps);
    for (int c = threadIdx.x; c < C; c += WarpSize) {
        float n = (s * ((float)residual[c] - m));
        normed[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
    }

    // 缓存均值和标准差倒数（由 Warp 0 号线程写入，避免重复写）
    if (threadIdx.x == 0) {
        mean[idx] = (floatX)m;
        rstd[idx] = (floatX)s;
    }
}

void fused_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                    const floatX* inp1, const floatX* inp2,
                    const floatX* weight, const floatX* bias,
                    int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    fused_forward_kernel3<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   版本4：向量化 + 单遍统计 + 流式访存 + 锯齿循环
   ----------------------------------------------------------------------------
   优化目标：在 v3 的合并访存基础上，进一步提升指令效率与缓存利用率。

   四项核心改进：
     1. 128bit 向量化：使用 x128（Packed128）一次性读写 4 个 float（FP32）或 8 个 BF16，
        访存指令数减少到原来的 1/4 ~ 1/8，大幅降低指令发射开销。
     2. 单遍计算均值 + 方差：使用「平方和公式」 var = E[x²] - E[x]²，
        在第一趟遍历中同时累加元素总和与平方和，省掉了第二趟访存读取 residual。
     3. 流式访存：使用 load128cs 避免污染缓存，
        让 weight / bias 这类复用数据能驻留在缓存中。
     4. 锯齿形 (Zigzag) 遍历：第二趟循环从 C 尾部向前读，与第一趟从前往后写的访问顺序正好相反。
        这样，第一趟最后写的 residual 尾部数据还留在缓存里，第二趟一上来就读到它们，大幅提升缓存命中率。

   性能权衡：
     单遍公式 var = E[x²] - E[x]² 会带来数值稳定性问题（大数相减会损失精度），但省掉了一次显存遍历。
     适用于 C 较大且推理场景下对数值精度容忍度较高的场合。
   ---------------------------------------------------------------------------- */
__global__ void fused_forward_kernel4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C) {
    using x128 = Packed128<floatX>;
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= N) return;
    // 指针偏移到当前 token
    residual += idx * C;
    normed += idx * C;
    inp1 += idx * C;
    inp2 += idx * C;
    const float eps = 1e-5f;

    // ---------- 第一趟：残差相加 + 单遍累加总和与平方和 ----------
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int c = threadIdx.x * x128::size; // 每线程负责一个 x128 包
    for (; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for (int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];                    // 元素总和 → 用于算均值
            sum_sq += (float)out[k] * (float)out[k]; // 平方和   → 用于算方差
        }
        store128(residual + c, out); // 普通存储：第二趟循环还要读这个残差结果，希望它留在缓存里
    }
    sum = warpReduceSum(sum);       // 总和
    sum_sq = warpReduceSum(sum_sq); // 平方和
    float m = sum / C;              // 均值
    float v = sum_sq / C - m * m;   // 单遍方差公式：var = E[x²] - E[x]²
    float s = rsqrtf(v + eps);      // 倒标准差

    // ---------- 第二趟：锯齿形倒序遍历（提升缓存命中率） ----------
    // 第一趟从前往后写 residual，第二趟从后往前读，利用 L2 缓存的 LRU 特性，刚写入的数据还在缓存里，先读它们命中率高。
    c -= WarpSize * x128::size; // 回退到最后一个有效块
    for (; c >= 0; c -= WarpSize * x128::size) {
        const x128 r = load128cs(residual + c); // 流式加载：residual 只在此处读一次
        const x128 w = load128(weight + c);     // 普通加载：weight 是全局共享的，缓存起来供所有 Token 复用
        const x128 b = load128(bias + c);       // 普通加载：bias 同理
        x128 out;
        for (int k = 0; k < x128::size; ++k) {
            float n = s * ((float)r[k] - m);
            out[k] = (floatX)(n * (float)w[k] + (float)b[k]);
        }
        store128cs(normed + c, out); // 流式存储：输出写完就不需要了
    }

    if (threadIdx.x == 0) {
        mean[idx] = (floatX)m;
        rstd[idx] = (floatX)s;
    }
}

void fused_forward4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                    const floatX* inp1, const floatX* inp2,
                    const floatX* weight, const floatX* bias,
                    int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    fused_forward_kernel4<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   版本5：全共享内存缓存
   ----------------------------------------------------------------------------
   优化目标：解决 v4 的三个痛点：
     1. weight / bias 重复读取：v4 每个 Token 都要从全局内存读一次 weight 和 bias；
        v5 中整个线程块只读一次，块内所有 Token 共享复用。
     2. 单遍方差的数值稳定性：v4 使用 E[x²]-E[x]² 存在大数相减误差；
        v5 使用标准的「先减均值再平方」两趟法，因为在共享内存中读取，多遍历一遍也影响不大。
     3. 不再依赖锯齿循环：共享内存中的数据不会被缓存自动驱逐，随时读取均为低延迟，
        因此无需像 v4 那样通过倒序遍历来提升 L2 缓存命中率。

   代价：共享内存占用大幅增加，需要 cudaFuncSetAttribute 显式申请超过 48KB 的动态共享内存，并提供失败时的回退方案。

   共享内存布局（按 floatX 元素个数计）：
     [s_weight] : C 个元素
     [s_bias  ] : C 个元素
     [s_res   ] : block_y × C 个元素（每个 Warp 一份私有的残差缓存）
     总计：(2 + block_y) × C 个元素
   ---------------------------------------------------------------------------- */
__global__ void fused_forward_kernel5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    extern __shared__ char params[];
    // 共享内存三段式划分（以 x128 包为单位）
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + 2 * (C / x128::size) + threadIdx.y * (C / x128::size);

    // 块内所有线程协作加载 weight / bias（因为 weight/bias 是整个块共享的，所以用整个块一起加载而不是单个 Warp 加载，更多线程并行加载能让访存合并更充分、加载更快）
    int sidx = (threadIdx.y * WarpSize + threadIdx.x) * x128::size;
    for (int i = sidx; i < C; i += blockDim.y * WarpSize * x128::size) {
        s_weight[i / x128::size] = load128(weight + i);
        s_bias[i / x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;  // 全局 warp 索引（每个warp负责一个token）
    if (idx >= N) return; // 越界检查放在共享内存加载之后，保证所有线程都参与了加载
    // 指针偏移到当前 token
    residual += idx * C;
    normed += idx * C;
    inp1 += idx * C;
    inp2 += idx * C;

    const float eps = 1e-5f;
    float sum = 0.0f;
    // 第一趟：残差相加 + 求和 + 缓存到共享内存
    for (int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for (int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
        }
        store128cs(residual + c, out); // 写全局内存（供反向传播使用）
        s_res[c / x128::size] = out;   // 写共享内存（供本 Warp 后续复用）
    }
    sum = warpReduceSum(sum);
    float m = sum / C;

    // 第二趟：从共享内存读，标准公式计算方差
    float v = 0.f;
    for (int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for (int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }
    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    // 第三趟：归一化 + 缩放平移（输入、权重、偏置全从共享内存读）
    for (int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for (int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m);
            out[k] = (floatX)(n * (float)w[k] + (float)b[k]);
        }
        store128cs(normed + c, out);
    }

    if (threadIdx.x == 0) {
        __stcs(mean + idx, m);
        __stcs(rstd + idx, s);
    }
}

void fused_forward5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                    const floatX* inp1, const floatX* inp2,
                    const floatX* weight, const floatX* bias,
                    int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);
    // CUDA 默认单 Block 动态共享内存上限为 48KB，超过需显式申请
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_forward_kernel5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if (status == cudaSuccess) {
        fused_forward_kernel5<<<grid_size, dim3(32, block_y), smem>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                      weight, bias, N, C);
    } else {
        // 回退方案：共享内存申请失败时使用 v4（无需共享内存）
        fused_forward_kernel4<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   版本6：三维 Block + 网格步长循环（Grid‑Stride Loop）（生产级自适应设计）
   ----------------------------------------------------------------------------
   核心变化：从「Warp 与 Token 一一对应」演进为「多个 Warp 组队处理一个 Token，一个 Block 同时处理多组 Token」。

   三维 Block 设计：dim3(32, block_y, block_z)
     - blockDim.x = 32: 一个完整 Warp 的线程数
     - blockDim.y = 单 Token 使用的 Warp 数（由 C 的大小动态决定）
     - blockDim.z = 块内并行 Token 数（由 block_size 和 blockDim.y 推导）

   自适应能力：
     - C 很大（如 4096）时，block_y 增大，让更多 Warp 协作处理一个 Token。
     - C 很小（如 128）时，block_y 减小，block_z 增大，让一个块处理更多 Token。
     - 无论 N 多大，都只启动固定数量的线程块，靠网格步长循环消化。

   Grid-Stride Loop 的意义：
     传统做法按数据量开块，N 越大块越多，N 越小硬件越闲置。
     Grid-Stride 改为「按硬件算力开固定块数」，块数只和 GPU 规格相关：
     块数 num_blocks = cuda_num_SMs * cuda_threads_per_SM / block_size
     这样 SM 占用率恒定，性能波动小，适合生产环境部署。
   ---------------------------------------------------------------------------- */
__global__ void fused_forward_kernel6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                      const floatX* inp1, const floatX* inp2,
                                      const floatX* weight, const floatX* bias,
                                      int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    extern __shared__ char params[];

    // ---------- 共享内存五段式划分（前 3 段是 floatX 类型，后 2 段是 FP32 类型（归约需要更高精度） ----------
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params + C * sizeof(floatX));
    // 残差缓存按 Token 划分，每个 Token 占一份，由 threadIdx.z 标识
    x128* s_res = reinterpret_cast<x128*>(params + 2 * C * sizeof(floatX) + threadIdx.z * C * sizeof(floatX));
    // 每个 Token 自己的均值/方差归约缓冲区，大小固定 32 个 float（对齐 Warp 大小）
    float* s_mean = reinterpret_cast<float*>(params + 2 * C * sizeof(floatX) + blockDim.z * C * sizeof(floatX) + threadIdx.z * 32 * sizeof(float));
    float* s_var  = reinterpret_cast<float*>(params + 2 * C * sizeof(floatX) + blockDim.z * C * sizeof(floatX) + blockDim.z * 32 * sizeof(float) + threadIdx.z * 32 * sizeof(float));

    int cidx = (threadIdx.y * WarpSize + threadIdx.x) * x128::size;  // 元素索引 = 单 Token 内所有线程的一维编号 × 向量化包大小
    int step = blockDim.y * WarpSize * x128::size;  // 循环步长：单 Token 内所有线程一轮能处理的总元素数

    // Block 内所有线程协作加载 weight / bias 到共享内存
    for (int c = cidx; c < C; c += step) {
        s_weight[c / x128::size] = load128(weight + c);
        s_bias[c / x128::size]   = load128(bias + c);
    }
    /* 注：此处省略 __syncthreads() 是安全的，因为：
       加载完 weight/bias 后，代码不会立刻读取共享内存里的它们，
       而是要先执行残差加法、均值归约、方差归约等一大段计算逻辑。
       这段计算指令本身会消耗足够的时钟周期，足以掩盖所有线程完成共享内存写入的延迟。
       等到真正需要读 s_weight / s_bias 时，加载已经完成。 */

    // ---------- Grid-Stride Loop：遍历所有 Token ----------
    for (int tidx = blockIdx.x * blockDim.z + threadIdx.z; tidx < N; tidx += gridDim.x * blockDim.z) {
        // 指针偏移到当前 token
        floatX* residual_bt = residual + tidx * C;
        floatX* normed_bt = normed + tidx * C;
        const floatX* inp1_bt = inp1 + tidx * C;
        const floatX* inp2_bt = inp2 + tidx * C;
        const float eps = 1e-5f;

        // ---------- 第一趟：残差相加 + 求和 ----------
        float sum = 0.0f;
        for (int c = cidx; c < C; c += step) {
            const x128 in1 = load128cs(inp1_bt + c);
            const x128 in2 = load128cs(inp2_bt + c);
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                out[k] = (floatX)((float)in1[k] + (float)in2[k]);
                sum += (float)out[k];
            }
            store128cs(residual_bt + c, out); // 写全局内存（供反向传播使用）
            s_res[c / x128::size] = out;      // 写共享内存（本 Token 后续复用）
        }

        // ---------- 两级规约求均值 ----------
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0) s_mean[threadIdx.y] = sum;
        // 必须同步：接下来所有线程要读 s_mean 里的全部 block_y 个值
        __syncthreads();
        float m = warpReduceSum(threadIdx.x < blockDim.y ? s_mean[threadIdx.x] : 0.f) / C;

        // ---------- 第二趟：从共享内存读，计算方差 ----------
        float v = 0.f;
        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            for (int k = 0; k < x128::size; ++k) {
                v += ((float)res[k] - m) * ((float)res[k] - m);
            }
        }
        v = warpReduceSum(v);
        if(threadIdx.x == 0) {
            s_var[threadIdx.y] = v;
        }
        __syncthreads();
        v = warpReduceSum(threadIdx.x < blockDim.y ? s_var[threadIdx.x] : 0.f) / C;
        float s = rsqrtf(v + eps);

        // ---------- 第三趟：归一化 + 缩放平移 ----------
        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            const x128 w = s_weight[c / x128::size];
            const x128 b = s_bias[c / x128::size];
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                float n = s * ((float)res[k] - m);
                out[k] = (floatX)(n * (float)w[k] + (float)b[k]);
            }
            // 普通存储：LayerNorm 的输出是下一层线性层的输入，很快会被读取，保留在缓存中
            store128(normed_bt + c, out);
        }

        // 缓存均值和标准差倒数（多 Warp 协作时，仅 0 号 Warp 的 0 号线程写入一次）
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            mean[tidx] = (floatX)m;
            rstd[tidx] = (floatX)s;
        }
    }
}

/* 「固定线程块数量 + 网格步长循环（Grid-stride Loop）」架构
 这个设计体现在两个地方：
   1. 主机启动函数：不按数据量计算块数，而是按硬件算力计算出最优的固定块数；
   2. 核函数内部：外层用 gridDim.x * blockDim.z 作为步长，循环遍历所有 Token。

 旧版本（kernel5及之前）：按数据量开块
 const int grid_size = ceil_div(N, block_y);  // kernel5 启动器
 需要的块数 = 总Token数 ÷ 每个块能处理的Token数。
 块数随数据量 N 线性变化：N 越大，开的块越多；N 越小，开的块越少。（N 太小时硬件闲置，N 太大时块数过多，调度开销飙升）

 新版本（kernel6）：按硬件算力开固定块
 const int num_blocks = max(1, cuda_num_SMs * cuda_threads_per_SM / block_size);  // kernel6 启动器
 - cuda_num_SMs：       GPU 的流多处理器（SM）总数
 - cuda_threads_per_SM：每个 SM 能同时驻留的最优线程数
 - block_size：         每个线程块的总线程数
 计算逻辑：整个 GPU 最多能同时跑多少个线程块，就开多少个块。这个数量只和硬件规格、block_size大小有关，和输入 Token 总数 N 无关。
 （比如 RTX 5060 Laptop 算出来最优是 64 个块，那不管 N 是 1000 还是 10000，都只开 64 个块，剩余的 Token 靠循环分批处理）

 这么设计的核心好处
 1. 硬件占用率稳定
    永远只开刚好让 GPU 满载的块数，不会因为 N 太小导致 SM 闲置，也不会因为 N 太大导致块数过多、调度开销暴涨。
 2. 性能可预测
    块数固定，每个 SM 的负载固定，性能波动小，适合生产环境部署。
 3. 适配任意数据量
    无论 Token 数 N 是多少，这套代码都能正常工作，不需要根据输入大小调整启动参数。                                   */
void fused_forward6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                    const floatX* inp1, const floatX* inp2,
                    const floatX* weight, const floatX* bias,
                    int N, int C, const int block_size) {
    int total_warps = block_size / 32;
    // 根据 C 的大小动态计算：单 Token 需要多少 Warp 协作
    int warps_per_token = max(1, C / Packed128<floatX>::size / 32);
    int block_z = max(1, total_warps / warps_per_token); // 块内并行 Token 数
    int block_y = max(1, total_warps / block_z);         // 单 Token 内实际的 Warp 数

    // 共享内存总大小 = 权重 + 偏置 + 残差缓存 + 归约缓冲区（s_mean + s_var，各 32 个 float）
    size_t smem = 2 * C * sizeof(floatX) + block_z * C * sizeof(floatX) + block_z * 64 * sizeof(float);

    // 标准的大共享内存配置流程
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if (status == cudaSuccess) {
        // 按硬件算力计算最优块数，与数据量 N 解耦
        const int num_blocks = max(1, cuda_threads_per_SM * cuda_num_SMs / block_size);
        fused_forward_kernel6<<<num_blocks, dim3(32, block_y, block_z), smem>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                                weight, bias, N, C);
    } else {
        // 回退方案：共享内存申请失败时使用 v4
        const int grid_size = ceil_div(N, total_warps);
        fused_forward_kernel4<<<grid_size, dim3(32, total_warps)>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                    weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Kernel 版本调度分发
// ============================================================================
void fused_forward(int kernel_num, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                   const floatX* inp1, const floatX* inp2,
                   const floatX* weight, const floatX* bias,
                   int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1: fused_residual_forward1(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
        case 2: fused_forward2(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
        case 3: fused_forward3(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
        case 4: fused_forward4(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
        case 5: fused_forward5(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
        case 6: fused_forward6(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
        default:
            printf("无效的内核编号\n");
            exit(1);
    }
}

// ============================================================================
// 主程序：正确性验证 + 性能基准测试 + 性能采集模式
// ============================================================================
int main(int argc, const char **argv) {
    setup_main();
    int B = 8;
    int T = 1024;
    int C = 768;
    //int C = 2048;
    //int C = 4096;

    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 主机端内存分配
    float* residual = (float*)malloc(B * T * C * sizeof(float));
    float* normed = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // GPU 端内存分配与数据迁移
    floatX* d_residual; floatX* d_normed;
    floatX* d_inp1; floatX* d_inp2;
    floatX* d_mean; floatX* d_rstd;
    floatX* d_weight; floatX* d_bias;
    cudaCheck(cudaMalloc(&d_residual, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_normed, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(memcpy_convert(d_inp1, inp1, B * T * C));
    cudaCheck(memcpy_convert(d_inp2, inp2, B * T * C));
    cudaCheck(memcpy_convert(d_weight, weight, C));
    cudaCheck(memcpy_convert(d_bias, bias, C));

    // 如果不是 profile 模式，则执行标准验证与测试流程
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        // 先验证核函数正确性
        residual_forward_cpu(residual, inp1, inp2, B * T * C);
        layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);

        int block_sizes[] = {32, 64, 128, 256, 512, 1024};
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("正在测试线程块大小 %d.\n", block_size);
            cudaCheck(cudaMemset(d_residual, 0, B * T * C * sizeof(floatX)));
            fused_forward(kernel_num, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                          B * T, C, block_size);
            // BF16 容差放宽到 5e-2，FP32 保持 1e-5
            float tol = std::is_same_v<floatX, float> ? 1e-5 : 5e-2;
            validate_result(d_residual, residual, "residual", B * T * C, tol);
            validate_result(d_mean, mean, "mean", B * T, tol);
            validate_result(d_rstd, rstd, "rstd", B * T, tol);
            validate_result(d_normed, normed, "normed", B * T * C, tol);
        }
        printf("所有结果验证通过，开始性能基准测试。\n\n");

        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            int repeat_times = 100;
            float elapsed_time = benchmark_kernel(repeat_times, fused_forward, kernel_num,
                                                  d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                                                  B * T, C, block_size);
            /* 理论总访存字节数（算法层面，非硬件实际访存量）：
               每个 Token 读写：
                 - 读 inp1 : C 个元素
                 - 读 inp2 : C 个元素
                 - 写 residual : C 个元素
                 - 写 normed : C 个元素
                 - 写 mean / rstd : 2 个标量
               合计：4*C + 2 个 floatX 元素
               该值除以执行时间得到「等效带宽」，衡量访存利用效率，越接近显卡峰值带宽越好。 */
            long memory_ops = B * T * (C * 4 + 2) * sizeof(floatX);
            float memory_bandwidth = memory_ops / elapsed_time / 1e6; // 换算为 GB/s
            float toks_per_msec = B * T / elapsed_time / 1e3;         // Token 吞吐率（ktok/ms）

            printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 带宽 %.2f GB/s | 吞吐率 %.2f ktok/ms\n",
                   block_size, elapsed_time, memory_bandwidth, toks_per_msec);
        }

        // 释放资源
        free(residual); free(normed); free(mean); free(rstd); free(weight); free(bias);
        free(inp1); free(inp2);
        cudaCheck(cudaFree(d_residual));
        cudaCheck(cudaFree(d_normed));
        cudaCheck(cudaFree(d_mean));
        cudaCheck(cudaFree(d_rstd));
        cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias));
        cudaCheck(cudaFree(d_inp1));
        cudaCheck(cudaFree(d_inp2));
        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     特别注意：
       版本1（非融合实现）会启动两个独立的 Kernel：residual_forward_kernel1, layernorm_forward_kernel1。
       版本2~6 各启动一个 Kernel。
       因此版本1~6 一轮的 Kernel 启动次数总和是：2 + 1 + 1 + 1 + 1 + 1 = 7 个 Kernel。
       预热 2 轮，总启动次数 = 2 × 7 = 14 个 Kernel。
       采集 1 轮，最后的 7 个 Kernel 对应：v1(2个), v2(1个), v3(1个), v4(1个), v5(1个), v6(1个)。

     编译指令示例：
       nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt fused_forward.cu -o fused_forward

     NCU 采集命令示例（注意跳过 14 次预热启动，采集接下来的 7 次正式启动）：
       ncu --set full --cache-control all --launch-skip 14 --launch-count 7 -o fused_forward_profile_128_768 ./fused_forward 0 profile

     系统级性能分析：
       nsys profile --stats=true -o fused_forward_nsys ./fused_forward 0 profile
     */
    {
        const int profile_block_size = 128;
        const int warmup_rounds = 2;

        // 预热 2 轮：依次调用版本 1 到 6
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= 6; ver++) {
                fused_forward(ver, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias, B * T, C, profile_block_size);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮：依次调用版本 1 到 6
        for (int ver = 1; ver <= 6; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            fused_forward(ver, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias, B * T, C, profile_block_size);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放资源
        free(residual); free(normed); free(mean); free(rstd); free(weight); free(bias);
        free(inp1); free(inp2);
        cudaCheck(cudaFree(d_residual));
        cudaCheck(cudaFree(d_normed));
        cudaCheck(cudaFree(d_mean));
        cudaCheck(cudaFree(d_rstd));
        cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias));
        cudaCheck(cudaFree(d_inp1));
        cudaCheck(cudaFree(d_inp2));
        return 0;
    }
}