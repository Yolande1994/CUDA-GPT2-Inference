/* ============================================================================
   LayerNorm 前向传播的 CUDA Kernel 实现
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward

   版本说明：
     版本1：从 CPU 代码朴素移植，在 B、T 维度并行，C 维度循环遍历。
     版本2：将 LayerNorm 拆分为均值、方差、归一化三个独立 Kernel，全维度并行。
     版本3：使用协作组（Cooperative Groups）在 B、T、C 全维度并行（Warp 级规约）。
     版本4：架构同上，但采用单趟方差公式 var(x) = mean(x²) - mean(x)²，减少一次全局内存遍历。
     版本5：采用 Block 级任务划分（多 Warp 协作处理一行），结合两级规约（Warp内用shuffle + Warp间用共享内存）。
     版本6：高性能实现，128bit 向量化访存 + 用共享内存缓存权重/偏置/输入数据。

   运行命令（数字对应版本号）：
     ./layernorm_forward 1
     ./layernorm_forward 2
     ...
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../common.h"

// ============================================================================
// CPU 参考实现
// ============================================================================
// GPT-2 LayerNorm 前向传播
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // 定位到输入张量的 inp[b,t,:] 位置
            const float* x = inp + b * T * C + t * C;
            // 计算均值
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            // 计算方差（不做无偏校正）
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // 计算标准差的倒数（rstd）
            float s = 1.0f / sqrtf(v + eps);
            // 定位到输出张量的 out[b,t,:] 位置
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m));        // 归一化输出
                float o = n * weight[i] + bias[i]; // 缩放与平移
                out_bt[i] = o;                     // 写入结果
            }
            // 缓存均值与标准差倒数，供后续反向传播使用
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================
// 朴素移植实现：直接将 CPU 逻辑搬入 Kernel，B、T 维度并行，C 维度循环遍历
__global__ void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 const float* inp, const float* weight, const float* bias,
                                 int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;
    if (idx < N) {
        const float* x = inp + idx * C;
        // 计算均值
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // 计算方差（深度学习 LayerNorm 统一不做偏差校正，除以通道总数 C 而不是 C-1）
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // 计算标准差的倒数
        float s = 1.0f / sqrtf(v + eps);
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m));        // 归一化输出
            float o = n * weight[i] + bias[i]; // 缩放与平移
            out_idx[i] = o;                    // 写入结果
        }
        // 缓存均值与标准差倒数
        mean[idx] = m;
        rstd[idx] = s;
    }
}

/* ----------------------------------------------------------------------------
   版本2：三 Kernel 拆分方案设计说明
   ----------------------------------------------------------------------------
   定位：过渡演示版本，逻辑拆分清晰，便于调试，但性能非最优。
   缺点：
     1. 三个 Kernel 串行启动，存在多次启动开销。
     2. 中间结果 mean/rstd 需写回全局内存，下一个 Kernel 再读回，额外增加两次全局内存读写，带宽压力大。

   线程粗化（Thread Coarsening）范式：
     for(i = 线程在工作组内的局部编号; i < 总长度; i += 工作组总线程数)
     - Warp 粒度工作组：工作组大小 = 32，编号 warp.thread_rank()，步长 warp.size()。通信方式：warp shuffle（寄存器互通）。
     - Block 粒度工作组：工作组大小 = block_size，编号 blockIdx.x，步长 block_size。通信方式：共享内存中转数据。

   二分迭代求和规约（Block 级）：
     1. 步长 stride 初始为 block 线程数的一半，每轮循环步长折半，直到 stride=1。
     2. 每轮先同步，保证上一轮的合并写入全部完成。
     3. 只有编号小于 stride 的线程工作：将 shared[tid] 和间隔 stride 位置的 shared[tid+stride] 累加合并。迭代完成后，shared[0] 即为当前行所有 C 个元素的总和。

   与 Kernel3 的对比：
     Kernel2：Block 级任务划分，1 个 Block 处理 1 行，
              块内多个 Warp 协作处理同一行，汇总结果必须跨 Warp 通信，须经过共享内存。并伴随 __syncthreads() 同步，开销较大。
     Kernel3：Warp 级任务划分，1 个 Warp 处理 1 行，
              warp 内通信直接用硬件寄存器原语（如 __shfl_down_sync），无需共享内存与 __syncthreads()，延迟极低。
     这也是 Kernel2 性能弱于 Kernel3 的原因之一：多 Kernel 拆分 + 多次全局读写 + Block 共享内存同步开销。
   ---------------------------------------------------------------------------- */
__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[]; // extern：声明动态大小的共享内存数组
    int idx = blockIdx.x;             // 取值范围 [0, B*T)
    int tid = threadIdx.x;            // 取值范围 [0, block_size)
    const float* x = inp + idx * C;   // 当前 block 负责 token 的通道起始地址
    // 线程粗化：block 内 block_size 个线程分摊整行 C 个元素
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum; // 将每个线程的部分和写入共享内存
    __syncthreads();
    // 二分迭代求和规约
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {  // 编号小于stride的前一半线程干活，后一半线程休息
            shared[tid] += shared[tid + stride];  // 把后半组所有数据，并行合并到前半组里
        }
    }
    // 将最终结果（由 0 号线程）写入全局内存
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = inp + idx * C;
    float m = mean[idx];
    // 线程粗化
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // 规约运算
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // 将最终结果写入全局内存
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}


/* ----------------------------------------------------------------------------
   版本3：Warp 级任务划分与两趟法
   ----------------------------------------------------------------------------
   任务划分：1 个 warp 处理 1 个 Token。
   规约方式：使用 Cooperative Groups 协作组做 warp 内规约。
   计算方式：两次遍历通道完成均值、方差计算。
   访存优化：引入流式访存 __ldcs/__stcs 规避缓存污染。
   ---------------------------------------------------------------------------- */
__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size 为一个线程块内的线程束数量，meta_group_rank 为当前线程束的索引
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // 全局 warp 编号，以 warp 为最小任务单元
    if(idx >= N) {
        return;
    }
    const float* x = inp + idx * C;  // 定位当前 warp 负责处理的输入行

    // 均值计算
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{}); // 底层靠硬件 warp shuffle 实现，线程直接互读寄存器，无需共享内存与__syncthreads
    float m = sum / C;  // sum 已经广播到所有线程，每个线程都能独立算出均值
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);  // __stcs(全局显存地址，值)
    }

    // 标准差倒数计算
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);  // rsqrtf()：硬件原生平方根倒数指令，比 1.0f / sqrtf(...) 更快，是高性能算子标准写法
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // 最终归一化，缩放平移
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m); // 流式加载，不驻留缓存，让权重与偏置获得更高的缓存命中率
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

/* ----------------------------------------------------------------------------
   版本4：用公式单趟求出方差，减少一次全局内存遍历（有代价）
   ----------------------------------------------------------------------------
   优化思路：使用公式 var(x) = mean(x²) - mean(x)² 计算方差，只需单次遍历读取输入数据。

   代价是数值稳定性问题：
     当输入数值很大、但方差很小时，计算误差会变大，极端情况会算出负数方差，开根号直接出 NaN。
     直观例子：假设所有元素都是 10000.1，均值就是 10000.1，真实方差非常小。
       - 版本3方法（先减均值再平方）：减去均值得到 0.1 左右的小数，平方后累加，数值很小，误差极小。
       - 版本4方法（E[x²]-E[x]²）：元素平方是 100002000.01 这种大数，平方均值和均值平方都是上亿的大数，两个大数相减，微小精度误差被放大，算出的方差不准，甚至可能为负。

   工程选型建议：
     - 训练场景：一般用版本3这种“先减均值再平方”的两趟法，训练对精度要求高，数值不稳定会影响收敛。
     - 推理场景：大多可以接受版本4，推理对精度损失感知不明显，性能收益更重要。
     - 改进方案：偏移法（减去近似中心值缩小数值范围，零额外开销）或并行 Welford 算法（精度接近两趟法，适合极高精度场景）。
   ---------------------------------------------------------------------------- */
__global__ void layernorm_forward_kernel4(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }
    const float* x = inp + idx * C;

    // 逐行线程粗化遍历，串行累加求和
    float sum = 0.0;  // 存储 sum(x)
    float sum2 = 0.0; // 存储 sum(x²)
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // 末尾执行线程束级规约
    sum = cg::reduce(warp, sum, cg::plus<float>{});   // sum(x)
    sum2 = cg::reduce(warp, sum2, cg::plus<float>{}); // sum(x²)
    sum /= C;  // mean(x)  均值
    sum2 /= C; // mean(x²) 平方均值

    // 计算均值、方差、标准差倒数
    float m = sum;
    float var = sum2 - sum * sum; // 套公式：方差 = 平方的均值 - 均值的平方
    float s = rsqrtf(var + 1e-5f);
    // 写入均值
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // 写入标准差倒数
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // 最终归一化，以及权重、偏置的缩放平移
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

/* ----------------------------------------------------------------------------
   版本5：Block 级任务划分与两级规约
   ----------------------------------------------------------------------------
   特点：每个 block 处理一行数据，而非单个 warp 线程束处理一行，适配更大通道维度 C，充分利用 Block 内更多线程。

   核心亮点：两级规约
     这是版本5最精髓的优化。
     - 版本2：纯共享内存二分规约，全程走共享内存，同步多、延迟高。
     - 版本3/4：纯 warp shuffle 规约，全程走寄存器，速度快，但只能 32 个线程协作。
     - 版本5：两级规约 = warp 内 shuffle 规约 + warp 间共享内存中转 + 二次 warp 规约。大部分操作走寄存器，只在 warp 之间走一次共享内存。

   实现步骤：
     1. 每个 warp 先内部规约，算出自己 warp 的局部总和。
     2. 所有 warp 把各自的局部总和写入共享内存对应位置。
     3. 每个 warp 都从共享内存里，把所有 warp 的结果全部读进自己内部的线程寄存器里，数量不够的位置用 0 补齐。
     4. 每个 warp 独立在自己内部做第二次规约，都能得到整个 block 的最终总和。

   为什么超出范围的线程要补 0.0f？
     共享内存里只有前 num_warps 个位置是有效数据，后面的都是未初始化的垃圾值，补 0 是为了避免把脏数据加进结果里。
   为什么不让多余线程直接跳过？
     因为 warp 规约是 32 个线程全员参与的硬件指令，用 if 让部分线程跳过会导致 warp 内分支发散，补 0 是最干净高效的做法。

   标准手写版 warp 规约：
     __device__ float warp_reduce_sum(float val) {
         for (int stride = 16; stride > 0; stride >>= 1) {
             val += __shfl_down_sync(0xffffffff, val, stride);
         }
         return val;
     }
     __shfl_down_sync 是硬件原生指令，线程直接读取前面间隔 stride 位置的线程寄存器值，无需共享内存中转，延迟极低。
   ---------------------------------------------------------------------------- */
__global__ void layernorm_forward_kernel5(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32];  // 只存每个 warp 的最终结果（block 最大 1024 线程，最多 32 个 warp，32 个位置刚好够）
    __shared__ float shared_sum2[32];
    int num_warps = blockDim.x / 32;  // 整个 block 里共有多少个 warp
    int warp_id = threadIdx.x / 32;   // 当前线程属于第几个 warp
    int lane_id = threadIdx.x % 32;   // 当前线程在自己 warp 里的编号（0~31）
    int idx = blockIdx.x;             // 当前处理的 token 索引（1 个 block 只处理 1 行）
    const float* x = inp + idx * C;
    
    float thread_sum = 0.0;  // 存储 sum(x)
    float thread_sum2 = 0.0; // 存储 sum(x²)
    // 线程粗化累加：更多线程分摊同一行
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }

    // 一、线程束级规约
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});   // sum(x)
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{}); // sum(x²)
    // 将线程束规约结果存入共享内存（warp 内所有线程的值完全一样，写相同值不会错，不用分支判断）
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // 从共享内存加载结果到线程，超出范围的线程补零
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;

    // 二、对所有线程束的规约结果做二次规约（每个线程里存的是所有 warp 的总和，规约完就得到整个 block 的最终总和）
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});   // sum(x)
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x²)
    // 计算均值、方差、标准差倒数
    float m = block_sum / C;            // mean(x)
    float var = block_sum2 / C - m * m; // mean(x²)
    float s = rsqrtf(var + 1e-5f);
    // 写入均值与标准差倒数
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // 最终归一化，以及权重、偏置的缩放平移
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

/* ----------------------------------------------------------------------------
   版本6：共享内存向量化
   ----------------------------------------------------------------------------
   核心优化：128bit 向量化访存 + 共享内存缓存权重/偏置/输入，以 x128 包为单位循环。
   任务划分：1 个 warp 处理 1 行数据，同 block 内多个 warp 共享权重偏置参数。
   精度特性：两趟法计算方差，数值稳定性好，兼顾性能与精度。

   二维 Block 设计（一维 grid + 二维 block）：
     <<< grid, (32, block_y) >>>
     - blockDim.x：固定 32 个线程（一个完整 warp）。
     - blockDim.y：一个 block 里有几个 warp。
     - threadIdx.x：线程在 warp 内的编号（0~31）。
     - threadIdx.y：warp 在 block 内的编号。

   共享内存大小计算示例：
     假设 C=768 通道，FP32，x128一包 4个float，一包 = 16 字节。
     整行需要 768/4 = 192 个 x128 包。
     block 内有 4 个 warp（blockDim.y = 4）。
     weight: 192 包
     bias:   192 包
     输入缓存: 4 × 192 = 768 包
     总包数 = 192 + 192 + 768 = 1152 包
     总字节 = 1152 × 16 = 18432 字节。
     启动 Kernel 时，第三个参数传入 18432。

   设计参考自 fused_residual_forward.cu 中的 fused_residual_forward_kernel5，并在其基础上统一以 x128 包为单位，消除重复除法
   ---------------------------------------------------------------------------- */
__global__ void layernorm_forward_kernel6(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    assert(blockDim.x == WARP_SIZE);  // 强制约束：block 的 x 维度必须是完整的 32 线程（1个warp）
    extern __shared__ char params[];  // char：1字节最小存储单位，方便偏移计算
    const int packs = C / x128::size; // 每行数据对应的 x128 包的数量
    // 动态共享内存三段划分
    x128* s_weight = reinterpret_cast<x128*>(params);  // 存权重
    x128* s_bias   = s_weight + packs;                 // 存偏置
    x128* s_inp    = s_weight + (2 + threadIdx.y) * packs;  // 存每个 warp 私有的输入

    // --------- 全 block 合力加载权重/偏置到共享内存（从 DRAM 读1次，后面流程复用） ---------
    // 整个 block 所有线程排成一队，按线程 ID 跨步循环，保证所有线程的访问地址是连续的
    int tidx = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int p = tidx; p < packs; p += blockDim.y * WARP_SIZE) {
        s_weight[p] = load128(weight + p * x128::size);
        s_bias[p]   = load128(bias   + p * x128::size);
    }
    __syncthreads();

    // --------- 计算当前 warp 负责的全局行号 ---------
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= N) { return; }
    inp = inp + idx * C;
    out = out + idx * C;
    const float eps = 1e-5f;
    float sum = 0.0f;

    // --------- 第一趟：逐元素累加求和 + 输入缓存到共享内存 ---------
    for (int p = threadIdx.x; p < packs; p += WARP_SIZE) {
        const x128 in_data = load128cs(inp + p * x128::size);
        for (int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_inp[p] = in_data; // 顺手把刚读的输入数据存到共享内存，后面直接复用
    }
    // warp 内规约算均值
    sum = warpReduceSum(sum);
    float m = sum / C;

    // --------- 第二趟：读共享内存算方差（两趟法，数值稳定） ---------
    float v = 0.0f;
    for (int p = threadIdx.x; p < packs; p += WARP_SIZE) {
        const x128 in_data = s_inp[p];  // 取出缓存的输入数据
        for (int k = 0; k < x128::size; ++k) {
            float diff = (float)in_data[k] - m;
            v += diff * diff;
        }
    }
    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    // --------- 第三趟：归一化 + 仿射变换 + 写回全局内存 ---------
    // 输入、权重、偏置全从共享内存读取，全程不碰全局内存。最后一次性向量化写回全局内存
    for (int p = threadIdx.x; p < packs; p += WARP_SIZE) {
        const x128 in_data = s_inp[p];
        const x128 w       = s_weight[p];
        const x128 b       = s_bias[p];
        x128 out_data;
        for (int k = 0; k < x128::size; ++k) {
            float normed = s * ((float)in_data[k] - m);
            out_data[k] = normed * (float)w[k] + (float)b[k];
        }
        store128cs(out + p * x128::size, out_data);
    }

    // --------- 写入当前行的均值与标准差倒数 ---------
    if (threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    if (threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
}

// ============================================================================
// Kernel 启动封装函数
// ============================================================================
// 版本1：朴素单线程版
void layernorm_forward1(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;  // 单线程处理一整行，总任务量就是 N 行
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// 版本2：三 Kernel 拆分版
void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // 前两个内核：每个 block 处理一行，分配共享内存做 block 级规约
    mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaCheck(cudaGetLastError());
    // 归一化阶段改成元素级全并行，每个线程只处理 1 个输出元素
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

// 版本3：warp 级两趟法
void layernorm_forward3(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// 版本4：warp 级一趟法（用 E[x²]-E[x]² 公式）
void layernorm_forward4(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel4<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// 版本5：block 级两级规约版（适合通道数 C 很大的场景）
void layernorm_forward5(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    assert(block_size <= 1024);
    const int N = B * T;
    const int grid_size = N;
    layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// 版本6：共享内存向量化工业级
void layernorm_forward6(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(float);

    // 手动申请更大的动态共享内存上限（CUDA 默认单 block 动态共享内存上限是 48KB）
    auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if (status == cudaSuccess) {
        layernorm_forward_kernel6<<<grid_size, dim3(32, block_y), smem>>>(out, mean, rstd, inp, weight, bias, N, C);
    } else {  // 调用失败时自动回退到不使用共享内存的实现版本
        const int grid_size = N;
        layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

// Kernel 版本调度分发
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1: layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size); break;
        case 2: layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size); break;
        case 3: layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size); break;
        case 4: layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size); break;
        case 5: layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size); break;
        case 6: layernorm_forward6(out, mean, rstd, inp, weight, bias, B, T, C, block_size); break;
        default:
            printf("无效的内核编号\n");
            exit(1);
    }
}

// ============================================================================
// 主程序：正确性验证 + 性能基准测试 + 性能采集模式
// ============================================================================
int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    //int C = 4096;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // 在主机端创建随机数内存
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // 数据迁移到 GPU 端
    float* d_out;   float* d_mean;  float* d_rstd;
    float* d_inp;   float* d_weight; float* d_bias;
    // 申请 GPU 显存
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    // 数据从 CPU 内存拷贝到 GPU 显存
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // 从命令行选择要测试的 kernel 版本
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 如果不是 profile 模式，则执行标准验证与测试流程
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        int block_sizes[] = {32, 64, 128, 256, 512, 1024};
        // 运行 CPU 参考实现
        layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

        // 遍历所有线程块大小，逐一验证 GPU 计算结果和 CPU 结果是否一致
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("正在测试线程块大小 %d.\n", block_size);
            layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
            validate_result(d_out, out, "out", B * T * C, 1e-5f);
            validate_result(d_mean, mean, "mean", B * T, 1e-5f);
            validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
        }
        printf("所有结果验证通过，开始性能基准测试。\n\n");

        // 测试不同线程块大小下的 kernel 耗时
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            int repeat_times = 100;
            float elapsed_time = benchmark_kernel(repeat_times, layernorm_forward,
                                                  kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                                  B, T, C, block_size);
            // 粗略估算显存带宽：总访存字节数 / 耗时
            long memory_ops = (2 * B * T * C) * 4;  // 读1次输入 + 写1次输出，每个 float 占 4 字节
            float memory_bandwidth = memory_ops / elapsed_time / 1e6;

            printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 带宽 %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
        }

        // 释放内存
        free(out); free(mean); free(rstd); free(inp); free(weight); free(bias);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_mean));
        cudaCheck(cudaFree(d_rstd));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias));

        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     特别注意：
       版本2（三 Kernel 拆分方案）会启动三个独立的 Kernel：mean_kernel, rstd_kernel, normalization_kernel
       所以版本1~6 一轮的 Kernel 启动次数总和是：1+3+1+1+1+1 = 8 个 Kernel。
       预热 2 轮，总启动次数 = 2 × 8 = 16 个 Kernel。
       采集 1 轮，最后的 8 个 Kernel 对应：v1(1个), v2(3个), v3(1个), v4(1个), v5(1个), v6(1个)。

     编译指令示例：
       nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward

     NCU 采集命令示例（注意跳过 16 次预热启动，采集接下来的 8 次正式启动）：
       ncu --set full --cache-control all --launch-skip 16 --launch-count 8 -o layernorm_profile_128 ./layernorm_forward 0 profile

     系统级性能分析：
       nsys profile --stats=true -o layernorm_nsys ./layernorm_forward 0 profile
     */
    {
        const int profile_block_size = 128;
        const int warmup_rounds = 2;

        // 预热 2 轮：依次调用版本 1 到 6
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= 6; ver++) {
                layernorm_forward(ver, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, profile_block_size);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮：依次调用版本 1 到 6
        for (int ver = 1; ver <= 6; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            layernorm_forward(ver, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, profile_block_size);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放内存
        free(out); free(mean); free(rstd); free(inp); free(weight); free(bias);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_mean));
        cudaCheck(cudaFree(d_rstd));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias));

        return 0;
    }
}