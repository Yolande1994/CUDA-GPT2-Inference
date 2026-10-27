/* ============================================================================
   Softmax 前向传播 CUDA Kernel 实现
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -lcublas -lcublasLt softmax_forward.cu -o softmax_forward

   算子说明：
     对输入矩阵的每一行独立做 Softmax 归一化，将一行任意实数转换为总和为 1、取值在 0~1 之间的概率分布。
     输入 inp 维度为 (N, C)，输出 out 维度为 (N, C)。
     其中 N = B*T（总 token 数），C 为特征维度。

   数值稳定性：
     Softmax 需要计算 exp(x)，若 x 很大（如超过 88）会溢出为 inf。
     标准做法是先减去行内最大值 maxval 再算指数，数学上等价但能避免上溢。

   版本说明：
     版本1：朴素移植（B、T 并行，C 串行循环）
     版本2：融合内核，B、T、C 全维度并行（共享内存二分规约）
     版本3：单 Warp 版，纯 Warp Shuffle 规约（必须 block_size=32）
     版本4：块级两级规约（Warp 内 shuffle + 跨 Warp 共享内存），支持 32 的倍数
     版本5：高性能实现（块级两级规约 + 循环展开 + 读写分离 + 缓存优化）
     版本6：在线 Softmax 朴素移植（单线程处理一行，C 串行）
     版本7：在线 Softmax + 协作组（一个 Warp 处理一行）
     版本8：在线 Softmax + 手写 Shuffle 规约（透明展示版）
     版本9：在线 Softmax + 块级两级规约（一个 Block 处理一行）

   运行命令：
     ./softmax_forward 1
     ./softmax_forward 2
     ...
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../common.h"

// ============================================================================
// CPU 参考实现
// ============================================================================

/* 标准 Softmax CPU 参考实现（三趟遍历）
   逐行处理：先求行内最大值 → 计算指数并求和 → 归一化。
   注意：使用 double 累加求和，保证参考结果的数值精度。 */
void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        // 第一趟：求行内最大值
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) maxval = inp_row[j];
        }
        // 第二趟：计算 exp(x - max) 并累加
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        // 第三趟：归一化
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

/* 在线 Softmax CPU 参考实现（基于论文《Online normalizer calculation for softmax》）
   核心思想：将「求最大值」和「求指数和」合并到一次遍历中完成。
   遍历过程中动态维护 (maxval, sum)：
     1. 若新元素 > 当前 maxval，更新 maxval，并把旧 sum 按 exp(旧max-新max) 折算
     2. 否则直接把 exp(新元素 - maxval) 累加到 sum

   数学等价性：全程只有加法和乘法，没有相消误差，浮点误差与标准版本几乎一致。
   相比标准三趟法，访存从 3 次降为 2 次（1读1写），在大 C（如词表 50k）场景下性能显著提升。 */
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        // 第一趟：同时更新最大值与求和
        float maxval = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            if (inp_row[j] > maxval) {  // 情况1：新元素大于当前最大值
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            } else {                    // 情况2：新元素不大于当前最大值
                sum += expf(inp_row[j] - maxval);
            }
        }
        // 第二趟：归一化
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   版本1：朴素移植（B、T 维度并行，C 串行循环）
   ----------------------------------------------------------------------------
   定位：最直接的 CPU 代码移植版本，作为性能基线（Baseline）。
   问题：每个线程独立处理一整行 C 个元素，导致：
     1. 线程级并行度严重不足（只有 N = B*T 个线程）
     2. 相邻线程访问地址间隔 C 个元素，合并访存失效
     3. 单线程串行遍历 C 次，长依赖链严重拖慢执行
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) maxval = inp_row[j];
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

/* ----------------------------------------------------------------------------
   版本2：融合内核，B、T、C 全维度并行（共享内存二分规约）
   ----------------------------------------------------------------------------
   核心改进：
     采用「1 个 Block 处理 1 行」的并行策略，块内所有线程协作处理同一行 C 个元素。
     通过 Thread Coarsening（线程粗化）让每线程负责 C/block_size 个元素，
     相邻线程访问连续地址，完美触发合并访存。

   规约方式：
     使用经典的共享内存二分规约。每轮 step 折半，只有编号小于 step 的线程干活。
     需要多次 __syncthreads() 保证跨 Warp 通信正确。

   性能瓶颈：
     共享内存读写 + 多次块级同步开销较大。支持任意 block_size，适合中等 C 场景。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel2(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[]; // 动态共享内存：大小 = block_size * sizeof(float)
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* x = inp + bid * C;

    // 一、求最大值（线程粗化 + 共享内存二分规约）
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();  // 每轮 stride 都需要同步
    }
    __syncthreads();// 可删
    float offset = shared[0];

    // 二、计算指数并写回全局内存
    for (int i = tid; i < C; i += block_size) {
        out[bid * C + i] = expf(x[i] - offset);
    }
    __syncthreads();

    // 三、求指数和（复用共享内存 + 二分规约）
    x = out + bid * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;  // 复用共享内存
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();// 可删
    float sum = shared[0];

    // 四、归一化写回
    for (int i = tid; i < C; i += block_size) {
        out[bid * C + i] = x[i] / sum;
    }
}

// ----------------------------------------------------------------------------
// Warp 规约函数（两种风格对比）
// ----------------------------------------------------------------------------
/* Warp 内最大值规约：基于 __shfl_down_sync（向下规约）
   核心 API：__shfl_down_sync(mask, var, offset)
     线程 i 读取同一 Warp 内线程 i+offset 的 var 值。
   执行完后只有 lane 0 持有完整的 Warp 最大值，其他线程的 val 是中间结果，
   需要额外用 __shfl_sync 广播才能让所有线程拿到。 */
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/* Warp 内求和规约：基于 __shfl_xor_sync（蝴蝶规约）
   核心 API：__shfl_xor_sync(mask, var, offset)
     线程 i 读取线程 i^offset 的 var 值，两两成对交换。
   执行完后 Warp 内所有线程都持有完整的 Warp 总和，无需额外广播。*/
__device__ float warpReduceSum_v2(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/* ----------------------------------------------------------------------------
   版本3：单 Warp 版（固定 blockDim.x = 32，纯 Warp Shuffle 规约）
   ----------------------------------------------------------------------------
   核心改进：
     将任务划分缩小到「1 个 Warp 处理 1 行」，块大小固定为 32（一个完整 Warp）。
     全程使用 Warp Shuffle 指令完成规约，完全没有共享内存读写和块级同步。

   与 v2 对比：
                    v2（共享内存块级规约）         v3（单 Warp 洗牌规约）
     块大小          灵活，32~1024 均可            固定 32
     规约载体        共享内存                      寄存器（Warp Shuffle）
     同步操作        多次 __syncthreads()         无显式同步（Warp 硬件天然锁步）
     共享内存占用     block_size × 4 字节          无
     并行度          高，适合大 C                  低，适合小 C
     延迟            较高（共享内存 + 同步）        更低（纯寄存器通信）

   适用场景：小维度 C（32 线程循环次数少），延迟优势最明显。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = inp + bid * C;

    // 1. 线程粗化遍历 + Warp 内规约，求行全局最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);
    // 此时只有 lane 0 持有完整的 Warp 最大值，需广播给所有线程
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);

    // 2. 计算指数项并写入全局内存
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = expf(x[i] - offset);
    }

    // 3. 线程粗化 + Warp 规约，求指数和
    x = out + bid * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);  // 基于__shfl_xor_sync特性，计算后所有线程结果一致，无需广播

    // 4. 归一化输出
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = x[i] / sumval;
    }
}

/* ----------------------------------------------------------------------------
   版本4：两级规约（Warp 内 Shuffle + 跨 Warp 共享内存）
   ----------------------------------------------------------------------------
   核心改进：
     结合 v2 的「支持任意 32 倍数 block_size」和 v3 的「Warp Shuffle 高效规约」，
     采用两级规约架构：
       第一级（Warp 内）：32 线程用 Shuffle 指令完成寄存器级规约，延迟极低。
       第二级（跨 Warp）：各 Warp 的 0 号线程把结果写入共享内存，由 block 内 0 号线程串行合并。

   共享内存用量对比：
     v2：block_size 个 float（每个线程都要写）
     v4：warpsPerBlock 个 float（每个 Warp 只写 1 个）
     例：block_size=1024 时，v2 需 1024 个 float，v4 仅需 32 个。
     共享内存占用降低 → SM 可驻留更多 Block → 硬件占用率提升。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[]; // 大小 = warpsPerBlock * sizeof(float)
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;  // 块内 warp 编号
    int laneId = threadIdx.x % 32;  // warp 内线程编号
    int warpsPerBlock = blockDim.x / 32;  // 块内 warp 总数
    const float* x = inp + bid * C;

    // ---------- 求行全局最大值 ----------
    // 线程粗化：每线程算自己的局部最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // 第一级：Warp 内 shuffle 规约
    maxval = warpReduceMax(maxval);
    // 每个 Warp 的 0 号线程把本 Warp 结果写入共享内存
    if (laneId == 0) shared[warpId] = maxval;
    __syncthreads();
    // 第二级：block 内 0 号线程串行合并所有 Warp 的结果
    if (tid == 0) {
        float val = shared[0];            // 1.先拿第 0 个 Warp 的最大值当初始值
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, shared[i]);  // 2.循环遍历剩下所有 Warp 的最大值，逐个比较取最大
        }
        shared[0] = val;                  // 3.最后把整个块的全局最大值，存回共享内存的第 0 位
    }
    __syncthreads();
    float offset = shared[0]; // 广播给所有线程

    // 计算指数并写回
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = expf(x[i] - offset);
    }

    // ---------- 求指数和（复用同一套两级规约框架） ----------
    x = out + bid * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);
    if (laneId == 0) shared[warpId] = sumval;
    __syncthreads();
    if (tid == 0) {
        float val = shared[0];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += shared[i];
        }
        shared[0] = val;
    }
    __syncthreads();
    float sum = shared[0];

    // ---------- 归一化写回 ----------
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = x[i] / sum;
    }
}

/* ----------------------------------------------------------------------------
   版本5：高性能实现（v4 全面优化版）
   ----------------------------------------------------------------------------
   核心架构：1 Block 处理 1 行，两级规约（与 v4 一致）。

   相对 v4 的 6 项关键优化：
     1. 循环展开：UNROLL_FACTOR=8，减少循环控制指令，提升指令级并行，隐藏访存延迟，访存更连续。
     2. 计算合并：计算 exp 写回的同时，寄存器内直接累加局部 sum，省去一整趟读取 exp 结果的全局内存访问。
     3. 共享内存分区：maxvals / sumvals 独立划分，逻辑清晰，无覆盖风险。
     4. 缓存优化：__ldcs 流式加载输入，避免污染缓存。
     5. 边界完善：读操作用 min 钳位保证性能，写操作加 if 判断保证安全。
     6. 读写分离：先批量加载、再集中计算，让编译器有充足的指令调度空间，用计算时间隐藏访存延迟。

   整体仍为 3 趟流程，但比 v4 少 1 次全局内存读取（省掉重新读 exp 结果的那趟）。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel5(float* out, const float* inp, int N, int C) {
    const int UNROLL_FACTOR = 8;               // 循环展开因子
    const int warpsPerBlock = blockDim.x / 32; // 块内 warp 数量
    extern __shared__ float shared[];  // 大小 = 2 * warpsPerBlock * sizeof(float)
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    float* maxvals = shared;                 // 前半段：存每个 Warp 的局部最大值
    float* sumvals = &shared[warpsPerBlock]; // 后半段：存每个 Warp 的局部指数和

    // 越界线程兜底：写入规约初始值后退出（大维度时多余）
    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }
    const float* x = inp + bid * C; // 输入
    float*       y = out + bid * C; // 输出

    // ---------- 第一趟：求最大值（8 倍循环展开 + min 地址钳位） ----------
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            // 读操作只做 min 钳位：越界索引卡到最后一个元素，重复读不影响最大值结果
            maxval = fmaxf(maxval, x[min(C - 1, i + u * blockDim.x)]);
        }
    }
    maxval = warpReduceMax(maxval);
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // 跨 Warp 合并
    if (tid == 0) {
        float val = maxvals[0];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    // ---------- 第二趟：计算 exp + 写回 + 寄存器内局部求和（三合一） ----------
    // 相比 v4 省掉了一趟「重新读 exp 结果求 sum」的全局内存访问
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        // 阶段 1：批量加载到寄存器（提高访存合并度，掩盖访存延迟）
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u * blockDim.x)]);
        }
        // 阶段 2：批量计算、写回、累加
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u * blockDim.x < C) { // 写操作必须严格判断越界（越界写将破坏其他行数据）
                float output = expf(reg_array[u] - offset);
                y[i + u * blockDim.x] = output;
                sumval += output; // 在寄存器内累加，避免再读一遍全局内存
            }
        }
    }
    // ---------- 指数和的两级规约 ----------
    sumval = warpReduceSum(sumval);
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    if (tid == 0) {
        float val = sumvals[0];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];

    // ---------- 第三趟：归一化写回（读写分离，计算时间掩盖访存延迟） ----------
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u * blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u * blockDim.x < C) {
                y[i + u * blockDim.x] = reg_array[u] / sum;
            }
        }
    }
}

// ============================================================================
// 在线 Softmax 相关结构与内核
// ============================================================================

/* ----------------------------------------------------------------------------
   版本6：在线 Softmax 朴素移植（B、T 并行，C 串行）
   ----------------------------------------------------------------------------
   与 v1 同为「单线程处理一行」的模式，但使用在线算法将访存量从 3 次读
   降为 2 次（1读输入 + 1写输出），没有 exp 中间值的全局内存中转。

   逐元素串行遍历，动态维护 (maxval, sum)：
     - 若当前元素刷新最大值，旧 sum 需按 exp(旧max-新max) 打折
     - 否则直接累加 exp(当前元素 - maxval)

   注：单线程模式下 expf 的串行延迟完全暴露，后续 v7/v8 改为 Warp 级并行缓解。
   ---------------------------------------------------------------------------- */
__global__ void online_softmax_forward_kernel6(float* out, const float* inp, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* inp_row = inp + i * C;
    float* out_row = out + i * C;
    float maxval = -INFINITY;
    // 一趟遍历同时求出全局最大值和指数总和
    double sum = 0.0;  // 维度C很大时，双精度可显著提升数值稳定性
    for (int j = 0; j < C; j++) {
        float maxval_prev = maxval;
        float current_val = inp_row[j];
        if (current_val > maxval) {
            maxval = current_val;
            sum = sum * expf(maxval_prev - maxval) + expf(current_val - maxval);
        } else {
            sum = sum + expf(current_val - maxval);
        }
    }
    for (int j = 0; j < C; j++) {
        out_row[j] = expf(inp_row[j] - maxval) / sum;
    }
}

/* 结构体 SumMax：将「局部最大值」和「基于该最大值的指数和」打包。
   设计为 8 字节对齐（两个 float），方便后续通过 warp shuffle 一次性交换。 */
struct __align__(8) SumMax {
    float maxval;
    float sum;
};

/* 在线合并算子：合并两个 SumMax 对 (a, b)，得到统一基准下的新 SumMax。
   核心逻辑：
     1. 选出更大的 maxval 作为合并后的基准
     2. 较小基准下的 sum 需乘以 exp(smaller_max - bigger_max) 折算到新基准
     3. 两个 sum 相加得到合并后的总指数和 */
__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}

/* ----------------------------------------------------------------------------
   版本7：在线 Softmax + 协作组（一个 Warp 处理一行）
   ----------------------------------------------------------------------------
   核心改进：
     任务划分从「一个线程一行」升级为「一个 Warp 一行」，充分利用并行度，同时合并访存。

   与 v4（两级规约块级实现）的对比：
                      v4（块级两级规约）            v7（Warp 级在线）
     处理一行的单位    整个 Block 协作              单个 Warp（32 线程）
     一个 Block 处理   1 行                        blockDim.x/32 行
     块内 Warp 关系   互相协作（共享内存通信）       完全独立，各算各行
     共享内存          必须有                       无
     块级同步          多次 __syncthreads()         无
     适用场景          C 很大                       C 不太大，一个 Block 可并行处理多行。

   细节：{ x[i], 1.0f } 表示单元素对应的 SumMax
     - 单元素的最大值 = 自身
     - 单元素的指数和 = exp(x[i] - x[i]) = 1.0f
   ---------------------------------------------------------------------------- */
__global__ void online_softmax_forward_kernel7(float* out, const float* inp, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) { return; }
    const float* x = inp + idx * C;

    // 每个线程的局部 SumMax
    SumMax sm_partial;
    sm_partial.maxval = -INFINITY;
    sm_partial.sum = 0.0f;

    // 线程粗化遍历，逐元素合并到局部 SumMax
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
    }

    // Warp 内规约：使用 reduce_sum_max_op 作为合并算子传入
    SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);

    // 归一化写回
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        __stcs(out + idx * C + i, expf(x[i] - sm_total.maxval) / sm_total.sum);
    }
}

/* ----------------------------------------------------------------------------
   版本8：在线 Softmax + 手写 Shuffle 规约（教学透明版）
   ----------------------------------------------------------------------------
   与 v7 做的事完全一样，性能基本一致。区别在于：
    v7 用的是官方库里 cg::reduce 规约，看不到 shuffle 的具体步骤；
    v8 手写整个 shuffle 规约循环，每一步 __shfl_down_sync、折算、累加的过程摆在明面上。
   价值：
    1. 展示在线 Softmax 的标准公式 sumval = sumval*exp(maxval-bigger) + exp(x[i]-bigger)，
       这个公式会在 Flash Attention 里反复出现。
     2. 理解 cg::reduce 底层做了什么。
   ---------------------------------------------------------------------------- */
__global__ void online_softmax_forward_kernel8(float* out, const float* inp, int N, int C) {
    const int warpsPerBlock = blockDim.x / warpSize;
    int tid = threadIdx.x;
    if (tid >= C) { return; }
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // 全局行号 = 块编号 × 每块Warp数 + Warp的块内编号
    int row = blockIdx.x * warpsPerBlock + warpId;
    if (row >= N) { return; }

    const float* x = inp + row * C;
    float* const y = out + row * C;

    // 主循环：单趟遍历同时维护 maxval 和 sumval（在线 Softmax 标准公式）
    float maxval = -INFINITY, sumval = 0.0f, bigger;
    for (int i = laneId; i < C; i += warpSize) {
        bigger = fmaxf(maxval, x[i]);
        sumval = sumval * expf(maxval - bigger) + expf(x[i] - bigger);
        maxval = bigger;
    }

    // Warp 内两两合并（基于 SumMax 结构的两级规约语义）
    float offsetMaxval, offsetSumval;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        __syncwarp();
        offsetMaxval = __shfl_down_sync(0xFFFFFFFF, maxval, offset);
        offsetSumval = __shfl_down_sync(0xFFFFFFFF, sumval, offset);
        if (offsetMaxval > maxval) {
            sumval *= expf(maxval - offsetMaxval);
            maxval = offsetMaxval;
        } else {
            offsetSumval *= expf(offsetMaxval - maxval);
        }
        sumval += offsetSumval;
    }
    // 广播最终结果（向下规约后只有 lane 0 持有完整值，所以每个线程都读取 lane 0 的值）
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);
    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

    for (int i = laneId; i < C; i += warpSize) {
        y[i] = expf(x[i] - maxval) / sumval;
    }
}

/* ----------------------------------------------------------------------------
   版本9：在线 Softmax + 块级两级规约（一个 Block 处理一行）
   ----------------------------------------------------------------------------
   架构与 v4 相似（两级规约），但合并算子换成了 reduce_sum_max_op（在线合并）。
   适用场景：C 很大的场合（需要整个 Block 的线程来分摊）。

   规约流程：
     1. 每个线程遍历自己负责的元素，累积局部 SumMax
     2. Warp 内 shuffle 规约：合并 32 个线程的 SumMax 到 lane0
     3. 跨 Warp 规约：各 Warp 的 lane0 结果写入共享内存，block 0 号线程串行合并
     4. 广播全局 maxval / sum，各线程归一化写回

   共享内存占用：warpsPerBlock * sizeof(SumMax) = warpsPerBlock * 8 字节
   相比 v2 的 block_size * 4 字节大幅降低。
   ---------------------------------------------------------------------------- */
__global__ void online_softmax_forward_kernel9(float* out, const float* inp, int N, int C) {
    extern __shared__ SumMax shared_sm[]; // 大小 = warpsPerBlock * sizeof(SumMax)
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;

    const float* x = inp + bid * C;

    // 1. 每线程累积局部 SumMax
    SumMax sm_partial;
    sm_partial.maxval = -INFINITY;
    sm_partial.sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
    }

    // 2. Warp 内 shuffle 规约：合并到 lane0
    SumMax sm_warp = sm_partial;
    for (int offset = 16; offset > 0; offset >>= 1) {
        SumMax other;
        other.maxval = __shfl_down_sync(0xFFFFFFFF, sm_warp.maxval, offset);
        other.sum    = __shfl_down_sync(0xFFFFFFFF, sm_warp.sum, offset);
        sm_warp = reduce_sum_max_op(sm_warp, other);
    }
    if (laneId == 0) shared_sm[warpId] = sm_warp;
    __syncthreads();

    // 3. 跨 Warp 规约：block 0 号线程串行合并
    if (tid == 0) {
        SumMax sm_total = shared_sm[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            sm_total = reduce_sum_max_op(sm_total, shared_sm[i]);
        }
        shared_sm[0] = sm_total;
    }
    __syncthreads();

    // 4. 广播全局 maxval / sum
    float global_max = shared_sm[0].maxval;
    float global_sum = shared_sm[0].sum;

    // 5. 归一化写回
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = expf(x[i] - global_max) / global_sum;
    }
}

// ============================================================================
// Kernel 启动封装函数
// ============================================================================
void softmax_forward1(float* out, const float* inp, int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward2(float* out, const float* inp, int N, int C, const int block_size) {
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel2<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward3(float* out, const float* inp, int N, int C, int block_size) {
    block_size = 32; // 该内核固定使用单 Warp
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel3<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward4(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    size_t shared_mem_size = block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward5(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel5<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void online_softmax_forward6(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N, block_size);
    online_softmax_forward_kernel6<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void online_softmax_forward7(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    online_softmax_forward_kernel7<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void online_softmax_forward8(float* out, const float* inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    online_softmax_forward_kernel8<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void online_softmax_forward9(float* out, const float* inp, int N, int C, int block_size) {
    int grid_size = N;
    size_t shared_mem_size = block_size / 32 * sizeof(SumMax);
    online_softmax_forward_kernel9<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

// Kernel 版本分发
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1: softmax_forward1(out, inp, N, C, block_size); break;
        case 2: softmax_forward2(out, inp, N, C, block_size); break;
        case 3: softmax_forward3(out, inp, N, C, block_size); break;
        case 4: softmax_forward4(out, inp, N, C, block_size); break;
        case 5: softmax_forward5(out, inp, N, C, block_size); break;
        case 6: online_softmax_forward6(out, inp, N, C, block_size); break;
        case 7: online_softmax_forward7(out, inp, N, C, block_size); break;
        case 8: online_softmax_forward8(out, inp, N, C, block_size); break;
        case 9: online_softmax_forward9(out, inp, N, C, block_size); break;
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
    int V = 50257; // 词表大小，模拟 GPT-2 输出层维度
    //int V = 768;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // 主机端内存分配
    float* out = (float*)malloc(B * T * V * sizeof(float));
    float* inp = make_random_float(B * T * V);

    // 在每行随机选 3 个位置放大数值，让部分位置成为概率峰值（模拟真实分布）
    const int* outliers = make_random_int(B * T * 3, V);
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < B * T; ++j) {
            inp[j * V + outliers[j * 3 + k]] *= 20;
        }
    }

    // 数据迁移到 GPU 端
    float* d_out;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * V * sizeof(float), cudaMemcpyHostToDevice));

    // 从命令行读取内核版本号
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 如果不是 profile 模式，则执行标准验证与测试流程
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        int block_sizes[] = {32, 64, 128, 256, 512, 1024};

        // CPU 参考实现 + 结果合理性检查
        softmax_forward_cpu(out, inp, B * T, V);
        {
            float max_el = -INFINITY;
            for (int i = 0; i < B * T * V; ++i) {
                max_el = max(max_el, out[i]);
            }
            assert(max_el > 1e-4);
            printf("最大输出值: %f\n", max_el);
        }

        // 遍历所有 block_size 验证正确性
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("正在测试线程块大小 %d.\n", block_size);
            softmax_forward(kernel_num, d_out, d_inp, B * T, V, block_size);
            validate_result(d_out, out, "out", B * T * V, 1e-4f);
        }
        printf("所有结果验证通过，开始性能基准测试。\n\n");

        // 性能基准测试
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            int repeat_times = 100;
            float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                                  kernel_num, d_out, d_inp, B * T, V, block_size);
            printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 单 token %.2f 微秒\n",
                   block_size, elapsed_time, elapsed_time * 1000 / (B * T));
        }

        // 释放资源
        free(out); free(inp); free((void*)outliers);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     特别注意：
       版本 1~9 每轮各启动 1 个 Kernel，一轮共 9 个 Kernel。
       预热 2 轮 = 18 个 Kernel，采集 1 轮 = 9 个 Kernel。

     编译指令示例：
       nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt softmax_forward.cu -o softmax_forward

     NCU 采集命令示例（跳过 18 次预热启动，采集接下来的 9 次正式启动）：
       ncu --set full --cache-control all --launch-skip 18 --launch-count 9 -o softmax_profile_512_50257 ./softmax_forward 0 profile

     系统级性能分析：
       nsys profile --stats=true -o softmax_nsys ./softmax_forward 0 profile
     */
    {
        const int profile_block_size = 512;
        const int warmup_rounds = 2;

        // 预热 2 轮：依次调用版本 1 到 9
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= 9; ver++) {
                softmax_forward(ver, d_out, d_inp, B * T, V, profile_block_size);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮：依次调用版本 1 到 9
        for (int ver = 1; ver <= 9; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            softmax_forward(ver, d_out, d_inp, B * T, V, profile_block_size);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放资源
        free(out); free(inp); free((void*)outliers);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        return 0;
    }
}