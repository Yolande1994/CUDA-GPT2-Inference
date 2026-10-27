/* ============================================================================
   融合分类器（Softmax + 交叉熵 + Logit 梯度反向传播）CUDA Kernel 实现
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -lcublas -lcublasLt classifier_fused.cu -o classifier_fused

   运行命令：
     ./classifier_fused 1
     ./classifier_fused 2
     ...

   算子说明：
     本文件实现语言模型最后一层「分类器」的融合前向 + 反向计算，每个位置独立完成：
       1. Softmax：P = softmax(logits)，把 logits 转成概率分布
       2. 交叉熵损失：L = -log(P[target])，输出每个位置的标量损失
       3. Logit 梯度反向：dlogits = (P - indicator(target)) * dloss
     常规实现需要把 (B, T, V) 的 probs 物化到显存；V=50257 时单样本 logits 就有
     200MB+，一整套 probs/dlogits 会吃掉大量 HBM 带宽。本算子把三步融合在一次
     kernel 内完成，probs 只在寄存器 / 共享内存中存在，避免 (B, T, V) 大张量的
     HBM 往返。

   融合可行性：
     dloss 在前向传播之前就能确定，因此可以在同一个 kernel 里同时算损失和梯度：
       - 预训练时 dloss 是常量 1/(B*T)；
       - 微调时 prompt 部分的 dloss 为 0，也是预先已知的。

   张量与形状约定：
     logits    输入 logits，形状 [B, T, P]，P 是 V 向上补齐到 64 的倍数
     probs     输出概率（softmax 后），形状 [B, T, P]，仅调试用，可传 NULL
     dlogits   输出 logit 梯度，形状 [B, T, P]
     losses    输出每个位置的损失，形状 [B, T]
     dlosses   上游传回的损失梯度，形状 [B, T]，可传 NULL（此时取 1/(B*T)）
     targets   每个位置的正确 token id，形状 [B, T]
     其中 B=批大小，T=序列长度，V=词表大小（GPT-2 为 50257），P=补齐后的词表宽度。

   版本说明（按演进顺序）：
     版本1：Warp 级融合（一个 warp 处理一行 logits，简单但 V=50257 时并行度不足）
     版本2：Block 级融合 + float4 向量化（整个 block 协作处理一行）
     版本3：Block 级融合（标量版本，与版本2对比 float4 向量化的收益）
     版本4：Block 级融合 + 低精度 IO（floatX = FP16/BF16，x128 打包访存）
     版本5：版本4 优化版（store128cs 流式写、循环拆分对齐、模板控制输出开关）

   参考资料：
     "Online normalizer calculation for softmax"（Milakov & Gimelshein, 2018）
     改编自 Andrej Karpathy 的 build-nanogpt 教学代码
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../common.h"

/* 编译期精度切换：
   仅 kernel 4、kernel 5 在 FP16/BF16 下有意义；kernel 1~3 仍是 FP32 实现。
   低精度版本只用于测试 kernel 在半精度 IO 下的带宽表现，
   输入输出仍由主机端按 FP32 准备，转换逻辑见 main 函数。 */
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
#elif defined(ENABLE_FP16)
typedef half floatX;
#else
typedef float floatX;
#endif
typedef Packed128<floatX> x128;

// ============================================================================
// CPU 参考实现（正确性基准）
// ============================================================================

/* ----------------------------------------------------------------------------
   softmax_forward_cpu —— 逐行 softmax 前向
   ----------------------------------------------------------------------------
   输入 inp、输出 out 形状均为 (N, C)，每一行独立做 softmax。
   标准三趟写法：求行最大值 → 求 exp 和 → 归一化。
   ---------------------------------------------------------------------------- */
void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp 形状 (N, C)，out 形状 (N, C)，每行独立 softmax
    for (int64_t i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        // 第一趟：求行最大值（减去最大值是 softmax 数值稳定的关键）
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        // 第二趟：求 exp 和
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        // 第三趟：归一化
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}

/* ----------------------------------------------------------------------------
   crossentropy_forward_cpu —— 逐位置交叉熵损失
   ----------------------------------------------------------------------------
   损失 = -log(probs[target])，每行只取 target 对应的那一个概率。
   ---------------------------------------------------------------------------- */
void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // losses 形状 (B, T)：每个位置一个标量损失
    // probs  形状 (B, T, V)：softmax 后的概率
    // targets 形状 (B, T)：每个位置正确 token 的 id
    for (int64_t bt = 0; bt < B * T; bt++) {
        const float* probs_bt = probs + bt * V;
        int ix = targets[bt];
        losses[bt] = -logf(probs_bt[ix]);
    }
}

/* ----------------------------------------------------------------------------
   crossentropy_softmax_backward_cpu —— Softmax + 交叉熵的融合梯度
   ----------------------------------------------------------------------------
   数学上：dlogits = (P - one_hot(target)) * dloss。
   把 softmax 和交叉熵的反向传播解析合并后正好是这个式子，
   不需要再单独保存中间的 softmax 雅可比。
   ---------------------------------------------------------------------------- */
void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    for (int64_t bt = 0; bt < B * T; bt++) {
        float* dlogits_bt = dlogits + bt * V;
        const float* probs_bt = probs + bt * V;
        float dloss = dlosses[bt];
        int ix = targets[bt];
        for (int i = 0; i < V; i++) {
            float p = probs_bt[i];
            float indicator = i == ix ? 1.0f : 0.0f;  // one-hot 指示
            dlogits_bt[i] = (p - indicator) * dloss;
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* Softmax 一趟扫描需要同时返回两个量：
   Scale = 1/sum(exp(x - max))，Offset = 行最大值。
   后续第二遍扫描写概率时同时用这两个量，避免再扫一遍。 */
struct SoftmaxParams {
    float Scale;
    float Offset;
};

namespace cg = cooperative_groups;

/* ----------------------------------------------------------------------------
   warpReduceMax —— Warp 内求最大值的蝴蝶规约
   ----------------------------------------------------------------------------
   与 warpReduceSum 同构，只是操作从加法换成 fmax。
   ---------------------------------------------------------------------------- */
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/* ----------------------------------------------------------------------------
   prepare_softmax —— Warp 级在线 softmax 参数计算（kernel 1 用）
   ----------------------------------------------------------------------------
   一个 32 线程的 warp 协作处理 logits 的一行（长度 V）：
     - 用"在线 softmax"递推，一遍扫描同时维护 max 和 sum；
     - 每个线程串行跨跨步（stride=32）处理 V 个元素；
     - warp 内先规约出全局 max，再据此修正 sum，最后规约出全局 sum。
   返回 {1/sum, max}，供第二遍算概率和梯度。
   ---------------------------------------------------------------------------- */
__device__ SoftmaxParams prepare_softmax(cg::thread_block_tile<32>& warp,
                                         int64_t idx, const float* inp, int V, int P) {
    // warp 处理 inp 的一行 inp[idx, :]（长度 V，实际物理行宽 P）
    const float* x = inp + idx * P;

    // 线程粗化：32 个线程跨步处理全部 V 个元素
    float maxval = -INFINITY;
    float sumval = 0.0f;
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float v = x[i];
        float old_maxval = maxval;
        // 在线 softmax 递推：新最大值出现时，旧的 exp 和要按 (old_max - new_max) 修正
        maxval = fmaxf(maxval, v);
        sumval *= expf((old_maxval - maxval));
        sumval += expf(v - maxval);
    }

    // warp 级规约：得到整行的全局最大值
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    // 用全局最大值修正本线程的 sum
    sumval *= expf((maxval - global_maxval));
    // warp 级规约：得到整行的全局 sum
    float global_sumval = cg::reduce(warp, sumval, cg::plus<float>{});

    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}

/* ----------------------------------------------------------------------------
   fused_classifier_kernel1 —— 版本1：Warp 级融合分类器
   ----------------------------------------------------------------------------
   并行策略：一个 block 内有 N_warps 个 warp，每个 warp 独立处理一行 logits；
             block_size=128 时即 4 行/block。
   计算流程：
     1. prepare_softmax 一趟扫描得到 (max, sum)；
     2. lane 0 单独计算交叉熵损失；
     3. 32 个线程跨步再扫一遍 logits，直接算梯度 (prob - indicator) * dloss。
   局限：V=50257 时一个 warp 串行处理 1570 个元素，并行度严重不足。
   ---------------------------------------------------------------------------- */
__global__ void fused_classifier_kernel1(float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // 例：B=4, T=1024, block_size=128 => grid_size=1024，每个 block 4 行
    int64_t idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) {
        return;
    }
    int b = idx / T;
    int t = idx % T;

    // 一趟扫描得到 max 与 sum（在线 softmax）
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P);

    // lane 0 计算该位置的交叉熵损失
    if(warp.thread_rank() == 0) {
        int ix = targets[b * T + t];
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -logf(prob);
    }

    // 第二遍：所有线程直接算梯度，prob 只在寄存器中物化，不落 HBM
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float prob = expf(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (prob - indicator) * dloss;
    }
}

/* 从 float4 里取第 k 个分量（按内存顺序）。 */
__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

/* ----------------------------------------------------------------------------
   prepare_softmax_blockwide —— Block 级在线 softmax 参数计算（kernel 2 用）
   ----------------------------------------------------------------------------
   与 prepare_softmax 的区别：
     - 整个 block（最多 1024 线程）协作处理一行 logits，并行度高；
     - 用 float4 一次加载 4 个 float，访存效率翻 4 倍；
     - 循环从尾部向前遍历，提高第二次读 logits 时的 L2 命中率。
   规约流程：线程内跨 warp 求和 → warp 内 shuffle → 跨 warp 共享内存 → warp 内 shuffle。
   ---------------------------------------------------------------------------- */
__device__ SoftmaxParams prepare_softmax_blockwide(cg::thread_block_tile<32>& warp,
                                                   int64_t idx, const float* inp, int V, int P) {
    // 整行按 float4 解读，一次加载 4 个 float
    const float4* x_vec4 = reinterpret_cast<const float4*>(inp + idx * P);

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // 反向遍历：prepare_softmax_blockwide 之后还会再读一遍同一行，
    // 反向顺序让第二次访问时数据仍在 L2 中概率更高
    for (int i = ceil_div(V, 4) + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float4 v4 = x_vec4[i];
        #pragma unroll
        for(int k = 0; k < 4; k++) {
            if (i*4+k >= V) {  // 物理行宽 P 可能大于 V，边界检查
                continue;
            }
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, vec_at(v4, k));
            thread_sumval *= expf(old_maxval - thread_maxval);
            thread_sumval += expf(vec_at(v4, k) - thread_maxval);
        }
    }

    // 两级规约：warp 内 shuffle → 跨 warp 共享内存 → warp 内 shuffle
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // 第一级：每个 warp 内规约 max
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // 第二级：第 0 个 warp 跨 warp 规约 max
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});

    // 用全局 max 修正本线程的 sum
    thread_sumval *= expf(thread_maxval - block_maxval);

    // 同样的两级流程规约 sum
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});

    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/* ----------------------------------------------------------------------------
   fused_classifier_kernel2 —— 版本2：Block 级 + float4 向量化
   ----------------------------------------------------------------------------
   与 kernel1 的关键区别：
     - 一个 block 只处理一行 logits，grid 大小 = B*T；
     - 用 float4 向量化加载，一次 16 字节；
     - probs 仅调试用可传 NULL，训练时只写 dlogits，省一半带宽；
     - 第二次读 logits 用 __ldcs（流式加载），因为这批数据不会再被读。
   注意：logits 和 dlogits 可能复用同一块显存，因此参数不加 __restrict__。
   ---------------------------------------------------------------------------- */
__global__ void fused_classifier_kernel2(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // 第一遍：算 softmax 的 max 与 sum
    SoftmaxParams sp = prepare_softmax_blockwide(warp, idx, logits, V, P);

    // 线程 0 单独算损失
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // dlosses 为 NULL 时默认均匀梯度 1/(B*T)，对应预训练场景
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);

    // 第二遍：再读一次 logits，直接算梯度，顺便可选写 probs
    const float4* logits_vec4 = reinterpret_cast<const float4*>(logits + idx * P);
    for (int i = threadIdx.x; i < ceil_div(V, 4); i += blockDim.x) {
        // 流式加载：这批数据写完梯度后就不再读，绕过 L1 避免污染
        float4 v4 = __ldcs(&logits_vec4[i]);

        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            int element = i*4 + k;
            float prob = expf(vec_at(v4, k) - sp.Offset) * sp.Scale;
            prob = (element < V) ? prob : 0.0f;  // 边界检查

            // 本 kernel 是 DRAM 带宽受限，内部分支开销可忽略
            if (probs != NULL) {
                probs[idx * P + element] = prob;
            }
            if (dlogits != NULL) {
                float indicator = element == ix ? 1.0f : 0.0f;
                dlogits[idx * P + element] = (prob - indicator) * dloss;
            }
        }
    }
}

/* ----------------------------------------------------------------------------
   prepare_softmax_blockwide_nofloat4 —— 标量版 block 级 softmax
   ----------------------------------------------------------------------------
   与 prepare_softmax_blockwide 逻辑完全一致，差别只是把 float4 换成标量 float，
   用于对照实验：量化 float4 向量化到底带来多少带宽收益。
   ---------------------------------------------------------------------------- */
__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cg::thread_block_tile<32>& warp,
                                                            int64_t idx, const float* inp, int V, int P) {
    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // 同样反向遍历以提高 L2 命中
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf(old_maxval - thread_maxval);
        thread_sumval += expf(v - thread_maxval);
    }

    // 两级规约（与 prepare_softmax_blockwide 相同）
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    thread_sumval *= expf(thread_maxval - block_maxval);

    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});

    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/* ----------------------------------------------------------------------------
   fused_classifier_kernel3 —— 版本3：Block 级标量版（无 float4）
   ----------------------------------------------------------------------------
   与 kernel2 唯一的区别：第二遍用标量 __ldcs 逐元素读 logits，
   用于测量 float4 向量化对整体带宽的提升。
   ---------------------------------------------------------------------------- */
__global__ void fused_classifier_kernel3(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);

    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        if (dlogits != NULL) {
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[idx * P + i] = (prob - indicator) * dloss;
        }
    }
}

/* ----------------------------------------------------------------------------
   prepare_softmax_blockwide2 —— 低精度版 block 级 softmax（kernel 4 用）
   ----------------------------------------------------------------------------
   与 kernel2 的 prepare_softmax_blockwide 对应，区别：
     - 输入是 floatX（FP16/BF16），用 x128 一次打包 8 个半精度数；
     - 用 common.h 提供的 warpReduceMax / warpReduceSum，替代 cg::reduce；
     - 计算仍在 FP32 累加，半精度只用于 IO。
   ---------------------------------------------------------------------------- */
__device__ SoftmaxParams prepare_softmax_blockwide2(int64_t idx, const floatX* inp, int V, int P) {
    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // 反向遍历 + 半精度 x128 打包加载
    for (int i = ceil_div(V, x128::size) + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128cs(x + i * x128::size);  // 流式加载，不保留在缓存
        for(int k = 0; k < packed_x.size; ++k) {
            if (i*x128::size+k >= V) {
                continue;
            }
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf(old_maxval - thread_maxval);
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // 两级规约：warpReduceMax / warpReduceSum 来自 common.h
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float warp_maxval = warpReduceMax(thread_maxval);
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    float block_maxval = warpReduceMax(warp_maxval);
    thread_sumval *= expf(thread_maxval - block_maxval);

    float warp_sumval = warpReduceSum(thread_sumval);
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = warpReduceSum(warp_sumval);

    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/* ----------------------------------------------------------------------------
   fused_classifier_kernel4 —— 版本4：低精度 IO + x128 打包
   ----------------------------------------------------------------------------
   与 kernel2 对应，但 logits / dlogits / probs 都是 floatX（FP16/BF16）：
     - 一次 x128 打包加载 8 个半精度数，带宽再减半；
     - expf、规约、累加都在 FP32 进行，只在写回时转回 floatX。
   已知问题：store128 没有加 .cs，导致后续 store 的缓存抖动，在 kernel5 修复。
   ---------------------------------------------------------------------------- */
__global__ void fused_classifier_kernel4(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    SoftmaxParams sp = prepare_softmax_blockwide2(idx, logits, V, P);

    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    float dloss = dlosses != NULL ? (float)dlosses[idx] : 1.0f / (B*T);

    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < ceil_div(V , x128::size); i += blockDim.x) {
        // 第二次读 logits：这批数据写完梯度后不再读，用流式加载
        x128 packed_logits_vec = load128cs(logits_vec + i * x128::size);
        x128 packed_probs;
        x128 packed_dlogits;
        for(int k = 0; k < packed_logits_vec.size; ++k) {
            int element = i*packed_logits_vec.size + k;
            if (element >= V) {
                continue;
            }
            float v = packed_logits_vec[k];
            float prob = expf(v - sp.Offset) * sp.Scale;
            packed_probs[k] = prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_dlogits[k] = (prob - indicator) * dloss;
        }
        // 注意：这里 store128 没有 .cs 提示，会污染缓存，kernel5 改用 store128cs 修复
        store128(dlogits + idx * P + i * packed_logits_vec.size, packed_dlogits);
        if (probs != NULL) {
            store128(probs + idx * P + i * packed_logits_vec.size, packed_probs);
        }
    }
}

/* ----------------------------------------------------------------------------
   prepare_softmax_blockwide3 —— 低精度版 softmax（kernel 5 用，最终优化）
   ----------------------------------------------------------------------------
   与 prepare_softmax_blockwide2 的区别：
     - 循环拆成两段：尾部未对齐元素单独处理（带边界检查），
       主循环按 x128::size 严格对齐，省掉每个元素的边界判断；
     - 主循环用 load128（普通加载），因为这行 logits 紧接着还要在
       fused_classifier_kernel5 的第二遍里再读一次，希望它留在 L2；
     - 用 common.h 的 blockReduce 模板替代手写的两级规约代码。
   ---------------------------------------------------------------------------- */
__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // 尾部未对齐元素单独处理（带边界检查）
    // 让主循环可以完全省掉边界判断
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break;
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // 主循环：严格按 x128 对齐，无需边界检查
    // 用普通 load128 保留在 L2，供 kernel5 第二遍再读
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128(x + i * x128::size);
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // 两级规约：max → 修正 sum → sum
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -FLT_MAX);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

/* ----------------------------------------------------------------------------
   fused_classifier_kernel5 —— 版本5：最终低精度融合分类器
   ----------------------------------------------------------------------------
   与 kernel4 的差异：
     - 第二遍循环同样拆成"x128 对齐主循环 + 尾部标量"两段；
     - 写 dlogits 用 store128cs（流式写），因为这些梯度写出去后不再读回，
       把 L2 留给"紧接着还要再读一次的 logits"；
     - 模板参数 WriteLogits / WriteProbs 在编译期决定是否写 probs 和 dlogits，
       分支完全消失，不产生运行时开销；
     - __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS) 告诉编译器
       每个 SM 至少驻留多少个 block，帮助寄存器分配。
   本 kernel 是低精度分类器的最终版本。
   ---------------------------------------------------------------------------- */
template <bool WriteLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
                fused_classifier_kernel5(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // 第一遍：算 softmax 参数（普通 load128，希望数据留 L2）
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // 线程 0 算损失
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = (floatX)(-logf(prob));
    }

    // 与预训练一致的默认 dloss
    float dloss = (dlosses != NULL) ? (float)dlosses[idx] : 1.0f / (B*T);

    // 第二遍：再读一次 logits，直接算梯度并就地覆盖写回
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // 普通 load128：依赖 prepare_softmax_blockwide3 已经把这行留在 L2
        x128 packed_logits_vec = load128(logits_vec + i * x128::size);
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteLogits){
            // 流式写 dlogits：梯度写出去就不再读回，绕过 L1/L2 保留给 logits
            store128cs(dlogits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // 处理最后一个 x128::size 倍数之后的剩余元素
    // 例：V=8003、x128::size=8 时，剩下 3 个元素走标量路径
    int unaligned_start = V & ~(x128::size - 1);  // 向下取整到 x128::size 的倍数
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteLogits){
            __stcs(dlogits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ============================================================================
// Kernel 启动器（host 端封装）
// ============================================================================

/* 版本1：block 内多 warp 各处理一行，rows_per_block = block_size / 32。 */
void fused_classifier1(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int rows_per_block = block_size / 32;
    const int grid_size = N / rows_per_block;
    fused_classifier_kernel1<<<grid_size, block_size>>>(dlogits, losses, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

/* 版本2：一个 block 处理一行，grid = B*T。 */
void fused_classifier2(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel2<<<grid_size, block_size>>>(dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

/* 版本3：与版本2 同布局，kernel 内走标量路径。 */
void fused_classifier3(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

/* 版本4：低精度路径，host 端按 FP32 准备，传入时强转 floatX*。 */
void fused_classifier4(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel4<<<grid_size, block_size>>>((floatX*)dlogits, (floatX*)losses, NULL, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

/* 版本5：最终低精度融合分类器，启用 WriteLogits、不写 probs。 */
void fused_classifier5(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel5<true,false><<<grid_size, block_size>>>((floatX*)dlogits, (floatX*)losses, NULL, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

/* 统一入口：根据 kernel_num 分发到对应版本。 */
void fused_classifier(int kernel_num, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    switch (kernel_num) {
        case 1: fused_classifier1(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
        case 2: fused_classifier2(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
        case 3: fused_classifier3(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
        case 4: fused_classifier4(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
        case 5: fused_classifier5(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ============================================================================
// 主函数（正确性校验 + 性能计时）
// ============================================================================

int main(int argc, char **argv) {
    srand(0);

    int64_t B = 8;              // 批大小
    int64_t T = 1024;           // 序列长度
    int64_t V = 50257;          // GPT-2 词表大小
    int64_t P = (V + 63) & ~63; // 物理行宽：V 向上补齐到 64 的最近倍数，便于向量化

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // 主机端随机数据
    float* logits = make_random_float(B * T * V);
    float* probs = make_random_float_01(B * T * V);
    float* dlogits = (float*)malloc(B * T * V * sizeof(float));
    float* losses = (float*)malloc(B * T * sizeof(float));
    float* dlosses = make_random_float(B * T);
    int* targets = make_random_int(B * T, V);

    // 随机 logits 经过 softmax 后概率几乎全为 0，测不准。
    // 人为在每行挑 3 个位置放大 20 倍，制造"少数大 logit"的真实分布。
    int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            logits[j * V +  outliers[j*3 + k]] *= 20;
        }
    }

    // 设备端分配（注意按补齐后的 P 分配）
    int *d_targets;
    float *d_logits, *d_losses;
    float *d_dlogits, *d_dlosses, *d_dlogits_no_pad;
    cudaCheck(cudaMalloc(&d_dlogits, B * T * P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_logits, B * T * P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlogits_no_pad, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_losses, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlosses, B * T * sizeof(float)));

    // 把 V 宽的主机 logits 按 P 为行宽拷到设备端（中间跳过 padding 区域）
    cudaCheck(cudaMemset(d_logits, 0xff, B * T * P * sizeof(float)));
    cudaCheck(cudaMemcpy2D(d_logits, P * sizeof(float), logits, V * sizeof(float), V * sizeof(float), B * T, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, B * T * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // 命令行选择 kernel 编号
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // 测试用 block 大小集合
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // CPU 参考结果
    softmax_forward_cpu(probs, logits, B * T, V);
    crossentropy_forward_cpu(losses, probs, targets, B, T, V);
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

#if defined(ENABLE_BF16) || defined(ENABLE_FP16)
    // kernel 4/5 在半精度下只用于测带宽，未做主机端 FP32→FP16 转换，
    // 因此低精度编译时跳过正确性校验，只跑性能。
    if (kernel_num < 4)
#endif
    {
        // 逐 block_size 做正确性校验
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("Checking block size %d.\n", block_size);
            fused_classifier(kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, P, block_size);
            validate_result(d_losses, losses, "losses", B * T, 1e-4f);
            // dlogits 在设备端是 P 宽，先抠出 V 宽再和 CPU 结果对比
            cudaCheck(cudaMemcpy2D(d_dlogits_no_pad, V * sizeof(float), d_dlogits, P * sizeof(float), V * sizeof(float), B * T, cudaMemcpyDeviceToDevice));
            validate_result(d_dlogits_no_pad, dlogits, "dlogits", B * T * V, 1e-4f);
        }
        printf("All results match. Starting benchmarks.\n\n");
    }

    // 性能计时：每个 block_size 重复 1000 次取平均
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_classifier,
                                              kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets,
                                              B, T, V, P, block_size);
        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // 释放主机与设备内存
    free(logits);
    free(probs);
    free(dlogits);
    free(losses);
    free(dlosses);
    free(targets);
    free(outliers);
    cudaCheck(cudaFree(d_dlogits));
    cudaCheck(cudaFree(d_losses));
    cudaCheck(cudaFree(d_logits));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_dlogits_no_pad));

    return 0;
}
