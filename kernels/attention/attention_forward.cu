/* ============================================================================
   多头注意力（Multi-Head Attention）前向传播 CUDA Kernel 实现
   ============================================================================

   编译示例：
     基础版本（版本 1~9，无需 cuDNN）：
       nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward
     启用 cuDNN（额外支持版本 10、11。路径改成本地）：
       nvcc -I/PATH/TO/cudnn-frontend/include -DENABLE_CUDNN -O3 --use_fast_math \
            -lcublas -lcublasLt -lcudnn -lnvrtc attention_forward.cu -o attention_forward
       nvcc -I/home/yolan/projects/cudnn-frontend-main/include -DENABLE_CUDNN -O3 --use_fast_math \
            -lcublas -lcublasLt -lcudnn -lnvrtc attention_forward.cu -o attention_forward
     说明：未安装 cuDNN 时，不定义 ENABLE_CUDNN 宏即可，版本 1~9 不受影响。

   算子说明：
     实现 GPT 类模型中的「多头因果自注意力」前向计算，每个注意力头独立执行：
       1. 打分：S = (Q @ Kᵀ) / √d_k，d_k 为单头维度
       2. 掩码：因果掩码（Causal Mask），第 t 个 token 只能关注 0~t 的位置
       3. 归一化：P = softmax(S)，每行权重总和为 1
       4. 聚合：O = P @ V，按权重对 Value 加权求和
     缩放因子 1/√d_k 的作用：点积结果随维度增大而增大，会把 softmax 推入梯度近乎为 0 的饱和区，缩放后可保持数值稳定。

   张量与形状约定：
     inp    输入，形状 [B, T, 3C]，Q、K、V 在最后一维拼接
     out    输出，形状 [B, T, C]
     preatt 原始注意力分数（缩放后、softmax 前），形状 [B, NH, T, T]，主要供反向传播使用
     att    注意力权重（softmax 后），形状 [B, NH, T, T]
     其中 B=批大小，T=序列长度，C=总通道数，NH=注意力头数，hs=C/NH=单头维度。

   版本说明（按演进顺序）：
     版本1 ：朴素三核分离（QK 点积、Softmax、加权求和各一个 kernel，性能基线）
     版本2 ：Warp 级融合 kernel（一个 warp 处理一行，融合 QK/Softmax/加权求和）
     版本3 ：Flash Attention 朴素实现（列优先遍历，仅作原理演示）
     版本4 ：Flash Attention-2（行优先遍历 + 序列维度并行 + 延迟归一化）
     版本5 ：Flash Attention-2 Warp 级优化（每 warp 负责连续多行 Q，warp 间零通信）
     版本6 ：cuBLAS + 自定义 Softmax
     版本7 ：cuBLAS + 融合在线 Softmax
     版本8 ：版本7 的 FP16/BF16 混合精度版（IO 用低精度、计算用 FP32）
     版本9 ：版本8 跳过排列操作（不对应真实场景，仅作性能对比基准）
     版本10：cuDNN 官方 Flash Attention（FP16/BF16）
     版本11：版本10 的全低精度链路（跳过 FP32/FP16 转换）

   运行命令：
     ./attention_forward 1
     ./attention_forward 2
     ...

   参考资料：
     Flash Attention（Dao et al., 2022）、Flash Attention-2（Dao, 2023）
     版本3 改编自 https://github.com/tspeterkim/flash-attention-minimal
     并参考 https://github.com/leloykun/flash-hyperbolic-attention-minimal
     cuDNN Attention：
       https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md
   ============================================================================ */

//#define ENABLE_CUDNN // 也可通过 nvcc 的 "-DENABLE_CUDNN" 参数启用
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#define ENABLE_BF16
#include "../common.h"

// ============================================================================
// CUDA & cuDNN 初始化
// ============================================================================

/* 全局标记：是否为首次运行。
   首次运行需要完整执行排列、格式转换等操作以验证正确性；
   后续性能基准循环中可跳过这些环节，节省测试时间。 */
static bool first_run_validation = true;

/* 条件编译：仅在定义 ENABLE_CUDNN 时编译 cuDNN 相关代码；
   未定义时整段忽略，保证未安装 cuDNN 也能正常编译。 */
#ifdef ENABLE_CUDNN
#include <cudnn_frontend.h>  // cuDNN 前端库头文件（新版计算图 API）
namespace fe = cudnn_frontend;
// 数据类型对齐：让 cuDNN 的 16 位类型与 cuBLAS 配置保持一致
#if CUBLAS_LOWP == CUDA_R_16BF
#define CUDNN_16BIT fe::DataType_t::BFLOAT16 // cuBLAS 使用 BF16 时，cuDNN 同样使用 BF16
#else
#define CUDNN_16BIT fe::DataType_t::HALF     // 否则 cuDNN 使用 FP16（HALF）
#endif

static cudnnHandle_t cudnn_handle;  // cuDNN 全局句柄：所有 cuDNN 操作通过它调度

// cuDNN 工作空间（临时显存）管理，按需动态分配（上限 256MiB）
static size_t cudnn_workspace_size = 0; // 当前已分配的工作空间大小
static void* cudnn_workspace = NULL;    // 工作空间指针

#define checkCudaErr(err) assert((int)err == 0);  // CUDA 接口返回值检查，非 0 即终止以便定位
#define checkCudnnErr(err) assert((int)err == 0); // cuDNN 接口返回值检查，非 0 即终止
#endif  // ENABLE_CUDNN 条件编译结束


// ============================================================================
// CPU 参考实现
// ============================================================================

/* 标准多头因果自注意力 CPU 参考实现，作为 GPU 各版本的正确性基准。

   参数与形状：
     inp    输入张量，形状 [B, T, 3C]，Q、K、V 在最后一维拼接
     out    输出张量，形状 [B, T, C]，注意力计算最终结果
     preatt 注意力原始分数（缩放后、softmax 前），形状 [B, NH, T, T]
     att    注意力权重（softmax 后），形状 [B, NH, T, T]

   单头内的矩阵运算（Q、K、V 与单头输出 O 形状均为 [T, hs]）：
     Q × Kᵀ = S    (T, hs) × (hs, T) = (T, T)
     P × V  = O    (T,  T) × (T, hs) = (T, hs)                       */
void attention_forward_cpu(float* out, float* preatt, float* att,
                           const float* inp,
                           int B, int T, int C, int NH) {
    int C3 = C*3;
    int hs = C / NH; // 单头维度
    float scale = 1.0 / sqrtf(hs); // 缩放因子 1/√d_k，防止点积过大导致 softmax 梯度消失
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;  // 定位当前 Q 向量
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;  // 定位当前注意力分数行
                float* att_bth  = att + b*NH*T*T + h*T*T + t*T;       // 定位当前注意力权重行
                // 第一步：计算 query 与每个 key 的点积，并记录行最大值
                float maxval = -FLT_MAX;  // 最小有限值，<float.h> 中定义的宏常量
                for (int t2 = 0; t2 <= t; t2++) {  // 因果掩码：只遍历 0~t 的位置
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C 跳过 Q，偏移到 key 部分
                    // q 向量与 k 向量做点积
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;         // 标准缩放：除以 √hs
                    if (val > maxval) {
                        maxval = val;     // 更新最大值
                    }
                    preatt_bth[t2] = val; // 保存原始分数（循环层次：b → h → t → t2）
                }
                // 调试对比用：因果区域外填充负无穷
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // 第二步：计算指数值并累加求和
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);  // 各分数先减行最大值，再求自然指数
                    att_bth[t2] = expv;  // 指数结果暂存 att 数组，后续统一归一化
                    expsum += expv;      // 累加得到分母总和
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum; // 带除零保护的倒数计算

                // 第三步：归一化得到 softmax 结果
                for (int t2 = 0; t2 < T; t2++) {  // 遍历完整一行，为历史/当前/未来位置全部赋合法值
                    if (t2 <= t) { // 历史/当前 token：有效参与计算的位置
                        att_bth[t2] *= expsum_inv;
                    } else {       // 未来 token：因果掩码屏蔽（此前未赋值，内存为脏数据）
                        att_bth[t2] = 0.0f;  // 严格来说无需置零，显式置零仅用于调试和与 PyTorch 结果对比
                    }
                }

                // 第四步：将加权后的 value 累加得到注意力输出
                float* out_bth = out + b * T * C + t * C + h * hs;  // 定位当前输出向量 out[b,t,h,:]（长度 hs，out 形状 [B,T,C]）
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; } // 累加器清零：凡使用 += 累加，初始值必须为 0
                for (int t2 = 0; t2 <= t; t2++) {   // 每次遍历：V 矩阵下移一行，权重行右移一个标量
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // 定位 V 向量，+C*2 跳过 Q、K
                    float att_btht2 = att_bth[t2];  // 标量权重预取，避免反复读取
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    } // 每个注意力头各输出长度 hs 的向量，在内存中紧密排列；h 循环结束后即拼接为 [B,T,C]，送入后续线性投影层
                }
            }
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   版本1（核 1/3）：QK 点积核 —— 计算注意力分数 S = Q @ Kᵀ / √hs
   ----------------------------------------------------------------------------
   定位：版本1 由三个独立 kernel 组成，本核为第一个，对应 CPU 的第一步。
   并行策略：一个线程计算输出张量的一个元素（分数矩阵的一个标量）。
   索引映射：输出为行优先 [B, NH, T, T] 布局，需从一维 idx 反解出 (b, h, t, t2) 四个坐标，反解公式为标准行优先布局的除法/取模。
   因果掩码：列号 t2 > 行号 t 的上三角位置直接写 -INFINITY 并返回，不参与点积计算。
   ---------------------------------------------------------------------------- */
__global__ void attention_query_key_kernel1(float* preatt, const float* inp,
                                            int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;  // 一个线程算一个元素
    if (idx < total_threads) {
        /* 从全局索引反解出四个维度坐标「批、头、行、列」，对应输出张量的内存布局。
           标准行优先布局的索引拆解公式：
             当前维度坐标 = (全局索引 ÷ 后面所有维度的大小乘积) 取余 当前维度大小
             最高维（变化最慢）直接用 idx ÷ 后面所有维度乘积
             最低维（变化最快）直接用 idx % 维度大小                            */
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
   版本1（核 2/3）：行 Softmax 核 —— 注意力分数 → 注意力权重
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
   版本1（核 3/3）：加权求和核 —— P @ V
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


/* ----------------------------------------------------------------------------
   版本2：Warp 级融合 Kernel（一个 Warp 处理一行，融合 QK/Softmax/加权求和）
   ----------------------------------------------------------------------------
   定位：将版本1 的三个 kernel 融合为一个，减少中间结果的全局内存往返。
   并行映射：
     - 一个 warp（32 线程）负责权重矩阵的一行，即一个 (b,h,t)
     - t = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank()
     - grid 三个维度分别对应：x=序列行（并行度最细）、y=注意力头、z=批次
   执行流程：
     1. QK 点积：warp 内 32 线程协作完成每个 key 的点积，同时跟踪行最大值
     2. 指数求和：warp 分工计算指数值，归约得到分母总和
     3. 归一化：直接写出最终权重（与下一步合并，减少一次内存往返）
     4. 加权求和：warp 按特征维度分工，遍历所有 token 累加得到输出
   性能问题：点积仍按 key 逐个串行计算，序列较长时延迟较高，
            后续版本通过分块（Flash Attention）或 cuBLAS 解决。
   ---------------------------------------------------------------------------- */
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


/* Warp 内最大值规约：基于 __shfl_down_sync（向下规约）。
   核心 API：__shfl_down_sync(mask, var, offset)
     线程 i 读取同一 Warp 内线程 i+offset 的 var 值。
   规约完成后仅 lane 0 持有完整的 Warp 最大值。 */
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}


/* ----------------------------------------------------------------------------
   自定义 Softmax 核：两级规约版（Warp 内 Shuffle + 跨 Warp 共享内存）
   ----------------------------------------------------------------------------
   定位：版本6 使用的行 Softmax 核，支持任意 32 倍数的 block_size。
   并行粒度：一个 Block 处理一行（共 N 行，每行 C 个元素）。
   规约方式：
     第一级（Warp 内）：shuffle 指令在寄存器中完成规约，延迟极低；
     第二级（跨 Warp）：各 Warp 的 0 号线程将结果写入共享内存，由 block 0 号线程串行合并。
   共享内存：仅需 2*warpsPerBlock 个 float（前半存最大值、后半存求和），少于每个线程都写共享内存的朴素二分规约。
   流程：找最大值 → 指数写回 → 求指数和 → 归一化。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel5(float* out, const float* inp, int N, int C) {
    int warpsPerBlock = blockDim.x / 32; // 每个块包含的 warp 数
    extern __shared__ float shared[];    // 需分配 2*warpsPerBlock 个元素，前半存最大值，后半存求和
    float* maxvals = shared;                    // 各 warp 的局部最大值
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


// float4 向量的下标访问工具，方便以 [index]（合法范围 0~3）方式访问 4 个分量。
__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}
__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}


/* ----------------------------------------------------------------------------
   融合在线 Softmax 核（缩放融合 + float4 向量化 + 在线递推 + 因果掩码）
   ----------------------------------------------------------------------------
   定位：版本7 使用的 Softmax 核。一个 Warp 处理一行，
        在 Softmax 计算的同时完成 QK 分数缩放，省掉独立的缩放 kernel。
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


/* ----------------------------------------------------------------------------
   融合在线 Softmax 核（低精度版）
   ----------------------------------------------------------------------------
   定位：功能与 softmax_forward_kernel6 完全一致，版本8 使用
        唯一区别是输入输出为 floatX（FP16/BF16），内部计算全程使用 FP32。
   工程范式（低精度算子标准做法）：
     IO 用低精度省显存带宽，内部用 FP32 保数值稳定。
     Softmax 的指数运算对精度敏感，FP16 下容易溢出或精度崩坏，必须升精度；
     类型转换是寄存器内硬件指令，仅几个时钟周期，相对上百周期的显存访问可以忽略。
   算子特性：Softmax 是典型的 IO 绑定（带宽受限）算子，计算量小，
            计算精度从 16 位升到 32 位几乎不影响性能。
   说明：本核不使用 floatX 向量指令，按 4 个连续元素一组做标量读取（读取后仍需逐个转 FP32，向量加载收益低，标量写法更简单稳定）。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel6_lowp(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    assert(T % 4  == 0);
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) { return; }
    int t = idx % T;
    int t_by_4 = t / 4;

    const floatX* x = inp + idx * T;
    float maxval = -FLT_MAX;
    float sumval = 0.0f;
    // 功能与 FP32 版相同，但不使用 float4，逐个元素计算
    for (int i = warp.thread_rank(); i < t_by_4; i += warp.size()) {
        float old_maxval = maxval;
        /* 4 个连续 floatX 为一组读取，逐个转 FP32 计算。
           这里的 4 是固定的循环展开粒度，与数据类型、向量宽度无关：
           按 4 个连续元素批量读取有利于内存对齐与缓存友好，也与 FP32 版的循环结构保持一致；
           由于内部仍需转 FP32，向量加载收益低。 */
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, (float)x[i*4 + k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * ((float)x[i*4 + k] - maxval));
        }
    }
    if(t_by_4*4 + warp.thread_rank() <= t) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[t_by_4*4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[t_by_4*4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    for (int i = warp.thread_rank(); i <= t; i += warp.size()) {
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));  // 输出转回 floatX
    }
}

/* ----------------------------------------------------------------------------
   低精度 QKV 拆分 + 维度重排融合核
   ----------------------------------------------------------------------------
   一次完成三件事：
     1. 拆分：把打包的 QKV 张量拆成 Q、K、V 三个独立张量
     2. 重排：内存布局从「序列优先」[B, N, 3, NH, d] 转为「头优先」[B, NH, N, d]，匹配后续 cuBLAS 批量矩阵乘的访存布局
     3. 量化：输入 FP32，输出压缩为 floatX（FP16/BF16），精度损失发生在此步
   ---------------------------------------------------------------------------- */
__global__ void permute_kernel_lowp(floatX* q, floatX* k, floatX* v,
                                    const float* inp,
                                    int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int nh_ = (idx / (N * d)) % NH;
        int n = (idx / d) % N;
        int d_ = idx % d;
        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;
        // 存储量化：FP32 压缩为 16 位 floatX 写入输出，Q/K/V 依次偏移 NH*d
        q[idx] = (floatX)inp[inp_idx];
        k[idx] = (floatX)inp[inp_idx + NH * d];
        v[idx] = (floatX)inp[inp_idx + 2 * (NH * d)];
    }
}

// 低精度反重排核：将输出从 (B, NH, N, d) 转回 (B, N, NH, d)。输入 floatX，输出转回 FP32。
__global__ void unpermute_kernel_lowp(const floatX* inp, float *out, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int nh_ = (idx / (N * d)) % NH;
        int n = (idx / d) % N;
        int d_ = idx % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = (float)inp[idx];
    }
}


/* ----------------------------------------------------------------------------
   版本3：Flash Attention 朴素实现（列优先，j 外 i 内）
   ----------------------------------------------------------------------------
   定位：Flash Attention 的教学演示版，改编自 flash-attention-minimal（见文件头参考资料）。
        该版本因列优先遍历、片上矩阵乘未优化，实测约比朴素版慢 3 倍，仅作原理演示。
   核心思想：分块计算 + 块级在线 Softmax + 共享内存，
            注意力大矩阵 [T,T] 的存取全部限制在片上，显存复杂度从 O(T²) 降为 O(T)。
   内存布局：标准 BHSD 布局 [B, H, T, d]（批→头→序列→维度），每个头的数据连续存放。

 * ============================================================
 *  FlashAttention 前向核（列优先 j外i内）
 * ============================================================
 *  注意力大矩阵 [T,T] 切成 Tr×Tc 个小块，小块形状 [Br,Bc]：
 *
 *        j=0    j=1    j=2    j=3
 *       ┌─────┬─────┬─────┬─────┐
 *  i=0  │  ①  │     │     │     │   遍历顺序：竖条式
 *       ├─────┼─────┼─────┼─────┤   固定 j，i 从 0→Tr-1
 *  i=1  │  ②  │     │     │     │   走完一列，j++
 *       ├─────┼─────┼─────┼─────┤
 *  i=2  │  ③  │     │     │     │
 *       ├─────┼─────┼─────┼─────┤
 *  i=3  │  ④  │     │     │     │
 *       └─────┴─────┴─────┴─────┘
 *
 *  单小块数据流水线：
 *    Qi[Br,d] ─┐
 *    Kj[Bc,d] ─┼→ S[Br,Bc] → exp(S-row_m) → ×Vj → pv[Br,d]
 *    Vj[Bc,d] ─┘                                     ↓
 *    m[T],l[T] ←──────────────────────────── 合并 → O[Br,d]
 * ============================================================

   遍历方式对比：
     列优先（复用 KV，本核）：每算完一块 KV，就要把 m、l、O 写回全局内存，下一块 KV 再读出，多出 Tc 次全局内存往返。
     行优先（复用 Q，版本4/5）：每个线程固定负责一行 Q，m、l、O 全程放在寄存器/共享内存中，遍历完所有 KV 块后才一次性写回，性能更优。

   本版是列优先，贴合 Flash Attention 原始论文的分块推导逻辑 —— 先讲 KV 分块、再讲 Q 遍历过去，直接对应数学公式。
   ---------------------------------------------------------------------------- */
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
    float* Qi = sram;                  // 逻辑形状 [Br, d]
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
                /* 因果掩码二次处理：未来位置的指数值直接强制置 0。
                   前面已置 -INFINITY，这里再判断的原因：
                     1. 双重保险：理论上 exp(-INFINITY)=0，但硬件可能产生极小噪声，
                        直接置 0 可保证因果掩码绝对严格；
                     2. 节省计算：直接赋值 0 比计算 exp(-INFINITY) 更快。 */
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


/* ----------------------------------------------------------------------------
   版本4：Flash Attention-2（行优先 + 序列维度并行 + 延迟归一化）
   ----------------------------------------------------------------------------
   相对版本3 的核心改进：
     1. 交换循环顺序：Q 块在外、KV 块在内，每个 block 固定处理一个 Q 块，Q 仅从 HBM 读取一次，行状态（m、l、输出累加器）全程驻留片上
     2. 序列维度并行：Grid 排布 (Tr, NH, B)，Q 分块放 x 轴，block 间无依赖、无通信；连续调度的 block 访问同一段 KV，提升 L2 缓存命中率
     3. 延迟归一化：内层循环只做乘法/累加、不做除法，遍历完所有 KV 块后统一除以指数和，每行仅 1 次除法
     4. 输出 logsumexp L = m + log(l)，替代分开存储 m、l，减少反向传播的 HBM 读取（纯推理场景可不写）
   ---------------------------------------------------------------------------- */
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
    // 除法移到外循环，每行仅 1 次
    float inv_l = 1.0f / row_l;
    for (int x = 0; x < d; x++) {
        float val = Out_buf[tx * d + x] * inv_l;
        O[qkv_offset + bx * qi_size + tx * d + x] = val;
    }
    // 写回 logsumexp L = m + log(l)，替代分开存 m、l，减少反向传播 HBM 读取（纯推理可不写）
    L[l_offset + q_row] = row_m + logf(row_l);
}


/* ----------------------------------------------------------------------------
   版本5：Flash Attention-2 Warp 级优化版
   ----------------------------------------------------------------------------
   核心特性：
     - split-Q 分块：每个 warp 独立负责连续 ROWS_PER_WARP 行 Q 的完整计算，
       warp 间零通信、零同步，消除跨 warp 规约开销
     - Warp 内归约：通过 cooperative_groups 高效完成行最大值、行指数和
     - 行优先架构：Q 分块常驻共享内存，KV 分块流式加载，降低 HBM 访问量
     - 序列维度并行：每个 block 独立处理一个 Q 块，充分利用 GPU 并行资源
     - 延迟归一化：内层仅累加未归一化分子，末尾统一做除法，减少冗余运算
     - Grid 排布 (Tr, NH, B)：Q 分块放 x 轴优先调度，优化 KV 的 L2 缓存命中
   配置：4 warp/block，每 warp 负责 8 行 Q，Q 分块行数 Br = 4*8 = 32。
   ---------------------------------------------------------------------------- */
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


/* ----------------------------------------------------------------------------
   QKV 拆分 + 维度重排核（FP32 版）
   ----------------------------------------------------------------------------
   输入为打包的 QKV 张量（形状 [B, N, 3, NH, d]），
   输出 Q、K、V 三个张量，形状均为 [B, NH, N, d]。
   一维线程网格，每个线程负责输出的一个元素：
     先把 idx 反解为输出侧四维坐标，再换算到输入侧的打包地址。
   ---------------------------------------------------------------------------- */
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 一维线程网格，每线程负责输出 1 个元素
    // 单张量总元素数 = B*NH*N*d，边界判断防止越界
    if (idx < B * NH * N * d) {
        // 输出侧：把一维 idx 拆解为四维坐标
        int b = idx / (NH * N * d);     // 批次索引：每个 batch 含 NH*N*d 个元素
        int rest = idx % (NH * N * d);  // 当前 batch 内的剩余偏移
        int nh_ = rest / (N * d);       // 注意力头索引：每个头含 N*d 个元素
        rest = rest % (N * d);          // 当前头内的剩余偏移
        int n = rest / d;               // 序列 token 索引：每个 token 含 d 个元素
        int d_ = rest % d;              // 特征维度索引
        // 输入侧：换算打包地址
        int inp_idx = \
            (b * N * 3 * NH * d)  // 跳过前 b 个完整 batch
            +   (n * 3 * NH * d)  // 跳过当前 batch 内前 n 个 token
            +       (0 * NH * d)  // Q 为第 0 个分量，偏移为 0
            +          (nh_ * d)  // 跳过当前分量内前 nh_ 个头
            +                d_;  // 特征维度偏移
        // K、V 在基地址上分别偏移 1、2 个步长 NH*d
        q[idx] = inp[inp_idx];                // Q = 第 0 个分量
        k[idx] = inp[inp_idx + NH * d];       // K = 第 1 个分量
        v[idx] = inp[inp_idx + 2 * (NH * d)]; // V = 第 2 个分量
    }
}

/* 反重排核（FP32 版）：将输出从 [B, NH, N, d] 转回 [B, N, NH, d]，
   匹配后续 FFN（按 token 计算）的输入格式。 */
__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * N * NH * d) {
        // 输入侧：把一维 idx 拆解为四维坐标
        int b = idx / (NH * N * d);
        int nh_ = (idx / (N * d)) % NH;
        int n = (idx / d) % N;
        int d_ = idx % d;
        // 输出侧：计算重排后的目标地址
        int out_idx = b * (NH * N * d) + n * (NH * d) + nh_ * d + d_;
        out[out_idx] = inp[idx];  // 数据搬运：(b,nh_,n,d_) → (b,n,nh_,d_)
    }
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


// ============================================================================
// Kernel 启动封装函数
// ============================================================================

/* 版本1：依次启动 QK 点积、Softmax、加权求和三个 kernel。 */
void attention_forward1(float* out, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // 启动 QK 点积核（一个线程算一个元素）
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);
    // 启动 Softmax 核与加权求和核（一个线程处理一行）
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);
}

/* 版本2：Warp 级融合 kernel，grid=(T/warps_per_block, NH, B)。 */
void attention_forward2(float* out, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    int warps_per_block = block_size / 32;
    dim3 grid_dim(ceil_div(T, warps_per_block), NH, B);
    attention_forward_kernel2<<<grid_dim, block_size>>>(out, preatt, att, inp, B, T, C, NH);
}

// 版本3：Flash Attention 朴素实现启动封装。
// 临时显存全部由调用方预分配传入，启动函数内零分配、零释放：
//   - lm   : 2*B*NH*T 连续显存，l 用前半、m 用后半
//   - qkvr : B*T*3*C 连续显存，指针偏移切出 q/k/v（与版本6 一致）
// 每轮仅保留 cudaMemset 重置统计量初值（对设备显存异步执行，不阻塞 host）。
void attention_forward3(float* out,
                       float* qkvr,
                       float* lm,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // 分块参数目前硬编码为 32
    const int Bc = 32;
    const int Br = 32;
    // 更多参数
    const int d = C / NH;
    const int Tc = ceil((float) T / Bc);
    const int Tr = ceil((float) T / Br);
    const float softmax_scale = 1.0 / sqrt(d);
    // 复用预分配的临时统计量：l、m 各占 B*NH*T
    float* l = lm;
    float* m = lm + B * NH * T;
    // 每轮重置统计量初值（kernel 内 j=0 迭代依赖 m=-10000、l=0）
    cudaCheck(cudaMemset(l, 0, B * NH * T * sizeof(float)));
    cudaCheck(cudaMemset(m, -10000.0f, B * NH * T * sizeof(float)));
    // 计算每个 block 所需的共享内存大小
    int col_tile_size = Bc * d;  // Kj、Vj 的大小
    int row_tile_size = Br * d;  // Qi 的大小
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // Kj、Vj
        + (row_tile_size * sizeof(float))    // Qi
        + (Bc * Br * sizeof(float));         // S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("最大共享内存：%d，请求共享内存：%d \n", max_sram_size, sram_size);
        printf("共享内存大小超出每个线程块的最大限制\n");
        printf("请尝试进一步减小 col_tile_size 或 row_tile_size\n");
        exit(1);
    }
    // 网格与线程块维度
    dim3 grid_dim(B, NH);  // 批次 × 注意力头数
    dim3 block_dim(Br);    // 每块 Br 个线程
    // kernel 期望 Q、K、V 形状为 (B, NH, T, d)，而输入为打包的 (B, T, 3, NH, d)，
    // 因此先用 permute kernel 做重排；q/k/v 从预分配的 qkvr 中指针偏移切出
    float* q = qkvr + 0 * B * T * C;
    float* k = qkvr + 1 * B * T * C;
    float* v = qkvr + 2 * B * T * C;
    int total_threads = B * T * NH * d;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, d);
    // 调用 Flash Attention 核
    attention_forward_kernel3<<<grid_dim, block_dim, sram_size>>>(
        q, k, v,
        T, d, Tc, Tr, Bc, Br, softmax_scale,
        l, m, out
    );
    // 输出形状为 (B, NH, T, d)，需反重排回 (B, T, NH, d)
    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, T, NH, d);
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));
    // 临时显存归调用方管理，无需在此释放
}


// 版本4：Flash Attention-2 算法优化版启动封装。
// 临时显存（L、q/k/v）由调用方预分配传入，启动函数内零分配、零释放。
void attention_forward4(float* out,
                        float* qkvr,
                        float* lm,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    const int Bc = 32;
    const int Br = 32;
    const int d = C / NH;
    const int Tc = ceil((float)T / Bc);
    const int Tr = ceil((float)T / Br);
    const float softmax_scale = 1.0f / sqrtf(d);
    // 复用预分配的 logsumexp 临时显存（替代分开的 l、m，减少反向传播读取次数）
    float* L = lm;
    // 计算共享内存大小（新增 Out_buf 缓冲区）
    int qi_size = Br * d;
    int kv_size = Bc * d;
    int s_size = Br * Bc;
    const int sram_size =
        qi_size * sizeof(float)       // Qi
        + 2 * kv_size * sizeof(float) // Kj + Vj
        + s_size * sizeof(float)      // S
        + qi_size * sizeof(float);    // Out_buf
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("最大共享内存：%d，请求共享内存：%d \n", max_sram_size, sram_size);
        printf("共享内存大小超出每个线程块的最大限制\n");
        printf("请尝试进一步减小Bc或Br\n");
        exit(1);
    }
    // ========== FA2 核心：序列长度维度并行 ==========
    // Grid：Q 分块数 × 注意力头数 × batch，block 间无依赖无通信
    dim3 grid_dim(Tr, NH, B);  // Q 分块放 x 轴首位，连续调度的 block 访问同一段 KV，提升 L2 命中
    dim3 block_dim(Br);
    // 张量重排（QKV 从 (B,T,3,NH,d) 转为 (B,NH,T,d)）
    float* q = qkvr + 0 * B * T * C;
    float* k = qkvr + 1 * B * T * C;
    float* v = qkvr + 2 * B * T * C;
    int total_threads = B * T * NH * d;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, d);
    // 调用 FA2 核
    attention_forward_kernel3_row_major_fa2<<<grid_dim, block_dim, sram_size>>>(
        q, k, v,
        T, d, Tc, Bc, Br, softmax_scale,
        L, out
    );
    // 反重排：(B,NH,T,d) 转回 (B,T,NH,d)
    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, T, NH, d);
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));
    // 临时显存归调用方管理，无需在此释放
}


// 版本5：Flash Attention-2 Warp 版启动封装。
// 临时显存（L、q/k/v）由调用方预分配传入，启动函数内零分配、零释放。
void attention_forward5(float* out,
                        float* qkvr,
                        float* lm,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // 分块配置
    const int Bc = 32;              // KV 分块列数
    const int Br = 32;              // Q 分块行数
    const int WARPS_PER_BLOCK = 4;  // 每个 block 的 warp 数
    const int block_threads = WARPS_PER_BLOCK * 32; // block 总线程数

    const int d = C / NH;            // 单头特征维度
    const int Tc = ceil((float)T / Bc); // KV 分块总数
    const int Tr = ceil((float)T / Br); // Q 分块总数
    const float softmax_scale = 1.0f / sqrtf(d); // Softmax 缩放系数
    // 复用预分配的 logsumexp 临时显存，供反向传播使用
    float* L = lm;
    // 计算共享内存总大小
    int qi_size = Br * d;
    int kv_size = Bc * d;
    int s_size = Br * Bc;
    const int sram_size =
        qi_size * sizeof(float)       // Q 分块缓冲区
        + 2 * kv_size * sizeof(float) // K、V 分块缓冲区
        + s_size * sizeof(float)      // 注意力分数缓冲区
        + qi_size * sizeof(float);    // 输出累加缓冲区
    // 检查共享内存是否超出硬件限制
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("最大共享内存：%d，请求共享内存：%d \n", max_sram_size, sram_size);
        printf("共享内存大小超出每个线程块的最大限制\n");
        printf("请尝试进一步减小Bc或Br\n");
        exit(1);
    }
    // Grid 维度排布：(Q 分块数, 注意力头数, batch 数)
    dim3 grid_dim(Tr, NH, B);
    dim3 block_dim(block_threads);
    // 张量重排：(B,T,3,NH,d) 转为 (B,NH,T,d)
    float* q = qkvr + 0 * B * T * C;
    float* k = qkvr + 1 * B * T * C;
    float* v = qkvr + 2 * B * T * C;
    int total_threads = B * T * NH * d;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, d);
    // 启动 FlashAttention-2 Warp 级核
    attention_forward_kernel3_row_major_fa2_warp<<<grid_dim, block_dim, sram_size>>>(
        q, k, v,
        T, d, Tc, Bc, Br, softmax_scale,
        L, out
    );
    // 反重排：(B,NH,T,d) 转回 (B,T,NH,d)
    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, T, NH, d);
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));
    // 临时显存归调用方管理，无需在此释放
}


/* ----------------------------------------------------------------------------
   版本6：cuBLAS + 自定义 Softmax（框架级标准基线）
   ----------------------------------------------------------------------------
   定位：两次计算密集的矩阵乘（Q@Kᵀ、P@V）全部交给 cuBLAS，
        仅 Softmax 这类带因果掩码的非标运算由自定义 kernel 完成，类似 PyTorch 的实现。
   流水线（六步）：
     维度重排 → cuBLAS QKᵀ → 缩放+因果掩码 → Softmax → cuBLAS P@V → 反重排
   临时显存：qkvr 为一块连续显存，通过指针偏移切分为重排后的 Q、K、V 三份。

   优缺点总结：
     架构       cuBLAS 矩阵乘 + 自定义 Softmax，工业标准基线架构
     显存复杂度  O(T²)，需完整存储 [T,T] 注意力矩阵，长序列受限
     性能       矩阵乘为硬件级优化，远高于手写朴素版（版本1/2）
     适用场景   短到中等序列长度的通用场景，深度学习框架默认实现范式
   与版本3 对比：版本6 实现简单、依赖成熟库但长序列显存 O(T²)；
                版本3 显存 O(T) 但手写矩阵乘性能低。
   ---------------------------------------------------------------------------- */
void attention_forward6(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    int HS = C / NH;
    // 从连续显存 qkvr 中切出重排后的 Q、K、V
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    /* 第一步：维度重排（permute）
       1. 拆分：打包的 QKV 拆成 Q、K、V 三个独立张量
       2. 转布局：从「序列优先」[B,T,3,NH,HS] 转为「头优先」[B,NH,T,HS] */
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    /* 第二步：cuBLAS 批量矩阵乘 —— Q @ Kᵀ
       数学公式：
         preatt[b,h] = Q[b,h] × Kᵀ[b,h]，形状 [T,HS]×[HS,T] = [T,T]
       接口说明：
         - 批量数为 B*NH，每个 batch、每个头各执行一次独立矩阵乘
         - cublasSgemmStridedBatched 批量执行多个相同尺寸的矩阵乘，一次性算出所有头的分数矩阵，比循环调用单次 gemm 开销小很多
         - cuBLAS 原生为列优先（Fortran 风格），我们的张量为行优先（C 风格），转置参数已做对应适配，最终结果与数学定义一致。 */
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N, // A 转置（K），B 不转置（Q）
                            T, T, HS,                 // m=行数, n=列数, k=公共维度
                            &alpha,
                            k, HS, T * HS,            // A=K：行步长 HS，批间步长 T*HS
                            q, HS, T * HS,            // B=Q：行步长 HS，批间步长 T*HS
                            &beta,
                            preatt, T, T * T,         // C=preatt：行步长 T，批间步长 T*T
                            B * NH));                 // 总批量数 = 批次×头数
    /* 第三步：缩放 + 因果掩码
       1. 缩放：乘以 1/√HS，防止点积过大导致 Softmax 梯度消失
       2. 因果掩码：上三角（未来位置）强制置 -INFINITY，保证自回归特性 */
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);
    /* 第四步：两级规约 Softmax 核
       输入：缩放后的分数 preatt [B,NH,T,T]
       输出：归一化权重 att [B,NH,T,T]
       一个 block 处理一行（共 B*NH*T 行），块内 warp 内寄存器规约 + 共享内存跨 warp 规约，高效求最大值与指数和 */
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel5<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);
    /* 第五步：cuBLAS 批量矩阵乘 —— Att @ V
       vaccum[b,h] = att[b,h] × V[b,h]，形状 [T,T]×[T,HS] = [T,HS]
       同样为批量计算，每个头独立完成。 */
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));
    /* 第六步：反维度重排（unpermute）
       输入：vaccum [B,NH,T,HS]
       输出：out [B,T,C]（C = NH×HS，所有头拼接）
       原因：后续 FFN 按 token 计算，序列优先布局下同一 token 的所有特征连续存放，访存效率最高。 */
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}


/* 版本7：cuBLAS + 融合在线 Softmax。
   除 Softmax 替换为 softmax_forward_kernel6 外，其余流程与版本6 完全一致。 */
void attention_forward7(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    int HS = C / NH;
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    // 第一步：维度重排，拆分 QKV 并转为头优先布局
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    // 第二步：cuBLAS 批量矩阵乘：Q @ Kᵀ → preatt [B,NH,T,T]
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));
    /* 第三步：融合在线 Softmax（缩放融合 + 在线递推 + float4 + 因果掩码）
       1. 缩放融合：不再单独走一遍缩放 kernel，在 Softmax 中同时完成缩放，减少一次完整 [B,NH,T,T] 矩阵的读写
       2. 在线 Softmax 替代两遍法：
          - kernel5 为两遍法：第一遍找最大值、第二遍算指数和，两次完整行遍历
          - kernel6 为在线递推：遍历元素的同时动态更新最大值与总和，旧总和按新最大值自动折算，只需一次行遍历，访存减半
       3. float4 向量化：以 4 个 float 为单位批量读取，一个 warp 处理一行，访存带宽利用率更高
       4. 因果掩码原生融合：只计算下三角区域，未来位置直接跳过，无需先整体置 -INFINITY，省掉无效区域的读写与计算 */
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel6<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    // 后半段与版本6 完全一致
    // 第四步：cuBLAS 批量矩阵乘：att @ V → vaccum [B,NH,T,HS]
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));
    // 第五步：反维度重排，头优先转回序列优先，输出最终结果
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}


/* ----------------------------------------------------------------------------
   版本8：FP16/BF16 混合精度版（版本7 的低精度适配）
   ----------------------------------------------------------------------------
   核心思路：所有中间驻留显存的张量使用 floatX 存储，显存减半；
            矩阵乘与 Softmax 内部自动升回 FP32 计算，精度损失仅发生在存储量化环节——低精度推理的标准实现。

   与版本7 的对比：
     维度          版本7（全 FP32）       版本8（混合精度）
     中间张量存储   FP32                   floatX（FP16/BF16），显存减半
     矩阵乘计算     FP32                   IO 为 FP16、内部 FP32
     Softmax      FP32                   IO 为 FP16、内部 FP32
     重排核        仅重排，无精度转换       重排 + 精度转换融合，一次 IO 完成
     cuBLAS 接口   普通版                 Ex 扩展版（可指定类型与计算精度）
     精度损失       无                     仅两次存储量化误差，计算无额外损失

   skip_permute 开关：
     - false（默认）：完整执行重排
     - true：跳过重排，单独测量「矩阵乘 + Softmax」的耗时，用于量化维度重排的 IO 开销（版本9 使用）
     - 首次运行（first_run_validation）强制执行重排，保证结果正确
   ---------------------------------------------------------------------------- */
void attention_forward8(float* out, floatX* vaccum, floatX* qkvr, floatX* preatt, floatX* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size, bool skip_permute=false) {
    int HS = C / NH;
    floatX *q = qkvr + 0 * B * T * C;
    floatX *k = qkvr + 1 * B * T * C;
    floatX* v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    /* 第一步：低精度重排 + 量化（带跳过开关）
       1. 拆分打包的 QKV 为三个独立张量
       2. 布局从「序列优先」转为「头优先」，适配 cuBLAS 批量矩阵乘
       3. FP32 → floatX 降精度，压缩存储
       首次运行强制执行；skip_permute=true 时仅在性能测试中跳过。 */
    if (!skip_permute || first_run_validation) {
        permute_kernel_lowp<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    }
    /* cuBLAS GEMM 计算精度模式与缩放系数适配（重点细节）：
       模式                    含义                          α/β 类型
       CUBLAS_COMPUTE_32F      IO 为 FP16、内部 FP32（默认）  必须传 FP32 的 α/β
       CUBLAS_COMPUTE_16F      全链路 FP16，更快但精度差       必须传 FP16 的 α/β
       类型不匹配时不会报错，但结果完全错误；
       代码根据 CUBLAS_LOWP_COMPUTE 自动选择对应精度的指针。 */
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const floatX alpha_lowp = (floatX)alpha;
    const floatX beta_lowp = (floatX)beta;
    // 根据当前低精度计算模式，自动选择 α/β 指针
    void* alpha_ptr = CUBLAS_LOWP_COMPUTE == CUBLAS_COMPUTE_16F ? (void*)&alpha_lowp : (void*)&alpha;
    void* beta_ptr  = CUBLAS_LOWP_COMPUTE == CUBLAS_COMPUTE_16F ? (void*)&beta_lowp : (void*)&beta;
    // 第二步：cuBLAS 批量矩阵乘，计算 Q @ Kᵀ
    // cublasGemmStridedBatchedEx 为扩展版，支持指定 IO 类型与内部计算精度，专为混合精度设计
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            alpha_ptr,
                            k, CUBLAS_LOWP, HS, T * HS,    // A=K：数据类型 + 行步长 + 批间步长
                            q, CUBLAS_LOWP, HS, T * HS,    // B=Q：数据类型 + 行步长 + 批间步长
                            beta_ptr,
                            preatt, CUBLAS_LOWP, T, T * T, // C=preatt：数据类型 + 行步长 + 批间步长
                            B * NH,
                            CUBLAS_LOWP_COMPUTE,    // 内部计算精度（默认 32 位混合精度，也可配置 16 位）
                            CUBLAS_GEMM_DEFAULT));  // GEMM 算法默认优选
    // 第三步：低精度在线 Softmax（IO 为 floatX，内部全程 FP32，保证数值稳定）
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel6_lowp<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    // 第四步：cuBLAS 批量矩阵乘，计算 att @ V
    // 参数逻辑与第一次矩阵乘完全一致，同样适配两种计算模式
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            alpha_ptr,
                            v, CUBLAS_LOWP, HS, T * HS,
                            att, CUBLAS_LOWP, T, T * T,
                            beta_ptr,
                            vaccum, CUBLAS_LOWP, HS, T * HS,
                            B * NH,
                            CUBLAS_LOWP_COMPUTE,
                            CUBLAS_GEMM_DEFAULT));
    // 第五步：反重排 + 升精度
    num_blocks = ceil_div(B * T * C, block_size);
    if(!skip_permute || first_run_validation) {
        unpermute_kernel_lowp<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    }
}


// 版本9：性能基准专用 —— 跳过 permute/unpermute 环节，单独测量「矩阵乘 + Softmax」本身的耗时（不对应真实部署场景）
void attention_forward9(float* out, floatX* vaccum, floatX* qkvr, floatX* preatt, floatX* att,
                     const float* inp,
                     int B, int T, int C, int NH,
                     const int block_size) {
    attention_forward8(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size, true);
}


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

/* 精度转换核：FP32 → floatX / floatX → FP32，仅首次验证时使用。 */
__global__ void fp32_to_lowp_kernel(floatX* out, const float* inp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (floatX)inp[idx];
}
__global__ void lowp_to_fp32_kernel(const floatX* inp, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (float)inp[idx];
}

/* ----------------------------------------------------------------------------
   版本10：cuDNN Flash Attention
   ----------------------------------------------------------------------------
   输入：inp（打包 QKV，形状 [B,T,3,NH,HS]，floatX）
   输出：out（形状 [B,T,C]，floatX）
   stats：Softmax 统计量（形状 [B,NH,T]），传 nullptr 走纯推理模式，不生成反向统计量、性能更高。
   特点：
     - 配合 stride 对打包 QKV 做零拷贝切片，无需 permute 核
     - 工作空间按需动态分配（上限 256MiB），不直接按最大值分配
     - 首次运行执行 FP32→低精度→FP32 的完整链路，用于与 CPU 结果验证
   ---------------------------------------------------------------------------- */
void attention_forward_cudnn(floatX* out,  // 输出：(B, T, NH, HS)
                             float* stats, // Softmax 统计量：(B, NH, T)
                             floatX* inp,  // 输入：(B, T, 3, NH, HS) QKV
                             float* in_fp32, float* out_fp32, // 仅用于验证的 FP32 输入/输出
                             int B, int T, int C, int NH,
                             bool skip_conversion = false) {
    int HS = C / NH;
    bool is_inference_only = (stats == nullptr);
    const int block_size = 64;
    // 精度转换：不跳过或处于验证阶段时执行
    if (!skip_conversion || first_run_validation) {
        int total_threads = B * T * C * 3;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        fp32_to_lowp_kernel<<<num_blocks, block_size>>>(inp, in_fp32);
    }
    // 获取计算图（缓存优先，无则新建）
    auto [graph, Q, K, V, attn_scale, O, softmax_stats] =
        lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);
    // 绑定真实物理显存地址（配合图中的 stride，实现零拷贝张量切片）
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;
    // 构建变体包（逻辑张量 → 物理地址映射表）
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {O, devPtrO}};
    // 训练模式下额外绑定 stats，供反向传播使用
    if (is_inference_only == false) {
        variant_pack[softmax_stats] = stats;
    }
    // 动态管理工作空间：按需扩容，不直接分配上限 256MiB
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }
    // 执行计算图：传入句柄、地址映射表与工作空间，cuDNN 按编译好的最优计划完成计算
    assert(graph->execute(cudnn_handle, variant_pack, cudnn_workspace).is_good());
    cudaCheck(cudaGetLastError());  // 检查内核是否正常启动
    // 精度转换：不跳过或处于验证阶段时执行
    if (!skip_conversion || first_run_validation) {
        int total_threads = B * T * C;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        lowp_to_fp32_kernel<<<num_blocks, block_size>>>(out, out_fp32);
    }
    cudaCheck(cudaGetLastError());
}

/* 版本11：cuDNN 全低精度链路（跳过 FP32/低精度转换）。
   定位：纯算子性能基准，对应全 FP16/BF16 网络的真实部署场景。 */
void attention_forward_cudnn_pure(floatX* out, float* stats, floatX* inp,
                                  float* in_fp32, float* out_fp32,
                                  int B, int T, int C, int NH) {
    attention_forward_cudnn(out, stats, inp, in_fp32, out_fp32, B, T, C, NH, true);
}
#endif // ENABLE_CUDNN


// ============================================================================
// Kernel 版本分发
// ============================================================================
void attention_forward(int kernel_num,
                       float* out, float* stats, float* vaccum,
                       float* qkvr, float* preatt, float* att,
                       float* inp, float* lm,
                       int B, int T, int C, int NH,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_forward1(out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 2:
            attention_forward2(out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 3:
            attention_forward3(out, qkvr, lm, inp, B, T, C, NH, block_size);
            break;
        case 4:
            attention_forward4(out, qkvr, lm, inp, B, T, C, NH, block_size);
            break;
        case 5:
            attention_forward5(out, qkvr, lm, inp, B, T, C, NH, block_size);
            break;
        case 6:
            attention_forward6(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 7:
            attention_forward7(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 8:
            attention_forward8(out, (floatX*)vaccum, (floatX*)qkvr,
                              (floatX*)preatt, (floatX*)att,
                              inp, B, T, C, NH, block_size, false);
            break;
        case 9: // 性能测试时跳过排列（模拟纯 16 位输入输出的性能）
            attention_forward9(out, (floatX*)vaccum, (floatX*)qkvr,
                              (floatX*)preatt, (floatX*)att,
                              inp, B, T, C, NH, block_size);
            break;

        #ifdef ENABLE_CUDNN  // 仅在开启 cuDNN 支持时编译
        case 10:
            // 传 nullptr 走纯推理模式，不生成反向统计量，性能更高；
            // 验证只关心最终输出 out
            attention_forward_cudnn((floatX*)vaccum, nullptr, (floatX*)qkvr, inp, out, B, T, C, NH);
            break;
        case 11:
            // 版本11：跳过精度转换，纯低精度计算，对应全 FP16/BF16 网络场景
            attention_forward_cudnn_pure((floatX*)vaccum, nullptr, (floatX*)qkvr, inp, out, B, T, C, NH);
            break;
        #endif

        default:
            printf("无效的核函数编号\n");
            exit(1);
    }
}


// ============================================================================
// 主程序：正确性验证 + 性能基准测试
// ============================================================================
int main(int argc, char **argv) {
    setup_main();
    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // 初始化 cuBLAS（可选 cuDNN）
    cublasCreate(&cublas_handle);
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    #ifdef ENABLE_CUDNN
    checkCudnnErr(cudnnCreate(&cudnn_handle));
    #endif

    // 主机（CPU）端内存
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    //float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float* inp = make_random_float(B * T * 3 * C);

    // 设备（GPU）端显存
    float* d_out;
    float* d_stats; // 用于 cuDNN
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    float* d_lm;   // v3 的 l/m、v4/v5 的 L 共用 scratch（大小 2*B*NH*T）
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_stats, B * NH * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    // 版本3/4/5 的中间统计量 scratch：v3 的 l/m 各占一半，v4/v5 的 L 用前半
    cudaCheck(cudaMalloc(&d_lm, 2 * B * NH * T * sizeof(float)));
    // 数据迁移到 GPU
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // 从命令行读取核函数编号
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用核函数 %d\n", kernel_num);

    // 如果不是 profile 模式，则执行标准验证与测试流程
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        // 标准（非 profile）模式下版本号必须合法；0 仅在 profile 模式表示"采集全部版本"
        if (kernel_num < 1 || kernel_num > 11) {
            printf("用法: ./attention_forward <版本1-11> [profile]\n");
            printf("       ./attention_forward 0 profile  # 采集全部版本（配合 NCU/NSYS）\n");
            return 1;
        }

        // 逐个测试多种块尺寸，寻找当前硬件上的最优配置
        int block_sizes[] = {32, 64, 128, 256, 512};
        // 版本 1~7（全 FP32）：误差阈值 1e-3；版本 8+（低精度）：放宽到 1e-2
        float accuracy_threshold = (kernel_num <= 7) ? 1e-3f : 1e-2f;

        // 第一步：CPU 参考实现计算标准答案
        attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
        // 第二步：遍历所有块尺寸验证正确性
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("正在验证块大小 %d。\n", block_size);
            attention_forward(kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, d_lm, B, T, C, NH, block_size);
            // 所有版本均验证最终输出 out
            validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
            // 跳过版本 3/4/5：分块在线计算，不生成完整 preatt/att 大矩阵，无中间结果可对比
            if (kernel_num != 3 && kernel_num != 4 && kernel_num != 5 && kernel_num < 8) {
                validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
            }
            // 跳过版本 7+ 的 preatt：版本 7 起缩放融合进 Softmax，preatt 与 CPU 版相差 1/√HS 倍，
            // 但最终输出正确
            if (kernel_num != 3 && kernel_num != 4 && kernel_num != 5 && kernel_num < 7) {
                validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);
            }
        }
        printf("所有结果匹配。开始基准测试。\n\n");

        // 第三步：性能基准测试
        first_run_validation = false; // 关闭验证开关
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            int repeat_times = 20;   // 重复多次取平均
            float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                                 kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att,
                                                 d_inp, d_lm, B, T, C, NH, block_size);
            printf("块大小 %4d | 耗时 %f 毫秒\n", block_size, elapsed_time);
        }

        // 资源释放与退出
        free(out);
        free(preatt);
        free(att);
        free(inp);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_vaccum));
        cudaCheck(cudaFree(d_qkvr));
        cudaCheck(cudaFree(d_preatt));
        cudaCheck(cudaFree(d_att));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_lm));
        cudaCheck(cudaFree(d_stats));
        cublasDestroy(cublas_handle);
        #ifdef ENABLE_CUDNN
        cudnnDestroy(cudnn_handle);
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        #endif
        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     注意（各版本每轮启动的 Kernel 数，经 ncu 实测固定）：
       v1 三核（qk/softmax/value）、v2 单核
       v3/v4/v5 各三核（permute + flash + unpermute）
       v6 六核（permute + QKᵀ + scale + softmax + PV + unpermute）
       v7/v8 各五核（permute + QKᵀ + softmax + PV + unpermute）
       v9 三核（跳过 permute/unpermute）、v10 三核（转换 + cuDNN + 转换）、v11 单核（纯 cuDNN）
       基础版 v1~v9 一轮共 32 个 Kernel，预热 2 轮 = 64 个；
       cuDNN 版 v1~v11 一轮共 36 个 Kernel，预热 2 轮 = 72 个。

     编译指令示例：
       # v1~v9（基础版）
       nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward
       # v1~v11（cuDNN 版，需本机已装 cudnn-frontend）（地址改成自己的）
       nvcc -I/home/yolan/projects/cudnn-frontend-main/include -DENABLE_CUDNN -O3 -lineinfo \
            --use_fast_math -lcublas -lcublasLt -lcudnn -lnvrtc attention_forward.cu -o attention_forward

     NCU 采集命令示例（跳过 2 轮预热启动，采集第 3 轮的正式启动）：
       # v1~v9（基础版）：跳过 64 次预热启动，采集接下来的 32 次
       ncu --set full --cache-control all --launch-skip 64 --launch-count 32 \
           -o attention_profile_128 ./attention_forward 0 profile
       # v1~v11（cuDNN 版）：跳过 72 次预热启动，采集接下来的 36 次
       ncu --set full --cache-control all --launch-skip 72 --launch-count 36 \
           -o attention_profile_cudnn_128 ./attention_forward 0 profile

     系统级性能分析：
       nsys profile --stats=true -o attention_nsys ./attention_forward 0 profile
     */
    {
        const int profile_block_size = 128;
        const int warmup_rounds = 2;
        first_run_validation = false; // 与基准测试一致：跳过首次运行的重排/转换检查

        #ifdef ENABLE_CUDNN
        const int max_ver = 11; // cuDNN 版：采集 v1~v11
        #else
        const int max_ver = 9;  // 基础版：采集 v1~v9
        #endif

        // 预热 2 轮：依次调用版本 1 到 max_ver
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= max_ver; ver++) {
                attention_forward(ver, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, d_lm,
                                  B, T, C, NH, profile_block_size);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮：依次调用版本 1 到 max_ver
        for (int ver = 1; ver <= max_ver; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            attention_forward(ver, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, d_lm,
                              B, T, C, NH, profile_block_size);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放资源
        free(out);
        free(preatt);
        free(att);
        free(inp);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_vaccum));
        cudaCheck(cudaFree(d_qkvr));
        cudaCheck(cudaFree(d_preatt));
        cudaCheck(cudaFree(d_att));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_lm));
        cudaCheck(cudaFree(d_stats));
        cublasDestroy(cublas_handle);
        #ifdef ENABLE_CUDNN
        cudnnDestroy(cudnn_handle);
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        #endif
        return 0;
    }

}
