/* ============================================================================
   GPT-2 位置编码前向传播的 CUDA Kernel 实现
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_forward.cu -o encoder_forward

   性能剖析（ncu/nsys）编译示例：
     nvcc -O3 -g -G --use_fast_math encoder_forward.cu -o encoder_forward -lcublas -lcublasLt

   版本说明：
     版本1：从 CPU 代码朴素移植，在 B、T 维度并行，C 维度循环遍历。
     版本2：在 B、T、C 全部维度上并行，每个线程直接处理 1 个输出元素。
     版本3：在版本 2 基础上使用 128bit 向量化访存（x128 数据包），每个线程一次处理一组连续通道。

   运行命令（数字对应版本号）：
     ./encoder_forward 1
     ./encoder_forward 2
     ./encoder_forward 3
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>

#define ENABLE_BF16 // 启用 BF16（脑浮点数16位）精度支持，common.h 据此开启 bf16 相关类型与算子实现
#include "../common.h"

// ============================================================================
// CPU 参考实现
// ============================================================================
// GPT-2 位置编码前向传播：out[b,t,:] = wte[inp[b,t],:] + wpe[t,:]
void encoder_forward_cpu(float* out,
                   const int* inp, const float* wte, const float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C; // 定位到词嵌入表 wte[ix,:]
            const float* wpe_t = wpe + t * C;   // 定位到位置编码表 wpe[t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i]; // 词嵌入与位置编码逐元素相加
            }
        }
    }
}

// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   低精度加法处理说明
   ----------------------------------------------------------------------------
   GPU 原生 half / __nv_bfloat16 不支持直接 + 运算符（Ampere 以前的硬件），有两条可行路线：
     方案 A（当前写法）：提升到 float 计算，再转回低精度（简单通用，可读性强）。
     方案 B：使用内置函数 __hadd2 / __bfloat16_add 直接低精度运算（性能略优，但代码不通用）。
   选型：采用方案 A，牺牲一点点性能，换取一套代码兼容 fp32/fp16/bf16。

   精度说明：单纯类型转换只是寄存器内操作，几乎没有开销；
            真正的误差来自低精度→fp32 计算后再舍入回低精度。
   ---------------------------------------------------------------------------- */

// 版本1：朴素移植实现，B、T 维度并行，C 维度循环遍历
__global__ void encoder_forward_kernel1(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (idx < N) {
        int b = idx / T;
        int t = idx % T;

        int ix = inp[b * T + t];
        const floatX* wte_ix = wte + ix * C;
        const floatX* wpe_t = wpe + t * C;
        floatX* out_bt = out + b * T * C + t * C;
        for (int i = 0; i < C; i++) {
            out_bt[i] = (floatX)((float)wte_ix[i] + (float)wpe_t[i]); // 提升到 float 相加，再转回低精度
        }
    }
}

/* ----------------------------------------------------------------------------
   版本2：全维度并行
   ----------------------------------------------------------------------------
   任务划分：在 B、T、C 全部维度上并行，每个线程只处理 1 个输出元素。
   相比版本1：把 C 维度的串行循环摊到更多线程上，并行度更高。
   ---------------------------------------------------------------------------- */
__global__ void encoder_forward_kernel2(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 单个元素的全局下标
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;
        floatX* out_btc = out + b * T * C + t * C + c;
        *out_btc = (floatX)((float)*wte_ix + (float)*wpe_tc);
    }
}

/* ----------------------------------------------------------------------------
   版本3：128bit 向量化访存
   ----------------------------------------------------------------------------
   核心优化：以 x128 数据包为单位读写，每个线程一次处理一组连续通道（FP32 模式 4 个、BF16 模式 8 个）。

   x128 是 Packed128<floatX> 的类型别名，16 字节对齐结构体，内部是 size 个连续的 floatX 元素；
   总大小固定 16 字节（128bit），刚好匹配 GPU 的 128bit 访存指令，满足硬件向量化对齐要求。
   x128::size 是类型自带的静态常量（static constexpr），用「类型::常量名」访问。

   下标换算：
     版本2：线程号直接当元素下标（idx = 线程号）。
     版本3：线程号 × size 作为数据包起始下标（idx = 线程号 × x128::size）。
     两套核函数的宿主调用代码完全一致，传参没有区别。

   store128 的作用：
     通过 int4 类型双关，强制编译器生成单条 128bit 宽存储指令，防止退化成多条 32bit/16bit 零散小规模访存；
     单纯用赋值 *(x128*)out = packed_out 存在类型别名与结构体内存布局的可移植风险，因此封装了 store128。
     访存策略：wte/wpe 用流式读 load128cs（不驻留缓存），输出用普通 store128，
     因为输出激活值后续注意力、FFN 层马上要读，留在 L1/L2 缓存里可提升后续命中率。

   优化效果：SM 前端指令数相比版本2 显著减少（size 越大收益越大）。
   ---------------------------------------------------------------------------- */
// 版本3：在版本2基础上使用 float4/x128 向量化读写
__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size; // 当前线程负责的数据包起始元素下标
    int N = B * T * C;
    if (idx < N) {
        int bt = idx / C; // 当前数据包属于第几个 token
        int b = bt / T;
        int t = bt % T;
        int c = idx % C; // 数据包起点通道，后面连续 size 个通道
        int ix = inp[b * T + t];
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;
        floatX* out_btc = out + b * T * C + t * C + c; // 指针指向数据包第一个通道

        x128 packed_out;               // 寄存器里的输出数据包
        x128 wte = load128cs(wte_ix);  // 流式读：一次读 size 个词嵌入元素打包进寄存器（局部变量名遮蔽了同名参数）
        x128 wpe = load128cs(wpe_tc);  // 流式读：一次读 size 个位置编码元素
        #pragma unroll
        for (int k = 0; k < wte.size; k++) { // wte.size 是编译期常量，可完整展开
            packed_out[k] = (floatX)((float)wte[k] + (float)wpe[k]);
        }
        store128(out_btc, packed_out); // 普通存储：输出留在缓存，供后续注意力、FFN 层读取
    }
}

// ============================================================================
// Kernel 启动封装函数
// ============================================================================

// 版本1：总线程数 = token 总数（B * T），每线程处理整行 C 个通道
void encoder_forward1(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// 版本2：总线程数 = 元素总数（B * T * C），每线程处理 1 个元素
void encoder_forward2(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// 版本3：总线程数 = 数据包总数（B * T * C / x128::size），每线程处理一组连续通道
void encoder_forward3(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// Kernel 版本调度分发
void encoder_forward(int kernel_num,
                     floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    switch (kernel_num) {
        case 1: encoder_forward1(out, inp, wte, wpe, B, T, C, block_size); break;
        case 2: encoder_forward2(out, inp, wte, wpe, B, T, C, block_size); break;
        case 3: encoder_forward3(out, inp, wte, wpe, B, T, C, block_size); break;
        default:
            printf("无效的内核编号\n");
            exit(1);
    }
}

// ============================================================================
// 主程序：正确性验证 + 性能基准测试
// ============================================================================
int main(int argc, char **argv) {
    setup_main(); // 全局初始化：随机种子、GPU 设备、cuBLAS、TF32 开关

    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // 在主机端创建随机数内存
    float* out = (float*)malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V);
    float* wte = make_random_float(V * C);
    float* wpe = make_random_float(T * C);

    // 数据迁移到 GPU 端
    floatX* d_out;
    int* d_inp;
    floatX* d_wte;
    floatX* d_wpe;
    // 申请 GPU 显存（大小按 floatX 自动适配：FP32=4 字节，BF16=2 字节）
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, V * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_wpe, T * C * sizeof(floatX)));
    // token id 是 int 类型，直接普通拷贝；权重是 float，需转成 floatX 再用封装好的转换拷贝函数
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(memcpy_convert(d_wte, wte, V * C));
    cudaCheck(memcpy_convert(d_wpe, wpe, T * C));

    // 从命令行选择要测试的 kernel 版本
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 先跑 CPU 版本，得到标准答案，再验证 GPU 结果
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

    // 遍历不同 block_size，逐一验证结果正确，再开始测速
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("正在测试线程块大小 %d.\n", block_size);
        encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);

// 根据精度模式设置误差容忍阈值
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;  // FP32 精度高，误差超过 1e-5 就算错
#else
        float tol = 1e-2f; // BF16/FP16 低精度本身有舍入误差，容忍度放宽到 1e-2
#endif
        validate_result(d_out, out, "out", B * T * C, tol); // 对比 GPU 结果与 CPU 标准答案，超误差即报错
    }
    printf("所有结果验证通过，开始性能基准测试。\n\n");

    // 测试不同线程块大小下的 kernel 耗时
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        // 封装的计时函数：内部用 CUDA 事件精确计时，返回平均单次耗时（毫秒）
        float elapsed_time = benchmark_kernel(repeat_times, encoder_forward,
                                              kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);

        // 粗略估算显存带宽：每个输出元素对应 4 次访存（读 inp/wte/wpe + 写 out），每次按 4 字节估算
        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 带宽 %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // 释放内存
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe));

    return 0;
}
