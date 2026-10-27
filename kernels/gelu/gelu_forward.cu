/* ============================================================================
   GELU 前向传播的内核实现
   ============================================================================
   编译示例：
   nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_forward.cu -o gelu_forward

   若编译报错 "error: identifier "M_PI" is undefined"，请在文件顶部添加以下两行：
   #define _USE_MATH_DEFINES
   #include <math.h>  或  #include <cmath>

   版本 1：朴素的 CPU 移植版本
   运行命令：./gelu_forward 1

   版本 2：基于 Packed128 数据结构的 bfloat16 版本
   运行命令：./gelu_forward 2
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "../common.h"

// ============================================================================
// CPU 参考实现
// ============================================================================
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        // 用 tanh 近似版 GELU 公式，而非精确的 erf 版。在精度损失可忽略的前提下大幅提升计算速度，是推理框架的通用实现。
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// ============================================================================
// GPU 内核函数
// ============================================================================
// 朴素版本（单元素读写，每次只加载2/4字节，无法打满显存带宽）
__global__ void gelu_forward_kernel1(floatX* out, const floatX* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

/* 向量化版本，每个线程负责一包 x128::size 个连续元素
 1.访存向量化
 逐元素算子是'显存带宽受限型'算子，计算量很小，瓶颈在显存读写。128bit 向量化读写可以：
 - 减少访存指令数量，降低指令发射与调度开销
 - 触发硬件合并访存，最大化显存带宽利用率
 这是逐元素算子收益最高、最基础的优化手段。

 2.计算标量化
 GELU 包含 tanh 超越函数，计算部分逐个标量执行，同时计算过程提升到 FP32 精度下保证精度

 3.缓存策略选型
 - 读操作用 load128cs（流式加载，避缓存）：输入数据只读一次，后面不会立刻复用，用流式加载避免污染缓存，留给后续层使用。
 - 写操作用 store128（普通存储，走缓存）：GELU 的输出是下一层的输入，很快会被再次读取，保留在缓存里提升后续操作的性能。
 */
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size; // 全局索引 × 向量包大小
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i);  // 不再复用，流式加载避免污染缓存
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        store128(out + i, packed_out); // 使用普通存储而非流式存储，GELU 的输出是下一层的输入，很快会被再次读取
    }
}

// ============================================================================
// 内核启动函数
// ============================================================================
void gelu_forward1(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward2(floatX* out, const floatX* inp, int N, const int block_size) {
    // grid 计算变为 ceil_div(N, block_size * x128::size)，因为每个线程处理的元素数变多了，需要的 block 数量对应减少
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// 内核版本调度函数
void gelu_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_forward1(out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(out, inp, B * T * C, block_size);
            break;
        default:
            printf("无效的内核版本\n");
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

    // 在主机端分配内存并生成随机数
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // 从命令行参数读取内核版本号
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 将数据迁移到 GPU
    floatX* d_out;
    floatX* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));

    // 默认模式（正确性验证+性能基准测试）
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        // 首先验证内核计算的正确性
        gelu_forward_cpu(out, inp, B * T * C);

        // 在不同线程块大小下测试内核性能
        int block_sizes[] = {32, 64, 128, 256, 512, 1024};
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("正在测试线程块大小 %d.\n", block_size);
            gelu_forward(kernel_num, d_out, d_inp, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
            float tol = 1e-5;  // FP32 用 1e-5 小容差
#else
            float tol = 1e-2f; // BF16/FP16 用 1e-2 大容差
#endif
            validate_result(d_out, out, "out", B * T * C, tol);
        }

        printf("所有结果验证通过，开始性能基准测试。\n\n");

        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            int repeat_times = 100;
            float elapsed_time = benchmark_kernel(repeat_times, gelu_forward,
                                                  kernel_num, d_out, d_inp,
                                                  B, T, C, block_size);

            // 估算实际显存带宽：每个元素对应 1 次读 + 1 次写，每个元素占 4 字节
            long memory_ops = B * T * C * 2 * (int)sizeof(floatX);
            float memory_bandwidth = memory_ops / elapsed_time / 1e6;

            printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 带宽 %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
        }

        // 释放内存
        free(out);
        free(inp);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        return 0;
    }

    // ========== 性能采集模式（配合NCU/NSYS使用） ==========
    /* 性能分析命令示例：
     nvcc -O3 --use_fast_math -lineinfo -lcublas -lcublasLt gelu_forward.cu -o gelu_forward

     全尺寸批量采集：
     ncu --set full \
         --kernel-name gelu_forward_kernel2 \
         --launch-skip 18 --launch-count 6 \
         --export gelu_kernel2_all_blocks.ncu-rep \
         ./gelu_forward 2 profile
     
     单尺寸精准采集：
     ncu --set full \
         --kernel-name gelu_forward_kernel2 \
         --launch-skip 20 --launch-count 1 \
         --export gelu_kernel2_128.ncu-rep \
         ./gelu_forward 2 profile
     
     系统级性能分析：
     nsys profile --stats=true -o gelu_nsys ./gelu_forward 2 profile
     */
    {
        int profile_blocks[] = {32, 64, 128, 256, 512, 1024};
        int num_blocks = sizeof(profile_blocks) / sizeof(int);
        int warmup_rounds = 3;
        printf("性能采集模式：%d 种线程块尺寸，%d 轮预热\n", num_blocks, warmup_rounds);

        // 预热轮次
        for (int round = 0; round < warmup_rounds; round++) {
            for (int i = 0; i < num_blocks; i++) {
                gelu_forward(kernel_num, d_out, d_inp, B, T, C, profile_blocks[i]);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集轮次
        for (int i = 0; i < num_blocks; i++) {
            gelu_forward(kernel_num, d_out, d_inp, B, T, C, profile_blocks[i]);
        }
        cudaCheck(cudaDeviceSynchronize());

        free(out);
        free(inp);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        return 0;
    }
}