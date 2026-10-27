/* ============================================================================
   残差连接前向传播的内核实现
   ============================================================================

   编译示例：
   nvcc -O3 --use_fast_math -lcublas -lcublasLt residual_forward.cu -o residual_forward
 
   版本 1：从 CPU 代码朴素移植到内核，逐元素运算
   运行命令：./residual_forward 1
 
   版本 2：128位向量化访存优化，流式加载绕过L1缓存
   运行命令：./residual_forward 2
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
void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ============================================================================
// GPU 内核函数
// ============================================================================
// 逐元素运算，基础并行实现
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// 128bit向量化优化版：单指令处理8个BF16元素，流式加载避免缓存污染
__global__ void residual_forward_kernel2(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx < N) {
        x128 packed_out;
        x128 packed_inp1 = load128cs(inp1 + idx); // 输入仅读取一次，流式加载
        x128 packed_inp2 = load128cs(inp2 + idx);
        for (int k = 0; k < packed_inp1.size; ++k) {
            packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);  // 输入BF16→计算FP32→输出BF16，避免低精度累加损失
        }
        store128(out + idx, packed_out); // 输出保留在缓存，供后续算子复用
    }
}

// ============================================================================
// 内核启动函数
// ============================================================================
void residual_forward1(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward2(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    residual_forward_kernel2<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

// 内核版本调度
void residual_forward(int kernel_num,
                     floatX* out,
                     const floatX* inp1,
                     const floatX* inp2,
                     int N,
                     int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
        case 2:
            residual_forward2(out, inp1, inp2, N, block_size);
            break;
        default:
            printf("无效的内核版本\n");
            exit(1);
    }
}


// ============================================================================
// 主程序：正确性验证 + 性能基准测试 + 性能采集模式
// ============================================================================
int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // 主机端内存分配与随机数据生成
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);

    // GPU端内存分配与数据拷贝
    floatX* d_out;
    floatX* d_inp1;
    floatX* d_inp2;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp1, inp1, B * T * C));
    cudaCheck(memcpy_convert(d_inp2, inp2, B * T * C));

    // 从命令行读取内核版本号
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 默认模式（正确性验证+性能基准测试）
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        residual_forward_cpu(out, inp1, inp2, B * T * C);

        int block_sizes[] = {32, 64, 128, 256, 512, 1024};
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("正在测试线程块大小 %d.\n", block_size);
            residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size);

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
            float tol = 1e-5;  // FP32 用 1e-5 小容差
#else
            float tol = 1e-2f; // BF16/FP16 用 1e-2 大容差
#endif
            validate_result(d_out, out, "out", B * T * C, tol);
        }

        printf("所有结果验证通过，开始性能基准测试。.\n\n");

        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            int repeat_times = 100;  // 重复次数
            float elapsed_time = benchmark_kernel(repeat_times, residual_forward,
                                                  kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size
                                                 );

            // 访存带宽估算：每个输出对应2次读+1次写，单元素4字节
            long memory_ops = B * T * C * 3 * 4;
            float memory_bandwidth = memory_ops / elapsed_time / 1e6;

            printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 带宽 %.2f GB/s\n",
                   block_size, elapsed_time, memory_bandwidth);
        }

        free(out); free(inp1); free(inp2);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp1));
        cudaCheck(cudaFree(d_inp2));
        return 0;
    }

    // ========== 性能采集模式（配合NCU/NSYS使用） ==========
    /* 性能分析命令示例：
     nvcc -O3 --use_fast_math -lineinfo -lcublas -lcublasLt residual_forward.cu -o residual_forward

     全尺寸批量采集：
     ncu --set full \
         --kernel-name residual_forward_kernel2 \
         --launch-skip 18 --launch-count 6 \
         --export residual_kernel2_all_blocks.ncu-rep \
         ./residual_forward 2 profile
     
     单尺寸精准采集：
     ncu --set full \
         --kernel-name residual_forward_kernel2 \
         --launch-skip 20 --launch-count 1 \
         --export residual_kernel2_128.ncu-rep \
         ./residual_forward 2 profile
     
     系统级性能分析：
     nsys profile --stats=true -o residual_nsys ./residual_forward 2 profile
     */
    {
        int profile_blocks[] = {32, 64, 128, 256, 512, 1024};
        int num_blocks = sizeof(profile_blocks) / sizeof(int);
        int warmup_rounds = 3;
        printf("性能采集模式：%d 种线程块尺寸，%d 轮预热\n", num_blocks, warmup_rounds);

        // 预热轮次
        for (int round = 0; round < warmup_rounds; round++) {
            for (int i = 0; i < num_blocks; i++) {
                residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, profile_blocks[i]);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集轮次
        for (int i = 0; i < num_blocks; i++) {
            residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, profile_blocks[i]);
        }
        cudaCheck(cudaDeviceSynchronize());

        free(out); free(inp1); free(inp2);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp1));
        cudaCheck(cudaFree(d_inp2));
        return 0;
    }
}
