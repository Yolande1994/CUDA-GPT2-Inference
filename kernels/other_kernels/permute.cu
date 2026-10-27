/* ============================================================================
   维度置换（Permute）算子的 CUDA 实现
   ============================================================================
   
   编译示例：
     nvcc -O3 permute.cu -o permute -lcublas -lcublasLt

   功能描述：
     将一个 4 维张量从原始形状 (dim1, dim2, dim3, dim4) 
     重排为新形状 (dim4, dim3, dim1, dim2)，元素值不变，仅改变内存布局。
   
   设计背景：
     纯 CPU 实现可通过指针偏移直接跨步访问，无需单独置换。但在 GPU 高性能计算中，依赖合并访存。
     以多头注意力为例，原生 (B, T, NH, hs) 布局下，单头数据存在大跨步，导致后续矩阵乘无法触发合并访存，带宽利用率暴跌。
     置换为 (B, NH, T, hs) 布局后，每个注意力头的矩阵内存连续，可充分发挥显存带宽。
   
   ----------------------------------------------------------------------------
   核心理论：线性索引与多维坐标的互转
   ----------------------------------------------------------------------------
   已知维度大小：dim1, dim2, dim3, dim4
   对于任意 4 维坐标 (i1, i2, i3, i4)，其线性索引（一维位置）为：
     linear_idx = i1 * (dim2 * dim3 * dim4) + i2 * (dim3 * dim4) + i3 * dim4 + i4
   
   反过来，已知线性索引 idx，求各维坐标（从高维到低维依次拆解）：
     i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
     i2 = (idx / (dim3 * dim4)) % dim2;
     i3 = (idx / dim4) % dim3;
     i4 = idx % dim4;
   
   规律说明：要求解任意一维的索引，先将线性编号 idx 除以「该维度右侧所有维度的乘积」，再对「当前维度的大小」取模。
   
   置换后的索引计算（从 (dim1, dim2, dim3, dim4) 置换为 (dim4, dim3, dim1, dim2)）：
     permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
   
   计算逻辑拆解：
     - i4 * (dim3 * dim1 * dim2)：当前 i4 块之前，共有多少个完整的 dim3 × dim1 × dim2 数据块。
     - i3 * (dim1 * dim2)：当前 i4 块内部，当前 i3 块的偏移量。
     - i1 * dim2：当前 i3 块内部，当前 i1 块的偏移量。
     - i2：当前 i1 块内部的元素偏移。
   
   最后，将原始矩阵中 idx 位置的元素，写入置换后矩阵的 permuted_idx 位置。
   同理，可扩展至任意维度的矩阵置换。
   */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common.h"

/* CPU 参考实现：4维张量置换
   万能规律：想求某一维的索引，就先用总位置除以「这一维右边所有维度的乘积」，再对这一维的大小取余数。
   
   比喻（年月日时）：
   把四个维度类比成 年→月→日→时，idx 就是「从元年1月1日0点开始数，第几个小时」。
   - 算当天第几小时（最内层 i4）：直接 idx % 24 —— 余数就是当天第几小时。
   - 算当月第几天（i3）：先 idx / 24 算出总天数，再对每月天数取余。
   - 算当年第几月（i2）：先 idx / (24×每月天数) 算出总月数，再对 12 取余。
   - 算第几年（i1）：先 idx / (24×每月天数×12) 算出总年数，再对总年数取余。
   
   新位置：哪个维度在最左边，它的索引就要乘以后面所有维度的总大小，再依次加起来。
 */
void permute_cpu(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    // 算总工作量：四个维度相乘，算出总共有多少个元素要搬。遍历每个元素都处理一遍
    int total_threads = dim1 * dim2 * dim3 * dim4;
    for (int idx = 0; idx < total_threads; idx++) {
        // 拆解多维坐标（从线性索引反推 4 维坐标）
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;
        // 计算置换后矩阵的新索引
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2; 
        // 搬运元素（把原数组里第 idx 个元素，放到新数组的第 permuted_idx 个位置上）
        out_matrix[permuted_idx] = matrix[idx]; 
    }
}

// CUDA 内核：并行执行 4 维矩阵置换
// 逻辑同样，利用 GPU 的海量线程并行加速。每个线程负责处理一个元素的坐标拆解、重算和搬运。
__global__ void permute_kernel(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    // 计算当前线程对应的全局线性索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查，确保索引不越界
    if (idx < dim1 * dim2 * dim3 * dim4) {
        // 拆解多维坐标（从线性索引反推 4 维坐标）
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;
        // 计算置换后矩阵的新索引
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
        // 写入新位置
        out_matrix[permuted_idx] = matrix[idx];
    }
}

/* 四维张量置换算子：优化笔记与工程思考
   优化要点：
   1. 性能瓶颈：
      朴素实现天生「读写合并不对称」——读地址连续可完美合并，写地址离散导致合并失效。
      写不合并是性能的主要瓶颈。保证写连续的收益通常高于读连续（写操作无多层缓存兜底）。
   2. 优化边界：
      仅读可做 128bit 向量化加速，写地址离散无法向量化。
      因此向量化不是通用解，必须匹配访存模式。
   3. 标准优化方案：
      共享内存分块中转（Shared Memory Tiling）——连续读入共享内存、片内完成换序，
      再连续写回全局内存，使读写均可打满带宽。
   4. 工程最优解：
      优先消除 permute 本身——与前后算子融合直接输出目标布局，或利用矩阵乘 stride 参数隐式处理，
      省掉一次全局显存读写。收益远高于继续手写 kernel 优化。
   */

int main() {
    int dim_1 = 24;
    int dim_2 = 42;
    int dim_3 = 20;
    int dim_4 = 32;

    // 初始化设备
    int deviceIdx = 0;
    cudaSetDevice(deviceIdx);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("设备 %d: %s\n", deviceIdx, deviceProp.name);

    // 分配主机端内存
    float* matrix = make_random_float(dim_1 * dim_2 * dim_3 * dim_4);
    float* permuted_matrix = (float*)malloc(dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));

    // 分配设备端内存
    float *d_matrix, *d_permuted_matrix;
    cudaMalloc(&d_matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));
    cudaMalloc(&d_permuted_matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));

    // 将矩阵从主机拷贝到设备
    cudaMemcpy(d_matrix, matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float), cudaMemcpyHostToDevice);

    // CPU 基准测试
    clock_t start = clock();
    permute_cpu(matrix, permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    clock_t end = clock();
    double elapsed_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // 配置并启动 CUDA 内核
    dim3 blockSize(256);
    int totalThreads = dim_1 * dim_2 * dim_3 * dim_4;
    int gridSize = (totalThreads + blockSize.x - 1) / blockSize.x; // 向上取整计算网格大小

    permute_kernel<<<gridSize, blockSize>>>(d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    cudaDeviceSynchronize(); // 等待内核执行完成

    // 校验结果正确性
    printf("正在校验结果正确性...\n");
    validate_result(d_permuted_matrix, permuted_matrix, "permuted_matrix", dim_1 * dim_2 * dim_3 * dim_4, 1e-5f);
    printf("所有结果验证通过。\n\n");

    // 内核性能基准测试
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, permute_kernel,
                                          d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    printf("GPU 耗时: %.4f 毫秒\n", elapsed_time);
    printf("CPU 耗时: %.4f 毫秒\n", elapsed_time_cpu);

    // 释放内存
    free(matrix);
    free(permuted_matrix);
    cudaFree(d_matrix);
    cudaFree(d_permuted_matrix);

    return 0;
}