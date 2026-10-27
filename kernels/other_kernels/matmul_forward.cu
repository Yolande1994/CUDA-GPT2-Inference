/* ============================================================================
   Matmul 前向传播 CUDA Kernel 实现
   ============================================================================

   编译示例：
     nvcc -O3 --use_fast_math -Xcompiler -fopenmp -lcublas -lcublasLt matmul_forward.cu -o matmul_forward

   算子说明：
     线性层（Linear Layer）前向传播，计算 out = inp @ weight.T + bias。
     输入 inp 维度为 (B, T, C)，权重 weight 维度为 (OC, C)，偏置 bias 维度为 (OC)，输出 out 维度为 (B, T, OC)。
     B 为批大小，T 为序列长度，C 为输入特征维度，OC 为输出特征维度。
   矩阵乘法形状（GEMM）：
     M = B*T（总 token 数），N = OC（输出通道），K = C（公共累加维度）。

   版本说明：
     版本1：朴素 GPU 实现（基线）
     版本2：手写非张量核 GEMM（共享内存 + float4 向量化搬运）
     版本3：手写非张量核 GEMM（优化 Grid 索引，weight 原生 [OC, C] 布局）
     版本4：cuBLAS（cublasSgemm）+ 单独 add_bias kernel
     版本5：cuBLASLt（epilogue 融合 bias / GELU，自动择优算法）
     
   运行命令：
     ./matmul_forward 1
     ./matmul_forward 2
     ...
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>
#include "../common.h"


// ============================================================================
// CPU 参考实现
// ============================================================================

/* 标准矩阵乘法 CPU 参考实现
   逐元素计算：out[b][t][o] = Σ inp[b][t][i] * weight[o][i] +  bias[o]
   使用 OpenMP collapse(2) 在 B、T 两个维度并行，加速 CPU 计算。 */
void matmul_forward_cpu(float* out,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o * C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}


// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   版本1：朴素 GPU 实现
   ----------------------------------------------------------------------------
   定位：最直接的 CPU 代码移植版本，作为性能基线（Baseline）。
   并行策略：
     二维 Grid，x 方向遍历 BT，y 方向遍历 OC，每线程负责一个输出元素。
   ---------------------------------------------------------------------------- */
__global__ void matmul_forward_kernel1(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;

    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc * C;
        const float* inp_bt = inp + bt * C;

        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }

        out[bt * OC + oc] = val;
    }
}

/* ----------------------------------------------------------------------------
   add_bias：逐元素给矩阵乘法输出加偏置
   ----------------------------------------------------------------------------
   定位：cuBLAS 标准 GEMM（cublasSgemm）只做纯矩阵乘法，不支持 epilogue 融合，需要单独启动此 kernel 补加 bias。
   实现：
     使用 grid-stride loop（网格跨步循环），不依赖具体的 Grid 大小，线程数不足时通过循环遍历所有元素。
   性能问题：
     单独启动 kernel 会多一次全局内存读和写。
     最优做法是通过 cuBLASLt 的 EPILOGUE_BIAS 融合到矩阵乘末尾，省掉一次全局内存往返（见版本5）。
   ---------------------------------------------------------------------------- */
__global__ void add_bias(float* out, const float* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 整个 Grid 的总线程数，作为循环步进

    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC; // 对 OC 取模，得到当前元素所属的输出通道
        out[i] += bias[col];
    }
}


// ----------------------------------------------------------------------------
// float4 向量化访存辅助函数
// ----------------------------------------------------------------------------
/* ld_vec：一次性加载连续 4 个 float（16 字节）
   通过 reinterpret_cast 将 float* 重新解释为 float4*，
   编译器生成一条 128 位的加载指令，访存指令数压缩为 1/4。 */
__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

/* st_vec：一次性写入连续 4 个 float（16 字节）
   与 ld_vec 对应，生成一条 128 位的存储指令。 */
__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}


/* ----------------------------------------------------------------------------
   版本2：手写 Tiled GEMM
   ----------------------------------------------------------------------------
   定位：手写的半优化 GEMM，展示共享内存分块和向量化访存的基本结构。
   并行配置：
     Block 大小 16×16（256 线程），每线程负责 8×8 个输出元素。
     单个 Block 输出 128×128 的 Tile，K 方向分块大小为 32。
   执行流程：
     1. 沿 K 方向分块遍历，每块 32 个元素
     2. 将输入和权重的 Tile 从全局内存通过 float4 搬运到共享内存
     3. 从共享内存读取数据，在寄存器中累加 8×8 个输出
     4. 全部 K 块遍历完后，将结果写回全局内存
   注意：此版本 Grid 维度和输出索引存在问题，优化版见版本3。
   ---------------------------------------------------------------------------- */
__global__ void __launch_bounds__(16*16) matmul_forward_kernel2(float* out,
                                                                const float* inp, const float* weight, const float* bias,
                                                                int C, int OC) {
    // 当前线程负责的 8 个连续输出通道的起始位置
    int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);
    __shared__ float lhs_s[128][32]; // 输入 Tile：128 行对应 M，32 列对应 K
    __shared__ float rhs_s[128][32]; // 权重 Tile：128 行对应 N，32 列对应 K

    // 指针预偏移到当前 Block 负责的 Tile 起始位置
    inp    += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out    += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    // 寄存器累加器，每线程持有 8×8 个输出
    float vals[8][8] = {};
    if (bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j + 0] = b.x;
                vals[i][j + 1] = b.y;
                vals[i][j + 2] = b.z;
                vals[i][j + 3] = b.w;
            }
        }
    }

    int si_start = 4 * (16 * threadIdx.y + threadIdx.x);
    // K 方向分块遍历，每块 32 个元素
    for (int so = 0; so < C; so += 32) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8  = threadIdx.x / 8;
        int xo = 4 * xmod8;
        // 将 16×16 线程重映射为 32×8 加载网格，铺满 128×32 共享内存
        for (int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        // 从共享内存读取，寄存器内累加
        for (int si = si_start; si < si_start + 32; si += 4) {
            float4 rhs[8];

            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);

                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    // 结果写回全局内存
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result);
        }
    }
}


/* ----------------------------------------------------------------------------
   版本3：手写 Tiled GEMM（优化索引）
   ----------------------------------------------------------------------------
   定位：与版本2 结构相同，但优化了以下问题：
     1. Grid 维度：x 方向对应 OC 分块，y 方向对应 BT 分块
     2. weight 保持原生 [OC, C] 行优先布局，K 方向连续访问，无需预转置
     3. 输出写回索引优化，块内行号和列号正确对应
   格式说明（GEMM NT）：
     对应 GEMM NT 格式：A（输入）不转置，B（权重）逻辑上转置，
     数学上等价于 out = inp @ weight.T。
     K 方向上 inp 和 weight 都是连续访问，完美触发合并访存。
   共享内存布局：
     rhs_s[128][32] 按 weight 行优先布局，128 行对应 OC，32 列对应 K。
     搬运时直接连续读取 32 个 K 元素，适配 float4 向量化读写。
   ---------------------------------------------------------------------------- */
__global__ void __launch_bounds__(16*16) matmul_forward_kernel3(float* out,
                                                                const float* inp, const float* weight, const float* bias,
                                                                int C, int OC) {
    __shared__ float lhs_s[128][32]; // 输入 Tile：128 行对应 M(BT)，32 列对应 K
    __shared__ float rhs_s[128][32]; // 权重 Tile：128 行对应 N(OC)，32 列对应 K

    // 当前线程负责的连续 8 个输出通道的起始列号
    int oc = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    // 指针预偏移到当前 Block 负责的 Tile 起始位置
    inp    += blockIdx.y * 128 * C;
    weight += blockIdx.x * 128 * C;
    out    += blockIdx.y * 128 * OC + blockIdx.x * 128;

    // 寄存器累加器，每线程持有 8×8 个输出
    float vals[8][8] = {};
    if (bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j + 0] = b.x;
                vals[i][j + 1] = b.y;
                vals[i][j + 2] = b.z;
                vals[i][j + 3] = b.w;
            }
        }
    }

    /* ----- 计算阶段：每个线程负责 4 个连续 K 位置 -----
     256 个线程按线性编号 tid = threadIdx.y * 16 + threadIdx.x （0~255）
     每线程负责 4 个连续 K 列，si_start 是它在 [0, 128) 范围内的起始列号
     128 / 4 = 32，刚好每线程一轮读完自己的 4 个 float，交错分布避免 bank conflict */
    int si_start = (threadIdx.y * 16 + threadIdx.x) * 4;
    /* ----- 加载阶段：把 16×16 线程重映射成 32 行 × 8 列的 float4 加载网格 -----
     目标：256 个线程要填满 128 行 × 32 列的共享内存，每次搬 float4（4 个 float）
     所以实际加载网格是 128 行 × 8 个 float4 = 1024 个 float4，每线程搬 4 次
     threadIdx.x 是 0~15，拆成两部分：
       高 1 位（x/8）决定行方向偏移 0 或 1 → 配合 threadIdx.y*2 得到 0~31 的行号
       低 3 位（x%8）决定列方向，对应 8 个 float4 位置（每列 4 个 float） */
    int xby8  = threadIdx.x / 8;  // 行偏移：0 或 1（高1位）
    int xmod8 = threadIdx.x % 8;  // 列编号：0~7（低3位）
    int xo    = xmod8 * 4;         // 共享内存列起始地址：0,4,8,...,28（float4 对齐）

    // K 方向分块遍历，每块 32 个元素
    for (int so = 0; so < C; so += 32) {
        __syncthreads();

        // 将 16×16 线程重映射为 32×8 加载网格，铺满 128×32 共享内存
        for (int y = threadIdx.y * 2 + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        // 从共享内存读取，寄存器内累加
        for (int si = si_start; si < si_start + 32; si += 4) {
            float4 rhs[8];

            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[threadIdx.x * 8 + u][si % 32]);
            }
            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[threadIdx.y * 8 + ii][si % 32]);

                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    // 结果写回全局内存
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            int bt = threadIdx.y * 8 + i; // 块内行号
            int oc = threadIdx.x * 8 + j; // 块内列号
            st_vec(out + bt * OC + oc, result);
        }
    }
}


// ============================================================================
// Kernel 启动封装函数
// ============================================================================

// 版本1：朴素 kernel
void matmul_forward1(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B * T, C, OC);
    cudaCheck(cudaGetLastError());
}

// 版本2：手写 Tiled GEMM
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     int sqrt_block_size) {
    sqrt_block_size = 16;

    dim3 gridDim(ceil_div(B * T, 8 * sqrt_block_size), ceil_div(OC, 8 * sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_forward_kernel2<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

// 版本3：手写 Tiled GEMM（优化索引）
void matmul_forward3(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     int sqrt_block_size) {
    sqrt_block_size = 16;
    int BT = B * T;
    // --------------------- 改动点：Grid维度交换顺序 ---------------------
    // 规范 dim3( gridX(列块OC), gridY(行块BT) )
    // grid.x → OC方向分块数量，grid.y → BT方向分块数量
    dim3 gridDim(ceil_div(OC, 8 * sqrt_block_size), ceil_div(BT, 8 * sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_forward_kernel3<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

/* 版本4：cuBLAS GEMM + 单独 add_bias kernel
   cuBLAS 列优先适配：
     目标计算（行优先）：out = inp @ weight.T
     cuBLAS 是列优先，同一段内存会被自动视为转置。
     令 cuBLAS 计算 out.T = weight @ inp.T：
       transa = CUBLAS_OP_T（weight 转置）
       transb = CUBLAS_OP_N（inp 不转置）
       m = OC, n = B*T, k = C
   关于偏置处理：
     cublasSgemm 不支持 epilogue 融合，bias 通过单独 add_bias kernel 补加。 */
void matmul_forward4(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasCheck(cublasSgemm(cublas_handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           OC, B * T, C,
                           &alpha,
                           weight, C,
                           inp, C,
                           &beta,
                           out, OC));
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

/* 版本5：cuBLASLt，epilogue 融合 bias / GELU
   核心优势：
     cuBLASLt 支持在 GEMM 的 epilogue 阶段融合 bias、GELU 等操作，
     矩阵乘算完后直接在寄存器中完成 bias 加法和激活，再写回全局内存，
     省掉单独 add_bias kernel 的一次全局内存往返。
   算法选择：
     通过 cublasLtMatmulAlgoGetHeuristic 让库根据硬件和矩阵形状自动选择最优算法，支持 split-K 等需要工作空间的算法。
   对齐要求：
     bias 指针必须 16 字节对齐（cuBLASLt 内部使用向量化访存）。
   参考文档：https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
   参考示例：https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu */
void matmul_forward5(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);
    int has_gelu = 0;

    // 检查偏置指针的内存对齐（cuBLASLt 要求 bias 指针 16 字节对齐，和 float4 向量化访存一样）
    if (((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    // epilogue 选择：根据是否有 bias 和 GELU 选择融合方式
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS; // 偏置 + GELU 一起融合
    } else if (has_bias) {
        epilogueBias = CUBLASLT_EPILOGUE_BIAS;      // 仅加偏置
    } else if (has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU;      // 仅做 GELU
    }

    // 创建运算描述符
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // 定义矩阵布局描述符：列优先，参数依次为（数据类型，行数，列数，前导维）
    cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B * T, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B * T, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

    // 创建偏好设置，指定最大工作空间
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // 查找最优算法
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
                                               weightLayout, inputLayout, outputLayout, outputLayout,
                                               preference, 1, &heuristic, &returnedResults));

    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
               B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    // 执行矩阵乘法
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, weight, weightLayout, inp, inputLayout, &beta,
                               out, outputLayout, out, outputLayout, &heuristic.algo,
                               cublaslt_workspace, cublaslt_workspace_size, 0));

    // 资源清理
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

// 版本分发
void matmul_forward(int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1: matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size); break;
        case 2: matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size); break;
        case 3: matmul_forward3(out, inp, weight, bias, B, T, C, OC, sqrt_block_size); break;
        case 4: matmul_forward4(out, inp, weight, bias, B, T, C, OC, sqrt_block_size); break;
        case 5: matmul_forward5(out, inp, weight, bias, B, T, C, OC); break;
        default:
            printf("无效的内核编号\n");
            exit(1);
    }
}


// ============================================================================
// 主程序：正确性验证 + 性能基准测试
// ============================================================================
int main(int argc, char **argv) {
    srand(0);
    // 测试参数：模拟 GPT-2 MLP 层输入输出维度
    int B = 32;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // MLP 中间层扩张 4 倍

    // 初始化 GPU 设备
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("设备 %d: %s\n", deviceIdx, deviceProp.name);

    // 初始化 cuBLAS 和 cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));

    int enable_tf32 = 0;
    // Ampere 及以上启用 TF32
    //int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("启用 TF32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // 分配 cuBLASLt 工作空间
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // 主机端数据
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // GPU 端显存分配
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));

    // 数据从主机拷贝到 GPU
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

    // 从命令行读取内核版本号，默认版本1
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 如果不是 profile 模式，则执行标准验证与测试流程
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        // CPU 参考结果
        matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

        // 正确性验证：遍历不同 Block 大小
        int sqrt_block_sizes[] = {4, 8, 16, 32};
        for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
            int sqrt_block_size = sqrt_block_sizes[j];
            printf("正在测试线程块大小 %d x %d。\n", sqrt_block_size, sqrt_block_size);
            matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
            validate_result(d_out, out, "out", B * T * OC, 1e-1f); // TF32 误差放宽
        }
        printf("所有结果验证通过，开始性能基准测试。\n\n");

        // 性能基准测试
        for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
            int sqrt_block_size = sqrt_block_sizes[j];
            int repeat_times = 100;
            float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                                  kernel_num, d_out, d_inp, d_weight, d_bias,
                                                  B, T, C, OC, sqrt_block_size);
            // 有效 TFLOPS = 2 × M × N × K / time
            float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
            printf("线程块大小 %4d | 耗时 %.4f 毫秒 | 算力 %.2f TFLOPS\n", sqrt_block_size, elapsed_time, tflops);
        }

        // 释放主机端内存
        free(out);
        free(inp);
        free(weight);
        free(bias);
        // 释放 GPU 端显存
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias));
        cudaCheck(cudaFree(cublaslt_workspace));
        // 销毁 cuBLAS / cuBLASLt 句柄
        cublasCheck(cublasDestroy(cublas_handle));
        cublasCheck(cublasLtDestroy(cublaslt_handle));
        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     编译指令示例：
       nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt matmul_forward.cu -o matmul_forward

     NCU 采集命令示例：
       ncu --set full --cache-control all --launch-skip 18 --launch-count 6 -o matmul_profile_16 ./matmul_forward 0 profile

     系统级性能分析：
       nsys profile --stats=true -o matmul_nsys ./matmul_forward 0 profile
     */
    {
        const int profile_sqrt_block_size = 16; // 固定块大小
        const int warmup_rounds = 3;  // 预热轮数
        const int total_versions = 5; // 版本 1~5

        printf("性能采集模式：%d 个内核版本，%d 轮预热\n", total_versions, warmup_rounds);

        // 预热
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= total_versions; ver++) {
                matmul_forward(ver, d_out, d_inp, d_weight, d_bias, B, T, C, OC, profile_sqrt_block_size);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集
        for (int ver = 1; ver <= total_versions; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            matmul_forward(ver, d_out, d_inp, d_weight, d_bias, B, T, C, OC, profile_sqrt_block_size);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放资源
        free(out);
        free(inp);
        free(weight);
        free(bias);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_weight));
        cudaCheck(cudaFree(d_bias));
        cudaCheck(cudaFree(cublaslt_workspace));
        cublasCheck(cublasDestroy(cublas_handle));
        cublasCheck(cublasLtDestroy(cublaslt_handle));
        return 0;
    }
}