/* ============================================================================
   common.h —— CUDA 算子公共工具库
   ============================================================================

   用途：
     本头文件提供所有 .cu 算子共用的基础工具，
     避免每个 kernel 重复写通用初始化、错误检查、规约、向量化访存、随机数据生成、结果校验和性能计时代码，
     让各 .cu 文件专注 kernel 本身的优化逻辑。

   提供的工具列表：
     1. 整数向上取整 ceil_div
     2. Warp / Block 级规约（warpReduceSum、blockReduce）
     3. CUDA / cuBLAS 错误检查宏（cudaCheck、cublasCheck）
     4. cuBLAS / cuBLASLt 句柄与 GPU 设备属性（由 setup_main 初始化）
     5. 128 位向量化读写封装（Packed128、load128、store128 等）
     6. 编译期精度切换（FP32 / FP16 / BF16，通过 ENABLE_BF16 / ENABLE_FP16）
     7. 主机端随机数据生成（make_random_float 等）
     8. CPU / GPU 结果对比（validate_result）
     9. Kernel 性能计时（benchmark_kernel，含 L2 冲刷）
   ============================================================================ */
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define WARP_SIZE 32U  // GPU 一个 warp 的固定线程数

extern cudaDeviceProp deviceProp;

// ============================================================================
// 规约原语
// ============================================================================

/* ----------------------------------------------------------------------------
   ceil_div —— 整数除法向上取整
   ----------------------------------------------------------------------------
   用途：计算 grid 大小时常用，例如 N=100、block=32 时，ceil_div(100, 32) = 4，避免线程数不足。
   公式：(a + b - 1) / b，整数向上取整的标准写法。
   ---------------------------------------------------------------------------- */
template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

/* ----------------------------------------------------------------------------
   warpReduceSum —— Warp 内求和规约
   ----------------------------------------------------------------------------
   原理：蝴蝶规约（butterfly reduction），5 步完成（16→8→4→2→1），
         使用 __shfl_xor_sync 在 warp 内直接交换寄存器数据，不需要共享内存，延迟远低于跨 warp 方案。
   返回值：32 个 lane 的 val 之和，每个 lane 都拿到相同结果。
   ---------------------------------------------------------------------------- */
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 规约函数指针类型：用于把 warp 级规约作为模板参数传入 blockReduce
using reduction_func_t = float(*)(float);

/* ----------------------------------------------------------------------------
   blockReduce —— 线程块内两级规约
   ----------------------------------------------------------------------------
   第一级：每个 warp 内部用 shuffle 规约成 1 个数；
   第二级：各 warp 的结果写入共享内存，第 0 个 warp 再规约一次。
   模板参数 warp_reduction 传入 warpReduceSum 或 warpReduceMax，
   一套代码同时支持求和、求最大值，编译期绑定无运行时开销。

   参数：
     final_sync    为 true 时，规约结束后再 __syncthreads()；
                   当共享内存在循环中被复用时必须开启，防止数据竞争。
     out_of_bounds 超出 num_warps 的 lane 用该值填充，避免读到脏数据。
   ---------------------------------------------------------------------------- */
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync, float out_of_bounds) {
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) shared_val[warp_id] = warp_val;
    __syncthreads();

    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) __syncthreads();
    return block_val;
}

// blockReduce 的简化版本：不需要最终同步时直接调用，参数更短。
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val) {
    return blockReduce<warp_reduction>(val, false, 0.0f);
}

// ============================================================================
// 错误检查
// ============================================================================

// CUDA 错误检查：失败时打印文件名、行号和错误字符串并退出。
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS 错误检查：失败时打印状态码、文件名、行号并退出。
void cublasCheck(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d at %s:%d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ============================================================================
// cuBLAS 句柄与 GPU 设备属性（全局变量，由 setup_main 一次性初始化）
// ============================================================================

static size_t cublaslt_workspace_size = 32 * 1024 * 1024;  // cuBLASLt 工作区 32 MiB（Hopper 架构需求较大）
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;

cublasHandle_t cublas_handle;      // cuBLAS 标准版句柄
cublasLtHandle_t cublaslt_handle;  // cuBLASLt 句柄（支持 BF16 / TF32 等新精度）
int cuda_arch_major = 0;           // GPU 架构主版本号（Ampere=8, Hopper=9）
int cuda_arch_minor = 0;
int cuda_num_SMs = 0;              // SM 总数
int cuda_threads_per_SM = 0;       // 每 SM 最大常驻线程数

// Ampere（sm_80）及 Hopper+（sm_90）每个 SM 可同时驻留 2 个 1024 线程 block，其他架构只能驻留 1 个；用于控制大 kernel 的并行度。
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ============================================================================
// 128 位向量化读写封装
// ============================================================================

/* ----------------------------------------------------------------------------
   Packed128 —— 16 字节（128 位）对齐的数据打包
   ----------------------------------------------------------------------------
   作用：把连续若干个 ElementType 打成一个 16 字节整体，强制编译器发出 LDG.128 / STG.128 指令，一次运输 4 个 float 或 8 个 BF16。
   设计：
     - alignas(16) 强制 16 字节对齐，硬件才能发射 128 位指令；
     - 元素个数由 sizeof(int4) / sizeof(ElementType) 在编译期推导；
     - 用 int4 做原始载体，通过 memcpy 做类型双关（type punning），避免直接 reinterpret_cast 的严格别名违规。
   ---------------------------------------------------------------------------- */
template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;

    // 从显存读出的原始 int4 构造，memcpy 按字节拷贝，保证类型安全的同时零开销。
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    // 生成所有元素均为 value 的包，用于初始化或常量填充。
    __device__ static Packed128 constant(ElementType value) {
        Packed128 result;
        for (int k = 0; k < size; ++k) result.payload[k] = value;
        return result;
    }

    __device__ static Packed128 zeros() { return constant(0); }
    __device__ static Packed128 ones()  { return constant(1); }

    __device__ ElementType&       operator[](int index)       { return payload[index]; }
    __device__ const ElementType& operator[](int index) const { return payload[index]; }

    // 写回显存时使用：把 payload 转回 int4，配合 STG.128 指令。
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }

    static constexpr int size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

typedef Packed128<float> f128;  // FP32 专用别名：一包 4 个 float

// 普通加载：走完整缓存路径（显存 → L2 → L1 → 寄存器）。适合数据后续会被重复读取的场景。
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

// 流式加载（__ldcs）：绕过 L1，数据只读一次时用，避免污染 L1 缓存。
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}

// 普通写入：走完整缓存路径，适合写出去还要再读回来的数据。
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

// 流式写入（__stcs）：绕过 L1，适合输出后不再读取的数据。
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}

// L2 缓存写入（__stcg）：绕过 L1、保留在 L2，适合中间结果。
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// ============================================================================
// 编译期精度切换
// ============================================================================

/* 通过 nvcc -D 宏选择精度，整套算子自动切换：
     -DENABLE_BF16  → floatX = __nv_bfloat16
     -DENABLE_FP16  → floatX = __half
     （默认）       → floatX = float
   cuBLAS 侧同步设置数据类型与计算类型（低精度 IO、FP32 累加）。 */
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#define CUBLAS_LOWP          CUDA_R_16BF
#define CUBLAS_LOWP_COMPUTE  CUBLAS_COMPUTE_32F
#elif defined(ENABLE_FP16)
typedef half floatX;
typedef half floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;  // 当前精度下的 128 位打包类型

/* 旧版 nvcc（<12.0）没有为 __nv_bfloat16 提供 __ldcs / __stcs 重载，
   这里通过 unsigned short 做兼容封装；新版编译器不会走到这段。 */
#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && \
    !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}
__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif

// ============================================================================
// 主机端随机数据生成
// ============================================================================

// 生成 [0, 1] 均匀分布的 float 数组。
float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX);
    }
    return arr;
}

// 生成 [-1, 1] 均匀分布的 float 数组，初始化权重时最常用。
float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    return arr;
}

// 生成 [0, V) 的随机整数数组，用于模拟 token id。
int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V;
    }
    return arr;
}

// 生成全 0 float 数组。
float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float));
    return arr;
}

// 生成全 1 float 数组。
float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ============================================================================
// H2D 拷贝与精度转换
// ============================================================================

/* ----------------------------------------------------------------------------
   memcpy_convert —— 主机端精度转换 + H2D 拷贝
   ----------------------------------------------------------------------------
   用途：CPU 侧读出的权重是 float，上传到 GPU 时需要转成 FP16/BF16 以省显存；
        本函数把"逐元素转换 + 设备拷贝"两步合一。
   注意：转换在主机端完成，临时缓冲区用完即释放。
   ---------------------------------------------------------------------------- */
template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (size_t i = 0; i < count; i++) converted[i] = (TargetType)h_ptr[i];
    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);
    return status;
}

// ============================================================================
// 全局初始化
// ============================================================================

/* ----------------------------------------------------------------------------
   setup_main —— 程序启动时一次性初始化
   ----------------------------------------------------------------------------
   内容：
     1. 固定随机种子，保证结果可复现；
     2. 选择 GPU 0，读取 SM 数、最大线程数、架构版本；
     3. 创建 cuBLAS / cuBLASLt 句柄并分配工作区；
     4. Ampere 及以上架构开启 TF32 Tensor Core。
   每个 .cu 的 main 函数第一行调用。
   ---------------------------------------------------------------------------- */
void setup_main() {
    srand(0);

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // Ampere 及以上启用 TF32 Tensor Core；老架构保持 FP32 默认数学。
    int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, math_mode));
}

// ============================================================================
// CPU / GPU 结果校验
// ============================================================================

/* ----------------------------------------------------------------------------
   validate_result —— 逐元素对比 GPU 结果与 CPU 参考值
   ----------------------------------------------------------------------------
   容差逻辑：
     有效容差 = 用户指定 tolerance + |参考值| × epsilon；
     低精度（BF16）尾数只有 7 位，相对误差上限更大，因此 epsilon 取 0.079；
     非有限值（masked 掉的 -inf / NaN）直接跳过。
   流程：
     1. GPU 结果拷回主机；
     2. 打印前 5 个元素供肉眼检查；
     3. 累计不匹配元素，满 10 个即终止。
   ---------------------------------------------------------------------------- */
template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name,
                     std::size_t num_elements, T tolerance = 1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));

#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;  // BF16：尾数 7 位，相对误差约 1/128
#endif

    int nfaults = 0;
    for (std::size_t i = 0; i < num_elements; i++) {
        if (!isfinite(cpu_reference[i])) continue;
        if (i < 5) printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);

        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n",
                   name, i, cpu_reference[i], (T)out_gpu[i]);
            if (++nfaults >= 10) { free(out_gpu); exit(EXIT_FAILURE); }
        }
    }
    free(out_gpu);
    if (nfaults > 0) exit(EXIT_FAILURE);
}

// ============================================================================
// Kernel 性能计时
// ============================================================================

/* ----------------------------------------------------------------------------
   benchmark_kernel —— 精确测量 kernel 平均耗时（毫秒）
   ----------------------------------------------------------------------------
   关键细节：
     1. 每次迭代前用 memset 冲刷一块 L2 Cache 大小的显存，排除缓存命中对性能的虚高；
     2. 用 cudaEventRecord 精确计时，避免主机端 sleep 的误差；
     3. 重复 repeats 次取平均。
   ---------------------------------------------------------------------------- */
template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceIdx));

    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    float elapsed = 0.f;
    for (int i = 0; i < repeats; i++) {
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(stop));

        float single;
        cudaCheck(cudaEventElapsedTime(&single, start, stop));
        elapsed += single;
    }

    cudaCheck(cudaFree(flush_buffer));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    return elapsed / repeats;
}
