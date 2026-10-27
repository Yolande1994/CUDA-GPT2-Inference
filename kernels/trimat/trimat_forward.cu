/* ============================================================================
   自回归注意力中的三角矩阵乘法 (Causal Attention Q @ Kᵀ)
   ============================================================================

   编译命令：
     nvcc -O3 --use_fast_math -lcublas -lcublasLt trimat_forward.cu -o trimat_forward -lcublas

   算子说明：
     本算子实现自回归注意力机制中的核心计算：注意力分数矩阵 S = Q @ Kᵀ。
     针对因果注意力（Causal Attention）中的下三角矩阵，本算子通过融合掩码逻辑，跳过无效的上三角计算，理论算力节省近 50%。

   版本说明：
     版本0：cuBLAS 基准内核
     版本1：朴素实现
     版本2：寄存器优化版
     版本3：向量化版
     版本4：共享内存版

   运行命令：
     ./trimat_forward 0
     ./trimat_forward 1
     ./trimat_forward 2
     ...

   为什么要单独写这个算子：
     1. 性能收益：如果用通用矩阵乘 + 单独掩码两步实现，需要多一次完整的显存读写，还多一次 kernel 启动开销；
     把掩码融合进矩阵乘内部，计算的同时就完成了掩码，零额外访存成本。
     2. 场景专属：自回归推理的因果注意力是大模型的核心操作，这个算子是注意力模块的专属优化组件。
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../common.h"

/* ============================================================================
   第一章 - 项目概述与参考实现
   ============================================================================
   本文件实现自回归注意力中的三角矩阵乘法（Q @ Kᵀ），针对因果掩码特性做专属优化：
   通用矩阵乘会计算完整的 T×T 矩阵，而因果注意力仅下三角（含对角线）有效，
   本项目通过直接跳过无效上三角计算，理论可节省近一半算力。

   文件包含三类实现：
   1. CPU 朴素参考实现：作为正确性校验的黄金标准，结果无误差；
   2. cuBLAS 官方库实现：作为性能基准，采用通用矩阵乘 + 前置维度重排，无三角优化；
   3. 4 个逐步优化的手写 CUDA 内核：从朴素实现到共享内存分块优化，逐步逼近 cuBLAS 性能。
   ============================================================================ */

/* ----------------------------------------------------------------------------
   trimul_cpu：CPU 朴素参考实现
   ----------------------------------------------------------------------------
   取自注意力前向传播过程。这个函数是自回归注意力中「Q @ Kᵀ」 注意力分数计算的 CPU 朴素参考实现，是整个工程的「正确性标准」。
   本函数用固定的 Query 向量和所有历史 Key 向量逐个做向量点积（适用于训练和 prefill 阶段，decode 阶段用不着掩码）。

   输入张量形状：(B, T, 3, NH, HS)  （已经投影 + 打包好的 QKV 合并张量）
   输出张量形状：(B, NH, T, T)      （每个样本、每个头对应的 T×T 注意力分数矩阵，仅下三角有效）

   数学公式：S = Q @ Kᵀ
   形状变化：(B, T, C) @ (B, C, T) = (B, T, T)
   矩阵含义：S[i][j] 就是「第 i 个 token 的 Q，和第 j 个 token 的 K 的点积相似度」，其值越大，代表第 i 个 token 越关注第 j 个 token 的信息。

   为什么输入是打包的 QKV：
   1. Q/K/V 内存连续，方便 CUDA 做向量化加载（本文件后续优化用了 float4）；
   2. 不用做额外的维度转置，指针计算简单：取 K 只要在 Q 的基础上偏移 C 个元素，取 V 偏移 2*C；
   3. 每个头的计算完全独立，天然适合 CUDA 大规模并行。

   多头注意力有两个天然的独立性，是 CUDA 优化的核心依据：
   1. 不同样本、不同头之间完全独立：没有任何数据依赖，可以同时计算，本文件把 B×NH 放到了线程块的 z 维度并行。
   2. 不同 query 位置 t 之间完全独立：每个 t 的分数行只依赖 Q 的第 t 行和所有 K 行，互不影响，可以拆给不同线程块/线程计算。
   ---------------------------------------------------------------------------- */
void trimul_cpu(float* out, const float* inp,
                int B, int T, int C, int NH) {
    int C3 = C*3;    // C3 是 QKV 三个矩阵的总特征维度，用于指针偏移
    int HS = C / NH; // 单头维度
    float scale = 1.0 / sqrtf(HS);  // 点积缩放因子
    // 外层循环：遍历「所有批量样本 × 所有 query token 位置 × 所有注意力头」
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int nh = 0; nh < NH; nh++) {
                const float* query_t = inp + b * T * C3 + t * C3 + nh * HS;  // 向量 Q [b][nh][t][:] = 输入 inp[b][t][0][nh][:]
                float* out_bth = out + b * NH * T * T + nh * T * T + t * T;  // 输出 out [b][nh][t][:]
                // 计算 Q 向量与 K 向量的点积，并记录最大值（本函数只负责 QK 分数计算，最大值是后续 softmax 计算才需要的）
                for (int t2 = 0; t2 <= t; t2++) {  // 只遍历 '0~t' 的历史 token（含自身），完全不碰 't+1 ~ T-1' 的未来 token
                    // 键向量 K[b][nh][t2][:] = 输入 inp[b][t2][1][nh][:]
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + nh * HS + C; // +C 因为这是键向量部分
                    // 计算 Q[b][nh][t][:] 与 K[b][nh][t2][:] 的点积
                    float val = 0.0f;
                    for (int i = 0; i < HS; i++) {
                        val += query_t[i] * key_t2[i];  // 长度为 HS 的两个向量点积得到标量相似度值，值越大语义相似度越高
                    }
                    val *= scale;
                     // 写入输出矩阵第 t 行第 t2 列：out[b][nh][t][t2] = val
                    out_bth[t2] = val;
                }
                // 对于未来 token 位置（t2 > t），统一赋值为 NAN
                for(int t2 = t + 1; t2 < T; t2++) {
                    out_bth[t2] = NAN;  // 因果掩码，使用 NAN 避免编译警告 —— 理论上应该用负无穷，但不影响结果，因为结果校验函数会忽略无穷值和 NAN
                }  // 到这里，只计算了注意力分数矩阵的第 t 行
            }
        }  // 当外层 t 从 0 遍历到 T-1，把所有行拼起来，就得到了完整的 T×T 下三角注意力分数矩阵，也就是整个 trimat 算子的最终输出。
    }
}

/* ----------------------------------------------------------------------------
   permute_kernel：维度重排内核
   ----------------------------------------------------------------------------
   功能：「拆分 QKV + 维度转置」，把 (B,T,3,NH,HS) 的打包输入，转换成三个形状为 (B,NH,T,HS) 的独立张量 q、k、v。

   这个内核是 cuBLAS 基准版本的前置数据转换模块，后面调用的 cublasSgemmStridedBatched 批量矩阵乘法有两个要求：
   1. Q、K、V 必须是三个独立的连续张量，不能打包在一起；
   2. 维度顺序必须是 (B, NH, T, HS)：把「批量 × 注意力头」作为批量维度，每个头的 (T, HS) 矩阵在内存里连续，才能被 cuBLAS 当成独立矩阵并行计算。

   输入参数 inp：打包的 QKV 张量，形状 (B, T, 3, NH, HS)
   输出参数 q/k/v：三个独立张量的指针，形状均为 (B, NH, T, HS)
   ---------------------------------------------------------------------------- */
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程负责搬运一个元素
    // Q[b][nh][t][hs] = inp[b][t][0][nh][hs]
    if (idx < B * NH * T * HS) {
        int b = idx / (NH * T * HS);    // 跨 1 个 batch 的步长是 NH*T*HS，得到 batch 索引
        int rest = idx % (NH * T * HS); // 剩余低 3 维的偏移
        int nh = rest / (T * HS);       // 跨 1 个头的步长是 T*HS，得到注意力头索引
        rest = rest % (T * HS);         // 剩余低 2 维的偏移
        int t = rest / HS;               // 跨 1 个 token 的步长是 HS，得到 token 位置索引
        int hs = rest % HS;              // 最后剩余的就是头维度内的索引
        // 根据反解出的 (b, nh, t, hs)，计算这个元素在原始打包输入 inp 里的偏移。输入形状是 (B, T, 3, NH, HS)，各维度步长：
        int inp_idx =
            (b * T * 3 * NH * HS)
            +   (t * 3 * NH * HS)       // 跨 batch： T * 3 * NH * HS
            +       (0 * NH * HS)       // 跨 token 位置： 3 * NH * HS
            +           (nh * HS)       // 跨 Q/K/V：          NH * HS（每个矩阵块的大小）
            +                 hs;       // 跨注意力头：              HS
        // 拆分 Q/K/V 并写入输出
        q[idx] = inp[inp_idx];                   // Q 在索引 0 位置，直接取 inp[inp_idx]
        k[idx] = inp[inp_idx + NH * HS];        // K 在 Q 后面偏移 NH*HS 的位置（第 3 维索引 1）
        v[idx] = inp[inp_idx + (NH * HS)*2];    // V 在 Q 后面偏移 2*NH*HS 的位置（第 3 维索引 2）
    }
}

static float* d_qkvr;   // cuBLAS 内核使用的临时缓冲区

/* ----------------------------------------------------------------------------
   trimul_cublas：cuBLAS 官方库版本的注意力分数计算实现
   ----------------------------------------------------------------------------
   整体流程概览：
    1. 打包输入 inp (B,T,3,NH,HS)
    2. 用 permute_kernel 维度重排 + 拆分成三个独立张量 q/k/v (B,NH,T,HS)
    3. cublasSgemmStridedBatched：B*NH 个小 GEMM 并行
    4. 完整输出 preatt (B,NH,T,T) —— 含上三角，cuBLAS 不做因果跳过

   这个函数的关键特点：
   1. 完整矩阵计算，无三角优化：cuBLAS 是通用矩阵乘法库，会计算完整的 T×T 矩阵，包括不需要的上三角部分，浪费近一半算力。
      后面的手写内核就是针对这点做优化，只算下三角。
   2. 额外的数据搬运开销：需要先调用 permute 内核做拆分和维度重排，还要额外的 d_qkvr 缓冲区存储拆分后的 Q/K/V。
      手写内核直接适配打包输入，省掉了这部分开销。
   3. 高度优化的通用实现：cuBLAS 内部用了 Tensor Core、分块计算、共享内存、寄存器复用等极致优化，性能非常高，作为基准来衡量手写内核的优化效果。
   4. 缩放因子融合：直接把注意力的 1/√HS 缩放作为 alpha 参数融合到矩阵乘法里，不用额外再做一次全矩阵缩放，省了一次内存遍历。
   ---------------------------------------------------------------------------- */
void trimul_cublas(float* preatt,
                   const float* inp,
                   int B, int T, int C, int NH) {
    int HS = C / NH;  // 单头维度
    float* q, * k, * v;  // 维度重排，将输入从 (B, T, 3, NH, HS) 拆分为 3 个形状为 (B, NH, T, HS) 的张量
    // d_qkvr 是提前申请的一块大显存缓冲区（大小 B*T*3*C），下面把它平均切成 3 块，每块大小 B*T*C，分别给 q、k、v 使用，偏移量 0/1/2 * B*T*C 正好对应 Q/K/V 三个矩阵
    q = d_qkvr + 0 * B * T * C;
    k = d_qkvr + 1 * B * T * C;
    v = d_qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, 256);
    // 启动 permute 内核：每个线程搬运一个元素，256 线程为一个 block，总线程数等于单个张量的总元素数
    permute_kernel<<<num_blocks, 256>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // 使用 cuBLAS 执行批量矩阵乘法
    const float alpha = 1.0f / sqrtf(HS);
    const float beta = 0.0f;
    /* 这里一次性并行跑 B*NH 次矩阵乘法，每次算的都是 q @ k^T，输入形状 (T, HS)，输出 (T, T) 的注意力分数矩阵。
     下面说个最容易懵的坑：cuBLAS 是列优先存矩阵的，咱们 C 代码是行优先，所以这个函数的参数看起来很反直觉，给你讲明白为啥这么传就对了：
     1. 咱们行优先的 (T, HS) 矩阵，在 cuBLAS 眼里会被当成 (HS, T) 的列优先矩阵，所以主维度要填 HS。
     2. 为了不用额外再跑一次转置内核浪费性能，我们故意把 K 放第一个参数、Q 放第二个，并且给 K 加转置。
        乍一看你会以为在算 K^T @ Q，其实转置后形状是 (T, HS) @ (HS, T) = (T, T)，形状正好对。
     3. 你可能会担心结果不对：按 K^T @ Q 算出来的元素是 k_i · q_j，咱们要的是 q_i · k_j 啊？
        别怕，点积满足交换律 k_i·q_j = q_j·k_i，再加上 cuBLAS 是列优先输出，内存布局转置一下，正好就是咱们行优先想要的 q_i·k_j 矩阵。
     4. 这么传参的好处是：完全不用额外做矩阵转置，零开销直接拿到正确结果，性能最优。
     说白了就是利用 cuBLAS 的列优先特性，调换一下 Q/K 的参数顺序，白嫖一次转置，不用额外跑内核。                                                                            */
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                          CUBLAS_OP_T, CUBLAS_OP_N, // 对第一个矩阵（k）做转置，第二个矩阵（q）不转置
                                          T, T, HS,         // 输出矩阵 m 行 n 列，内部累加维度 k
                                          &alpha,           // 乘法缩放因子
                                          k, HS, T * HS,    // 第一个矩阵 A、主维度、批量间步长
                                          q, HS, T * HS,    // 第二个矩阵 B、主维度、批量间步长
                                          &beta,             // 输出矩阵原有值的缩放因子
                                          preatt, T, T * T, // 输出矩阵 C、主维度、批量间步长
                                          B * NH));          // 批量数，并行执行多少次矩阵乘法
}

/* ============================================================================
   第二章 - 手写内核通用调度框架
   ============================================================================
   以下为所有手写 CUDA 内核的通用调度层，采用模板 + 函数指针的设计实现逻辑解耦：
   将所有版本通用的线程划分、地址计算、块级因果掩码跳过逻辑固定在框架中，
   后续不同优化版本仅需实现核心的 8×8 区域矩阵计算逻辑，即可复用整套调度，无需重复编写通用代码。

   并行划分规则：
   1. 输出 T×T 矩阵划分为多个 128×128 的线程块（block），每个 block 负责一个大块的计算；
   2. 每个 block 内配置 16×16=256 个线程，每个线程负责 8×8 的结果计算区域（16×8=128，正好覆盖一个 block 的 128×128 范围）；
   3. 批量 B×注意力头 NH 维度放到 block 的 z 轴，所有样本、所有头完全独立并行。
   模板为编译期展开，无运行时额外开销，性能与手写固定版本完全一致。
   ============================================================================ */

/* ----------------------------------------------------------------------------
   模板函数核心设计价值
   ----------------------------------------------------------------------------
   代码复用 + 灵活插拔，把所有算子都一样的通用逻辑抽出来固定好，后续不同的优化算子只需要写自己不一样的核心计算部分，就能直接套进去用，不用写重复代码。

   1. 极致复用：把'掩码跳过、地址计算、线程划分'这些所有版本都一样的逻辑抽出来，
      后面 4 个优化内核只需要专注于“怎么算 8×8 子块更快”，不用重复写调度代码，大大简化了优化迭代的成本。
   2. 零开销抽象：模板是编译期展开的，函数指针在编译时就能确定，不会有运行时的额外开销，性能和直接写死内核完全一致。
   3. 和 cuBLAS 版本的核心差异：直接适配打包输入，省掉了 permute 拆分的显存搬运和额外缓冲区，这也是手写内核能和 cuBLAS 对标的优势之一。

   「模板」: 写一次通用逻辑，给它传不同的 “材料”（参数），编译器就能自动生成不同版本的具体代码，不用重复写。
   - 通用逻辑（比如线程调度、地址计算、因果跳过）写一次，放在模板里；
   - 可变的部分（比如矩阵计算、寄存器优化计算、向量化计算）作为 “材料” 传给模板；
   - 编译器自动生成对应版本的完整代码，性能和手写死的一模一样，没有额外开销。

   using 语法为函数指针创建别名：
   using 新名字 = 旧类型：就是给复杂的类型起别名，比如 using 整数 = int; 以后写 '整数 a' 就等价于 'int a'。
   void(*)(参数列表)：
      - 开头的 void：表示这类函数没有返回值；
      - 括号里的一长串：表示这类函数必须带这 9 个参数（输出矩阵指针、步长 T、K 指针、K 步长 C3、Q 指针、Q 步长 C3、序列长度 T、单头维度 HS、缩放因子），顺序和类型都不能错。
   整体含义：以后 matmul_fn_ptr 这个名字，就代表「所有返回值为 void、参数正好是这 9 个的函数」。

   template<matmul_fn_ptr matmul_tri>:
   定义一个通用模板，要求传入一个符合上面标准的计算函数，然后把该函数嵌到通用框架里，生成完整的可运行代码。
   1. template<...>：声明这是一个模板，尖括号里是要接收的 “材料” 参数。
   2. matmul_fn_ptr matmul_tri：表示这个模板要接收的材料，是一个 matmul_fn_ptr 类型的函数指针，参数名叫 matmul_tri。
      - 这里不是传类型，是传'一个具体的函数（函数的地址）'，属于'非类型模板参数'；
      - 传哪个函数进去，模板里调用 matmul_tri(...) 的时候，就会执行哪个函数的逻辑。

   对应到后面的代码，模板下面就是通用的 CUDA 内核 trimul_global，它负责做所有版本都一样的事：
   → 跳过无效上三角块、计算地址、定位 Q/K 指针，最后调用 matmul_tri 做核心的 8×8 矩阵计算。

   __launch_bounds__(256, 2) 是给编译器的优化提示：每个 block 最多 256 线程，每个 SM 最多同时跑 2 个 block，避免寄存器占用过高导致并行度下降。
   对角块（x=y）的处理：对角块不会被块级判断跳过（因为 y=x 不满足 y>x），它会进入计算流程。在具体的计算函数里会做更细粒度的子块级掩码。
   ---------------------------------------------------------------------------- */
using matmul_fn_ptr = void(*)(float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha);

template<matmul_fn_ptr matmul_tri>
__global__ void __launch_bounds__(256, 2) trimul_global(float* out, const float* inp, int T, int C, int NH) {
    if(blockIdx.y > blockIdx.x) return;  // 列大于行的块退出，上三角跳过（块级别的粗粒度跳过，块内所有 256 个线程都会执行这个判断）
    int C3 = C*3;
    int HS = C / NH;
    float scale = 1.0 / sqrtf(HS);
    // 将「批次 B × 注意力头 NH」维度放到 block 的 z 轴索引上
    int b  = blockIdx.z / NH;
    int nh = blockIdx.z % NH;
    // 获取当前批次、当前注意力头对应的基地址 （形状：inp (B, T, 3, NH, HS)， Q (B, NH, T, HS)， K (B, NH, T, HS) ）
    const float* q = inp + b * T * C3 + nh * HS;      // Q[b][nh][:][:] = inp[b][:][0][nh][:]
    const float* k = inp + b * T * C3 + nh * HS + C;  // K[b][nh][:][:] = inp[b][:][1][nh][:]
    float* r = out + (b*NH + nh)*T*T;                  // 输出 out[b][nh][:][:]
    /* 调用具体计算内核，开始矩阵乘法运算
       matmul_tri 是模板参数传进来的函数指针，在编译时就已确定了具体是哪个函数。
       目标函数会用这些参数，让当前块内的 256 个线程，分别计算自己负责的 8×8 子块的矩阵乘法结果，最终把所有结果写回到 `r` 指向的输出内存里。 */
    matmul_tri(r, T, k, C3, q, C3, T, HS, scale);
}

/* ----------------------------------------------------------------------------
   trimul_launcher：CPU 端启动封装函数
   ----------------------------------------------------------------------------
   尺寸校验：强制要求序列长度 T 是 128 的倍数，因为我们按 128×128 分块，非整数倍会导致结果错误。
   线程块划分：
   网格（grid）维度：dim3(T/128, T/128, NH*B)
   - x/y 轴：把 T×T 输出矩阵分成 (T/128)×(T/128) 个 128×128 的大块；
   - z   轴：所有批量和注意力头并行。
   线程块（block）维度：dim3(16, 16)，每个 block 共 256 线程，对应 16×16 的线程排布，
   每个线程负责计算 8×8 的结果子块（16×8=128，正好覆盖一个 128×128 的大块）
   ---------------------------------------------------------------------------- */
template<matmul_fn_ptr matmul_tri>  // 模板参数是有作用域的，每个模板函数都需要单独声明自己的模板参数
void trimul_launcher(float* out, const float* inp, int B, int T, int C, int NH) {
    assert(T % 128 == 0);  // 这里假设输入尺寸是规整的。没必要为了支持奇怪的、实际不会用到的尺寸把代码搞得一团糟
    trimul_global<matmul_tri><<<dim3(T / 128, T / 128, B * NH), dim3(16, 16)>>>(out, inp, T, C, NH);  // 不需要向上取整除法，反正如果不是 128 的倍数，结果本来就是错的
    cudaCheck(cudaGetLastError());
}

/* ============================================================================
   第三章 - 朴素实现（性能基线）
   ============================================================================
   第一个手写内核为无优化朴素实现，完全照搬 CPU 版本的计算逻辑，仅做最基础的并行化，无任何内存访问优化。
   核心性能瓶颈：严重的重复全局显存读取。计算 8×8 结果区域时，同一行 Q 向量会被重复读取 8 次，同一行 K 向量也会被重复读取 8 次；
   全局显存访问延迟远高于计算延迟，导致硬件计算单元大部分时间处于闲置等待状态，利用率极低。
   基准性能：作为后续所有优化版本的性能对比基线。
   ============================================================================ */

/* ----------------------------------------------------------------------------
   matmul_tri_naive：朴素实现
   ----------------------------------------------------------------------------
   照搬 CPU 版本的计算方式，只做了最基础的并行化，没有任何内存优化
   （基准实现：耗时 20 ms）

   索引坐标系说明：
   | - - - - - j 轴（列）
   |
   |
   i 轴（行）
   ---------------------------------------------------------------------------- */
__device__ void matmul_tri_naive(float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha) {
    // 全局输出矩阵 T×T → 128×128 线程块（blockIdx.x/y）→ 线程内 8×8 计算区域（threadIdx.x/y → i_base/j_base）→ 区域内偏移 io/jo → 全局坐标 i/j → 特征维度 hs
    int i_base = blockIdx.x * 128 + threadIdx.x * 8;  // 当前线程负责的 8×8 子块的'行'起始位置（query 位置起始）
    int j_base = blockIdx.y * 128 + threadIdx.y * 8;  // 当前线程负责的 8×8 子块的'列'起始位置（key 位置起始）
    /* 线程级因果掩码判断：如果当前线程负责的 8×8 计算区域完全落在上三角（列起始 > 行起始），当前线程直接退出不计算。
       位于对角线上的线程块（blockIdx.x == blockIdx.y）中，部分 8×8 计算区域会跨上下三角；
       我们没有对 8×8 区域内的单个上三角元素做细粒度跳过，直接计算也不影响结果 —— 后续结果校验时，这些上三角元素会被 CPU 参考实现的因果掩码忽略。
       也可以在下方最内层循环加 if(i > j) 逐个跳过无效元素，但 if 分支判断会拖慢执行效率，因此这里选择用极少量的无效计算换更高的执行性能。  */
    if(j_base > i_base) return;
    // 单线程内逐个计算 8×8=64 个结果元素
    for(int io = 0; io < 8; ++io) {
        int i = i_base + io;     // 当前计算的结果行（query 位置）
        for(int jo = 0; jo < 8; ++jo) {
            int j = j_base + jo; // 当前计算的结果列（key 位置）
            float val = 0;
            /* 三个步长（QS/KS/PS）的由来
               步长（stride）是行优先存储矩阵中，跨一行需要偏移的元素个数，完全由'Q/K/输出矩阵'的内存布局决定：
               1. QS（Q 矩阵行步长）：Q 是从打包输入 inp 里取的，inp 形状 (B,T,3,NH,HS)，每个 token 的 Q 后面紧跟着'K 和 V'，因此 Q 矩阵里下一个 token 的 Q 要偏移 C3 个元素，所以 QS=C3
               2. KS（K 矩阵行步长）：K 同样在打包输入里，每个 token 的 K 后面紧跟着'V 和下个 token 的 Q'，跨一行也是偏移 C3，所以 KS=C3
               3. PS（输出矩阵行步长）：输出是标准行优先的 T×T 矩阵，下一行偏移 T 个元素，所以 PS=T

               inp 形状（打包 QKV）：(B,T,3,NH,HS)
               全局逻辑形状：Q(B,NH,T,HS) @ K(B,NH,T,HS) = 输出(B,NH,T,T)
               单头计算形状：Q(T,HS) @ K(T,HS) = 输出(T,T)
               逻辑形状 ≠ 实际内存布局
               虽然单头 Q/K 的逻辑形状是 (T, HS)，但因为它们是从打包的 inp 里取的，内存并不是'连续'的 (T,HS) 布局：
               - 每个 token 的 Q 后面，紧跟着同 token 的 K 和 V，因此跨一行（下个 token 的 Q）需要偏移 C3 个元素，而不是 HS 个。 */
            for (int hs = 0; hs < HS; ++hs) {  // 公共维度在最内层，低效版本
                val += q[i * QS + hs] * k[j * KS + hs]; // 计算 Q 第 i 行 和 K 第 j 行 的 HS 维点积
            }
            p[i * PS + j] = val * alpha; // 乘缩放因子，写入输出矩阵
        }
    }
}

/* ============================================================================
   第四章 - 寄存器复用优化
   ============================================================================
   第二个内核为寄存器复用优化版，核心优化手段是重排循环顺序：
   把原本需要反复从全局显存读取的 Q/K 行，预先加载到寄存器中重复利用，大幅减少全局显存访问，将 Q/K 行的全局显存读取次数从原来的 8 次降低为 1 次。
   受限于 GPU 单线程寄存器容量限制，单线程最多负责 8×8 结果区域，无法进一步扩大计算范围。
   ============================================================================ */

/* ----------------------------------------------------------------------------
   matmul_tri_registers：寄存器复用优化版
   ----------------------------------------------------------------------------
   朴素版的问题：
   朴素版计算每个 (i,j) 元素时，都要重新从全局显存读一整行 Q 和一整行 K：
   - 8×8=64 个元素，Q 的每一行会被重复读 8 次，K 的每一行也会被重复读 8 次；
   - 全局显存访问延迟极高，计算单元 90% 以上的时间都在等数据，硬件利用率极低。

   寄存器版的优化：
   通过重排循环顺序，把 “先算每个元素的完整点积” 改成 “先处理每个特征维，批量加载后完成所有元素的该维累加”：
   - 每个 Q 元素、每个 K 元素仅需要从全局显存读 1 次，就能在寄存器里被复用 8 次；
   - 全局显存读取量直接降到朴素版的 1/8，大部分时间都在做寄存器内的高速计算，硬件利用率大幅提升。
   ---------------------------------------------------------------------------- */
__device__ void matmul_tri_registers(float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha) {
    int i_base = blockIdx.x * 128 + threadIdx.x * 8;
    int j_base = blockIdx.y * 128 + threadIdx.y * 8;
    if (j_base > i_base) return;
    // 将指针偏移到当前线程负责区域的位置
    q += i_base * QS;           // 偏移后 q[u * QS + hs] 就等价于原来的 q[(i_base+u) * QS + hs]
    k += j_base * KS;
    p += i_base * PS + j_base; // 偏移后 p[i * PS + j] 就等价于原来的 p[(i_base+i) * PS + (j_base+j)]

    float vals[8][8] = {};     // 8×8 的寄存器数组
    for (int hs = 0; hs < HS; ++hs) {  // 外层公共维度循环
        float lhs[8];
        float rhs[8];
        // 批量加载到寄存器：把当前特征维下，8 个 query 行的第 hs 维元素（lhs[8]）和 8 个 key 行的第 hs 维元素（rhs[8]），一次性从全局显存读到寄存器数组
        for (int u = 0; u < 8; ++u) {
            lhs[u] = q[u * QS + hs];
            rhs[u] = k[u * KS + hs];
        }
        // 寄存器内完成 8×8=64 次乘加累加
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                vals[i][j] += lhs[i] * rhs[j];
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            p[i * PS + j] = vals[i][j] * alpha;
        }
    }
}

/* ============================================================================
   第四章 - 向量化加载优化
   ============================================================================
   第三个内核为向量化加载优化版，在寄存器复用的基础上进一步优化内存访问效率：
   使用 float4 向量指令一次加载/存储 4 个 float 元素，实现全局显存的合并访问，充分利用显存带宽；
   同时输出阶段也采用 float4 向量写入，减少存储指令数量。
   本版本核心计算逻辑与寄存器复用版完全一致，仅优化内存访问方式，无算法层面改动。
   ============================================================================ */

// 封装 float4 读写，提升可读性
__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}
__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

/* ----------------------------------------------------------------------------
   matmul_tri3：向量化加载优化版
   ----------------------------------------------------------------------------
   逻辑与上一个内核相同，只是改用 float4 加载以提升内存合并度。

   为什么向量化能让性能翻倍？

   1. 寄存器版的访存问题
   寄存器版里，每个线程每次只读 1 个 float，虽然特征维 hs 是连续的，但 warp 内不同线程访问的是不同 u 行的元素，
   而 u 行之间的步长是 QS=3*C（跨步很大），导致 warp 内的访问地址不连续，合并度很低，大量显存带宽被浪费。

   2. float4 向量化的优化效果
   用 float4 后，每个线程一次读 4 个连续的特征维元素（hs 到 hs+3），warp 内的线程访问的地址块变成连续的 128 位块，
   GPU 可以完美合并这些访问，把显存带宽打满；同时访存指令数量减少到原来的 1/4，指令开销也大幅降低。

   注意：向量化只能在'内存连续的维度'上做，这里只有最内层的特征维 hs 是连续的，所以步长加在 hs 上； u 维度是跨步的，不能做向量化。
   ---------------------------------------------------------------------------- */
__device__ void matmul_tri3(float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha) {
    int i_base = 128 * blockIdx.x + 8 * threadIdx.x;
    int j_base = 128 * blockIdx.y + 8 * threadIdx.y;
    if (j_base > i_base) return;
    q += i_base * QS;
    k += j_base * KS;
    p += i_base * PS + j_base;
    /* 为什么 lhs 不放在外层和 rhs 一起？
       1. 两者的复用生命周期不同
       - rhs（K 的 8 行的 4 个特征维）：需要在所有 8 次 i 循环里被复用 —— 每个 Q 行都要和 8 个 K 行做乘加，所以必须放在外层，加载一次后保留到整个 hs 循环结束。
       - lhs（Q 第 i 行的 4 个特征维）：只在当前 i 的 8 次 j 循环里被复用，下一个 i 就需要加载新的 Q 行，不需要保留 8 个 Q 行的数据。所以放在 i 循环里加载即可，用完就可以释放寄存器。
       2. 大幅减少寄存器占用
       - 如果把 lhs 也放在外层做成 lhs[8]，需要 8 个 float4 寄存器，也就是 32 个 float 寄存器；
       - 现在放在 i 循环里，只需要 1 个 float4 寄存器（4 个 float），寄存器占用直接减少 28 个 float。
       这种写法完全没有降低复用率 —— lhs 在 i 循环里加载一次，还是会在 j 循环里复用 8 次，和放在外层的复用次数完全一致，只是加载位置从外层移到了内层，不影响性能，还省了寄存器。 */
    float vals[8][8] = {};
    for (int hs = 0; hs < HS; hs += 4) {  // 步长从 1 变成 4，一次处理 4 个特征维
        float4 rhs[8];  // 以 float4 形式加载，提升内存合并度
        for (int u = 0; u < 8; ++u) {
            rhs[u] = ld_vec(k + u * KS + hs); // 一次加载 K 第 u 行的 4 个连续特征维
        }
        for (int i = 0; i < 8; ++i) {
            // lhs 不需要在 i 循环外保留，它只在 j 循环中被复用
            float4 lhs = ld_vec(q + i * QS + hs); // 一次加载 Q 第 i 行的 4 个连续特征维
            for (int j = 0; j < 8; ++j) {
                // 4 个特征维分别乘加，等价于原来 4 次单维乘加
                vals[i][j] += lhs.x * rhs[j].x;
                vals[i][j] += lhs.y * rhs[j].y;
                vals[i][j] += lhs.z * rhs[j].z;
                vals[i][j] += lhs.w * rhs[j].w;
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) { // 列步长改为 4，一次写 4 个连续输出元素
            float4 result;
            result.x = vals[i][j + 0] * alpha;
            result.y = vals[i][j + 1] * alpha;
            result.z = vals[i][j + 2] * alpha;
            result.w = vals[i][j + 3] * alpha;
            st_vec(p + i * PS + j, result);
        }
    }
}

/* ============================================================================
   第五章 - 共享内存分块优化
   ============================================================================
   第四个内核为共享内存分块优化版，是本项目性能最优的手写实现，进一步逼近 cuBLAS 性能：
   将 128×32 的 Q/K 数据分片预先由全线程块协作加载到共享内存中，后续所有计算都从延迟远低于全局显存的共享内存读取数据，进一步降低全局显存访问压力。

   核心优化思路：
   1. 把整个线程块（128×128 输出区域）需要的 128 行 Q、128 行 K 的 32 维公共维度分片，通过'全线程协作、完全合并访问'一次性加载到共享内存；
   2. 后续所有计算都从延迟仅 20~30 周期的共享内存读取，不再访问全局显存；
   3. 通过地址偏移避免共享内存 bank 冲突，把共享内存访问效率拉满。
   优化细节：
   1. 数据加载阶段采用全线程协作，保证全局显存合并访问；
   2. 数据加载需要所有线程共同参与，因此上三角线程块不能在计算开始前提前退出，仅在最终写入结果阶段做因果掩码判断，跳过无效区域写入。
   ============================================================================ */

// matmul_tri4：共享内存分块优化版
__device__ void matmul_tri4(float* p, int PS, const float* k, int KS, const float* q, int QS, int T, int HS, float alpha) {
    int i_base = blockIdx.x * 128 + threadIdx.x * 8;
    int j_base = blockIdx.y * 128 + threadIdx.y * 8;
    /* 数据加载需要所有线程共同参与，因此即使某些线程不负责有效结果，也不能提前退出。
       这里的提前退出和之前版本完全不同：只做 block 级的上三角判断（整个块完全在上三角才退出），不做 8×8 线程级的判断。
       原因是：后面的数据加载需要整个线程块的 256 个线程共同参与，哪怕某个线程负责的 8×8 区域在上三角、最终结果没用，它也必须参与加载共享内存的数据，
              不然它负责的那部分数据没人搬，其他线程就拿不到完整的输入。所以加载阶段不能提前退出，只有等所有计算做完、写结果的时候，才会判断无效区域不写。 */
    if (blockIdx.y > blockIdx.x) return;
    // 这里的指针偏移是'偏移到整个线程块负责的 128 行 Q/K 的起始位置'，而不是单线程的 8 行起始 —— 因为后面要加载整个块的 128 行数据到共享内存，要 block 的基地址
    q += blockIdx.x * 128 * QS;
    k += blockIdx.y * 128 * KS;
    __shared__ float lhs_s[128][32]; // 存 128 个 Q 行的 32 维公共维度分片
    __shared__ float rhs_s[128][32]; // 存 128 个 K 行的 32 维公共维度分片

    float vals[8][8] = {};
    // 公共维度 HS 被拆成多个 32 维的块，每次处理一个 32 维分片，逐块加载到共享内存计算，避免共享内存容量不够
    for (int so = 0; so < HS; so += 32) {
        /* 全线程协作加载阶段：临时重组线程分工，实现全局内存完全合并访问。
           加载阶段的线程分工和计算阶段完全不同，只为了让 warp 内 32 个线程访问的全局内存地址完全连续，实现 100% 合并访问：
           1. threadIdx.y 两两分组：同组两个线程分别负责一行的前 16 列和后 16 列，凑齐一行 32 列
           2. 8 组线程按行号错开，每组负责每隔 8 行的 16 行，8 组正好覆盖 128 行
           3. 每个线程每次加载一行的 16 个连续元素，warp 内地址完全连续，带宽利用率拉满
           注意：即使部分线程加载的数据自己计算时用不上，但这是整个线程块的协作任务，必须所有线程参与。
           最终整个块会加载 128 个 Q 行、128 个 K 行的当前 32 维分片到共享内存。                      */
        __syncthreads();
        for(int y = threadIdx.y / 2; y < 128; y += 8) {
            int xo = (threadIdx.y % 2) * 16;
            lhs_s[y][threadIdx.x + xo] = q[y * QS + so + threadIdx.x + xo];
            rhs_s[y][threadIdx.x + xo] = k[y * KS + so + threadIdx.x + xo];
        }
        __syncthreads();

        /* 计算阶段：基于共享内存中的 32 维数据分片，计算 128×128 输出区域的部分点积
           每个线程负责其中 8×8 个输出元素的累加：
           - threadIdx.x 方向：每个线程负责 8 个连续的 Q 行，16 个线程正好覆盖 128 个 Q 行
           - threadIdx.y 方向：每个线程负责 8 个连续的 K 行，16 个线程正好覆盖 128 个 K 行
           外层 so 循环每处理一个 32 维分片，就把当前分片的点积结果累加到 vals 中；
           所有分片处理完后，vals 中就是完整的 8×8 点积结果，最后只写下三角的有效区域。  */
        for (int si = 0; si < 32; ++si) {
            float rhs[8];
            for (int u = 0; u < 8; ++u) {
                 // 列索引用 (threadIdx.x+si)%32 而非直接用 si：把 warp 内线程的访问列打散到不同 bank，避免共享内存 bank 冲突
                rhs[u] = rhs_s[threadIdx.y * 8 + u][(threadIdx.x + si) % 32];
            }

            for (int ii = 0; ii < 8; ++ii) {
                float lhs = lhs_s[threadIdx.x * 8 + ii][(threadIdx.x + si) % 32];
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs * rhs[ji];
                }
            }
        }
    }
    // 完全在上三角的区域直接返回，不写结果
    if (j_base > i_base) return;
    // 写回用 float4 向量化，和上版本一致
    for (int ii = 0; ii < 8; ++ii) {
        for (int ji = 0; ji < 8; ji += 4) {
            int i = i_base + ii;
            int j = j_base + ji;
            float4 result;
            result.x = vals[ii][ji + 0] * alpha;
            result.y = vals[ii][ji + 1] * alpha;
            result.z = vals[ii][ji + 2] * alpha;
            result.w = vals[ii][ji + 3] * alpha;
            st_vec(p + i * PS + j, result);
        }
    }
}

/* ============================================================================
   内核调度与性能测试
   ============================================================================
   以下为内核统一调度入口与主函数：
   1. trimul_gpu 为统一调度入口，支持通过 kernel_num 参数选择不同计算内核：
      0=cuBLAS 基准，1=朴素版，2=寄存器优化版，3=向量化优化版，4=共享内存优化版；
   2. 主函数完成显存/内存分配、随机数据初始化、正确性校验（与 CPU 参考实现结果对比）、重复性能基准测试，
      最终输出手写内核与 cuBLAS 的性能对比结果。
   ============================================================================ */

void trimul_gpu(int kernel_num,
                float* out,  const float* inp,
                int B, int T, int C, int NH) {
    switch (kernel_num) {
        case 0:
            trimul_cublas(out, inp, B, T, C, NH);
            break;
        case 1:
            trimul_launcher<matmul_tri_naive>(out, inp, B, T, C, NH);
            break;
        case 2:
            trimul_launcher<matmul_tri_registers>(out, inp, B, T, C, NH);
            break;
        case 3:
            trimul_launcher<matmul_tri3>(out, inp, B, T, C, NH);
            break;
        case 4:
            trimul_launcher<matmul_tri4>(out, inp, B, T, C, NH);
            break;
        default:
            printf("无效的内核编号\n");
            exit(1);
    }
}

int main(int argc, char **argv) {
    setup_main();

    // 让 cuBLAS 走 FP32（不开启 TF32 张量核）：与手写内核同精度公平对比，
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;
    int HS = C / NH;

    // 在主机端创建随机数内存
    float* out = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // 迁移到 GPU 端
    float* d_out;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS 使用的缓冲区
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));

    // 从命令行读取内核编号
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("使用内核版本 %d\n", kernel_num);

    // 如果不是 profile 模式，则执行标准验证与基准测试流程
    if (!(argc > 2 && strcmp(argv[2], "profile") == 0)) {
        // 首先校验内核的正确性：与 CPU 参考实现对比
        trimul_cpu(out, inp, B, T, C, NH);
        trimul_gpu(kernel_num, d_out, d_inp, B, T, C, NH);
        validate_result(d_out, out, "out", B * NH * T * T, 1e-4f);

        printf("所有结果验证通过，开始性能基准测试。\n\n");

        // 测试内核的运行速度
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, trimul_gpu,
                                              kernel_num, d_out, d_inp,
                                              B, T, C, NH);

        float cublas_time = benchmark_kernel(repeat_times, trimul_gpu,
                                             0, d_out, d_inp,
                                             B, T, C, NH);

        // 统计 FLOPs（一次乘加 = 2 FLOPs）：cuBLAS 计算完整 T×T 矩阵，手写内核只算下三角（含对角线）
        double full_flops = (double)B * NH * T * T * 2 * HS;              // cuBLAS：全矩阵 FLOPs
        double tri_flops  = (double)B * NH * (T * (T + 1) / 2) * 2 * HS;  // 手写内核：下三角有效 FLOPs
        double my_flops   = (kernel_num == 0) ? full_flops : tri_flops;   // 被测内核按实际计算口径取值

        printf("被测内核 %.2f ms | %.2f GFLOPs | %.2f TFLOPS\n",
               elapsed_time, my_flops / 1e9, my_flops / elapsed_time / 1e9);
        printf("CuBLAS    %.2f ms | %.2f GFLOPs | %.2f TFLOPS（全矩阵）\n",
               cublas_time, full_flops / 1e9, full_flops / cublas_time / 1e9);

        // 释放内存
        free(out);
        free(inp);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_qkvr));
        cublasDestroy(cublas_handle);

        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    /* profile 专用分支，仅在传入 "profile" 参数时执行

     特别注意：
       版本0（cuBLAS 基准）每次调用会启动 1 个 permute_kernel，外加 cuBLAS 内部的 gemm kernel；
       版本1~4（手写内核）每次调用各启动 1 个 trimul_global kernel。
       按 cuBLAS 单次 gemm 启动 1 个 kernel 估算，一轮的总启动次数约 = 2+1+1+1+1 = 6。
       预热 2 轮 ≈ 12 个 kernel，采集 1 轮 ≈ 6 个 kernel（对应 v0 2 个、v1~v4 各 1 个）。
       注意：cuBLAS 内部 kernel 数量可能随 CUDA 版本变化，若实测数量不一致，以 nsys 看到的实际启动序列为准

     编译指令示例：
       nvcc -O3 -lineinfo --use_fast_math -lcublas -lcublasLt trimat_forward.cu -o trimat_forward

     NCU 采集命令示例（跳过 12 次预热启动，采集接下来的约 6 次正式启动）：
       ncu --set full --cache-control all --launch-skip 12 --launch-count 6 -o trimat_profile ./trimat_forward 1 profile

     系统级性能分析：
       nsys profile --stats=true -o trimat_nsys ./trimat_forward 1 profile
     */
    {
        const int warmup_rounds = 2;

        // 预热 2 轮：依次调用版本 0 到 4
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 0; ver <= 4; ver++) {
                trimul_gpu(ver, d_out, d_inp, B, T, C, NH);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮：依次调用版本 0 到 4
        for (int ver = 0; ver <= 4; ver++) {
            printf("正在采集版本 %d ...\n", ver);
            trimul_gpu(ver, d_out, d_inp, B, T, C, NH);
        }
        cudaCheck(cudaDeviceSynchronize());

        // 释放内存
        free(out);
        free(inp);
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_inp));
        cudaCheck(cudaFree(d_qkvr));
        cublasDestroy(cublas_handle);

        return 0;
    }
}