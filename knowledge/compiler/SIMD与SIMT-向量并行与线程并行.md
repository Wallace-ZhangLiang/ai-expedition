# SIMD 与 SIMT：向量并行与线程并行

> 编译器 / 硬件并行模型里的两个核心概念。读 Triton / GPU / NPU 源码和 IR 时，理解它们能快速看懂「底层到底在怎么并行」。

## 一句话

**SIMD 是「一条指令并行算多个数据」；SIMT 是「一条指令同时驱动多个线程」。** 前者显式、后者隐式。

## SIMD —— Single Instruction, Multiple Data

一条指令同时作用于多个数据元素，数据并行是**显式**暴露给程序的。

- 用**向量寄存器**一次性算 N 个数（一条指令把 8 个 float32 加起来）。
- 向量宽度 N 是**固定的、架构相关**的（AVX 256bit = 8×fp32，AVX-512 = 16×fp32，ARM NEON，NPU 的 vector 单元）。
- **没有「线程」概念**，只有一条控制流，代码里明摆着写「这是向量操作」。
- 典型：CPU 的 SSE/AVX/NEON，NPU 的向量计算单元。

## SIMT —— Single Instruction, Multiple Threads

很多「线程」执行同一条指令，但每个线程各有自己的寄存器/数据，数据并行是**隐式**的。

- 写的是**每线程的标量代码**（普通 if/else/循环），硬件把一群线程打包成 **warp**（NVIDIA 32 线程）**lockstep 锁步执行**——一条指令同时驱动这 32 个线程。
- 各线程数据不同，但**共享同一条指令流**。
- 线程可走不同分支（**divergence**），硬件用 predication/masking 处理：分歧的两条路都串行执行一遍，用 mask 屏蔽，有性能代价。
- 典型：GPU（CUDA / OpenCL / AMD）。

## 核心区别

| | SIMD | SIMT |
|---|---|---|
| 并行对象 | 数据（向量 lane） | 线程（warp 里的 thread） |
| 并行是否显式 | 显式（写向量指令） | 隐式（写标量代码，硬件聚合） |
| 控制流 | 单一控制流 | 线程可有分歧（divergence） |
| 编程心智 | 「一条指令算 N 个」 | 「N 个线程各算各的」 |
| 典型硬件 | CPU SIMD、NPU 向量单元 | GPU（CUDA）、SIMT 风格 NPU |

## 为什么两者能共存

底层硬件其实都是「一条指令并行处理多份数据」，区别在**抽象层次**：

- SIMD 把这个能力**直接暴露成向量**；
- SIMT 在外面包一层「线程」抽象，让你写标量代码，硬件再把你骗进锁步执行。

所以 **SIMT 可以理解为「用线程做外壳、底下仍是 SIMD 式并行」的一种编程模型**。

## 和 Triton / NPU 后端的关系

Triton 的编程模型（SPMD、block/tile、`tl.program_id`、**没有 `threadIdx`**）更像 **SIMT 那套抽象**：你写「一块怎么算」，后端决定怎么铺到线程 / warp 上，与 GPU 心智天然对齐。

一些 Triton 后端（如 triton-ascend）会提供 `simd` / `simt` 两种编译模式，对应两条 lowering 路线：

- **SIMT 路线**：把 block 内运算映射成「线程 + warp 锁步」语义，更贴近 NVIDIA / TritonGPU 那套 block 布局，方便复用上游优化。
- **SIMD 路线**：更直接落到 NPU 的**纯向量单元**语义。

同一 kernel 走两条路线，产出的 IR 与优化策略不同，某些算子就可能在某一条路径下暴露问题（例如对齐失败、片上缓冲区溢出）。
