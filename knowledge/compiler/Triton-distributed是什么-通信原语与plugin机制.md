# Triton-distributed 是什么：通信原语 + Triton plugin 机制

> **定位**：了解 Triton-distributed（字节 Seed 出的「分布式 Triton」）。它和 triton-ascend 是同一脉络的东西——都在「扩展 Triton」，区别只是前者往「多卡通信」方向扩、后者往「昇腾后端」方向扩。搞懂它还能顺便理解 Triton 3.7 新引入的 plugin 机制。

## 〇、一句话

**Triton-distributed 把「跨卡通信」也变成 kernel 里可以自由编排的一条条指令（通信原语），让你像写普通 Triton kernel 一样写出「计算和通信重叠」的多卡算子；而它加载进原生 Triton 的方式，靠的是 3.7 新增的 plugin 机制（`TRITON_PLUGIN_PATHS`）——不用再 fork 整个 Triton 源码树。**

## 一、它解决什么问题：通信在拖后腿

大模型多卡训练/推理里，通信和计算经常是**串行**的：算一段 → 停下来等 AllReduce / AllToAll / 专家并行(EP)的 dispatch-combine → 再算下一段。**通信那段时间，卡是闲着的。**

```
串行（慢）：        ┌────计算────┐   ┌──通信──┐   ┌────计算────┐
                   └────────────┘   └────────┘   └────────────┘
                                      ↑ 这段时间卡在干等

重叠（快）：        ┌────计算────┐  ┌────计算────┐
                   └────┬───────┘  └────┬───────┘
                     ┌──┴──通信──┐  ┌──┴──通信──┐   ← 通信藏在计算里，几乎「免费」
                     └───────────┘  └───────────┘
```

Triton-distributed 就是干「重叠」这件事的。官方数据：它写的低延迟 AllToAll，32 卡 H800 上 137µs，比 DeepEP 的 182µs 还快。

## 二、通信原语：把「跨卡搬数据」变成 kernel 里的指令

### 2.1 什么是原语

**「原语」= 最小、不可再分的基础操作单元**，用来搭更复杂的系统。你熟悉的 `tl.load` / `tl.store` / `tl.program_id` 是「单卡内的计算/访存原语」；**通信原语**就是「跨卡搬数据/同步」的最小操作单元。

传统 MPI/NCCL 的通信是**库调用**（在 kernel 外面喊一句「帮我同步」）；Triton-distributed 把它们**下沉成 kernel 里的一条条指令**（类似 NVSHMEM 的「单边通信」模型），于是编译器和程序员能把通信指令和计算 tile **交错、重叠**。

### 2.2 底层原语分三类

| 类别 | 原语 | 作用 | 用你熟悉的东西类比 |
|---|---|---|---|
| **数据搬运** | `putmem_nbi_block(...)` | 非阻塞把一块内存**写到远端卡**（`nbi`=non-blocking immediate，发出即返回） | 跨卡版的 `tl.store` |
| | `getmem(...)` | 从远端**读** | 跨卡版的 `tl.load` |
| **同步/保序** | `fence()` | 保证之前的写对远端可见（内存屏障） | 跨卡的内存序保证 |
| | `signal_op` / `signal_wait_until` | 用信号量做生产者-消费者握手（我发完通知你 / 我等你） | 跨卡的条件等待 |
| **寻址/标识** | `tid(axis=0)` | 拿到线程/向量 lane 的 id | 比 `tl.program_id` 更细一层的线程号 |
| | `rank` / `WORLD_SIZE` | 第几张卡 / 一共几张卡 | 分布式常识 |

### 2.3 一段真实片段（AllToAll kernel 里的几行）

```python
# 算好源/目标偏移 → 异步 put 到远端 → fence 保序 → signal 通知对方
libshmem_device.putmem_nbi_block(data_dst_ptr, data_src + src_off * HIDDEN, nbytes, pid)
libshmem_device.fence()
libshmem_device.signal_op(signal_ptr, call_count, libshmem_device.NVSHMEM_SIGNAL_SET, pid)
libshmem_device.signal_wait_until(signal + ..., NVSHMEM_CMP_EQ, call_count)
```

> **关键直觉**：这些原语是「积木」，用它们可以拼出高层算子——AllReduce、AllToAll、MoE 的 dispatch/combine。所以官方说「目前发布了**底层原语**，**高层原语**（直接给现成算子的抽象）计划中」。这和你用 `tl.load`/`tl.dot` 拼 GEMM 是同一套「积木」思维，只是积木换成了跨卡通信。

## 三、Triton 3.7 的 plugin 机制：扩展不用再 fork

### 3.1 为什么要有它

改 Triton 源码树里的 C++（pass/dialect），每次都要重编整个 `libtriton.so`——就是因为 **3.6 没有 plugin 机制**：加后端只能 **fork + 打补丁**（triton-ascend 早期就是这么干的，`third_party/ascend/` 整棵都长在 Triton 树里）。3.7 的 plugin 机制就是解这个痛点：**扩展与内核解耦**，各维护各的仓库。

```
老方式（3.6）：                       新方式（3.7 plugin）：
  Triton 源码树                         原生 Triton（不动）
   ├─ ...（Triton 自己）                 +  你的插件仓库
   └─ third_party/ascend/（长在里面）        └─ 编成一个 libxxx.so
   → 改后端 = 重编整个 Triton           → 改后端 = 只重编自己的 .so
```

### 3.2 怎么工作（三个要点）

1. **入口是环境变量 `TRITON_PLUGIN_PATHS`**：一串 `.so` 路径（`:` 分隔）。Triton 在**第一次 import**（初始化 `libtriton`）时，**惰性、且只加载一次**地 `dlopen` 每个路径，结果缓存。所以**必须在 `import triton` 之前设好**，否则加载不到。

2. **插件 `.so` 只导出唯一一个 C 入口** `tritonGetPluginInfo()`，返回一个 `PluginInfo` 结构体，登记它贡献的三样东西——**dialect 列表 / pass 列表 /（可选）op builder 列表**：

   ```cpp
   TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
     static plugin::DialectInfo dialects[] = {          // 我注册了哪些 dialect
       {"DistributedDialect", ..., registerDistributedDialect},
       {"SIMTDialect", ..., registerSIMTDialect}, ...
     };
     static plugin::PassInfo passes[] = {               // 我注册了哪些 pass
       {"convert_triton_distributed_to_tritongpu", ..., addConvertTritonDistributedToTritonGPU, ...},
       ...
     };
     static plugin::PluginInfo info = { TRITON_PLUGIN_API_VERSION, "TritonDistributed",
       ..., passes, /*numPasses*/..., dialects, /*numDialects*/..., /*ops*/nullptr, 0,
       TRITON_VERSION };
     return &info;
   }
   ```

   Triton 加载后把这些登记项注册进自己的 `DialectRegistry` / `PassRegistry`，就像它们是原生的。

3. **构建是 out-of-tree**：插件在自己仓库里编，`find_package(MLIR)` + 指向「已单独编好的 `libtriton.so` 头文件/tablegen 产物」；链接时 Triton/MLIR/LLVM 符号**留空**（`--allow-shlib-undefined`），运行时才从 `libtriton.so`（`RTLD_GLOBAL`）解析。**关键约束**：插件必须和运行时 import 的那个 Triton 用同一套 LLVM/pybind11 编，否则 ABI 对不上，加载即崩。

> **还不够的点**：plugin 机制并非万能。triton-distributed 源码注释里明确说，它仍打了两个最小补丁（AxisInfo 访问器注册钩子、软件流水线 predication 钩子）——这两个扩展点 3.7.1 还没通过 plugin 机制暴露。所以它现在是「plugin + 少量 patch」的混合态。

## 四、和 triton-ascend 的关系

| | triton-distributed | triton-ascend |
|---|---|---|
| 扩展方向 | **多卡通信**：加 Distributed/SIMT 方言 + 通信原语 | **昇腾后端**：把 lowering 后半段换成华为毕昇系 |
| 当前形态 | 3.7 上已走 **plugin**（`TRITON_PLUGIN_PATHS` 加载 `.so`） | 早期是 **in-tree fork**（`third_party/ascend/` 长在树里） |

两者是同一件事的两面——**都在「扩展 Triton」**。triton-distributed 已经走通 3.7 的 plugin 路线；等 triton-ascend rebase 到 3.7，理想情况下它也能像 triton-distributed 一样，把后端 pass 编成 plugin `.so`，**不再 fork 整个 Triton**，把「改 C++ → 重编 libtriton」的流程变成「改 .so → 只重编插件」。

## 五、该掌握的 / 不用掌握的

| 该掌握的 | 不用掌握的 |
|---|---|
| 通信原语 = 跨卡搬数据/同步的最小指令（put/fence/signal） | 手写 NVSHMEM 通信代码 |
| 「计算-通信重叠」为什么能提性能 | 每个原语的精确语义/参数 |
| plugin 机制三要素：`TRITON_PLUGIN_PATHS` + `tritonGetPluginInfo()` + 返回 `PluginInfo`（dialects/passes/ops） | 手写 `PluginInfo` 结构体 |
| plugin 必须 `import triton` 之前设好、和 Triton 同版本编译 | 3.7 那些 patch 的具体实现 |
