# Triton 入门到上手：一份学习指南

> **写给谁**：没写过 Triton、但有 Python 基础的工程师（懂点 CUDA / 算子更好，不懂也能跟）。
> **目标**：从「零基础」走到「能在一个 Triton 后端上独立完成一个真实 issue / 算子改单」。
> **主线**：以昇腾后端 **triton-ascend** 为主线，但前半段的核心概念与硬件无关，任何 Triton 后端（NVIDIA / AMD / 昇腾）都通用。
> **怎么读**：分五个阶段，每阶段都有「动手」和「搞懂」两部分。建议每天 1~2 小时连续投入。

---

## 〇、先用五分钟建立正确认知

### 0.1 Triton 是什么：写「怎么分块算」，而不是「怎么算」

**Triton 是一种「写算子的高级语言」**：你写一段长得像 Python 的代码，编译器帮你翻译成硬件（GPU / 昇腾 NPU）能跑的指令。

它解决的是这样一个痛点——一个模型跑起来，拆开是无数个 **kernel（算子/内核）**：

```
y = W·x + b        ← 矩阵乘 GEMM
Attention / MoE / LayerNorm / KV cache 读写 / 量化 ...
```

这些 kernel 是「写给人看的代码」，加速卡不认识，必须翻译成硬件指令。翻译有两条路：

| | 算子库（如 CANN 算子库） | Triton 编译 |
|---|---|---|
| 比喻 | 预制菜 / 大厨招牌菜 | 现场照着菜谱现做 |
| 谁写的 | 人工预调优 | 编译器自动翻译 |
| 覆盖 | 常见通用算子（GEMM / 卷积 / LayerNorm） | **任意新 kernel** |
| 性能 | 最好 | 取决于编译器质量 |

**关键结论**：新模型带来的麻烦，几乎全在「算子库没有的那部分」——新 attention 变体、MoE 专家路由、KV cache、量化 kernel——**这些只能靠 Triton 编译**。这正是 Triton 的价值所在。

### 0.2 一个贯穿全程的心智模型（编译旅程）

```
你写的 Triton 代码（高层语言，跟硬件无关）
      ↓ Triton 编译器【前端，跟硬件无关】
中间表示 IR（TTIR → 更底层 IR）
      ↓ 某硬件后端【翻译官】
硬件能跑的指令
```

**这段话是后面所有内容的总纲**。你以后遇到的每一个「跑不通 / 跑得慢」的 kernel，本质都是这条链路上某一段卡住了——前端语法不支持？后端 lowering 没接？性能优化不到位？先定位是哪一段。

### 0.3 三个必须扭过来的认知（最容易踩的坑）

1. **kernel 不是被「执行」的，是被「编译」的。** 函数体在编译期被解释成 IR；里面的张量变量是**符号值**，不是真实数据。所以**调试靠「看 IR + host 侧对答案」，不能靠 `print` 一个张量的值**。等价于：你在写「生成机器码的模板」。
2. **编程模型是 SPMD 的「块（tile）」模型。** `tl.program_id` ≈ CUDA 的 `blockIdx`；每个 program 一次处理**一整块**向量运算。**没有线程（threadIdx）的概念**——别去找，那是 CUDA 的事。
3. **性能问题最终都归到三件事**：数据搬运多不多、片上缓冲区够不够、流水线排得好不好。

---

## 一、阶段 0：先跑通第一个 kernel（半天 ~ 1 天）

### 1.1 环境准备

在昇腾服务器上，一套能跑 triton-ascend 的环境由四件套组成：**驱动 + CANN toolkit + torch / torch_npu + triton-ascend**。先确认机器到哪一步了：

```bash
# 1. 驱动：能识别到昇腾卡吗？
npu-smi info

# 2. CANN toolkit 是否已装
ls /usr/local/Ascend/ascend-toolkit/

# 3. Python 环境
python --version && which python
```

**装包**（版本号务必对照昇腾官方版本对应表，别照抄）：

```bash
conda create -n triton_ascend python=3.10 -y && conda activate triton_ascend
pip install torch            # 与 torch_npu 版本严格对应
pip install torch_npu        # 与 torch 版本严格对应
pip install triton-ascend    # 或从源码装（阶段三再做）
```

> ⚠️ **最重要的一个坑**：跑任何 NPU 示例前，**必须先 source 环境变量**，否则编译器/驱动找不到：
> ```bash
> source <你的 CANN toolkit 路径>/set_env.sh
> ```

### 1.2 第一个 kernel：向量加法（理解「Triton 代码怎么变成结果」）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                          # 我是第几个「块」
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 我负责哪些元素
    mask = offsets < n_elements                          # 别越界
    x = tl.load(x_ptr + offsets, mask=mask)              # 读一整块
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)        # 写一整块

n = 1024
x = torch.arange(n, dtype=torch.float32)
y = torch.arange(n, dtype=torch.float32)
out = torch.empty_like(x)

grid = (triton.cdiv(n, 256),)
add_kernel[grid](x, y, out, n, BLOCK_SIZE=256, device="npu")  # device 决定跑在哪

print("结果正确:", torch.equal(out, x + y))
```

**预期**：打印 `结果正确: True`。这就打通了 `Python → Triton → 后端 → 硬件` 整条链路，是**第一个里程碑**。

> 💡 **入门技巧**：Triton 是**多后端**的。入门阶段可以先用 `device="cpu"` 跑通（不占卡、报错好排查），确认代码逻辑对了再切到目标设备。同一个 kernel，前端代码几乎不变，只是换个后端。

### 1.3 看懂刚才那几行（零基础重点）

| 原语 | 干嘛的 | 对应你熟悉的事 |
|---|---|---|
| `tl.program_id(axis=0)` | 我是第几块 | 工牌号 |
| `tl.arange(0, BLOCK)` | 我这一块负责哪些下标 | 我管哪片区域 |
| `tl.load(ptr+offsets, mask)` | 从显存读**一整块** | 一趟卡车拉一车货 |
| `tl.store(ptr+offsets, val, mask)` | 写**一整块**回显存 | 卸货 |
| `tl.dot(a, b)` | 矩阵乘法 | 算 W·x |

三个要点：
1. **块思维**：kernel 不是一次处理 1 个元素，而是切成很多块，每块 `BLOCK_SIZE` 个。
2. **mask 的作用**：防止越界（当元素总数不是 `BLOCK_SIZE` 整数倍时）。
3. **最反直觉的一点**：Triton 里**没有显式的 for 循环遍历元素**。所有原语都是「一整块一起处理」，编译器负责把块并行化。这和普通 Python 是两套思维，要刻意切换。

---

## 二、阶段 1：搞懂原理（3 ~ 5 天）

跑通 demo 后，回头搞懂那几行代码在干嘛。

### 2.1 块（tile）思维：性能的起点

```
你不是一个人干完全部 1024 个元素，
而是切成 4 块、每块 256 个，4 个「工人」一人扛一块
```

- `grid = (4,)` → 开 4 个工人
- `pid = tl.program_id(0)` → 「我是第几号工人」
- `offsets = pid*BLOCK + tl.arange(0, BLOCK)` → 我管哪片区域
- `mask = offsets < n` → 边界护栏

**关键直觉**：性能起点在「块切得好不好」——块太大浪费算力，块太小调度开销大。看真实 kernel 时，`BLOCK_SIZE` 怎么取是第一件要留意的事。

### 2.2 一段代码的完整旅程

```
python 调用 add_kernel[grid](...)   ← 你发起
      ↓
Triton 前端解析这段函数           ← 编译器看懂你的意图
      ↓
生成 IR（中间表示）               ← 与硬件无关
      ↓
后端                            ← 翻译官
      ↓
硬件上跑起来                     ← 硬件干活
```

> 这解释了「为什么有的 kernel 在某个硬件上跑不了/慢」——中间某个环节（某个 Triton 语法后端还不支持）卡住了。

### 2.3 进阶：从「会写」到「会看」——IR 分层

Triton 不是把 Python 一步变成机器码，而是经过一串「逐层下降（lowering）」的中间表示。编译后缓存目录（`~/.triton/cache/`）里每个文件就是这条流水线某一层的快照，**越靠上越像你写的 Python，越靠下越像机器**：

```
Python @triton.jit 源码
   ▼ AST 下降
TTIR（.ttir）                ← 最高层，最接近代码，最好读
   ▼ 后端 lowering
后端 IR（.ttadapter 等）       ← 硬件相关，已 bufferize 成 memref 形式
   ▼ 代码生成
设备二进制（.npubin 等）       ← 真正上硬件跑的机器码
```

| 文件 | 是什么 | 人读吗 |
|---|---|---|
| `*.ttir` | 做完 TTIR 层优化后的 IR | ✅ **重点读这个** |
| `*.ttadapter`（昇腾） | 后端 IR：已 bufferize 成 memref | 读，看数据搬运/对齐 |
| `*.json` | 编译选项 + launch 元数据 | 看，了解用了什么 flag |
| `*.mlirbc` / `*.bcmlir` / 设备二进制 | 二进制 | ❌ 不看 |

**MLIR 语法 5 分钟速成**（读 IR 只要记住几条规则）：

1. 所有操作都是 `方言.名字`：`tt.load`、`arith.mulf`、`tensor.extract_slice`、`scf.for`、`linalg.transpose`……
2. **SSA 单赋值**：每个结果绑一个 `%名字`，永不重名。
3. `loc(...)` 是源码位置，告诉你这行 IR 对应 `.py` 第几行——这是读 IR 的**最强大工具**。

**自己 dump 每一层**：

```bash
source <你的 CANN toolkit 路径>/set_env.sh
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 python your_kernel.py
# 去 ~/.triton/cache/<hash>/ 看 *.ttir / *.ttadapter
```

> 一条关键的源码 ↔ IR 对应关系（练熟这一条，看 IR 就通了）：
> ```python
> x = tl.load(ptr + offsets, mask=mask)   # 源码
> ```
> ```mlir
> %p = tt.splat %ptr ...                  # 把指针铺成张量
> %p2 = tt.addptr %p, %offsets ...        # 指针 + 偏移
> %x = tt.load %p2, %mask, %cst ...       # 带 mask 读
> ```

---

## 三、阶段 2：看懂编译器（LLVM / MLIR 背景，半天）

学 Triton 源码到处撞见 LLVM / MLIR / IR，这里一句话讲清它们是什么、和 Triton 什么关系。

### 3.1 LLVM：编译器世界的「万能中转站」

世界上有 N 种语言、M 种硬件，老式做法要写 N×M 个编译器。LLVM 的做法是：

```
语言前端（Clang/Rust/Swift）→ LLVM IR → 硬件后端（x86/ARM/CUDA/NPU）
     只管「语言 → IR」                     只管「IR → 机器码」
```

**IR（中间表示）= 语言和硬件之间的中间语言/中转站**。三段式架构：前端（管语言）→ 中端（通用优化）→ 后端（管硬件）。

### 3.2 MLIR：多层 IR 框架，Triton 的直接地基

LLVM IR 只有一层（已经很底层），而 **MLIR 允许自定义多层 IR**——先在高层（贴近算法语义）优化，再一层层往下降。编译 GPU/NPU 这种「从算法到硬件距离很远」的场景特别需要它。

**Triton 就是用它家的 MLIR 搭起来的**：定义了自己的方言 TTIR（`tt.load`、`tt.dot` 这些指令的「语法」），一路 lower 下去。仓库目录就是干这个的：

- `lib/Dialect/` —— 定义 IR 方言
- `lib/Conversion/` —— 方言之间的层层转换（lowering pass）
- 各后端的适配层目录 —— **改 bug 的主战场**（昇腾是 `third_party/ascend/`）

### 3.3 两条后端路线（triton-ascend 和原版的分叉点）

```
            Python kernel（@triton.jit）
                     ↓
          TTIR → TritonGPU IR      ← 前半段：MLIR，两边一样
                ↓                    ↓
      ┌─ NVIDIA 路线 ─┐      ┌─ 昇腾路线 ──────────┐
      │ LLVM IR → PTX │      │ AscendNPU-IR(毕昇系)│
      │   → cubin     │      │   → 昇腾机器码      │
      └──────────────┘      └────────────────────┘
```

- **NVIDIA**：TritonGPU IR → LLVM IR → PTX（GPU 汇编）
- **昇腾**：后半段换成华为**毕昇编译器**（Bisheng，本身是 LLVM 的一个分支）的 NPU 路线。

> **一句话总结关系**：Triton 的骨架（IR 定义、多层 lowering）来自 LLVM 的 MLIR；NVIDIA 后端直接用 LLVM IR 落地；昇腾后端把最后一段换成自家毕昇系编译器——但追根溯源整个体系都是 LLVM 系的。

---

## 四、阶段 3：上手开发（1 ~ 2 周，边做边学）

到这一步你已经：环境稳定 ✅、能跑通 demo ✅、懂 block/load/store/dot 原理 ✅、能读 IR ✅。接下来进入真正的开发。

### 4.1 从源码编译（改 C++ 的人绕不开）

改 Python 直接生效，**改 C++（MLIR pass）必须重编 C++ 扩展**。以 editable 方式安装源码：

```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
pip install -e .
```

改完 C++ 后的**增量重编**（快，无需重跑 pip install）：

```bash
cd triton-ascend/build/<你的构建目录>
ninja   # 只重编改动的 .o + 重链接 .so
```

> 完整构建需要 clang/lld、cmake、ninja 等工具链，以及（可选）AscendNPU-IR 子模块；首次编译较慢，后续增量重编很快。详见仓库 README。

### 4.2 接需求的标准工作流（背下来）

1. **找到失败的 Triton kernel**（来自 vLLM / SGLang / FlagGems / 业务算子）
2. **最小复现**（删到只剩出错的 op）
3. **dump 每级 IR，定位「在哪一级丢了/错了」**
4. **找到对应的 pass 源码 + 它的 lit 测试，读契约**
5. **修 pass / 加 converter；补 lit 测试 + pytest 用例**
6. **提 PR**（先读 `CONTRIBUTOR.md`）

### 4.3 调 bug 三件套

卡住时的标准动作，按顺序来：

1. **dump IR**：`TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1` 重跑，看 `*.ttir → *.ttadapter → *.npuir` 在哪一层坏了
2. **查 lit 测试**：`unittest/Conversion/**/*.mlir` 是每个 pass 的「产品说明书」
3. **和 torch 对答案**：`torch.testing.assert_close(输出, torch参考)`

### 4.4 四类常见需求方向

| 需求类型 | 主战场 |
|---|---|
| 新算子降级补齐 | 后端的 lowering 适配层，加 Converter |
| 性能优化 | 编译选项、multibuffer、软件流水 |
| 精度问题 | dtype/转换规则 |
| 新硬件特性 | SIMT 路径、NPU IR |

---

## 五、学习路线总览与建议节奏

```
阶段 0  跑通第一个 kernel（半天~1天）      → 块思维 + 三原语
阶段 1  搞懂原理 + 会读 IR（3~5天）        → 编译旅程 + IR 分层
阶段 2  看懂编译器背景（半天）             → LLVM / MLIR / 两条后端路线
阶段 3  上手开发（1~2周，边做边学）        → 源码编译 + 改 bug 工作流
阶段 4  实战复盘（贯穿始终）               → 独立挑一个 issue 走一遍完整定位流程
```

- **节奏**：建议 `1h 概念 + 1.5~2h 动手`，每天连续投入，比周末突击有效。
- **每个 kernel 记一行性能数字**，建立「性能直觉」。
- **第一个 PR 目标**：一个 pass 小修复 + 一个 lit/pytest 测试（小而完整），别一上来就啃大需求。
- **阶段 4 的实战案例**：两个真实 issue 的完整定位过程，各一篇——[`tl.flip` 对齐失败与 UB 溢出](../deep-dives/triton-backend-issue复盘/Triton后端issue复盘-tl-flip案例.md)、[no-op `tl.permute` 导致 UB 溢出](../deep-dives/triton-backend-issue复盘/Triton后端issue复盘-tl-permute案例.md)。

## 六、资源清单

| 资源 | 用途 |
|---|---|
| [Triton 官方文档](https://triton-lang.org) | 语言参考（`tl.*` API） |
| [triton-ascend 仓库](https://github.com/triton-lang/triton-ascend) | 昇腾后端说明、安装、适配层源码 |
| 仓库 `python/tutorials/` | 官方教程（vector-add / softmax / matmul / dropout / layernorm / attention） |
| 仓库后端 `tutorials/` | 后端适配版教程（和上游版 diff 一下，差异处就是后端特性） |
| 仓库 `unittest/pytest_ut/` | 端到端用例（每个教程对应一个可跑的 test） |
| 仓库 `unittest/Conversion/**/*.mlir` | pass 契约（lit 测试，读它理解每个 pass） |
| vLLM / vllm-ascend 源码 | 真实 kernel 素材（拿真实算子对照概念读，比看教程快 10 倍） |
| 昇腾官方文档 | CANN / torch_npu 版本对应表 |
| [实战复盘：`tl.flip` 对齐失败与 UB 溢出](../deep-dives/triton-backend-issue复盘/Triton后端issue复盘-tl-flip案例.md) | 阶段 4 配套：bit-reversal → 原生 flip 的定位过程 |
| [实战复盘：no-op `tl.permute` 导致 UB 溢出](../deep-dives/triton-backend-issue复盘/Triton后端issue复盘-tl-permute案例.md) | 阶段 4 配套：no-op 转置被降成真转置的定位过程 |

## 七、常见坑 FAQ

| 症状 | 原因 / 解决 |
|---|---|
| `import triton` 报找不到编译器/驱动 | 没 source CANN 的 `set_env.sh` |
| 改了 C++ 代码重跑 kernel 没变化 | 没重编 C++ 扩展，或装到了别的环境 |
| 反复编译但结果一样 | 命中缓存，`rm -rf ~/.triton/cache` 或 `TRITON_ALWAYS_COMPILE=1` |
| 想精确看单个 pass 的前后 IR | 用 `triton-opt` 手动跑 pass + `--mlir-print-ir-before/after` |
| 数值对了但怀疑某优化没生效 | 数值对 ≠ 优化命中，要加 IR 级断言（白盒） |
