# Triton 插件化是什么：后端与 pass 即插即用

> **定位**：讲清 Triton 的「插件化」到底要做什么——把「加一个新硬件后端、加一个新 pass」从「改上游源码、整个 Triton 重编译」改成「独立编成一个 .so 插件、运行时动态加载、`pip install` 即用」。

## 〇、一句话

**Triton 插件化 = 把「加一个新硬件后端、加一个新 pass」从「改上游源码、整个 Triton 重编译」改成「独立编成一个 .so 插件、运行时动态加载、`pip install` 即用」。**

## 一、两个名词别混淆

| 名词 | 指什么 |
|---|---|
| **上游插件机制（Triton Plugin）** | Triton 主线自带的扩展点：`.so` 动态库、运行时 `dlopen` 加载、可注册 MLIR pass / dialect / 后端。这是**上游已做好的基础设施** |
| **插件化（某后端的方案）** | 把某个后端**接进这套机制**。不是造新机制，而是搭桥 + 把后端代码组织方式改成插件形态 |

## 二、它解决的核心痛点（为什么要有它）

上游 Triton 的现状（Triton 3.6 时代）：每个新硬件后端都要在**上游源码树里改 `compiler.py` 等文件**、插入自家 pass、改流水线，然后**整个 Triton 重新编译**才生效。官方文档原话：

> "Currently when we want to insert passes… it is required for the compiler.py file itself to be modified and all of Triton to be recompiled."

痛点三层：

1. **编译耦合**——改一个后端，动整个编译器，构建一次几十分钟；
2. **发布耦合**——后端代码混在上游里，上游发版节奏卡住自家特性上线；
3. **升级耦合**——每次跟上游主线都冲突，维护 fork 成本高。

## 三、机制怎么工作（3 个关键点）

1. **形态**：扩展编译成**共享库（.so）**、做成**独立 Python wheel**，`pip install` 即可，靠 `__init__.py` 里一行 `libtriton.extend_with(...)` 注册。官方原话："Extensions are loaded by Triton by their `__init__.py` file"。运行时动态加载，不碰上游源码。

2. **载体**：上游配套出了 `triton-lang/triton-ext` 仓库（原名 triton-plugins）——一个**树外扩展集合**，目录按类型分：

   | 目录 | 装什么 | 例子 |
   |---|---|---|
   | `backend/` | 新设备后端 | Apple 的树外 GPU 后端（TTGIR 直接降到 Metal Shader Language） |
   | `dialect/` | 新 MLIR dialect | — |
   | `pass/` | 新 MLIR pass | intensity 等 |
   | `extensions/` | 打包一整套 dialect+pass+语言绑定 | utlx 等 |

3. **三个前置/开关**（迁移成本所在）：
   - Triton 编译时 `TRITON_EXT_ENABLED=ON`（3.7 起默认开）；
   - **LLVM 必须 asserts 构建**（`LLVM_ENABLE_ABI_BREAKING_CHECKS=1`），否则 `.so` 里 `ilist_iterator` 的 mangled 类型对不上，`dlopen` 时报 undefined symbol（链接期不报，运行期才爆）；
   - 插件入口实现 `tritonGetPluginInfo` API，用 `TRITON_PLUGIN_PATHS` 环境变量加载。

> 上游还有一层 **pass 插件**：不只能加后端，还能通过 `inspect_stages_hook` 往现有编译流水线（TTIR/TTGIR 各阶段）里**插入自定义 pass**——改流水线不用再改 `compiler.py` 和重编整个 Triton。

## 四、插件化之前 vs 之后

| 现状 | 插件化之后 |
|---|---|
| 后端代码内嵌在某个 fork 仓，`compiler.py`/`driver.py` 跟着走 | 后端是独立的 `.so` + wheel，`import` 即注册 |
| 后端特有优化 pass 写死在自家流水线里 | pass 以插件注入，可被上游流水线钩子插入，不覆盖上游管线 |
| 每次升 Triton 版本都要重新适配、重新编译 | 对着**已安装的 Triton wheel** 编译，升版本只需重链接插件 |
| 与上游共享同一套源码，merge 冲突 | 树外独立迭代，随装随用 |

## 五、一句话总结

> 插件化 = 把 Triton 的「新后端 + 新 pass」从「改上游源码、重编整个编译器」变成「独立 .so 插件、运行时动态加载」。后端接入上游这套基础设施后，能以独立 wheel 形式迭代、不 fork 主线、随 Triton 版本升级即插即用。

## 附：一手材料来源

- 上游插件机制文档：triton-lang/triton `examples/plugins/README.md`
- 树外扩展集：github.com/triton-lang/triton-ext（原名 triton-plugins）
- Apple 树外后端示例：triton-ext `backend/AppleGPU`
