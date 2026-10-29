# Triton 后端 issue 实战复盘：`tl.flip` 对齐失败与 UB 溢出

> **阶段 4「实战复盘」配套材料**。定位方法统一为：**复现 → 抓多级 IR → 追 lowering 链 → 定位到某个 pass / 函数 → 改**。
> 以开源 triton-ascend（昇腾后端）为例，方法与具体硬件无关，任何 Triton 后端都适用。

## 现象

`compile_mode: "simd"` 下，对 `64×64` 张量执行 `tl.flip(x, dim=1)`，报 `cannot align N axis` 和 `ub overflow`。

## 定位过程

1. **复现**：写最小脚本，确定「去掉 `tl.flip` 就过、`BLOCK_SIZE` 从 64 降到 32 也过」→ 锁定是 `tl.flip`。
2. **抓 IR**：dump 出 `ttir / 后端 IR / npuir` 多级 IR。
3. **锁阶段**：`grep expand_shape` 发现 `memref.expand_shape` 只出现在最底层 IR、不在 linalg → 生成点在底层 bufferization。
4. **追链**：`tt.reshape`（ttir）→ `tensor.reshape`（linalg）→ `memref.expand_shape`（npuir）→ 对齐 pass 报错。
5. **定位代码**：在报错信息附近定位到「对齐传播」pass 无法穿过这种 reshape。

## 根因（一句话）

`tl.flip` 没有原生 reverse 算子，前端把它展开成 **bit-reversal**——把长度 64 的轴 reshape 成 6 个 2 相乘（`2×2×2×2×2×2`），这个 reshape 在下游对齐 pass 里无法传播对齐信息，于是报错。

## 方案选择（最值得学的一步）

| 落点 | 归属 | 能否直接提 PR |
|---|---|---|
| 改底层对齐 pass | 独立的底层 IR 仓库 | ❌ 只能提 issue |
| 改通用前端里 flip 的实现 | 上游 Triton 通用代码 | ❌ 通用文件里塞后端私活，上游不接 |
| **在后端适配层接线原生 flip** | **后端自己的适配层** | ✅ **落这里** |

## 为什么选「接线」

仓库里其实**早就埋好了一个原生 flip 算子，只是一直没人用**（孤儿 op）。整条原生链每一段都真实存在，唯一缺的是「没有代码去制造它」。所以解法是：**写一个 pattern（图案匹配器），把 bit-reversal 那套固定图案认出来，整段重写成原生 flip**，绕过崩人的 reshape。

**一句话记住这个方案**：*不造新轮子，把已经铺好的高速公路的入口匝道接上。*

## 结果

末维 flip 从 bit-reversal 变成单条原生向量翻转，典型尺寸下从几百 µs 降到十几 µs（约 20× 加速）；更大的尺寸下 bit-reversal 直接编译失败，而原生 flip 能过。

## 能带走的方法论

1. **改 bug 前先想「落点归属」**——底层依赖、上游通用代码、还是自己的适配层？只有最后一个能直接提 PR。
2. **孤儿 op 是最省事的修法**——如果仓库里已有原生算子、只是没接线，就别去改底层。
3. **pattern 的规矩是「宁可错过，不能改错」**——每一步严格校验，任何一步对不上就原样返回，绝不乱改别的 op。
