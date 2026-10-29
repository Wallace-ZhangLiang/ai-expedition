# Triton 后端 issue 实战复盘：no-op `tl.permute` 被降成真转置导致 UB 溢出

> **阶段 4「实战复盘」配套材料**。定位方法统一为：**复现 → 抓多级 IR → 追 lowering 链 → 定位到某个 pass / 函数 → 改**。
> 以开源 triton-ascend（昇腾后端）为例，方法与具体硬件无关，任何 Triton 后端都适用。

## 现象

`ub overflow`（约 2.17× 超量）。

## 根因（一句话）

`tl.permute(x, (1, 0))` 把 `(1, 4096)` 转成 `(4096, 1)`，两个形状的行主序**完全相同**（等价于一次 reshape，数据一字不动）。但它在后端被**无条件降成真正的 `linalg.transpose`**，底层又把这个转置当矩阵转置用分块布局做，缓冲区放大 8 倍，片上内存爆掉。

## 决定性的实验（定位 no-op 的经典手法）

```python
# 同一个 kernel，只改一行：
tl.permute(floored, (1, 0))      # tt.trans → ub overflow ❌
tl.reshape(floored, (4096, 1))   # tt.reshape → 编译通过 ✅
```

两者产出的 `(4096,1)` 数据逐位相同，唯一区别是 `tt.trans` vs `tt.reshape`。这证明：**罪魁就是 no-op 转置被当成了真转置**。

## 修复方向

在转置的统一出口里加一个 no-op 判断——**若 permutation 只挪动 size-1 的维、非单例维的相对顺序不变，就是 no-op 转置，直接降成 reshape**，而不是 `linalg.transpose`。判定是标准的 O(rank)：

```cpp
// 非单例维在 perm 下的目标位置必须严格递增；有任何一个往回退就是真转置。
static bool isNoOpPermutation(shape, perm) {
  int64_t lastPos = -1;
  for (int i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 1) continue;           // 单例维可自由移动
    if (perm[i] <= lastPos) return false;  // 非单例维往回退 → 真转置
    lastPos = perm[i];
  }
  return true;
}
```

## 能带走的方法论

1. **语义上「等价的形状变换」要优先走更便宜的路径**——no-op 转置本质是 reshape，降成 reshape 能避免不必要的分块转置开销。
2. **定位 no-op 问题，用「对照实验」最快**：把嫌疑 op 换成语义等价、但走不同 lowering 的 op（这里是 `tl.permute` vs `tl.reshape`），看行为差异。
3. **修复落点选在后端适配层**，不碰底层依赖、不碰上游通用代码，才能直接提 PR。
