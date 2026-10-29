# 如何阅读 Triton IR：TTIR 到 TTAdapter 逐层对照

> **定位**：学会看 Triton 编译产物。跑完 kernel 后 `~/.triton/cache` 里生成了 `*.ttir`、`*.ttadapter`、`*.npubin` 一堆文件，本文讲清每个是啥、怎么读。

## 〇、一句话

**Triton 不是把 Python 一步变成机器码，而是经过一串「逐层下降（lowering）」的中间表示（IR）**。cache 里每个文件就是这条流水线某一层的快照：**越靠上越像你写的 Python（TTIR），越靠下越像机器（npubin）**。学看 IR 就是学会在每一层认出「我源码里的那个操作变成了什么」。

## 一、编译流水线全景图

```
Python @triton.jit 源码
   │  AST 下降
   ▼
TTIR（.source / .ttir）          ← 最高层，最接近代码，最好读
   │  后端 lowering（如 triton-ascend 的 TTAdapter）
   ▼
TTAdapter IR（.ttadapter）        ← 后端特有，已 bufferize 成 memref 形式
   │  转 LLVM 方言 / 后端 IR → 机器码
   ▼
.npubin（设备二进制）              ← 真正上硬件跑的机器码（类比 CUDA 的 .cubin）
```

> 关键：**越靠上越像 Python，越靠下越像机器。** 读 IR 的方法 = 在每一层把 IR 行映射回源码行。

## 二、cache 目录里每个文件是什么

Triton 用「源码 + 编译选项」的哈希做目录名，同一段代码同一参数会命中缓存。目录内每个后缀含义：

| 文件 | 是什么 | 人读吗 |
|---|---|---|
| `*.source` | AST 下降后、**还没做优化**的 TTIR | 读，最贴近源码 |
| `*.ttir` | 做完 TTIR 层优化后的 TTIR | ✅ **重点读这个** |
| `*.ttadapter` | 后端 IR：已 bufferize 成 memref、带硬件属性 | 读，看数据搬运/对齐 |
| `*.mlirbc` | MLIR **字节码**（二进制序列化） | ❌ 不看 |
| `*.bcmlir` | LLVM bitcode 二进制 | ❌ 不看 |
| `*.npubin` | 最终设备二进制（上硬件跑的） | ❌ 二进制 |
| `*.json` | 编译选项 + launch 元数据 | 看，了解用了什么 flag |
| `launcher_*.so` | CPU 侧启动器（把参数喂给核函数） | ❌ |

`.json` 里几个值得注意的字段，直接说明这次编译的环境：

```json
"backend": "npu", "arch": "Ascend910B3",
"num_warps": 32, "num_stages": 2,
"sanitize_overflow": true,   ← 编译器往 IR 里插溢出检查（= ttir 里那堆 tt.assert）
"debug": true,                ← 因为源码里写了 print()，所以 ttir 里才有 tt.print
"kernel_name": "my_kernel"
```

## 三、MLIR 语法速成（5 分钟）

MLIR 文本由「方言 + 操作」组成，看懂几条规则就能读：

**1. 所有操作都是 `方言.名字`**

| 前缀 | 方言 | 常见操作 |
|---|---|---|
| `tt.` | Triton | `tt.load` `tt.store` `tt.make_range` `tt.addptr` `tt.splat` `tt.expand_dims` `tt.broadcast` `tt.get_program_id` |
| `arith.` | 算术 | `arith.extf`(转类型) `arith.mulf/subf/addf`(乘/减/加) `arith.cmpi`(比较) |
| `tensor.` | 张量 | `tensor.extract_slice`(切片) `tensor.insert_slice`(插回) |
| `scf.` | 控制流 | `scf.for`(循环) |
| `linalg.`/`memref.`/`bufferization.` | 更底层张量/缓冲区 | 见 §六 |
| `func.` | 函数 | `func.func` |

**2. SSA 单赋值**：每个结果绑一个 `%名字`，永不重名。所以会看到 `%qk_offsets_9`、`%qk_offsets_29`——同一个逻辑变量被改写 29 次，每次都是新名字。

**3. 类型系统**：`i32`/`i64`/`f32` 标量，`bf16`/`f16` 半精度，`tensor<4x16xf32>` 张量，`!tt.ptr<bf16>` 指针。

**4. 结构层级**：
```
module {                         顶层模块，带目标属性
  tt.func @名字(参数...) {        函数，参数是 %qk_ptr 等
    ...一串操作...
    scf.for %i = ... {            循环 = 一个 region
      ...循环体...
    }
    tt.return
  }
}
```

**5. `loc(...)` 是源码位置**（source location），告诉你这行 IR 对应源码第几行——这是读 IR 的**最强大工具**。

## 四、拿一个 RoPE kernel 逐段对照源码（重点）

对照源码与生成的 `.ttir`，看每个 Python 操作落成了什么 IR。

### ① 函数签名 → 参数变裸指针 + stride

Python 里传的张量 + 标量，到 IR 里全部变成**指针 + stride**：

```mlir
tt.func @kernel(
  %qk_ptr: !tt.ptr<bf16>, %qk_row_stride: i32,
  %cos_ptr: !tt.ptr<bf16>, %cos_row_stride: i32, ...
```

`tl.constexpr` 参数直接**编译期内联**，函数参数里看不到——它们已变成下文常量（`dense<16>`、`end = 16`）。

### ② `tl.arange` → `tt.make_range`

```python
cos_sin_offsets = tl.arange(0, pad_rope_dim // 2)   # 0..7
```
```mlir
%cos_sin_offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
```

### ③ 二维 offset：`[:, None]` → `expand_dims` + `broadcast`

```python
qk_offsets = tl.arange(0, n)[:, None] * hd + tl.arange(0, d)[None, :]
```
```mlir
%qk_offsets_10 = tt.expand_dims %qk_offsets_9 {axis=1}     // [:, None] → 4x1
%qk_offsets_16 = arith.muli %qk_offsets_10, %qk_offsets_6  // * hd(=16)
%qk_offsets_18 = tt.expand_dims %qk_offsets_17 {axis=0}    // [None, :] → 1x16
%qk_offsets_19/20 = tt.broadcast ...                        // 广播成 4x16
%qk_offsets_29 = arith.addi %qk_offsets_19, %qk_offsets_20 // 相加
```

### ④ 那堆 `tt.assert` 是你没写的——溢出检查

因为 `sanitize_overflow: true`，编译器自动插入的 int32 溢出防护：

```mlir
tt.assert %qk_offsets_15, "int32 overflow detected for operation mul"
```
读 IR 时**先跳过这些**，它们只是护栏，不是业务逻辑。

### ⑤ `tl.load` → `splat 指针` + `addptr` + `load`

```python
qk_tile = tl.load(qk_start_ptr + qk_offsets, mask=qk_mask, other=0).to(tl.float32)
```
```mlir
%qk_tile   = tt.splat %qk_start_ptr_36 : !tt.ptr<bf16> -> tensor<4x16x!tt.ptr<bf16>> // 复制指针
%qk_tile_47 = tt.addptr %qk_tile, %qk_offsets_29 ...                                 // 指针+偏移
%qk_tile_48 = tt.load %qk_tile_47, %qk_mask_34, %cst ...                             // 带 mask 读
%qk_tile_49 = arith.extf %qk_tile_48 : tensor<4x16xbf16> to tensor<4x16xf32>         // .to(f32)
```
- `splat` = 把标量/指针铺成张量
- `addptr` = 指针加偏移
- `load` 第三操作数 `%cst`（全 0）= `other=0`
- `arith.extf` = 类型扩展（extend float），bf16→f32

### ⑥ `extract_slice` → `tensor.extract_slice`

```mlir
%qk_tile_1 = tensor.extract_slice %qk_tile_49[0, 0] [4, 8] [1, 1] : tensor<4x16xf32> to tensor<4x8xf32>
%qk_tile_2 = tensor.extract_slice %qk_tile_49[0, 8] [4, 8] [1, 1] : tensor<4x16xf32> to tensor<4x8xf32>
```
语法 `extract_slice %源[起点][大小][步长]`。`[0,0][4,8][1,1]` = 从第0行第0列取 4×8 块，步长 1。

### ⑦ RoPE 核心数学 → `arith.mulf/subf/addf`

```python
new_qk_tile_1 = qk_tile_1 * cos_row - qk_tile_2 * sin_row
```
```mlir
%new_qk_tile_1_51 = arith.mulf %qk_tile_1, %new_qk_tile_1_50   // qk_tile_1 * cos_row
%new_qk_tile_1_54 = arith.mulf %qk_tile_2, %new_qk_tile_1_53   // qk_tile_2 * sin_row
%new_qk_tile_1_55 = arith.subf %new_qk_tile_1_51, %new_qk_tile_1_54
```
`cos_row`（`tensor<8xf32>`）先 `expand_dims` 成 `1x8` 再 `broadcast` 成 `4x8`，才能和 `4x8` 的 `qk_tile_1` 相乘。

### ⑧ `insert_slice` → `tensor.insert_slice`

```mlir
%qk_tile_58 = tensor.insert_slice %new_qk_tile_1_55 into %qk_tile_49[0, 0] [4, 8] [1, 1]
```
把左半边插回原张量，再插右半边，最后 `arith.truncf`（f32→bf16 截断）+ `tt.store` 写回。

### ⑨ `tl.range` 循环 → `scf.for`

```python
for row_idx in tl.range(pid, num_tokens, row_block_size):
```
```mlir
scf.for %row_idx = %pid_7 to %c1_i64 step %0 : i64 { ... }
```

### ⑩ `print()` → `tt.print`

```mlir
tt.print " qk_tile=: " {hex=false} : %qk_tile_49 : tensor<4x16xf32>
```
源码里的每个 `print(...)`，对应一个 `tt.print`。

## 五、顺带：`tt.get_program_id` / `tt.get_num_programs`

```python
pid = tl.program_id(0)         nblocks = tl.num_programs(0)
```
```mlir
%pid = tt.get_program_id x : i32
%nblocks = tt.get_num_programs x : i32
```
之后 `arith.extsi`（符号扩展 i32→i64）是为了参与 64 位指针运算。

## 六、`.ttadapter` 是什么（后端独有，别处看不到）

这是后端和上游 Triton 最大的分叉点。上游 NVIDIA 用 **TTGIR**（带 layout 编码）；部分后端（如昇腾）改用 **TTAdapter**，先做 **bufferize（缓冲化）**，把 `tt.load/tt.store` 的指针张量形式转成 `memref` + `memref.alloc/copy` + `bufferization.to_tensor`：

```mlir
%qk_tile_12 = memref.alloc() : memref<4x16xbf16>          // 分配片上缓冲区
memref.copy %qk_tile, %qk_tile_12 : ...                    // 全局内存拷进片上
%qk_tile_13 = bufferization.to_tensor %qk_tile_12 ...      // 转回张量做计算
%new_qk_tile_1_16 = linalg.broadcast ins(%cos_row_6 : tensor<8xf32>) ... // 广播
bufferization.materialize_in_destination %1 in writable %reinterpret_cast  // 写回
```

函数属性还带硬件特有信息：

```mlir
attributes {global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"}
```

这一层已能看出**硬件资源**：`memref.alloc` 分配片上缓冲区（如 UB），`memref.copy` 是显式搬运。读这一层，目的是看**数据从全局内存 → 片上的搬运、对齐、有无多余 copy**——性能问题和对齐 bug 都发生在这一层附近。

## 七、实操：自己 dump 每一层

```bash
# 1) 编译时把每层 IR 打到屏幕
source <你的 CANN toolkit 路径>/set_env.sh
TRITON_PRINT_IR=1 python your_kernel.py

# 2) 强制每次重新编译并落盘（避免命中缓存）
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 python your_kernel.py

# 3) 对 .mlir 手动跑 pass，加 --mlir-print-ir-after-all 看每个 pass 前后的差异
triton-opt repro.mlir --some-pass --mlir-print-ir-after-all

# 4) 看最终设备二进制/汇编：.npubin 已是二进制，可看 .json 里的编译选项
```

## 八、学习建议

1. **先只看 `.ttir`**（它是「会动的 Python」），练到不用看 `loc()` 就能说出每行 IR 对应源码哪一行。
2. **再进 `.ttadapter`** 看数据搬运、对齐。
3. **`.ttir` → `.ttadapter` 之间的变换**，正是后端优化 pass 的战场。

> **该掌握的**：IR 分层下降概念、cache 目录各后缀含义、`tt./arith./tensor./scf.` 几个常用方言、`extf/slice/load/store` 与 `tl.` API 的对应关系、用 `loc()` 回溯源码。
> **不用掌握的**：`.mlirbc`/`.bcmlir`/`.npubin` 的二进制格式、手写完整 TTIR、读后端 IR 源码。能对照源码读懂 `ttir` + `ttadapter` 就够应付看 IR 的任务了。
