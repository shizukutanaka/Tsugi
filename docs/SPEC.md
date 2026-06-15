# Tsugi Specification

> 仕様が実装を駆動する（C11）。本書の各opは src/tsugi/ir/ の dialect 定義と1:1対応。
> v0.1 対象: NVIDIA（PTX）/ AMD（AMDGCN）。

---

## 1. Tsugi Tile DSL（Layer C・ユーザーが書く層）

### 1.1 設計方針
Triton型のブロックレベルプログラミング。スレッド個別でなく**タイル（ブロック）単位**で記述。コンパイラがスレッドへ展開・autotune。

### 1.2 最小文法（Python埋め込み）

```python
import tsugi
from tsugi import tile

@tsugi.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tsugi.constexpr,
                  BLOCK_N: tsugi.constexpr,
                  BLOCK_K: tsugi.constexpr):
    pid_m = tsugi.program_id(0)
    pid_n = tsugi.program_id(1)

    acc = tile.zeros((BLOCK_M, BLOCK_N), dtype=tsugi.float32)
    for k in range(0, K, BLOCK_K):
        a = tile.load(a_ptr, (pid_m, k), (BLOCK_M, BLOCK_K))
        b = tile.load(b_ptr, (k, pid_n), (BLOCK_K, BLOCK_N))
        acc += tile.dot(a, b)          # → 行列コアへlowering（R1の核）

    tile.store(c_ptr, (pid_m, pid_n), acc.to(tsugi.float16))
```

### 1.3 グリッド起動

```python
grid = (triton_cdiv(M, BLOCK_M), triton_cdiv(N, BLOCK_N))
matmul_kernel[grid](a, b, c, M, N, K,
                    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32)
```

### 1.4 型
| Tsugi型 | 説明 |
|--------|------|
| `tsugi.float16` / `bfloat16` | 半精度（行列コア対象） |
| `tsugi.float32` | 単精度（accumulator標準） |
| `tsugi.int32` / `int8` | 整数 |
| `tsugi.constexpr` | コンパイル時定数（autotune対象） |

### 1.5 v0.1 対応組み込み関数
| 関数 | 意味 |
|------|------|
| `tsugi.program_id(axis)` | ブロックID |
| `tile.load(ptr, offset, shape)` | グローバル→タイルロード |
| `tile.store(ptr, offset, value)` | タイル→グローバルストア |
| `tile.dot(a, b)` | 行列積（行列コア経由） |
| `tile.zeros(shape, dtype)` | ゼロ初期化 |
| `tile.reduce(x, axis, op)` | 軸reduce（sum/max） |
| `tile.exp/sqrt/rsqrt` | 要素関数 |
| `x.to(dtype)` | 型変換 |

---

## 2. IR階層（Layer A・MLIR dialect）

### 2.1 tsugi.tile dialect（高位・ベンダー非依存）

```mlir
// matmul の高位表現
%acc = tsugi.tile.zeros : tensor<128x128xf32>
%a = tsugi.tile.load %a_ptr[%pid_m, %k] : tensor<128x32xf16>
%b = tsugi.tile.load %b_ptr[%k, %pid_n] : tensor<32x128xf16>
%d = tsugi.tile.dot %a, %b, %acc : tensor<128x128xf32>
tsugi.tile.store %c_ptr[%pid_m, %pid_n], %d : tensor<128x128xf16>
```

主要op:
| op | 説明 | lowering先 |
|----|------|-----------|
| `tsugi.tile.dot` | 行列積 | → gpu.matrix → 各社intrinsic |
| `tsugi.tile.load` | タイルロード | → gpu.shared_load + coalesce |
| `tsugi.tile.store` | タイルストア | → gpu.global_store |
| `tsugi.tile.reduce` | reduce | → gpu.warp_reduce |
| `tsugi.tile.elementwise` | 要素演算 | → arith/math |

### 2.2 tsugi.gpu dialect（中位・GPU実行モデル・まだベンダー非依存）

```mlir
tsugi.gpu.block_dims = [256]
%lane = tsugi.gpu.lane_id
%smem = tsugi.gpu.alloc_shared : memref<128x32xf16, #shared>
tsugi.gpu.barrier
%m = tsugi.gpu.matrix_mma %fa, %fb, %fc { shape = [16,16,16] }
```

主要概念: block / warp(=lane group) / shared memory / barrier / matrix_mma（抽象行列命令）。

### 2.3 vendor lowering（ここで初めて分岐）

| tsugi.gpu | NVIDIA (NVVM) | AMD (ROCDL) |
|-----------|---------------|-------------|
| `gpu.matrix_mma` | `nvvm.wmma.mma.sync` | `rocdl.mfma.f32.16x16x16f16` (CDNA) / `rocdl.wmma` (RDNA3+) |
| `gpu.barrier` | `nvvm.barrier0` | `rocdl.s.barrier` |
| `gpu.alloc_shared` | addrspace(3) | addrspace(3) (LDS) |
| `gpu.lane_id` | `nvvm.read.ptx.sreg.laneid` | `rocdl.workitem.id.x` |

---

## 3. lowering パイプライン（パス順序）

```
tsugi.tile
  │ 1. tile-to-gpu       タイル→スレッド展開・shared mem staging・software pipelining
  │ 2. autotune-annotate タイルサイズ/レイアウト/ステージ数を候補付与
tsugi.gpu
  │ 3. vendor-split      ターゲット指定で分岐
  ├─ NVIDIA:
  │    4n. gpu-to-nvvm   matrix_mma→wmma.mma等
  │    5n. nvvm-to-llvm  LLVM IR (nvptx target)
  │    6n. llvm-to-ptx   NVPTX backend → PTX文字列
  └─ AMD:
       4a. gpu-to-rocdl  matrix_mma→mfma等
       5a. rocdl-to-llvm LLVM IR (amdgcn target)
       6a. llvm-to-gcn   AMDGPU backend → AMDGCN object
```

各パスは `mlir-opt` 互換のパスとして src/tsugi/backend/ に実装。

---

## 4. autotuning 仕様

探索パラメータ（各ベンダー独立）:
| パラメータ | 候補例 | NVIDIA | AMD |
|-----------|-------|--------|-----|
| BLOCK_M/N/K | 64/128/256, 32/64 | warp=32前提 | wavefront=64前提 |
| num_stages | 2,3,4,5 | pipelining深度 | 同 |
| num_warps | 4,8 | 4 warps=128 lanes | 1 wavefront=64 |
| matrix shape | 16x16x16等 | WMMA形状 | MFMA形状 |

探索戦略: グリッドサーチ（v0.1）→ ベイズ最適化（将来）。結果は `.tsugi_cache/` にキャッシュ。

---

## 5. TorchInductor バックエンド（Layer C・楔の本体）

```python
import torch
import tsugi_torch  # backend登録

model = MyTransformer()
compiled = torch.compile(model, backend="tsugi")
out = compiled(x)   # Tsugi経由でカーネル生成・両ベンダー対応
```

責務:
1. TorchInductor の lowering IR を受け取る
2. `tsugi.tile` IR へ変換
3. §3 パイプラインでコンパイル
4. 標準op（GEMM等）でベンダーライブラリが勝つ場合は escape-hatch

---

## 6. Runtime API（Layer B・最小・IREE HAL候補）

```cpp
// 自作する場合の最小インターフェース（or IREE HALへbridge）
tsugi_device_t   tsugi_device_get(tsugi_vendor_t v);  // NVIDIA|AMD
tsugi_buffer_t   tsugi_malloc(tsugi_device_t, size_t);
void             tsugi_memcpy(tsugi_buffer_t, void*, size_t, tsugi_dir_t);
tsugi_module_t   tsugi_module_load(tsugi_device_t, const char* binary);
void             tsugi_launch(tsugi_module_t, const char* kernel,
                              tsugi_grid_t, void** args);
```

実装は薄い dispatch:
- NVIDIA: CUDA Driver API（`cuModuleLoadData` / `cuLaunchKernel`）
- AMD: HIP（`hipModuleLoad` / `hipModuleLaunchKernel`）

---

## 7. 対応op一覧（v0.1 DoD）

| カーネル系統 | op | Phase |
|-------------|----|----|
| GEMM | `tile.dot` (FP16/BF16) | 1-2 |
| Fused Attention | dot + softmax + dot | 3 |
| LayerNorm/RMSNorm | reduce + rsqrt + elementwise | 3 |
| Elementwise融合 | add/mul/exp/gelu fusion | 3 |

---

## 8. バージョニング・互換性

- SemVer。v0.x = POC段階（公開API不安定）。
- DSL文法の破壊的変更は MINOR（v0.x間）→ v1.0でAPI凍結。
- IR dialect は内部実装（公開APIでない）→ 自由に変更可。

---

## 9. 非対応・将来（スコープ外明示）

- Intel XMX / SPIR-V backend（v1.0以降）
- Apple Metal（v1.0以降）
- 訓練特化（backward最適化・collective）
- グラフィックス相互運用
- 動的形状の完全対応（v0.1は静的形状中心）
