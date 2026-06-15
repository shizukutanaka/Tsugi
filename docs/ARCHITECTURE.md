# Tsugi Architecture

> 継ぎ — GPU ベンダーを接合する移植検証層
> 統一GPU計算レイヤー。NVIDIA / AMD 横断（v0.1）。Intel / Apple は v1.0 以降。

本書は Tsugi の全設計判断の基準。実装はこの構造に従う（仕様が実装を駆動する・C11）。

---

## 1. 設計原理（なぜこの形か）

CUDA の堀は **言語でなくライブラリ（cuDNN/cuBLAS）と PyTorch 統合**。ゆえに Tsugi は CUDA API のクローンを作らない。代わりに **コンパイラ/IR 層に集約**し、**`torch.compile` のバックエンド**として刺さる。

3社の唯一の共通基盤は **LLVM/MLIR**（NVPTX・AMDGPU が upstream バックエンド）。ランタイム層に共通APIは存在しない（CUDA driver / ROCr / Level Zero は別物）。ゆえに統一は**コンパイラ層が最も現実的**で、ランタイムは薄い dispatch に留める（または IREE HAL を再利用）。

設計哲学: Carmack（性能）× Martin（単一責任）× Pike（簡潔）。

---

## 2. 3層アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│  Layer C: Frontend / Kernel DSL                             │
│  - Tsugi Tile DSL（Triton型・Pythonバインディング）          │
│  - torch.compile / TorchInductor バックエンドアダプタ        │
│  ※ CUDAバイナリ変換は一切しない（法的回避・ADR-002）          │
└────────────────────────┬────────────────────────────────────┘
                         │ 生成: Tsugi Tile IR (MLIR dialect)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer A: Unified Compiler / IR（中核）                      │
│                                                             │
│   tsugi.tile dialect  （高位: タイル演算・matmul/load/store）│
│        │ lowering passes                                    │
│        ▼                                                    │
│   tsugi.gpu dialect   （中位: スレッド/ワープ/共有メモリ）   │
│        │ vendor split                                       │
│        ├──────────────┬──────────────────────┐             │
│        ▼              ▼                       ▼             │
│   NVVM/NVPTX      ROCDL/AMDGPU          SPIR-V (fallback)   │
│   (LLVM IR)        (LLVM IR)            (Intel/Vulkan・将来) │
└────────┬──────────────┬──────────────────────┬─────────────┘
         │              │                      │
         ▼              ▼                      ▼
       PTX            AMDGCN                SPIR-V
    (ptxas→SASS)   (lld→GCN bin)        (Level Zero/Vulkan)
         │              │
         ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer B: Unified Runtime（薄いdispatch・IREE HAL再利用候補）│
│  - device管理 / memory / kernel launch                      │
│  - NVIDIA: CUDA Driver API (cuLaunchKernel)                  │
│  - AMD:    HIP/ROCr runtime                                  │
│  - escape-hatch: cuBLAS/rocBLAS委譲（codegenが負ける箇所）   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 各層の責務（単一責任・C8）

### Layer A — Unified Compiler / IR（最重要・差別化の核）
- **tsugi.tile dialect**: ユーザーが書く高位タイル演算。`tile.matmul` `tile.load` `tile.store` `tile.reduce` `tile.dot`。形状とデータ型のみ。ベンダー非依存。
- **tsugi.gpu dialect**: 中位。block/warp/lane・shared memory・barrier。まだベンダー非依存だがGPU実行モデルを表現。
- **vendor lowering**: ここで初めて分岐。
  - NVIDIA: `tsugi.gpu` → NVVM dialect → NVPTX backend → PTX
  - AMD: `tsugi.gpu` → ROCDL dialect → AMDGPU backend → AMDGCN
  - Intel/Vulkan（将来）: `tsugi.gpu` → SPIR-V dialect → SPIR-V
- **autotuning**: タイルサイズ・行列コアレイアウト・ステージ数を探索。各ベンダーで独立スケジュール。

### Layer B — Unified Runtime（薄く保つ）
- 自作は最小限。**IREE HAL 再利用を第一候補**（cuda/hip/vulkan/metal/cpu 抽象済み）。
- 責務: device列挙・メモリ確保/転送・カーネルロード/launch・同期。
- **escape-hatch**: 標準GEMM等でcuBLAS/rocBLASが勝つ場合は委譲（19年の蓄積に正面勝負しない・R5対策）。

### Layer C — Frontend
- **Tsugi Tile DSL**: Python。`@tsugi.jit` デコレータでタイルカーネル記述。Triton類似だが独自IR。
- **TorchInductorバックエンド**: `torch.compile(backend="tsugi")`。これが楔の本体（ADR-003）。
- 新DSL or ソースレベルのみ。**バイナリCUDA変換禁止**（ADR-002）。

---

## 4. Tensorコア抽象（最難関・R1・make-or-break）

GEMM が LLM 実行時間を支配（prefill 87.6% / decode 76.2% @ llama3.2-1B f16, arXiv 2505.06461）。3社の行列命令は不透明・世代間で挙動変化。

### 設計判断: MLIR intrinsic経路（Vulkan coopmat非依存）
| ベンダー | 行列命令 | lowering経路 |
|---------|---------|-------------|
| NVIDIA | WMMA / HMMA | NVVM `wmma.*` intrinsic（NVPTXがLLVM intrinsicとして公開） |
| AMD CDNA | MFMA | ROCDL `mfma.*` intrinsic / rocWMMA |
| AMD RDNA3+ | WMMA | ROCDL `wmma.*` intrinsic |
| Intel（将来） | XMX / DPAS | SPIR-V cooperative matrix |

**Vulkan `VK_KHR_cooperative_matrix` に依存しない理由**（ADR-004）:
- compute-only かつ HW gated（AMD は RDNA3+ のみ）
- 真に有用な `VK_NV_cooperative_matrix2` は NVIDIA 専有 → 移植性が崩れる
- NVIDIA反復パターン（拡張→Khronos標準化→次世代は再びNVIDIA専有）に巻き込まれない

`tile.matmul` を抽象opとして持ち、各社intrinsicへ直接lowering。SPIR-V coopmatは移植フォールバックのみ。

---

## 5. データフロー（1カーネルの旅）

```
[Python] @tsugi.jit def matmul(a, b): return tile.dot(a, b)
   │ trace
   ▼
[tsugi.tile IR]  tile.dot %a, %b : tensor<128x128xf16>
   │ tile→gpu lowering (tiling, shared mem staging, pipelining)
   ▼
[tsugi.gpu IR]   gpu.block / gpu.barrier / matrix ops
   │ vendor split + autotune schedule
   ├─ NVIDIA ─▶ [NVVM IR] nvvm.wmma.mma ─▶ [PTX] ─▶ ptxas ─▶ SASS
   └─ AMD    ─▶ [ROCDL IR] rocdl.mfma   ─▶ [AMDGCN] ─▶ lld ─▶ GCN binary
   │
   ▼
[Layer B runtime] load module → cuLaunchKernel / hipModuleLaunchKernel
```

---

## 6. リポジトリ構造

```
tsugi/
├── docs/                    # 完成形ファイル（C11・実装を駆動）
│   ├── SPEC.md              # DSL文法・IR階層・lowering規則
│   ├── ARCHITECTURE.md      # 本書
│   ├── FAQ.md
│   ├── BENCHMARK.md
│   └── adr/                 # 設計判断記録
│       ├── ADR-001-mlir-over-spirv.md
│       ├── ADR-002-no-binary-cuda.md
│       ├── ADR-003-torch-backend-first.md
│       └── ADR-004-tensorcore-abstraction.md
├── src/tsugi/               # C++ コンパイラ（MLIR）
│   ├── ir/                  # tsugi.tile / tsugi.gpu dialect 定義
│   ├── backend/             # NVPTX / AMDGPU lowering
│   ├── runtime/             # Layer B dispatch（or IREE HAL bridge）
│   └── frontend/            # tile DSL パーサ
├── python/tsugi_torch/      # torch.compile バックエンド
├── tests/
│   ├── correctness/         # 両ベンダー数値正答（Phase1 DoD）
│   └── perf/                # Triton/cuBLAS/rocBLAS比較
├── examples/
├── cmake/
└── .github/workflows/       # CI（lint/test/sbom）
```

---

## 7. 技術スタック・依存（再利用優先・C6×車輪の再発明回避）

| 層 | 採用技術 | ライセンス | 理由 |
|----|---------|-----------|------|
| IR/コンパイラ | MLIR + LLVM (NVPTX/AMDGPU) | Apache-2.0 w/ LLVM exc | 3社共通基盤・upstream |
| ランタイム | IREE HAL（候補） | Apache-2.0 | cuda/hip抽象済み・再構築回避 |
| DSL前例 | Triton | MIT | 競合でなく貢献検討 |
| 移植フォールバック | SPIR-V | — | Intel/Vulkan（将来） |
| Pythonバインド | nanobind or pybind11 | BSD | 軽量 |
| ビルド | CMake + Ninja | — | LLVM標準 |

依存はゼロにできない（コンパイラ基盤は自作非現実的）。ただし**全てpermissive**（GPL/AGPL不採用・C4準拠）。

---

## 8. スコープ境界（やらないことの明示）

- ❌ CUDAバイナリ変換（ZLUDA型）— ADR-002
- ❌ cuDNN/cuBLAS全面再実装 — escape-hatchで委譲
- ❌ 統一ランタイムAPI自作 — IREE HAL再利用
- ❌ グラフィックス — 計算特化
- ❌ Intel/Apple — v1.0以降
- ❌ 訓練特化 — v1.0まで推論優先

---

## 9. 性能目標（BENCHMARK.md と接続）

| 指標 | 目標 |
|------|------|
| 標準GEMM（FP16）vs cuBLAS/rocBLAS | 20〜30%以内 |
| カーネル4系統 vs Triton | 20%以内 |
| ランタイムdispatch overhead | 無視可能（<1%） |
| correctness | 両ベンダー完全一致（先行・性能は後） |

---

## 10. 既知の制約（正直に）

- 本リポジトリ初期実装は **LLVM/MLIR・GPU が必要**。CPUのみのCI環境では correctness テストの一部（SPIR-V emit検証等）に留まる。実GPU検証は NVIDIA・AMD実機で実施。
- 主張と実装の一致（IZANAGI教訓）: 動作未検証の経路は「未検証」と明記。誇張しない。
