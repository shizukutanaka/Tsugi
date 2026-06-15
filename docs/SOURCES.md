# Tsugi 一次情報源（出典）

> 主張と実装の一致。HW 定数・設計判断の根拠を一次情報源に紐付ける。
> 実装と乖離したら公式（一次情報源）を確認して再実装（ハーネス権限スタック準拠）。

## occupancy HW 定数

### NVIDIA H100 (Hopper, CC 9.0)
- NVIDIA Hopper Tuning Guide（docs.nvidia.com/cuda/hopper-tuning-guide）
  - 共有メモリ: 228 KB/SM（carveout 構成可・block あたり最大 227 KB）
  - 同時常駐 warp: 64/SM（2048 threads）
  - レジスタファイル: 64K（65536）32bit registers/SM
  - 最大 thread blocks: 32/SM ・ 最大 registers/thread: 255
  - warp size: 32

### AMD MI300X (CDNA3, gfx942)
- ROCm GPU architecture hardware specifications（rocm.docs.amd.com gpu-arch-specs）
  - wavefront size: 64 ・ LDS: 64 KiB/CU ・ VGPR file: 512 KiB/CU ・ CU 数: 304
- ROCm Compute Profiler — Pipeline descriptions（rocm.docs.amd.com rocprofiler-compute）
  - 命令バッファ: 8 wavefronts/SIMD（32 wavefront slots/CU）
  - VGPR 512 KiB/CU → 131072 32bit slots/CU（4 SIMD × 128 KiB）

### AMD RX 7900 XTX (RDNA3, gfx1100)
- AMD GPUOpen — Occupancy explained（gpuopen.com/learn/occupancy-explained）
  - 1536 VGPR/SIMD（wave32）・SGPR は固定割当で常時充足
- AMD RDNA3 ISA Reference Guide
  - wave32 既定 ・ LDS（usable per CU 概算 64 KiB）

注: occupancy はアーキ依存。上記は代表 SKU の実値。別アーキは `tsugi.occupancy.HW`
を上書きして使う。granularity（割当粒度）まではモデル化していない（保守的近似）。

## feasibility per-block 上限（起動可否を分ける壁）

> occupancy.HW が「常駐数を決める容量」なのに対し、こちらは「越えたら launch/compile が
> 失敗する per-block のハード上限」。`tsugi.feasibility.LIMITS`。両者は別物。

| ベンダー | smem/LDS per block | threads/block | regs/thread | 出典 |
|---------|--------------------|---------------|-------------|------|
| NVIDIA (H100/Hopper) | 227 KB（opt-in dynamic） | 1024 | 255 | Hopper Tuning Guide / CUDA C Programming Guide |
| AMD (MI300X/CDNA3) | 64 KiB（LDS/workgroup） | 1024 | 256（VGPR/lane） | ROCm GPU arch specs |
| AMD (RX7900XTX/RDNA3) | 64 KiB（LDS usable/workgroup） | 1024 | 256（VGPR/lane, wave32） | GPUOpen / RDNA3 ISA |

要点（移植の崖）: **smem/LDS per block は NVIDIA 227KB vs AMD 64KiB と 3.5 倍差**。
NVIDIA の広い smem を前提にチューニングした構成は AMD で *起動すらしない*
（docs/PERSPECTIVE-launch-feasibility.md）。これは性能差でなく単一ソース約束の破綻。

## 設計判断（リサーチ由来）

- CUDA の堀＝ライブラリ＋PyTorch 統合＋QA。言語でない（ADR-003）。
- NVIDIA は SPIR-V を CUDA 計算の first-class 入力としない → PTX 経路必須（ADR-001）。
- NVIDIA EULA 翻訳層禁止（2021〜・CUDA 11.6 明文化）・AMD ZLUDA 撤回（2024-08）（ADR-002）。
- 行列コア: Vulkan KHR coopmat は compute-only・HW gated、有用な NV coopmat2 は専有
  → MLIR intrinsic 直叩き（ADR-004）。
- GEMM が LLM 実行時間を支配（prefill 87.6%/decode 76.2% @ llama3.2-1B f16, arXiv:2505.06461）。
- AMD の弱点は性能でなく QA 文化（SemiAnalysis 2024）→ クロスベンダー検証が楔
  （docs/PERSPECTIVE-cross-vendor-verification.md）。

## envelope dtype 数値限界（IEEE 754）

> `tsugi.envelope.DTYPE_LIMITS`。本番入力が認証エンベロープ内かを検査する基準。

| dtype | 最大正規数 | 最小正規数 | exp-overflow 閾値 ln(max) |
|-------|-----------|-----------|---------------------------|
| float16 (IEEE binary16) | 65504 | 6.1035e-5 | 11.09 |
| bfloat16 | 3.3895e38 | 1.1755e-38 | 88.72 |
| float32 (IEEE binary32) | 3.4028e38 | 1.1755e-38 | 88.72 |

要点: fp16 は範囲が狭く **overflow** が主リスク（softmax 生 logit > 11.09 で exp が inf）。
bf16 は範囲が f32 並みに広いが仮数 7bit で **precision/denormal** が主リスク。
denormal（最小正規数未満）は GPU の FTZ（flush-to-zero）有無でベンダー差を生む
（docs/PERSPECTIVE-runtime-envelope.md）。
