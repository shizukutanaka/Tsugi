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

## 検証器の検出限界（detectability floor）

> `tsugi.calibration.detectability_floor`。許容ベース等価判定が見逃す系統誤差の下限。

相対 floor = safety·√K·u（u = 単位丸め誤差・scale 非依存）。これ未満の系統バグは
max_abs 等価判定で偽OK になる。K で √K 拡大するのが要点（大K ほど盲点が広がる）。

| K | fp16 検出限界（safety=4, u=2⁻¹¹） |
|---|---|
| 256 | 3.1% |
| 2048 | 8.8% |
| 8192 | 17.7% |

救済: scale/K 不変な RMS（エネルギー）比。乱雑な累積発散は zero-mean ゆえ ≈0、
系統バグ（スケール/バイアス）は相関ゆえ検出可能。max_abs（乱雑）と RMS 比（系統）は
相補的で、合成判定（fail-safe）が偽OK を消す（docs/PERSPECTIVE-verifier-calibration.md）。

## 非決定性のノイズフロア（run-to-run）

> `tsugi.nondeterminism.measure_noise_floor`。GPU の atomic 加算は和の順序が
> run-to-run で変わり結果が揺れる。出力は固定点でなく分布。

第二の床（HW の床）= run-to-run ノイズ。検証器の実効分解能 = max(数値検出限界, ノイズフロア)。
クロスベンダー差がこの床未満なら「A vs B」は「A vs A（別 run）」と区別不能 →
INDISTINGUISHABLE（等価判定が原理的に未定義）。

- 主因: atomicAdd ベースの reduction / split-K（スレッド到着順 = 浮動小数の和順）。
  関連: NVIDIA は「同一ビット再現性は保証されない」（cuBLAS/cuDNN docs の reproducibility 節）。
  PyTorch も `torch.use_deterministic_algorithms` で一部のみ決定化可能（atomic 経路は性能犠牲）。
- 含意: 単一 run 比較は方法論的に不健全。noise_floor を実測してから tolerance に供給する
  （tolerance.derive_tolerance の noise_floor 引数。既定 0 = 決定論仮定は誤り）。
  CPU 実装は atomic 非決定の擬似再現（明示）。実機では run_fn を実カーネルにするだけ。
  （docs/PERSPECTIVE-nondeterminism.md）

## バッチ不変性（batch invariance）— LLM 推論の支配的非決定源（2025）

> `tsugi.nondeterminism.measure_batch_variance`。第三の床（run-to-run とは独立・決定論的だが
> バッチ変動で生じる）。

最新研究の知見:
- **支配的な非決定源はバッチ不変性であり、atomic 並行性ではない**。あるサンプルの出力が
  forward の *バッチサイズ* に依存する（バッチ依存のタイル/縮約順序が丸めを変える）。
  matmul/RMSNorm/attention が影響を受け、バッチ不変な縮約カーネルで解消できる。temp=0 で
  同一プロンプト 1000 回 → 80 種の出力が、修正後は全ビット一致。GPU 固有でなく CPU/TPU でも生じる。
  - Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (2025)
    https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  - "Impacts of floating-point non-associativity on reproducibility…" SC'24 (arXiv:2408.05148)
- **浮動小数ノイズは独立ガウスでなく構造的（相関）** — 「誤差は独立ガウス」という仮定を覆す。
  これは calibration の系統（RMS 比）検出が *必要* である根拠を外部から裏づける。
  - "On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix Multiplication"
    (arXiv:2511.00025)

含意（Tsugi への取り込み）: 実効床 = max(run-to-run ノイズ, **batch-invariance 床**, 数値検出限界)。
本番でバッチが変動するなら batch-variance を等価判定に織り込む。クロスベンダーでは「タイルが
違う＝実効バッチが違う」ため、各ベンダーが個別に決定論的でも発散しうる。
