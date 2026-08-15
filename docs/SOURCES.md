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

## Microscaling (MX) / NVFP4 低精度フォーマット（2025-26）

> `tsugi.tolerance.UNIT_ROUNDOFF` / `tsugi.equivalence.TOLERANCE` / `tsugi.envelope.DTYPE_LIMITS`
> の mxfp4_e2m1 / mxfp6_e2m3 / mxfp6_e3m2 エントリの根拠。

確度高（一次スペックで確認可能）:
- **OCP Microscaling Formats (MX) Specification v1.0**（OCP: Open Compute Project）
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - block=32 要素ごとに 1 個の共有スケール（E8M0・2^-127〜2^127・power-of-two のみ）。
  - MXFP4 = 要素 E2M1（仮数 1 bit・max=6.0・min_normal=1.0）→ u=2⁻¹（全 dtype 中最粗）。
  - MXFP6 = E2M3（仮数 3 bit・max=7.5）または E3M2（仮数 2 bit・max=28）。
  - 丸めモードは実装定義（RNE/確率的丸め=SR）として仕様に残る——クロスベンダー発散源になりうる。
- **NVIDIA Blackwell アーキテクチャ**（NVIDIA developer blog: "NVIDIA Blackwell Delivers
  World-Record DeepSeek-R1 Inference Performance" 等の Blackwell Tensor Core 解説）と
  **AMD CDNA4 (MI350/MI355) アーキテクチャ**（AMD "Introducing AMD CDNA 4 Architecture" blog）
  は共に MXFP4/MXFP6/MXFP8 を HW ネイティブ対応する——これが Tsugi が MX ファミリーを
  「クロスベンダー共通フォーマット」としてテーブル化する根拠。
- **NVFP4**（NVIDIA blog: "Introducing NVFP4 for Efficient and Accurate Low-Precision
  Inference"）: block=16・スケールは E4M3（power-of-two に限らない・2 段スケーリング）で
  MX spec の NVFP4 ではない NVIDIA 独自拡張。AMD に対応 HW が存在しない。
  → NVFP4 は Tsugi の 3 テーブルに**意図的に含めない**（対象外）。NVFP4 で量子化した
  モデルはクロスベンダー移植それ自体が不能——これは許容誤差のチューニング問題でなく
  「そもそもこの dtype を選んで良いか」という移植性判断であり、数値許容モデルの範囲外。

確度中（検索サマリ由来・一次確認前——将来の精緻化候補として記録のみ）:
- テンサーコアのビット精度モデル（arXiv 2512.07004、2511.10909 とされる）: クロスベンダー
  matmul の発散を累積幅＋truncation/RNE 差として決定論的にモデル化できるとする報告。
  `tsugi.equivalence.simulate_vendor_matmul` の外部裏づけになりうるが、arXiv ID は
  検索サマリ経由で得たものであり一次確認前（ハードコード前に確認要）。
- TBIK（arXiv 2511.17826 とされる）: 縮約木トポロジーが一致すれば TP サイズに依らず
  ビット一致するという報告。既存の batch-invariance 節（「タイルが違う＝実効バッチが違う」）
  の一般化に相当しうるが同様に一次確認前。

## torch.compile / Triton の数値精度 API（2025-26 動向）

> `tsugi.tolerance` の TF32 記述・`tsugi.nondeterminism` カタログの根拠追補。

- **PyTorch 2.9: fp32 精度 API の変更**。`torch.backends.cuda.matmul.allow_tf32`（bool）は
  非推奨となり `torch.backends.cuda.matmul.fp32_precision = 'ieee' | 'tf32'`（文字列）へ移行。
  `set_float32_matmul_precision('highest')`→ieee ／ `'high'`・`'medium'`→tf32 に対応。
  PyTorch リリースノート・`torch.backends.cuda` ドキュメント参照。
  - **FlexAttention のデフォルト精度がリリース間で ieee→tf32 に回帰した実例**
    （GitHub pytorch/pytorch issue #161022 とされる）——同じコードでも PyTorch の
    バージョンが変わると数値が変わりうる実例。provenance の stale 検出が必要な根拠。
- **vLLM batch-invariant モード**（`VLLM_BATCH_INVARIANT=1`。vLLM ドキュメント／
  Thinking Machines Lab のバッチ不変カーネルを取り込んだもの）: RMSNorm/matmul/attention の
  縮約順序をバッチサイズ非依存に固定する。**NVIDIA（Compute Capability ≥ 8.0）専用——
  ROCm/AMD は未対応**。クロスベンダー比較では「片側だけ batch-invariant」という
  新たな非対称が生じうる（AMD 側は従来通りバッチ変動の影響を受ける）。
- Triton: `tl.dot(input_precision=)` に `ieee`/`tf32`/`tf32x3`、環境変数
  `TRITON_F32_DEFAULT` で既定精度を制御可能（Triton ドキュメント）。
- NeurIPS 2025「Understanding and Mitigating Numerical Sources of Nondeterminism in
  LLM Inference」（arXiv 2506.09501 とされる）・SGLang の deterministic mode ブログ:
  vLLM のバッチ不変性知見と独立に類似の LLM 推論非決定性対策が業界で並行して進んでいる
  という傍証（検索サマリ由来・一次確認前）。

## 丸め誤差境界: 確率的（√K）と決定論的最悪ケース（K）（2019-2023）

`tolerance.expected_gemm_abs_error` の次元依存項 √K の学術的根拠と、その **仮定** の記録。

- **古典的な決定論的境界（Wilkinson）**: 長さ K の内積/累積の前進誤差は
  γ_K = K·u/(1−K·u) ≈ **K·u** で抑えられる（worst-case・仮定なし）。
- **確率的境界（Higham & Mary, 2019, "A New Approach to Probabilistic Rounding Error
  Analysis", SIAM J. Sci. Comput.）**: 丸め誤差を **独立・平均 0** の確率変数と
  モデル化すると、worst-case 境界の次元定数を **その平方根で置き換えた** 境界が
  高確率で成り立つ（K·u → **√K·u**）。実務上、決定論的境界は悲観的すぎるため
  この √n 型の境界の方が実測に近い。
  - 関連: Connolly, Higham & Mary の確率的解析（内積の前進誤差 ~√(K ln K)·u）、
    確率的丸め（stochastic rounding）の分散ベース解析（Bienaymé–Chebyshev 由来の O(√n u)）。
- **GPU テンサーコアへの適用**: Fasi, Higham, Lopez, Mary, Mikaitis,
  "Matrix Multiplication in Multiword Arithmetic: Error Analysis and Application to
  GPU Tensor Cores" — 確率的誤差解析で NVIDIA テンサーコアの精度問題を説明し、
  パラメータ化ブロック和で性能/精度のトレードオフを改善している。
  本ライブラリが対象とする「テンサーコアの累積順序・精度差」と直接に関係する一次資料。

**本ライブラリでの扱い（重要な帰結）**: 既定の √K は *確率的* 境界であり **保証ではない**。
独立・平均 0 の仮定が破れる典型が **系統誤差**（相関した丸め・一方向のバイアス）であり、
本ライブラリはそれを `calibration.check_systematic` で検出する層を別に持つ——つまり
「仮定が破れうる」ことを設計として認めている。そのため
`derive_tolerance(..., model="worstcase")` で古典的 K·u 境界を選べるようにし、
`explain()` は既定使用時に「√K は確率的境界」「最悪ケースとの開き」「系統誤差検査への
誘導」を明示する（verify.py 不変条件 70）。K=2048・fp16 では両者の開きは約 45 倍。

> 出典の確度: Higham & Mary の √n 置換則と Wilkinson の γ_n は数値解析の標準的結果
> （複数の二次資料で一致）。テンサーコア論文は検索サマリ由来のため、実装値として
> ハードコードする前に一次資料の確認が要る（本節は *モデル選択肢の根拠* としてのみ使用し、
> 具体的な係数は導入していない）。

## 非決定下での検証: 証拠の累積（2025-26）

「参照実装自体が非決定的なとき、どう検証するか」は本ライブラリの
`nondeterminism.INDISTINGUISHABLE`（クロス差がノイズに埋もれ判定未定義）と同じ問題設定。

- **DiFR / Token-DiFR, "Inference Verification Despite Nondeterminism"**（arXiv:2511.20621）:
  推論が正しく実行されたかを検証したいが、同じ推論を 2 回走らせても良性の数値ノイズで
  結果が変わるため、正当な変動と本当の異常を単発では区別できない。DiFR は PRNG シードを
  強く同期して *唯一の変動源を logits の浮動小数ノイズに限定* し、**多数トークンに証拠を
  累積**して検出する。logprob の SNR を指標に、Llama 3.1 8B で数千トークン以内に
  量子化・シード・温度の設定誤りを検出できると報告。
- 関連: LLM-42（検証つき投機実行で決定性を得る・arXiv:2601.17768）、
  DASH（決定論的 attention スケジューリング・arXiv:2601.21824）、
  BEAVER（決定論的 LLM 検証器・huggingface.co/papers/2512.05439）。

**本ライブラリでの取り込み**: 従来 `INDISTINGUISHABLE` は *終端* 状態（「等価判定は未定義」で
行き止まり・ユーザーに次手が無い）だった。DiFR と同型の「証拠の累積」を採り、
`nondeterminism.runs_to_resolve(cross_diff, noise_floor, confidence)` を追加——独立な run を
平均すると平均のノイズは σ/√N に縮む一方、系統差 d は縮まないため SNR = d·√N/σ が伸びる。
必要条件 d > z·σ/√N より **N > (z·σ/d)²**。`compare_stable` は INDISTINGUISHABLE 時に
「1 ベンダーあたり約 N run を平均すれば分離可能」を報告し、行き止まりを **実行可能な次手** に
変える（verify.py 不変条件 72）。

> 出典の確度: DiFR の枠組み（シード同期＋証拠累積＋SNR 指標）は検索サマリ由来で
> 一次確認前（arxiv.org は本環境の egress proxy で遮断）。**本ライブラリが採用したのは
> 「独立試行の平均でノイズが √N で縮む」という初等統計の帰結のみ**で、論文固有の
> 数値・閾値は一切ハードコードしていない。

## outlier feature / massive activations（2022-2026）

`envelope.check_outlier_features` と torch 経路の `audit_fx(sample=)` が「単一 scale 仮定の
破綻」を検出する根拠。

- **LLM.int8()（Dettmers et al., NeurIPS 2022）**: transformer がスケールすると
  大きな magnitude を持つ **outlier feature** が出現し、全層とその量子化に強く影響する。
- **massive activations**: 大きさが 100 を超え、隠れ状態の **中央値の約 1000 倍** に達する
  活性が存在する。normal outlier が全トークンに渡って現れるのに対し、massive outlier は
  **少数のトークンに限局** するのが主な違い。FFN の down-projection 入力（特に 2 番目と
  最後から 2 番目のデコーダ層）で 1000 超の値が観測される。
- 対策研究: SmoothQuant / SmoothRot（channel-wise scaling ＋ rotation）・
  DuQuant（dual transformation で outlier を分散）・QLLM（channel reassembly）。
  いずれも「単一の scale では表現しきれない」ことを前提にした手法。

**本ライブラリでの帰結**: 代表テンソルを与えずに `scale=1.0` を仮定すると、実 LLM 活性では
認証 atol が **桁で** 誤る。`audit()` は `certify_from_sample`（B-1a）で既に実測していたが、
製品の想定入口である **torch 経路（`audit_fx`）には無かった**（FEATURE-AUDIT.md A-3）。
`audit_fx(sample=)` を追加し、実 RMS scale・`channel_scale_spread`（外れチャネル検出）・
`empirical_cond`（増幅 op の実測条件数）を測るようにした（verify.py 不変条件 73）。

> 出典の確度: LLM.int8() は一次資料（NeurIPS 2022）。massive activations の
> 「中央値の ~1000 倍」「down-projection 入力に局在」は検索サマリ由来で一次確認前。
> **本ライブラリは倍率等の具体値をハードコードせず**、外れ広がりを実測して閾値
> （`spread_warn=10.0`）で警告するに留める。

## 定数の実機校正 — 許容限界（tolerance interval）の統計

> `tsugi.calibration.tolerance_factor_normal` / `wilks_confidence` / `wilks_min_runs` /
> `calibrate_safety`。SAFETY=4.0（`constants.py`）は「4σ 相当」という*経験的*
> ヘッドルームで、一度も実機ノイズで校正されていない（FEATURE-AUDIT.md A-2）。
> 手順書は `docs/GPU-BRINGUP.md`。

**問題の性質**: SAFETY は許容 `atol = SAFETY·√K·u·scale` と検出限界
`rel = SAFETY·√K·u` の *両方* を一律にスケールする。誤っていれば 13 検証層すべての
判定が同じ向きに狂う（大きすぎ → 偽OK の盲点が広がる／小さすぎ → 良性ノイズを
偽BLOCK する）。「検証器が実機で正しい」という主張の最終根拠がこの定数で止まっている。

**「4σ」が成立する条件**: σ が *既知* のときだけ。有限標本から σ を推定する現実では、
必要な係数は片側許容限界の k(n, coverage, confidence) であり、常に k > z_coverage。

- **正規理論**: Natrella (1963) の近似式（NIST/SEMATECH e-Handbook §7.2.6.3 に同式）

      a = 1 − z_conf² / (2(n−1)),  b = z_cov² − z_conf² / n
      k = (z_cov + √(z_cov² − a·b)) / a

  実装は公表表（coverage=0.99・confidence=0.95）を 1.5% 以内で再現し、系統的に
  表より **小さい** 側に外れる（= 要求 SAFETY を過小 = 許容を締める = 偽BLOCK 側の
  誤り方）。n→∞ で z_0.99≈2.326 に収束（verify.py 不変条件 74）。

  | n（独立 run 対） | 10 | 20 | 30 | 100 | ∞ |
  |---|---|---|---|---|---|
  | k (0.99/0.95) | 3.98 | 3.30 | 3.06 | 2.68 | 2.33 |

- **非パラメトリック**: Wilks (1941) の順序統計量。標本最大を上側許容限界に使うとき
  達成信頼度 = 1 − coverage^n、必要標本数 n ≥ ln(1−confidence)/ln(coverage)。
  0.99/0.95 で **299 対**、0.999/0.95 で 2995 対。実機校正が「16 run 回して終わり」
  で済まないことの定量的根拠（n=16 対の達成信頼度は僅か 14.9%）。
  - S. S. Wilks, "Determination of Sample Sizes for Setting Tolerance Limits",
    Ann. Math. Statist. 12(1), 1941.

**両者の max を採る理由（文献）**: GPU の浮動小数ノイズを i.i.d. ガウスと見なす
作業仮定は実測で棄却されている —— 誤差は構造的・高相関で、fp16 では全誤差分散の
**約半分が共分散の非対角項**に乗り、「ランダムな静電ノイズ」ではなく協調した
方向性のある摂動として振る舞う。
  - "On the Structure of Floating-Point Noise in Batch-Invariant GPU Matrix
    Multiplication" (arXiv:2511.00025)

ゆえに正規理論の k 単独を信じる根拠が無く、分布仮定に依らない標本最大と併用して
下回らないようにする（保守側）。

**run-to-run 標本は下界**: 同一ベンダー内の揺れは *縮約順序差* だけを含み、
クロスベンダー発散（タイル形状・行列コア・ライブラリ実装の差を含む）の部分集合に
すぎない。ゆえに run-to-run 校正は SAFETY を **上げる根拠にはなるが下げる根拠には
ならない**（下げれば未測定のクロス成分を許容から外すことになり偽OK に倒れる）。
`calibrate_safety(source=...)` がこの非対称を判定に焼き込む（verify.py 不変条件 75）。

> 出典の確度: Wilks 1941 と Natrella 1963 の式は標準的な統計手法で、実装は公表表との
> 一致で機械検証済み（不変条件 74）。arXiv:2511.00025 の「分散の約半分が非対角」は
> 検索サマリ由来で一次確認前 —— **本ライブラリはこの数値を一切ハードコードせず**、
> 「正規理論単独を信じない（標本最大と max を採る）」という*設計判断の根拠*として
> のみ用いる。校正の結論そのものは実測標本から導く。

## 正規化層の数値安定性（LayerNorm vs RMSNorm）

> `tsugi.propagation` の kind `layer_norm` / `rms_norm`。`empirical_cond(x, "layer_norm")`。
> FEATURE-AUDIT.md A-5。**この節は「検証してから導入」ガードレールが前提の誤りを
> 捕まえた事例**——当初「正規化は増幅しない（≤1）はず」と想定していたが、数値実験で
> LayerNorm は平均優勢入力で *増幅* すると判明し、設計が反転した。

**ヤコビアンと相対増幅**（一次の摂動解析）:

- LayerNorm `y = g·(x−μ)/√(σ²+eps)` のヤコビアンは
  `J = (g/√(σ²+eps))·(I − 11ᵀ/d − ŷŷᵀ/d)`。射影項が **平均方向（1）と半径＝スケール
  方向（ŷ）の 2 つの特異値を消す**（この 2 方向の摂動は出力を一切変えない:
  `LN((1+c)x + b·1) = LN(x)`）。残る最大特異値は `g/√(σ²+eps)` で、相対 RMS 増幅の
  上界は `RMS(x)/√(σ²+eps) = 1/√(1−(μ/RMS)²)`（eps→0）。零平均で ≈1、
  **平均優勢（μ/RMS→1）で ≫1**。
- RMSNorm `y = g·x/RMS(x)` は `J = (g/r)(I − ŷŷᵀ)`——μ を引かないので σ でなく RMS で
  割り、消えるのは半径方向のみ。相対増幅は **無条件に ≤1**。

**本ライブラリでの実測**（`tests/correctness/test_propagation.py`・32×512 標準正規＋shift）:

| shift（μ/RMS） | LayerNorm 実測 amp | 上界 RMS/√(σ²+eps) | RMSNorm 実測 amp |
|---|---|---|---|
| 0 | 1.00 | 1.01 | ≤1.00 |
| 3 | 3.15 | 3.37 | ≤1.00 |
| 10 | 10.10 | 10.72 | ≤1.00（shift=100 でも worst 1.0021） |

**文献**:
- "Numerical stability analysis of large language models"（arXiv:2503.10251）——
  LayerNorm の forward error は **outlier の background entries の大きさに支配される**
  一方、**RMSNorm は unconditional forward stability** を示す。条件数の比較では
  どちらか一方が常に優位というわけではない。
- LayerNorm のヤコビアンで平均・分散を動かす摂動が出力を変えない（2 つの消失特異値）、
  ゲインが活性スケールに逆比例する（implicit gain control）という解析。
  ε 付き LayerNorm の Lipschitz 定数は `ε^(-1/2)·max|γ|·N`（LipsFormer 系）。
- PyTorch `torch.nn.LayerNorm` ドキュメント——`eps` 既定値 **1e-5**。

**設計判断**:
- `layer_norm` は増幅 op（`amp = max(1, cond)`）。cond は sample から
  **行ごと `RMS/√(σ²+eps)` の max** で実測する。median でなく max なのは、零平均の
  多数派に紛れた平均優勢の外れ行（massive activations 型・本文書の outlier feature 節）を
  median が隠し偽OK になるため。reduce の `Σ|x|/|Σx|` と違い eps ガードで有界
  （≤ RMS/√eps）なので max が暴走しない。
- `rms_norm` は **非増幅（amp=1.0 固定）**。実測は ≤1 だが、**1 未満の減衰係数は入れない**
  ——未検証係数の禁止であり、過大な dilution は偽OK の温床になる。amp=1 は実測 ≤1 を
  決して過小評価しない保守側。
- eps=1e-5 はハードコードした魔法数ではなく **torch.nn.LayerNorm 既定値＝実装が実際に割る数**。
  それより小さい custom eps の近定数行では上界を超えうる（病的ケース・正直な限界）。

> 出典の確度: ヤコビアンと相対増幅の式は本リポジトリの数値実験で機械検証済み
> （verify.py 不変条件 76・実 LayerNorm 実装に対して上界性と零空間の両方を固定）。
> arXiv:2503.10251 の記述は検索サマリ由来で一次確認前——**本ライブラリは論文中の
> 数値を一切ハードコードせず**、「RMSNorm を非増幅に、LayerNorm を増幅に置く」という
> *設計判断の裏づけ* としてのみ用い、cond は常に実測から導く。
