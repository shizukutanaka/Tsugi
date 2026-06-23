# Tsugi 検証層 仕様書（SPEC-verification）

> 仕様が実装を駆動する（C11）。本書は `tsugi` の **検証 API** の規範仕様。各関数の契約
> （入力・出力・判定意味論）と、**保証すること / しないこと（盲点）** を定める。
> コンパイラ/DSL の仕様は [SPEC.md](SPEC.md)、全体マップは [VERIFICATION.md](VERIFICATION.md)。
> 不変条件は `verify.py`（72）、実行可能な真値は `tests/correctness/`（167 関数）。

状態: v0.x（API 未凍結）。CPU リファレンスで動作・検証済み。GPU 実行は未検証（要実機）。

---

## 0. 設計契約（全層共通）

- **深刻度モデル**: `report.Risk` = `OK < INFO < WARN < BLOCK`（IntEnum）。所見は `report.Finding`
  `(risk, op, message)`。所見リスト型レポートは `report.FindingReport` を継承し `max_risk`/`ok`
  （= `max_risk < BLOCK`）/`to_text` を共有する。
- **統一インターフェース**: 全レポート（portability/feasibility/envelope/calibration/decision/
  equivalence/StabilityReport）は `max_risk` を持つ。スカラ計量型（EquivalenceReport）は基底を
  継承しないが `risk`/`max_risk`/`ok` を備える。
- **fail-safe**: 非対称コスト（偽OK ≫ 偽BLOCK）ゆえ、不確実なら BLOCK 寄りに倒す。
- **honesty 原則**: 各判定は「何を見たか」だけでなく「何を見ないか（盲点）」を明示する。
  緑は「検証可能範囲で問題なし」を意味し、「正しさの証明」ではない。

---

## 1. 静的層（デプロイ前・traced IR ＋構成のみ）

### 1.1 portability — 移植リスク静的解析
`analyze(module, target, *, block_dims=None, cfg=None) -> PortabilityReport`
- 入力: traced `ir.Module`、target ∈ {nvidia, amd_cdna, amd_rdna}。
- 出力: `PortabilityReport`（findings・`max_risk`・`portable`）。
- 判定: warp/wavefront 不整合・MMA 形状非対応・bf16 行列コア差・累積順序を Risk 付きで列挙。
- 保証: IR から静的に分かる移植リスクの列挙。**しない**: 実機での数値一致（実行が要る）。

### 1.2 feasibility — 起動可能性ゲート
`check(cfg, vendor, *, regs_per_thread=64, dtype_bytes=2) -> FeasibilityReport` /
`cross_vendor_feasibility(cfg, vendors=...) -> dict[str, FeasibilityReport]`
- 判定: per-block 上限（smem/LDS・threads・regs）超過で `launchable=False`（占有率より上流）。
- 保証: 起動不能（=移植ブロッカー）の categorical 検出。占有率 0% を性能 WARN と誤分類しない。

### 1.3 occupancy — 占有率推定
`estimate(cfg, vendor, *, regs_per_thread=64, dtype_bytes=2) -> OccupancyEstimate` /
`occupancy_gap(cfg, a, b) -> float`
- 出力: 占有率（0–1）・律速資源。HW 定数は一次情報源（SOURCES.md）。
- 保証: 同一構成のベンダー別占有率差の推定。**しない**: 実時間性能（モデル近似）。

### 1.4 tolerance — 導出許容誤差
`derive_tolerance(K, dtype, scale=1.0, noise_floor=0.0, safety=SAFETY) -> {atol,rtol,derived,noise_floor}`
- 規約: 固定値でなく `atol = max(safety·√K·u·scale, noise_floor)`（u=単位丸め）。SAFETY は
  `constants.SAFETY`（単一情報源・根拠は docstring）。
- 保証: K・dtype・scale から「数学が許す発散幅」を導出。**しない**: scale 既定 1.0 は実データと
  乖離しうる（呼び出し側が代表 scale を与えるべき）。

### 1.5 propagation — 合成的等価性（per-model）
`propagate(ops: list[GraphOp], input_div=0.0) -> PropagationReport` / `model_tolerance(ops) -> float` /
`propagate_dag(nodes, input_div=0.0, *, correlated=False) -> PropagationReport` /
`merge_divergence(divs, *, correlated=False) -> float`
- 規約: 通常 op `δ_out = amp·δ_in + local`、残差 op（`GraphOp(residual=True)`）
  `δ_out = √(δ_in² + (amp·local)²)`（skip 接続による希釈・~√L）。`propagate_dag` は `propagate`
  の一般化で、`nodes` 要素が `GraphOp`（直列）か `list[list[GraphOp]]`（フォーク→合流）。合流は
  `merge_divergence`: `correlated=False` で √(Σδ²)（独立な丸め・希釈）、`True` で Σδ（系統共有・
  worst-case）。attention 並列ヘッド・残差・concat の series-parallel DAG を表現。
- 増幅は *相対* 誤差を増やす op のみ（`reduce`/`softmax`/`exp`）。`empirical_cond(sample,kind)` で
  データ依存 cond を実測可能。
- 保証: per-kernel 等価 ⇏ per-model 等価の定量化（一次近似）。深さ（`propagate`）と幅
  （`propagate_dag` のフォーク合流）の両方向。**しない**: 静的 cond=1 は下界（真の増幅は実データ要）。
  series-parallel まで——交差辺をもつ一般 DAG（重み共有・cross-attention 往復）は SP 近似に留まる。

---

## 2. 数値等価性（2 出力の照合）

`compare(a, b, dtype) -> EquivalenceReport`（固定許容）/
`compare_gemm(a, b, K, dtype, scale=None, noise_floor=0.0) -> EquivalenceReport`（導出許容）
- 出力: `equivalent`・max_abs/rel・mismatch・`risk`（EQ→OK / DV→BLOCK）。

`classify_divergence(a, b, K, dtype) -> {EQUIVALENT, LAYOUT, DIVERGENT}`
- 規約: element-wise 不一致でも値の多重集合が一致すれば `LAYOUT`（転置/タイル順=整列バグ・数値は
  正しい）、多重集合も崩れれば `DIVERGENT`。
- 保証: BLOCK の *原因*（数値発散 vs レイアウト）の区別。

---

## 3. 動的層（本番・実データ・oracle 不要）

### 3.1 envelope — 認証エンベロープの実行時検査
`certify_gemm(K, dtype, scale, cond=1.0, noise_floor=0.0) -> Envelope`
`check_tensor(x, env) -> EnvelopeReport` / `check_softmax_input(logits, env) -> EnvelopeReport`
`check_outlier_features(x, axis=-1, spread_warn=10.0) -> EnvelopeReport`
- 判定: NaN/Inf・dtype overflow・denormal(FTZ)・scale 逸脱（認証 atol 無効）・logit>ln(max)・
  outlier feature（チャネル scale 広がり＝単一 scale 仮定の破綻）。
- 保証: 単一ベンダー・oracle 不要で本番入力の認証前提逸脱を検出。

### 3.2 nondeterminism — 出力は分布・ノイズ床
`measure_noise_floor(run_fn, n_runs=16) -> {spread, spread_robust, ...}`（robust=10-90 パーセンタイル・
外れ値頑健）/ `measure_batch_variance(run_of_batch, batch_tiles) -> {...}`（バッチ不変性床）
`attribute(cross_diff, noise_floor, tol) -> {INDISTINGUISHABLE, EQUIVALENT, DIVERGENT}`
`compare_stable(run_a, run_b, K, dtype, n_runs=16, batch_floor=0.0, robust=False) -> StabilityReport`
- 規約: 実効床 = max(run-to-run, batch-invariance)。クロス差 ≤ 床 → `INDISTINGUISHABLE`（等価判定
  未定義）。
- 保証: 決定論を仮定しない健全な比較。**しない**: 単一 run 比較（フレーク）は不健全と明示。

### 3.3 decision — タスクレベル等価（最終単位）
`flip_rate(a,b)` / `topk_flip_rate(a,b,k)` / `nucleus_flip_rate(a,b,p,temperature)` /
`tie_rate(logits,eps)` / `margin(logits)` / `predicted_flip_bound(ref,delta)` /
`compare_decisions(a,b,*,flip_budget,ref,topk,top_p,temperature) -> DecisionReport` /
`regression_flip_rate(a,b,*,atol,rtol)` / `binary_flip_rate(a,b,*,threshold)` /
`binary_margin(a,*,threshold)` / `ranking_flip_rate(scores_a,scores_b,*,k)` /
`compare_task(a,b,*,task,flip_budget,...) -> TaskReport`（新視点11）
- 規約: 多クラス分類は argmax フリップ率（スケール不変）。**非分類タスク（新視点11）**: 回帰は
  |a-b|>atol+rtol·|a|、バイナリ sigmoid は threshold 跨ぎ、ランキングは top-k 集合変化。
  未知タスク種別は ValueError で弾く（静かに誤計算しない）。スケール不変（argmax/top-k）/
  確率依存（nucleus）。bound は残差（argmax 保存的アフィン系統成分を除く）で評価。
- 保証: 数値発散をタスク影響に翻訳。**しない**: 同点（`tie_rate` 高）では argmax は規約依存＝
  フリップ誤帰属に注意（WARN）。

### 3.4 rollout — 自己回帰的等価（生成単位・新視点9）
`sequence_survival(p, length) -> float`（=(1−p)^length）/ `expected_divergence_step(p) -> float`
（=1/p）/ `safe_generation_length(p, confidence=0.99) -> int` / `divergence_step_quantile(p, q)` /
`flip_rate_upper_bound(flips, n, confidence=0.95) -> float`（Wilson 片側上限）/
`analyze_rollout(p, target_length, *, confidence=0.99) -> RolloutReport` /
`rollout_from_logits(a, b, target_length, *, confidence=0.99, conservative=True, decode="greedy", topk=5, top_p=0.9, temperature=1.0)` /
`simulate_rollout(p, length, trials, seed) -> float`
- 規約: per-token フリップ率 p を生成長 L に合成。シーケンス一致確率 survival=(1−p)^L、初回発散
  期待位置 1/p。verdict は safe_len 内=OK / survival≥0.5=WARN / 未満=BLOCK。`rollout_from_logits`
  は既定で p の上側信頼限界を使い小標本の過信を防ぐ（fail-safe）。`decode` で運用のデコード方式に
  p を整合（greedy=argmax / topk・nucleus=候補集合フリップ）—— サンプリング生成では候補集合の分岐が
  per-token 発散になり、greedy argmax フリップ率は過小評価しうる。`audit_runtime` の rollout 層も
  同じ上側信頼限界 p を使う（点推定でなく fail-safe）。
- 保証: per-token 許容 ⇏ per-sequence 許容（複利増幅）を露出。**しない**: survival は *完全一致*
  の確率（意味等価でない・厳しい側）。フリップ率の定常性を仮定（位置非依存）。propagation の自己回帰版。

### 3.5 worstcase — 最悪ケース発散探索（能動探索・新視点10）
`divergence(fn_a, fn_b, x, *, relative=True) -> float` /
`search_worst_input(fn_a, fn_b, x0, *, radius=1.0, steps=400, seed=0, restarts=4, bounds=None) -> (x_worst, div_worst)` /
`analyze_worst_case(fn_a, fn_b, samples, *, tol, radius=1.0, steps=400, seed=0, bounds=None) -> WorstCaseReport`
- 規約: 既存層が *代表データ上の率* を測る受動検証なのに対し、box 制約（＝認証エンベロープ）内で
  発散を最大化する入力を微分フリー探索（黒箱・勾配なし・ランダム再開ヒルクライム）。verdict は
  worst > tol（エンベロープ内に許容超過の反例）=BLOCK / worst ≤ tol だが典型の ×10 以上=WARN /
  それ以外=OK。返す `x_worst` は seed 固定で再現する監査可能な反例。
- 保証: 平均ケース等価 ⇏ 最悪ケース等価（代表データが良性でも踏みうる反例）を露出。envelope と
  閉ループ（envelope が領域を定義し、worstcase がその内部を突く）。**しない**: 反例生成であって
  健全性証明でない（見つからない＝等価の証明ではない・探索の網羅性依存）。ヒルクライムは局所最適に
  嵌りうる（薄い manifold は取り逃す）。box の与え方が結論を左右（envelope の認証領域と一致させる）。

---

## 4. メタ層（検証器・oracle の信頼性）

### 4.1 calibration — 検証器の検証
`detectability_floor(K, dtype, scale, safety) -> {abs, rel}`（max_abs が見逃す誤差下限・√K で拡大）
`check_systematic(a,b,K,dtype) -> CalibrationReport`（RMS 比で系統バグ・scale/K 不変）
`is_equivalent_combined(a,b,K,dtype) -> bool`（max_abs + 系統の fail-safe 合成）
`roc_sweep(strengths,K,seeds) -> list[{strength, false_ok_max_abs, false_ok_combined}]`
`detect_shared_mode(a,b,oracle,K,dtype) -> {OK, DIVERGENT, SHARED_MODE}`
- 規約: 偽OK が cardinal metric（非対称）。`SHARED_MODE` = a≈b だが両方 oracle と不一致
  （cross-vendor 一致は必要十分でない）。
- 保証: 検出限界と偽OK 率の定量化。**しない**: oracle 無し（本番）では SHARED_MODE 検出不能。

### 4.2 oracle_check — オラクル自体の検証
`verify_oracle(seed=0, rtol=1e-4) -> FindingReport` / `oracle_is_trustworthy(seed) -> bool`
- 規約: 実装非依存のメタモルフィック関係（matmul 恒等・分配則・sum(ones)=n・exp(a+b)=exp(a)exp(b)・
  softmax 和=1・shift 不変・rsqrt 恒等）＋高精度(float64)照合。
- 保証: 第二オラクル無しでオラクルを検証（無限後退を断つ）。**しない**: 必要条件であって完全な
  正しさ証明ではない（性質を満たす誤実装はありうる）。

---

## 5. 時間軸（証明書の有効性）

### 5.1 provenance — 陳腐化検出
`certify(verdict, **env) -> Certificate` / `is_stale(cert, **env) -> bool` /
`changed_fields(cert, **env) -> {field: (old, new)}`
- 規約: verdict を環境フィンガープリント（python/numpy/platform ＋ 実機の cuda/rocm/driver/
  compiler）に束ねる。環境が変われば stale（再検証要）。
- 保証: 「一度認証＝永遠に有効」を排す。verdict はそれが計算されたスタックでのみ有効。

---

## 6. 統合ファサード

`audit(module, cfg, *, targets, block_dims, ref_logits=None, provenance=None) -> Audit`
- 静的層（portability/feasibility/occupancy/tolerance+envelope+detectability_floor/propagation）を
  まとめ 1 判定に集約。`ref_logits` でタスクフリップ率上界も併記。実行時層は pending として列挙。
  `provenance={...}` で verdict を環境 fingerprint に束ねる（§5.1・`Audit.is_stale`）。

`audit_runtime(a_out, b_out, K, *, dtype, env, noise_floor, logits_a, logits_b, flip_budget, oracle=None, provenance=None, gen_length=0) -> Audit`
- 実データで envelope/equivalence(+noise 3 状態)/systematic/decision を回し 1 判定に。
  `oracle` を渡すと correctness 層（detect_shared_mode）も算入。`provenance={...}` で verdict をスタンプ。
  `gen_length>0` かつ logits 指定で rollout 層（per-token→生成長合成・新視点9）も算入。

`audit_cross_vendor(run_a, run_b, K, *, ..., run_batch=None, batch_tiles, provenance=None) -> Audit`
- 実機入口: 各ベンダーの noise/batch 床を実測 → audit_runtime。`run_*` は seed/tile → 出力 callable。
  `provenance={...}`（実 GPU の rocm/cuda/driver）を audit_runtime へ素通し、実機 verdict を束ねる。

`Audit`: `phases`（各 `AuditPhase(name, when∈{decided,pending}, max_risk, lines)`）・`max_risk`
（decided 層のみ）・`portable`・`to_text`（検証ライフサイクルを一望）・`certificate`（provenance
スタンプ）・`is_stale(**env)`（スタック更新で再検証要を判定）。各 facade は `provenance={...}` で
cuda/rocm/driver 等を渡すと verdict をそのスタックに束ねる（§5.1 と接続）。

---

## 7. 検証連鎖（接地構造）

```
portability(移植可) → equivalence/decision(正しい・oracle 照合で shared-mode)
  → oracle_check(oracle 自体が信頼できる) → provenance(その verdict はまだ現スタックで有効)
```
各段が下の段に錨を下ろす。すべて CPU で実行・検証可能。実機 GPU 経路は未検証。
