# ソクラテス式問答 50 回 — 改善点の洗い出し

現コードベース（python/tsugi 2,512 行・95 テスト・verify 38 不変条件・8 視点＋audit）に
対し、前提を 50 問で問い直し改善点を洗い出す。各問は実コードの観察に基づく。
末尾に優先度（P0 即修正 / P1 近く / P2 後）で総括。

---

## A. safety 係数とマジックナンバー（Q1–6）

**Q1.** tolerance の `safety=4.0` は全許容を決める最重要定数だが、根拠は？
→ ✅ **修正済**: `tsugi.constants.SAFETY` の docstring に根拠（√K ランダムウォーク 1σ に掛ける
~4σ ヘッドルーム・実機 noise で校正すべき初期値）を明記。

**Q2.** safety は 5 箇所に重複（tolerance/calibration/propagation 等）。一元化されているか？
→ ✅ **修正済**: `tsugi.constants.SAFETY` に集約し 5 箇所すべてがそれを参照（単一情報源・DRY）。

**Q3.** safety=4.0 は fp16/bf16/fp32 で同じでよいか？
→ dtype で丸めの統計的性質は変わりうる。safety は dtype 非依存だが *unit roundoff*（u）を
dtype 別に掛けることで実効許容は dtype 別になる（u: fp16 4.9e-4 / bf16 3.9e-3 / fp32 6e-8 /
**fp64 1.1e-16**）。✅ **一部修正済（外部調査ベース）**: PyTorch `torch.testing.assert_close` の
dtype 別デフォルト（fp16=1e-3 / fp32=1e-4 / **fp64=1e-8**）を参照根拠として明記。さらに
**fp64 が `UNIT_ROUNDOFF`/`TOLERANCE` 両方で欠落し float32 にフォールバック → 8 桁緩い偽OK 源**
だった欠陥を発見・修正（fp64 を両 dict に追加）。残: safety 自体の dtype 別チューニングは実機校正待ち。

**Q4.** 検出限界 `0.1*max_normal`（overflow 近接 WARN の 10%）の 0.1 は？
→ envelope の閾値。任意。**改善: 閾値群（0.1・0.7・1.5・0.5*thresh 等）を名前付き定数化。**

**Q5.** decision の `confident_k`/裾判定 `0.5 * overall_margin_median` の 0.5 は？
→ 「near-tie 裾に集中していない」判定の閾値が任意。**改善: 根拠か感度分析を添える。**

**Q6.** これらの定数群はテストで固定されているが、値を変えた時の挙動は誰が守る？
→ ✅ **修正済**: `test_systematic_threshold_is_sensitive_to_safety_constant` が定数
`SAFETY` から閾値 `thresh=SAFETY·u` を算出し、境界±で判定が反転すること（`1.01·thresh`→BLOCK /
`0.99·thresh`→WARN / `0.4·thresh`→OK）を固定。定数の値が判定境界を *実際に* 支配することを
証明し、silent な定数 drift を捕える番人になる。（残: 他閾値群へ同型の感度テスト拡張は P2）

## B. 静的な条件数の盲目（propagation）（Q7–12）

**Q7.** propagation の `cond` 既定 1.0 は何を意味するか？
→ 全 op を well-conditioned と仮定。**改善: これは propagation が存在する理由（ill-conditioned
増幅の検出）を既定で無効化している。最低限ドキュメントで強く警告し、cond 推定の道筋を示す。**

**Q8.** audit の `_graph_ops` は実カーネルから cond をどう得るか？
→ 得ない（常に 1）。✅ **修正済**: (1) `_AMPLIFYING` を *相対*増幅する op のみ（reduce/
softmax/exp）に是正（div/reciprocal/add は相対条件数 ~1 で増幅しないと実測確認）。
(2) `empirical_cond(sample,kind)` でデータ依存 cond を実測（reduce=Σ|x|/|Σx| の相殺・
exp=max|x|）。(3) audit は増幅 op があるのに静的 cond=1 を当てる時 *下界* と WARN し
empirical_cond/audit_runtime を案内（過小評価を隠さない）。

**Q9.** 現トレーサは reduce/exp/softmax を IR に出すか？
→ ✅ **修正済**: トレーサは `reduce`/`exp`/`sqrt`/`rsqrt`/`max` を含む全 14 op を IR に出す
（`tracer.EMITTABLE_OPS`）。`audit._graph_ops` が `_AMPLIFY_KINDS`（reduce/exp/rsqrt/div 等）を
実グラフから拾い、propagation 層が空回りせず増幅 op を算入する。

**Q10.** つまり「発散が深さで ~2000倍」の実証は audit 経路では再現されないのでは？
→ ✅ **修正済**: softmax カーネル（reduce×2/exp/div を出す）を trace → `audit()` に通す統合テスト
（`test_audit_propagates_amplification_through_traced_softmax`）。propagation 層が実グラフから
増幅 op を抽出し「静的 cond=1 は *下界*（過小評価）」と WARN、empirical_cond/audit_runtime へ誘導
することを固定（過小評価を隠さない）。

**Q11.** model_divergence は相対だが、各 op の scale 変化（正規化層）を追えているか？
→ 追っていない（amp≈1 と単純化）。**改善: scale を伝播する版（正規化で発散がリセットされる
効果）を検討。**

**Q12.** propagation は線形 op 列のみ。分岐・残差接続（transformer）は？
→ ✅修正済（2 段）: (1) 残差は `GraphOp(residual=True)` で δ_out=√(δ_in²+(amp·local)²) と
random-walk 希釈。(2) 一般のフォーク→合流は `propagate_dag(nodes, correlated=)` ＋
`merge_divergence`（√Σδ² / Σδ）で series-parallel DAG（attention 並列ヘッド・concat）を表現。
numpy の 2 ブランチ合流で実測発散を上界することを検証（test_propagation）。**残: 交差辺を
もつ一般 DAG（重み共有・cross-attention 往復）は SP 近似に留まる。**

## C. scale=1.0 という暗黙仮定（Q13–17）

**Q13.** certify_gemm/derive_tolerance/detectability_floor の `scale=1.0` 既定は現実的か？
→ 実テンソルの RMS は 1 でない。audit も `certify_gemm(K,"float16",1.0)` と固定。
**改善: audit が代表 scale を（与えられれば）使う・無ければ「scale=1 仮定」と明示。**

**Q14.** envelope は check 時に RMS scale を測るが certify は引数 scale。整合は誰が保証？
→ 呼び出し側任せ。ズレると認証 atol が無意味。**改善: certify に代表サンプルを渡して scale を
測る補助を用意。**

**Q15.** propagation→decision 橋は `scale=RMS(logits)`。logit scale と GEMM 出力 scale は同一か？
→ 別物（logit は最終層出力）。相対発散を logit に適用する近似の妥当域を要明記。
**改善: 橋の仮定（相対発散が最終 logit にそのまま乗る）を明文化し限界を書く。**

**Q16.** bf16 と fp16 で scale の効き（denormal 域）が違うのに一律 RMS でよいか？
→ envelope は dtype 別だが scale 推定は一律。**改善: scale 推定に dtype 別の下限（denormal）考慮。**

**Q17.** 全 scale 既定 1.0 は「とりあえず動く」値。テストも scale~1 のデータばかりでは？
→ ✅ **修正済**: `test_tolerance_tracks_scale_across_extremes` が scale=1e-3/1.0/1e3 で
(a) 導出 atol が scale に厳密線形追従、(b) 正当なクロスベンダー順序差が両極で過剰検出されない
（envelope/tolerance が data scale に追従）、(c) scale 比例の 1% 系統バグを fail-safe
（`is_equivalent_combined`）が両極で捕捉（RMS 比ゆえ scale 不変）を実証。

## D. 橋の分布仮定（propagation→decision）（Q18–22）

**Q18.** flip_bound は P(margin<2δ)。これは δ が「どの向きにも等確率」を暗黙仮定。系統発散では？
→ calibration が示した通り、実発散は系統的（相関）でありうる。系統バイアスは argmax を
一方向に押す → 上界が緩すぎ/きつすぎになりうる。**改善: 系統成分（バイアス）と乱雑成分を
分け、系統分は全 logit を平行移動（argmax 不変）、乱雑分のみ flip に効く、と精密化。**

**Q19.** δ_abs = δ_rel·RMS は最悪/平均どちら？
→ 平均的スケール。最悪サンプル（小 logit）では相対誤差が増幅。**改善: per-sample の δ を使う
（一律 RMS でなく |logit| 依存）版を検討。**

**Q20.** 「margin<2δ が必要」は十分でない（向きが要る）。上界は実測の何倍緩い？
→ 実測 5.8% vs 上界 18.7%（約 3.2 倍）。**改善: 期待値版（向きを確率で割引）を併記し、上界と
推定値の両方を返す。**

**Q21.** 代表 logit 分布はどこから来る？本番分布と違えば予測は外れる。
→ 呼び出し側が渡す前提。**改善: 「代表 logit はキャリブレーション集合から」とガイドし、
分布シフト時は再評価が要ると明記。**

**Q22.** top-1 argmax だけ見るが、top-k / beam / sampling は？
→ 非対応。**改善: top-k 一致・分布距離（KL）など task 多様性への拡張余地を明記。**

## E. torch.compile の楔が検証を届けていない（Q23–28）

**Q23.** 製品の入口は `torch.compile(model, backend="tsugi")`。今そこで検証は走るか？
→ ✅ **修正済**: `tsugi_torch.fxbridge.audit_fx` が FX グラフに propagation 静的監査を走らせ、
backend が codegen 前でも増幅 op・モデル発散を warn する（検証だけ先に届ける＝楔の早期価値）。
実行は引き続き eager 素通し。

**Q24.** backend は torch 無し環境で全くテストされていない。回帰は誰が捕まえる？
→ 誰も。**改善: torch を test extra に入れ、最小 FX グラフで backend 登録と素通しを検証。**

**Q25.** backend が eager 素通しなのにユーザーは「両ベンダー対応」と思い込まないか？
→ ✅ **修正済**: backend が `verification-only (no codegen yet)` を warn するようにした。

**Q26.** FX グラフ → `_graph_ops` の橋は無い。audit は traced tile IR 専用。二重路線では？
→ ✅ **修正済**: `fxbridge.fx_to_graph_ops` が aten op 名（addmm/bmm/softmax/mean/...）を
propagation GraphOp へ写す（duck-typed・torch 不要でテスト）。実 torch.fx 結線は torch 環境が
要る（本環境では stand-in で検証・実 FX は未検証）。

**Q27.** escape-hatch（cuBLAS/rocBLAS 委譲）時、その区間の等価性は誰が保証？
→ 委譲先はベンダー実装で発散源そのもの。**改善: escape-hatch 区間こそ audit_runtime の対象、
と設計に明記。**

**Q28.** backend 登録は import 副作用（自動 register）。テスト分離・冪等性は？
→ ✅ **修正済**（第11回）: `tsugi_torch._BACKEND_REGISTERED` フラグで冪等化。
二度目の `register()` 呼出しは即 return（torch 有り時）。

## F. タスクモデルの狭さ（decision）（Q29–32）

**Q29.** decision は分類 argmax 前提。回帰・生成・検出は？
→ ✅修正済（新視点11）: `compare_task(task=regression/binary/ranking)` 追加。
  回帰は `|a-b|>atol+rtol·|a|`、バイナリは threshold 跨ぎ、ランキングは top-k 集合変化。

**Q30.** margin = top1−top2 は多クラス前提。2 値 sigmoid（しきい値 0.5）では？
→ ✅修正済（新視点11）: `binary_margin(a, threshold=0.5)` = |a − threshold| を追加。
  near-tie（threshold 付近）での flakiness が見えるようになった。

**Q31.** flip は「正しさ」でなく「一致」を測る。両ベンダーとも同じく誤るケースは？
→ flip=0 でも両方間違いはある。**改善: oracle がある検証集合では accuracy 差も併記。**

**Q32.** 温度・top-p サンプリング下では同 logit でも出力トークンが確率的に違う。
→ 非対応。**改善: サンプリング下の「分布一致」（決定論的 argmax でなく）を将来課題に。**

## G. テスト/verify の構造（Q33–38）

**Q33.** verify.py の不変条件は test_*.py と大きく重複。二重保守では？
→ 重複多数。**改善: verify は「機械可読な主張のサマリ」と位置づけ、重複を意図的と明記するか、
test から不変条件を生成する。**

**Q34.** 単一巨大 main() の verify は失敗箇所の局所化が弱い。
→ 1 関数に全 check。**改善: セクション分割（既に番号付きだが関数化）。**

**Q35.** テストは固定 seed の単発。数値主張なのに fuzz/property test が無い。
→ ✅ **修正済**: test_properties.py に 10 性質 × 200 試行のゼロ依存 property 検査を追加
（derive_tolerance の K 単調性・residual≤total・flip_rate スケール不変・残差 bound は上界・
アフィン系統は無フリップ・attribute 領域・envelope overflow 等）。hypothesis 不使用（ゼロ依存）。

**Q36.** calibration corpus は 9 ケースの合成。「TRUSTWORTHY=偽OK 0」は統計的に弱い。
→ ✅ **修正済**: `roc_sweep(strengths, seeds)` でバグ強度を連続掃引し偽OK率を測る。合成判定は
系統閾値（≈safety·u）超で偽OK=0、max_abs 単独は一様スケールを吸収し広範囲で偽OK。閾値未満
（~0.2%）は合成判定でも見逃す残存盲点を *正直に* 露出（test で両側を固定）。

**Q37.** GPU 経路は全 SKIP。SKIP が緑に紛れ「検証済み」と誤読されないか？
→ ✅ **修正済**: run.py 末尾に SUMMARY（CPU suites PASS 件数 + SKIPPED 件数の列挙 + 「緑は CPU
検証可能範囲のみ」注記）を出す。GPU 未検証を緑に紛れさせない。

**Q38.** カバレッジ計測が無い。147 関数中どれが未実行？
→ 不明。**改善: coverage を CI で測り閾値ゲート。**

## H. 再現性・依存・CI（Q39–43）

**Q39.** numpy 版が未固定。noise_floor/RNG 挙動は版で変わりうる。
→ ✅ **修正済**: pyproject に `numpy>=1.22` を宣言（さらに発覚した packaging 欠陥も修正——
`tsugi` パッケージ自体が build 対象外で numpy 依存も未宣言だった。`include=["tsugi*"]`＋deps 追加）。

**Q40.** version は全変更を通じ 0.1.0 のまま。SemVer 運用は？
→ ✅ **修正済**: 0.2.0 に bump（pyproject + `__version__`）。CHANGELOG に `[0.2.0] 2026-06-16`
見出しを追加。0.x は API 未凍結である旨も明記。

**Q41.** .github/workflows は除外で CI が実際には回らない。「CI」主張は願望では？
→ ✅ **修正済**: CONTRIBUTING に「CI について（正直な現状）」節を追加——GitHub Actions は
無効で、当面の CI 代替は ローカル `run.py`（SUMMARY 付）+ `verify.py`。protected-main の
「CI 全通過」記述も実態に合わせて修正。

**Q42.** ライセンス/依存監査は手動。permissive 主張の自動チェックは？
→ ✅ **修正済**（commit 4605479）: `verify._declared_dependencies()`/
`_undocumented_dependencies()` が `pyproject.toml` の宣言済み依存を正規表現で抽出し
（`tomllib` は Python 3.10 で標準ライブラリに無いため未使用）、permissive ライセンス
許容リスト（`numpy`=BSD-3-Clause・`torch`=BSD-3-Clause 系）と照合する。新規依存が
リストに無ければ CI が落ちる。plant-and-detect の自己検証で検出器自体の動作を保証済み。
pip-licenses 等の外部ツール・ネットワークアクセス不要（CPU-only ポリシーと整合）。

**Q43.** 乱数依存テストの flake 耐性は？seed 固定だが境界ケースは脆い。
→ 一部境界（test_single_run_flaky は midpoint で堅牢化済）。**改善: 全乱数テストの境界余裕を点検。**

## I. API/意味の一貫性（Q44–47）

**Q44.** equivalence.EquivalenceReport は report.FindingReport を継承していない（統合の取り残し）。
→ ✅ **修正済**: 等価判定はスカラ計量で所見リスト型でないため FindingReport は継承せず（理由を
docstring に明記）、共通の判定インターフェース `risk`/`max_risk`/`ok` を備えて第一級レポート化。

**Q45.** AuditPhase.when は "static"/"runtime" だが audit_runtime は実データ層を "static" と付ける
（意味の二重定義）。
→ ✅ **修正済**: when を "decided"（verdict に算入）/"pending"（実機データ待ち）に改名し意味を
一致させた。`decided_phases` プロパティ（旧 `static_phases` は後方互換エイリアス）。

**Q46.** 多くの関数が関数内 import（遅延）。一貫していない（top と function 内が混在）。
→ ✅ **修正済**（`CONTRIBUTING.md` 「Import 方針」節）: 標準ライブラリ/numpy は
モジュール先頭、facade 層（audit.py 等）のサブモジュール間 import は「呼ばれない phase の
コスト回避」「循環 import 予防」の 2 点を理由に関数内遅延、他モジュール非依存の葉モジュール
（report.py・constants.py 等）はモジュール先頭でよい、という基準を明文化した。

**Q47.** report の severity は Risk(IntEnum) だが equivalence は bool(equivalent)。粒度不一致。
→ ✅ **修正済**: EquivalenceReport.risk が equivalent→OK / divergent→BLOCK を返す。全レポートが
`max_risk` を持つ統一インターフェースに。

## J. 主張の統計的厳密さ（Q48–50）

**Q48.** 「per-model 発散 ~2000倍」等の数字は単一 seed・単一構成。一般性は？
→ 例示値。**改善: 複数 seed の分布（中央値±）で示し、example でなく統計と明記。**

**Q49.** noise_floor は spread(max-min)。外れ値 1 つで過大評価では？
→ ✅ **修正済**: `measure_noise_floor`/`measure_batch_variance` が `spread_robust`（10-90
パーセンタイル幅）も返す。`compare_stable(robust=True)` で選択。実証: 測定グリッチ 1 個で
max-min は ~4 万倍に膨張するが robust 床は不変（偽BLOCK 化を防ぐ）。

**Q50.** これら検証層は「実機で正しい」ことをまだ一度も確認していない。CPU 擬似のみ。
→ 最大の未検証。**改善: 実機での最小 e2e（1 カーネルを両ベンダーで走らせ audit_cross_vendor）
を最優先の次マイルストーンに据える（lowering.py 実装が前提）。**

---

## 総括（優先度）

**P0（即修正・低リスク・高価値）**
- ✅ **Q9/Q10（本パスで修正済）**: tracer が reduce/exp/sqrt/rsqrt/maximum＋elementwise
  （sub/mul/div）を IR に記録するようにした。softmax がトレース可能になり、増幅 op が IR と
  audit の propagation グラフに現れる（perspective4 の実効化）。残: Q8（cond 推定で増幅の
  *大きさ* を出す）は P1。
- ✅ Q2/Q1（修正済）: safety を `tsugi.constants.SAFETY` に集約＋根拠を docstring に明記。
- ✅ Q45（修正済）: AuditPhase.when を decided/pending に改名（意味の一致）。
- ✅ Q37（修正済）: run.py に SUMMARY（CPU PASS 件数 + SKIP 件数 + 「緑は CPU 範囲のみ」注記）。

**P1（近く）**
- ✅ Q8（修正済）: 相対増幅 op の是正＋ empirical_cond（data-driven）＋ 静的下界の WARN。
- ✅ Q23/Q25/Q26（修正済）: torch backend が FX→GraphOp 写像で静的 audit を走らせ warn
  （検証だけ先に届ける）。実 torch.fx 結線は torch 環境が要る（stand-in で検証）。
- ✅ Q18/Q20（修正済）: 橋を系統/乱雑成分に分け残差ベース bound に（前イテレーション）。
- ✅ Q44/Q47（修正済）: equivalence に共通 risk/max_risk/ok インターフェースを付与。
- ✅ Q35/Q36（修正済）: property test 10×200・calibration を roc_sweep で ROC 化。

**P2（後・要設計/実機）**
- ✅ Q12（修正済）: propagation の DAG 対応（`propagate_dag`＋`merge_divergence`・series-parallel）。
  残 Q11: scale 伝播（正規化での発散リセット）と一般 DAG の交差辺。
- ✅ Q29/Q30（修正済）: 新視点11 `compare_task(task=regression/binary/ranking)` で argmax 外のタスクを対応。
- ✅ 新視点12（修正済）: `tsugi.attribution` — per-layer 発散 prefix scan。`layer_divergences`/
  `find_onset`/`find_spike`/`attribute`/`bisect_onset`。O(L) 手動デバッグ → O(log L)。
  propagation.dominant（理論）と attribution.spike（実測）の照合で理論と実験の接続点を提供。
  tests/correctness/test_attribution.py 27 テスト、verify.py invariants 27-28（77/77）。
- ✅ 新視点13（修正済）: `tsugi.blame` — ベンダー責帰。`accuracy_relative`/`compare_accuracy`/
  `layer_blame`。oracle との dist_a / dist_b を比較し「どちらのベンダーを修正するか」を特定。
  attribution（どの層）と組み合わせて "layer X の vendor Y を直せ" という完全診断チェーンを完成。
  oracle_check（shared mode 検出）と相補的。tests/correctness/test_blame.py 19 テスト。
  verify.py invariants 29-30（82/82）。
- Q29–32: decision の task 多様性（回帰/生成/サンプリング）。
- Q50: 実機 e2e（最重要だが GPU 必須）。

最大の構造的発見: **B 群（Q9/Q10）— audit に流れる実グラフは単一 matmul なので、目玉の
propagation（per-kernel⇏per-model 増幅）が統合経路では発火していなかった。** さらに調べると
tracer が reduce/exp 等を IR に *全く記録していなかった*（softmax/rmsnorm はトレース不能）
ことが根因。**本パスで tracer を拡張して修正済**（softmax がトレースでき、増幅 op が audit に
流れる）。増幅の *大きさ*（cond 推定・Q8）が次の最優先。

---

## G. 過不足（excess/deficiency）を機械的に探す（Q51–56・第11-19回まとめ）

第11-18回で「機能は実装済みだが facade（audit.py）に未接続」という同型の欠陥を7件
発見・修正した（certify_from_sample・empirical_cond・robust noise floor・compare_task・
attribution.diagnose・worstcase.analyze_worst_case・equivalence.classify_divergence）。
第19回はこれを「不足（未接続）」でなく「過剰（無接続のまま放置された実装コスト）」の
視点から問い直した。

**Q51.** 「実装されているが呼ばれない」関数は全て同じ問題か？
→ いいえ、2 種に分かれる。(a) テストからは呼ばれるが facade からは呼ばれない
（＝意図された public API だが接続漏れ・「不足」側）。(b) テストからも一切呼ばれない
（＝真のデッドコード・「過剰」側、実装コストを払ったのに一切価値を届けていない）。
機械的スキャン（各モジュールの `def` 名を全ソース・全テストと照合）でこの 2 種を分離できる。

**Q52.** 実際に (b) 型の完全デッドコードは存在したか？
→ ✅ **修正済**（第19回）: `rollout.divergence_step_quantile`（初回発散ステップの q 分位）が
唯一の完全デッドコードだった。`analyze_rollout` も対応する `RolloutReport` もこれを一切
参照していなかった。

**Q53.** 見つかった過剰関数は削除すべきか、接続すべきか？
→ ケースバイケース。今回は削除でなく接続を選んだ: `divergence_step_quantile(p, 0.5)`
（中央値）は `expected_divergence_step(p)`（平均=1/p）と対になる統計的に意味のある値
だったため。単なる実験の残骸なら削除が正しい判断になる。

**Q54.** なぜ平均だけでなく中央値も要るのか？
→ 初回発散ステップは幾何分布に従い右に裾を引く。平均(1/p)は右裾の少数の長生存 run に
引っ張られ、中央値(≈ln2/p)より系統的に大きい（p=0.01 で平均100・中央値69）。平均だけの
報告は「典型的にはもっと長く保つ」と楽観視させる —— fail-safe の観点では危険な省略。
✅ **修正済**（第19回）: `analyze_rollout`/`RolloutReport` に `median_step` を追加、
`to_text()` に両方を表示。

**Q55.** 他にも「歪んだ分布を単一点推定だけで報告している」箇所はないか？
→ ✅ **修正済**（第20回）: `nondeterminism.measure_noise_floor` は `spread`（max-min）と
`spread_robust`（10-90パーセンタイル幅）の両方を返す（Q49）。`calibration.check_systematic`
の `bias` も単一点推定のままだったが、`systematic_divergence_stderr`（ブートストラップ）を
追加し `bias_upper_bound = |bias| + stderr` で判定するよう修正（rollout.flip_rate_upper_bound
と同じ「点推定でなく上側限界で判定」パターン）。N=4 の小テンソルで 1 要素だけ 5% 摂動させると
bias 点推定はたまたま極小（旧ロジックなら OK）になるが、上側限界は閾値を大きく超え正しく
BLOCK になることを実証（偽OK 修正）。大 N（典型的な GEMM 出力）では stderr が無視できるほど
小さく挙動は不変（回帰なし）。

**Q56.** この機械的スキャン手法自体をどう維持するか？
→ 現状は手動実行（第18・19回のように都度 Python ワンライナーで scan）。
**改善: verify.py に恒常的な invariant として組み込み、新しい「意図せぬデッドコード」や
「facade 未接続」を CI で検出する。** ただし false positive（module-private helper）の
除外リスト維持が必要になるため、費用対効果を見て判断（未着手・P2）。

**Q57.** Q55 の「点推定でなく上側限界」パターンは calibration 以外にも波及するか？
→ ✅ **修正済**（第21回）: `decision.predicted_flip_bound(ref_logits, delta)` = P(margin<2δ)
も、代表 logit n 件からの点推定に過ぎなかった。n が小さい代表集合ではたまたま
margin<2δ 該当が 0 件でも、母集団の真の確率は 0 でない（rollout.flip_rate_upper_bound
の rule-of-three と同じ問題）。0 件観測を「フリップ率 0%」と過信するのは偽OK の温床。
`rollout.flip_rate_upper_bound`（Wilson 上側限界・既存実装を再利用）へ委譲するよう修正。
n=20 の代表集合で margin<2δ が 0 件のケースで、旧ロジックなら bound=0.0（過信）だったのが
新ロジックで bound≈0.12（正しく不確実性を反映）になることを実証。n が大きい（数千件）
既存テスト・property test（200 試行）はすべて無回帰で通過（大 N では Wilson 上限が点推定に
収束するため）。`flip_bound_from_divergence`（propagation→decision 橋）にも自動波及。
残る同型候補: `decision.regression_flip_rate`/`binary_flip_rate`/`ranking_flip_rate` も
n 件のサンプルからの単純比率（点推定）。小標本での過信リスクは同様に残る（未着手・P2）。

**Q58.** Q57 は `predicted_flip_bound`（第2ベンダー実行前の *予測*）を直したが、
`compare_decisions` 自体の予算判定（第2ベンダーを *実際に* 走らせた後の主判定）は
どうか？
→ ✅ **修正済**（第22回）: `compare_decisions` の BLOCK/WARN 判定は観測 `flip_rate`
（点推定 k/n）を直接 `flip_budget` と比較していた——これは `decision` モジュールの
主判定であり `predicted_flip_bound` より製品経路として遥かに中心的。n が小さい評価
バッチ（例 n=30）でたまたま観測フリップが 0 件でも、母集団の真のフリップ率が予算を
超えている可能性は排除できない。`DecisionReport.flip_rate_ub`（Wilson 上側限界・
`rollout.flip_rate_upper_bound` を再利用）を追加し、予算判定はこちらを使うよう修正。
n=30 で観測フリップ 0 件・真の率が予算超のケースで、旧ロジックなら OK（偽OK）だったのが
新ロジックで WARN 以上に正しく倒れることを実証。既存の大 N テスト（n=2000-4000）は
すべて無回帰で通過（Wilson 上限は大 N で点推定に相対誤差 20% 未満で収束）。
