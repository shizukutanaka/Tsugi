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
→ ✅ **修正済**: `envelope._OVERFLOW_WARN_FRAC`/`_EXP_WARN_FRAC`/`_SCALE_BLOCK_RATIO` として
名前付き定数化済み。`test_envelope_thresholds_are_sensitive_to_their_constants`
（`tests/correctness/test_envelope.py`）が Q6 の `SAFETY` 感度テストと同型の境界±固定
（閾値直上/直下で判定が実際に反転することを実証）を追加し、値が判定境界を支配することを
silent drift から守る。

**Q5.** decision の `confident_k`/裾判定 `0.5 * overall_margin_median` の 0.5 は？
→ ✅ **修正済**: `decision._NEAR_TIE_MARGIN_FRAC` として名前付き定数化済み。
`test_near_tie_threshold_is_sensitive_to_its_constant`（`tests/correctness/test_decision.py`）
が同型の境界±固定を追加。根拠自体（0.5 という値の妥当性）は依然経験則だが、
値を変えた時に判定が実際に反応することは機械的に保証されるようになった。

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
→ ✅修正済（3 段）: (1) 残差は `GraphOp(residual=True)` で δ_out=√(δ_in²+(amp·local)²) と
random-walk 希釈。(2) 一般のフォーク→合流は `propagate_dag(nodes, correlated=)` ＋
`merge_divergence`（√Σδ² / Σδ）で series-parallel DAG（attention 並列ヘッド・concat）を表現。
numpy の 2 ブランチ合流で実測発散を上界することを検証（test_propagation）。
(3) **FEATURE-AUDIT A-12 Round 1/2**: `audit()` の `_graph_ops` が SSA の use-def
（`Op.operands`/`Op.result`）からフォークを検出し `propagate_dag` に接続。従来は
実グラフを線形化し `propagate_dag` は audit から一度も呼ばれていなかった（テスト/verify 専用）。
Round 1 は恒等路つきフォーク（residual・softmax の `row-reduce(row)` 再利用）→`[[], branch]`。
Round 2 は恒等路の無い計算 2 分岐（attention ヘッド和・row を exp/reduce の 2 経路が消費し
add で合流）→`[[A], [B]]`。偽OK 対策として `audit()` は `propagate_dag(correlated=True)`
（合流を線形和で合成・並列分岐の過小評価を防ぐ）を使い、DAG 発散が線形版を下回らないことを
保証。Round 3 で 2 分岐限定を撤廃し N（≥2）分岐へ一般化（`dot(a,b,acc)` の 3 operand 合流を実証・
不変条件 68）。検出できない形は線形（保守側）に落とす。verify.py 不変条件 63/64/68。
**残: 交差辺をもつ一般 DAG（重み共有・cross-attention 往復）のみ（SP 表現の設計上の限界）。**

## C. scale=1.0 という暗黙仮定（Q13–17）

**Q13.** certify_gemm/derive_tolerance/detectability_floor の `scale=1.0` 既定は現実的か？
→ 実テンソルの RMS は 1 でない。audit も `certify_gemm(K,"float16",1.0)` と固定。
**改善: audit が代表 scale を（与えられれば）使う・無ければ「scale=1 仮定」と明示。**

**Q14.** envelope は check 時に RMS scale を測るが certify は引数 scale。整合は誰が保証？
→ 呼び出し側任せ。ズレると認証 atol が無意味。**改善: certify に代表サンプルを渡して scale を
測る補助を用意。**

**Q15.** propagation→decision 橋は `scale=RMS(logits)`。logit scale と GEMM 出力 scale は同一か？
→ ✅**修正済（A-8 解消）**: `audit(ref_logits=)` の propagation phase が橋の仮定を *レポートに
明示* する（docstring だけでなく出力に出す＝このプロジェクトの「暗黙化しない」慣例）。
δ_rel（op グラフ相対発散）を最終 logit にそのまま乗せる近似の妥当域——正規化の scale
リセット・最終射影の条件数（logit scale ≠ GEMM 出力 scale）・分布シフト——を明記し要再評価を促す。
verify.py 不変条件 67。

**Q16.** bf16 と fp16 で scale の効き（denormal 域）が違うのに一律 RMS でよいか？
→ ✅**修正済（A-8 一部解消）**: `check_tensor` が denormal を *率* で区別する。従来は非ゼロ
最小値が dtype の `min_normal` 未満なら（＝単一の denormal 値でも）一律 WARN だったが、
「偶発的な単一 denormal」と「値の大半が denormal（scale が dtype に対し小さすぎ＝認証 atol の
前提が崩れる）」を denormal 率（`_DENORMAL_FRAC_WARN`=1%）で区別し、後者に rescale/再認証を
促す強い警告を出す。verify.py 不変条件 66。残: Q15（橋の仮定明文化）。

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
→ ✅ **修正済**: `flip_bound_from_divergence` が per-sample RMS とグローバル RMS の
**max** を δ_abs に使う（`derive_tolerance` の max(derived, noise_floor) と同じ保守側
パターン）。グローバル RMS のみだと、低スケール多数派に紛れた高スケール near-tie
サンプルの δ が過小評価され margin<2δ を満たさず偽OK になる——これをテストと
verify.py 不変条件 61 番で固定。`predicted_flip_bound` は delta のスカラ/配列両対応に。

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
→ ✅修正済（A-9 一部解消）: `audit_runtime(logits_oracle=)` が各ベンダーの判断誤り率
（`decision.flip_rate` を oracle 相手に再利用）を併記し、A↔B が一致（低フリップ率）でも
両方 oracle 判断と食い違えば task レベル shared-mode として WARN する（tensor レベルの
`detect_shared_mode` の task 版）。flip=0 でも両方誤りを見逃さない。verify.py 不変条件 65。

**Q32.** 温度・top-p サンプリング下では同 logit でも出力トークンが確率的に違う。
→ 非対応。**改善: サンプリング下の「分布一致」（決定論的 argmax でなく）を将来課題に。**

## G. テスト/verify の構造（Q33–38）

**Q33.** verify.py の不変条件は test_*.py と大きく重複。二重保守では？
→ ✅**修正済**: `verify.py` のモジュール docstring に両者の役割分担を明文化した。
tests/correctness は *挙動* を網羅 exercise するスイート、verify.py は製品の *主張* を
名前つき不変条件で列挙した *機械可読なサマリ（契約）*。重複は冗長でなく
belt-and-suspenders（テストが挙動を、verify が「その挙動が意図した保証である」ことを
別レイヤで主張）。test から自動生成する案は却下——「テストのどれが *約束* か」の人間の
選別が失われるため、手書きの主張リストとして維持する（重複を *意図的* と明記）。

**Q34.** 単一巨大 main() の verify は失敗箇所の局所化が弱い。
→ ✅ **修正済**: 61 セクションをテーマ別の 12 個の `_check_*()` 関数
（禁止パターン／柱／calibration／audit facade／伝播と床／レポート診断／dtype 3 表／
非決定カタログ／後期 facade 接続／統計的厳密さ／形状ガード／メタ整合性）に分割。
`main()` は順に呼ぶだけの薄い関数。挙動・実行順・check 文言・件数は分割前と
一字一句同一（リファクタ前後の出力 diff で機械的に確認）。

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
→ ✅**修正済**: `coverage_report.py`（標準ライブラリのみ・`sys.settrace` の call イベント）で
**関数カバレッジ 238/256 = 93.0%** を計測。coverage.py を使わないのは実行時依存を numpy のみに
保つため（不変条件 60 の前提を動かさない）。行でなく関数単位にしたのは、本プロジェクトが
繰り返し踏んだ欠陥が「実装済みだが誰からも呼ばれない」型（B-1 で 11 件）だから。
不変条件 57（*静的* な facade 未接続スキャン）に対し本ツールは *動的*（実行されたか）で相補的。
**実際にギャップを発見**: `tile.sqrt`/`rsqrt`/`maximum` は `EMITTABLE_OPS` にあり「対応済み」と
主張されていたのに、どのテストからも一度も実行されていなかった →
`test_sqrt_rsqrt_maximum_are_traced_and_lowerable` を追加して解消。

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
→ ✅**修正済**: 懸念を *仮定でなく実測* で潰した。中核判定（`is_equivalent_combined`）の
検出境界を多数 seed で掃引すると、境界は理論値 **SAFETY·u**（fp16 で ~0.195%）に一致し、
その ±1% で判定が **全会一致**（下側=全 seed が等価／上側=全 seed が非等価）になる——
つまり判定は seed 非依存でバグ強度のみに支配される。よって既存の固定 seed テストは
「たまたま通っている」のではないことが機械的に保証された。
test: `test_detection_verdict_is_seed_independent_at_safety_times_u`・verify.py 不変条件 71。
（Q6 の「定数 SAFETY が境界を支配する」を seed 横断に一般化した形でもある。）

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
→ ✅**修正済**: 60 seed で再測し、`docs/PERSPECTIVE-error-propagation.md` の表を
**中央値＋p10-p90** に置き換えた。結論（1 層で無視できる発散が 12 層で 2 桁以上に育つ）は
seed に依らず頑健だが、**倍率自体は seed で 2 倍以上ばらつく**（中央値 ≈2,400 倍・
p10-p90 = 1,600-3,300 倍）——「約 2000 倍」は代表値であって精密な定数ではないことを明記。
test: `test_depth_amplification_is_stable_across_seeds`（15 seed で結論の頑健性・
中央値のレンジ・ばらつきの存在の 3 点を固定）。

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
  （検証だけ先に届ける）。実 torch.fx 結線も検証済み（第 60 回・torch 不在時は skip）。
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

## K. ゲートは本当に全部を走らせているか（Q59–Q63・第61回）

第 59-60 回で codegen を L2 まで到達させ、想定ユーザーの経路も機械語まで繋いだ。
そこで「では**検証基盤そのもの**は健全か」を問い直したのが本節。長所として掲げてきた
「不変条件 200 件超」という *厚み* が、実は *深さ* を保証していなかった。

**Q59.** `python check.py` が緑のとき、ディスク上のテストは全部走っているか？
→ ❌ **走っていなかった**（第61回で発見・修正済）。`tests/correctness/run.py` の
`cpu_suites` に **`test_attribution.py` / `test_blame.py` / `test_worstcase.py` /
`test_tsugi_torch_compile.py` の 4 本が入っていなかった**（ディスク上 29 本中 4 本＝14%）。
つまり **attribution（新視点12）・blame（新視点13）・worstcase（能動探索）・
torch.compile backend は、ゲートが緑でも一度も実行されていなかった**。
4 本とも単体では PASS するので「腐っていた」のではなく「ゲート外にあった」。

原因は検査が **一段浅かった**こと。`verify.py:_orphan_tests()`（不変条件 56）は
「各 `test_*` 関数が *そのファイルの* `main()` に登録されているか」を見ていたが、
「*そのファイル自体が* `run.py` のスイート表に載っているか」を問うていなかった。
関数→ファイルの階層は見て、ファイル→ゲートの階層を見ていない。
→ 4 本を登録し、`_orphan_tests()` にファイル単位の検査を追加（不変条件 106）。

**教訓（本ラウンド最大）: 不変条件の *数* は検査の *深さ* を保証しない。**
「検査を足す」文化は、検査自身の階層に穴があると空回りする。数を誇る長所欄には
この限界を併記すべきで、ASSESSMENT の長所 2 を弱めて短所 7 を新設した。

**Q60.** 緑は「全部検証した」を意味するか？ GPU 側は Q37 で塞いだが CPU 側は？
→ ❌ **CPU 側に同型の穴が残っていた**（修正済）。torch が無い環境では意味論照合と
実 FX 結線が丸ごと skip されるのに、サマリは `CPU suites: PASS` としか言わなかった。
`[SKIP]` 行を数えて「環境起因の skip: N 件」を SUMMARY に出し、`check.py` の 1 行
サマリにも透過する。**実行されなかった検査を緑に含めて読ませない**（不変条件 106）。

**Q61.** ゲートは互いに独立した別プロセスなのに、なぜ直列に走らせているのか？
→ 部分的に **並列化した**（第61回）。ただし**素朴な結論は実測で覆った**:
`run.py` 内部の 29 スイートを 4 並列にすると 28s → 15.6s と効いたが、その上で
`check.py` の 4 ゲート（lint/suites/invariants/smoke）も並列にしたところ
**40s → 104s と大幅に悪化した**——4 コア環境でスイート自身が既に 4 ワーカーを
使っており、そこへ verify.py と smoke 3 本を重ねると oversubscription で全員が遅くなる。
**並列化すべき階層は 1 つだけ**だった。効かなかった部品は入れずに戻した
（Musk 第 2 段階）。結果 **40s → 26s**、かつスイートは 25→29 本に *増えて* いる。

**Q62.** 第 60 回で doc の *API 参照* の腐りを不変条件 104 で止めたが、
doc の *数値主張* はどうか？
→ ❌ **6 件すべて腐っていた**（修正済）: 不変条件件数 190→実 206、CPU スイート 23→29、
モジュール 30→34、docs 29→32、verify.py 1,400 行→倍近く、ゲート 17〜18s→実測 40s。
API 参照だけ検査して数値主張を素通ししたのは、**同じ種類の腐りに対する検査の穴**。

**Q63.** ではその 6 件も機械照合すべきか？
→ ❌ **違う。数える価値のない数値は文書から削るのが正解**（Musk 第 2 段階
「最良の部品は無い部品」）。行数・秒数・docs 本数・モジュール数は保守コストだけ高く
情報価値が薄いので文章から消し、所要時間は `python check.py` が毎回実測を印字する形に
変えた。**不変条件の総件数も照合しない**——`len(INVARIANTS)` は検査中の途中値で
最終値と 1 ずれる（自己参照）ため、**正しく書けない数値は書かない**のが筋。
機械照合するのは **CPU スイート数ただ 1 つ**（実際に欠陥が出た唯一の数値であり、
かつ安定して定義できる）。不変条件 107。検査する数値を減らすほど検査は信頼できる。

**Q64.** Q59/Q60 は *開発ゲート* の「緑は何を意味するか」を狭めた。では
*製品のレポート* は同じ規律に従っているか——利用者は「何が検査されなかったか」を
知れるか？
→ ❌ **知れなかった**（第61回で発見・修正済）。このプロダクトは「13 検証層」を掲げるが、
`python -m tsugi k.py` の既定出力に現れるのは **5 層**で、残り 10 層は *実データが
無いので走っていない*。レポートはその不在を一切告げないので、利用者は
「移植可（要注意点あり）」を **全層を通した判定** と読む。開発ゲートで塞いだ穴と
**まったく同じ偽OK の類型が製品側に残っていた**。

→ `audit.LAYER_CATALOG`（層 → その層を走らせるのに要るもの）を持ち、実行時
チェックリストに **走らなかった層を名指しし、何を渡せば走るかまで書く**:

```
この監査で走らなかった層: 11/15（下記は *検査していない* ——判定に含まれない）
  equivalence: 未実行 — 実機データ: 両ベンダーの出力 a_out/b_out
  worstcase:   未実行 — 実機データ: 実行可能なカーネル fn_a/fn_b
  correctness: 未実行 — oracle: 真値（一致≠正しさ・共有モード障害の検出）
  …
```

さらに掃くと **3 つの入口すべて**に規律が要ると分かった。`audit()` にだけ付けても
`audit_runtime()`（実データ）と `audit_torch()`（楔ユーザー）は無言のまま——実際
`audit_runtime(a, b, K)` は **15 層中 1 層しか走らない**のに何も告げていなかった。
**実データを渡した入口ほど「徹底的に調べられた」と読まれる**ので最も危ない。
`_coverage_phase()` を単一定義にして 3 入口が共有し、フェーズとして持つので
`to_dict()` にも載る（CI が被覆をそのまま機械可読に読める）。

不変条件 108 が「未実行の層を名指しすること」「走った層を未実行と誤報しないこと」
「3 入口すべてが開示すること」を固定する。
**教訓: 検査の規律は開発ゲートと製品出力の両方に、かつ全入口に適用しないと片肺になる。**

なお既存テスト `test_runtime_phase_excluded_from_verdict` が「pending フェーズは 1 つ」と
*件数* を固定しており、被覆フェーズ追加で落ちた。固定すべきは件数でなく
**「pending は verdict に算入されない」という契約**なので、テストの意図を書き直した
（`test_pending_phases_excluded_from_verdict`）——**偶然を契約として固定しない**。

**Q65.** 被覆を報告するようになったが、**その目録自体**は完全か？
→ ❌ **漏れていた**（第61回で発見・修正済）。`LAYER_CATALOG` に載っていない層が出ると、
被覆計算はそれを「走った」とも「走らなかった」とも数えず **黙って落とす**。実際
`torch/FX 静的監査`——**楔ユーザー経路の中核をなす層**——が目録から漏れており、
その経路の被覆が過小に報告されていた。→ 目録に追加し、**3 入口が出しうるフェーズ名が
すべて目録か `META_PHASES` に属する**ことを不変条件 109 で固定した。

これで本ラウンドは *同じ形の欠陥を 4 段* 見つけたことになる:
関数→ファイル（Q59）→ ファイル→ゲート（Q59）→ ゲート→製品出力（Q64）→
全入口（Q64 追補）→ **目録そのもの**（Q65）。
**教訓: 「検査を足す」たびに「その検査自身は誰が検査するのか」を一段問う。**

**Q66.** では層数の食い違い（文書 7 箇所が「13 検証層」・実体 15 層）にも
検査を足すべきか？
→ ❌ **足さない**（Q63 の原則の適用）。散文から層数を消し、`audit.LAYER_CATALOG` を
単一情報源にした。禁止用の正規表現を足すと、**欠陥を説明した歴史的記述**（「13 と
書いてあったが実体は 15 だった」）まで誤検出し、除外リストの保守が始まる。
**誤報の多いゲートは無視される**ので、検査を足すより数値を消す方が強い。
足したのは「目録が実在の層を漏らさない」という*構造*の検査（Q65）だけで、
これは数値でなく集合の包含関係なので腐らない。

---

## L. 製品が出す数値そのものを問う（第62回）

第61回までは「検査を検査する」階層を掘った。今度は視点を変え、**楔ユーザーが実際に
受け取る 1 行**を読んでみた。

**Q67（最重要）.** `audit_torch` が返す `model_divergence` / `task_flip_bound` は
**予測**か、それとも**天井**か？
→ ❌ **天井だった**（第62回で発見・修正済）。同じモデルを CPU で「2 ベンダー」として
走らせて実測すると、静的値は最悪クラスの実測の **約 200 倍**、典型（累積順序差）の
**1000〜1700 倍**大きい:

| model | 静的 δ | 予測 flip | 実測 δ 典型 | 実測 δ 最悪クラス | 実測 flip |
|---|---|---|---|---|---|
| MLP64 | 9.0e-2 | 35.5% | 5.2e-5 | 4.4e-4 (f16acc) | 0.00% |
| MLP+LN+GELU | 3.6e-2 | 24.3% | 2.2e-5 | 4.6e-4 (f16acc) | 0.00% |
| deep8 | 2.4e-1 | 66.3% | 3.4e-4 | 9.8e-4 (f16acc) | 0.00% |

乖離は調整不足でなく**構造的**である。伝播モデルは格納 dtype の `u(fp16)` を発散単位に
するが、両ベンダーが f32 で累積するなら跨ベンダー差は `u(f32)` スケール——2¹³≈8192 倍
小さい。よって静的値は「これ以上は絶対に出ない」という**許容の天井**であり、
**予測ではない**。→ 表示を「許容の天井」に改め、判定は実測に移した（不変条件 110）。

**Q68.** 天井を「予測」と呼ぶことの何が悪いのか。真な上界ではあるのに？
→ 真だが**無情報**である。楔ユーザーには毎回「判断フリップ率 ≤ 24〜66%」と出る。
どのモデルでも出るのだから、モデルを区別しない＝情報量がゼロで、しかも既定予算
（0.1%）を必ず超えるので**常に BLOCK**。この製品は一貫して「偽OK ≫ 偽BLOCK」を
掲げてきたが、その裏側を見ていなかった——**偽BLOCK が常態化すると、警告は読まれなく
なり、偽OK と同じく判定が信号を失う**。fail-safe は「常に赤」と同義ではない。

**Q69.** なぜ 61 回も気づかなかったのか？
→ **出力を読む人の視点で検査していなかったから**。不変条件 89 は「BLOCK は利用者の
`flip_budget` 超過のときだけ」という*契約*を固定していたが、その BLOCK が**いつも**
出ることは誰も問うていない。契約の検査は「間違った BLOCK を出さない」を保証するが、
「意味のある BLOCK を出す」は保証しない。→ 天井と実測の比を不変条件 110 に固定した
（比が 10 倍未満に縮んだら、伝播モデルか文書のどちらかが変わったということ）。

**Q70.** では模倣は実機の代わりになるのか？
→ ❌ **ならない**。模倣が含むのは既知の 4 クラス（累積順序・f16 累積・TF32 入力・RTZ
丸め）だけで、実機固有の要因（超越関数の実装差・FMA contraction の有無・レイアウト
依存の縮約順）は含まない。したがって**実測は実機発散の下界、静的天井は上界**であり、
真値はその間にある。両方を報告し、`audit_cross_vendor` による実機照合の必要性は
一切下がらない——そう `SimulationReport.to_lines()` に毎回書かせる。

**Q71.** 実測フリップ 0 件・n=256 のとき、Wilson 上界は 1.05% で予算 0.1% を超える。
これは BLOCK か？
→ ❌ **BLOCK にしない**（WARN にする）。上界超過が意味するのは「壊れている」ではなく
「**標本が足りない**」である。それを BLOCK にすると n=256 の全モデルが赤くなり、
Q68 で直したはずの偽BLOCK 常態化に逆戻りする。深刻度を分け、WARN 側には
**要求標本数**（この予算なら 0 フリップで n≥2703）を添えて「どうすれば通るのか」を
示した。要求数は rule of three の近似定数 3/budget を書き込むのでなく、実装済みの
`flip_rate_upper_bound` を二分探索で反転して求める（`samples_for_flip_budget`）——
**未検証の数値定数を導入しない**という既存の規律をここでも守る。

**Q72.** 模倣を製品経路（`torch.compile`）へ結線したとき、報告される標本数が
入力の行数と合わない。なぜか？
→ ❌ **代表入力が重み行列だった**（第62回で発見・修正済）。dynamo は重みを引数へ
持ち上げるので `example_inputs[0]` は `nn.Parameter` である。A-3 以来「sample 実測:
scale=…」は活性でなく**重みの統計**を報告していた。同じ根で模倣の束縛も壊れ、代表入力
1 本を全記述子へ配ると重みの位置に活性が入る。**実測と称して別のものを測るのは、
静的仮定を残すより悪い**——「(cond=1 lower bound)」なら利用者は下界だと分かるが、
「実測 scale=0.0718」は活性の実測だと読まれる。→ `nn.Parameter` を型で除いて活性を
選び、束縛は位置対応が取れるときだけ行い、取れなければ理由を述べて諦める
（不変条件 111）。**教訓: 「実測へ移す」ときは、測っている対象が本当にそれかを問う。**
これは第 59-60 回の「使う人が最初に打つコマンドを本物の依存で通す」規律の続きであり、
本ラウンドも *結線してみて初めて* 見つかった——スタンドインのグラフでは重みが
持ち上がらないので、この欠陥は永久に現れなかった。

**Q73.** 書いたばかりの模倣に同じ問いを向ける。**4 クラスのうち 2 つが
「相対発散 0.00e+00」**と報告している。これは「差が無い」のか？
→ ❌ **「その差が表現できない」だった**（第62回で発見・修正済）。`tf32` は入力を
10 仮数へ丸めるが fp16 の仮数はもともと 10 bit なので丸めが恒等になり、`rtz` は
`input_precision="ieee"` のままで `truncate_to_tensorcore` が仮数 23 bit で即 return
するため丸めモードの指定ごと捨てられていた。0 と出せば読み手は「TF32 起因の発散は
無い」と読む——**偽BLOCK を直しに来て偽OK を作っていた**。しかも隠れていたのは
最小のクラスではなく **最大のクラス**で、正しい f32 格納で測ると `rtz` の発散
（9.3e-4）は f16 累積（4.6e-4）を上回る。→ クラスごとに発現する格納 dtype を固定し、
測っていない／ビット同一だったクラスは 0 でなく「非適用＋理由」として名指しする
（不変条件 112）。**教訓: 新しい計測器を入れたら、その計測器に対して同じ問いを
向ける。「測って 0」と「測れないから 0」は表示上 区別がつかない。**

**Q74.** 「最悪クラス」はどう選ぶべきか。フリップ上界の最大でよいか？
→ ❌ **不足**。小標本ではフリップが全クラス 0 になり、Wilson 上界が同値で並ぶ。
`max` は同値のとき登録順の先頭を返すので、「最悪クラス」として実際には最小の
`order` が名指しされていた。相対発散で決着させる。**同値が起きうる比較で
tie-break を書かないのは、順序の偶然を結果として報告するのと同じ**。
