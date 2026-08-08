# Tsugi 機能過不足の台帳（FEATURE AUDIT）

このドキュメントは Tsugi の機能を「不足（あるべきだが無い・未接続）」「過剰（実装したが
価値未提供）」「適正（十分に機能している）」に選別した台帳である。**前提知識ゼロの
読者（人間・AI を問わず）が、このファイルだけを読んで作業を引き継げる**ことを目標に書く。
会話履歴やセッション文脈への参照は置かない。根拠が必要な箇所は commit hash を付す。

## 用語（この台帳を読むのに必要な最小セット）

- **Tsugi**: PyTorch `torch.compile` 向けの GPU ベンダー間（NVIDIA↔AMD）移植性検証
  ライブラリ。数値等価性・起動可能性・タスク影響を GPU 実機なしの CPU シミュレーションで
  検証する。コードは `python/tsugi/`（30 モジュール）と `python/tsugi_torch/`（2 モジュール）。
- **facade**: 製品の主要入口となる統合関数。`python/tsugi/audit.py` の
  `audit()`（静的検証）・`audit_runtime()`（実データ検証）・`audit_cross_vendor()`（実機入口）、
  および `python/tsugi_torch/__init__.py` の `_tsugi_compile()`（torch.compile バックエンド）。
  **個別モジュールに機能があっても、facade から呼ばれていなければユーザーには届かない。**
- **偽OK / 偽BLOCK**: 偽OK = 発散を等価と誤判定（silent に誤りを出荷・検出不能・最も致命的）。
  偽BLOCK = 等価を発散と誤判定（開発者が気づける・回復可能）。コストは非対称。
- **fail-safe**: 不確実なら BLOCK 側に倒す設計原則。偽OK の温床（点推定の過信・暗黙の既定値）
  を潰すことがこのプロジェクトの一貫した改善軸。
- **Risk**: 全レポート共通の深刻度。`OK < INFO < WARN < BLOCK`（`python/tsugi/report.py`）。
- **検証基盤の規模**: `verify.py` に 165/165 の機械検証可能な不変条件。
  `tests/correctness/` に 27 テストファイル。すべて CPU で実行可能（`python verify.py`）。
- **関連文書**: この台帳（機能の過不足）に対し、`docs/ASSESSMENT.md` はプロダクト・
  プロセス・運用まで含めた長所短所改善案の評価。改善案は `docs/INSTRUCTIONS-OPUS.md`
  （設計判断つき中規模ラウンド）・`docs/INSTRUCTIONS-SONNET.md`（スコープ確定済み反復）
  に実行指示として分解されている（会話履歴なしで作業開始できる）。

---

## 過不足一覧（インデックス）

全項目を 1 行 1 件で見渡す表。ID から本文（A/B 節）の詳細に辿れる。
散文を読まずにこの表だけで全体像を把握できることを意図している。

| ID | 分類 | 優先度 | 対象 | 状態 | 一行要約 |
|---|---|---|---|---|---|
| A-1 | 過剰(点推定→上側限界) | — | `decision.py` `compare_task()` | 解消済み(6e044f3) | regression/binary/ranking の予算判定を flip_rate_ub（Wilson 上側限界）に修正済み |
| A-2 | 不足 | P0 | `tests/gpu/` ／実機全般 | 環境待ち(GPU) | 実機 GPU での end-to-end 検証がゼロ |
| A-3 | 不足 | P1 | `tsugi_torch/__init__.py` `_tsugi_compile()` | 一部解消(efdc027) | nondeterminism 警告は接続済み。worstcase/attribution/LAYOUT/タスク別 decision は依然未接続 |
| A-4 | 不足 | P1 | `lowering.py` ／GPU codegen | 環境待ち(LLVM/MLIR) | PTX/AMDGCN 生成が無い（対応表のみ・Phase 4） |
| A-5 | 不足 | P1 | `propagation.py` `propagate()` | 一部解消(425fc22) | 数値モデルは未対応(要検証)だが、torch 経路の警告で過大評価バイアスを可視化済み |
| A-6 | 過剰(接続済) | — | facade 未接続スキャン全般 | 解消済み(88846ec) | デッドコード／未接続検出を verify.py の恒常不変条件として CI 化 |
| A-7 | 不足 | P2 | `decision.py` 統計判定 | 一部解消 | per-sample δ（Q19）は解消済み。多 seed 分布報告（Q48）が残る |
| A-8 | 不足 | — | scale 推定 | 解消済み | dtype 別 denormal 率検査(Q16)＋propagation→decision 橋の仮定をレポートに明示(Q15)、両方解消 |
| A-9 | 不足 | P2 | タスクモデル拡張 | 一部解消 | oracle 有り時の accuracy 差併記(Q31)は解消——audit_runtime(logits_oracle=)が task レベル shared-mode を WARN。残: beam search・温度サンプリング下の分布一致 |
| A-10 | 不足 | P2 | 検証基盤の構造 | 一部解消 | main() 分割（Q34）・乱数境界点検（Q43・検出境界の seed 非依存性を実測で固定）は解消済み。カバレッジ計測（Q38）が残る |
| A-11 | 不足 | — | 開発運用（Q4/Q5/Q42/Q46） | 解消済み(4605479/afa52ef/+) | 閾値定数の境界感度テスト・依存ライセンス監査・遅延 import 方針、すべて解消 |
| A-12 | 不足 | — | `audit.py:_graph_ops()` / `propagation.propagate_dag` | 解消済み | 恒等路つき（residual/softmax）＋計算 N 分岐（N≥2・attention ヘッド和/N-ary 合流）を SSA から抽出し propagate_dag(correlated=True・保守側)に接続。残: 交差辺のある一般 DAG のみ（SP 近似の設計上の限界） |
| B-1a | 過剰(接続済) | — | `envelope.certify_from_sample` | 解消済み(e288b7f) | scale=1 仮定の解消関数が `audit()` に未接続だった |
| B-1b | 過剰(接続済) | — | `propagation.empirical_cond` | 解消済み(2ed0a96) | データ依存 cond 実測が `audit()` から呼ばれていなかった |
| B-1c | 過剰(接続済) | — | `nondeterminism` robust noise floor | 解消済み(4d68287) | 外れ値頑健な床が `audit_cross_vendor()` に未接続だった |
| B-1d | 過剰(接続済) | — | `decision.compare_task` | 解消済み(f44f889) | 非分類タスク判定が `audit_runtime()` から呼ばれていなかった |
| B-1e | 過剰(接続済) | — | `attribution.diagnose` | 解消済み(fc55388) | 層別診断＋責帰の集大成関数が未接続だった |
| B-1f | 過剰(接続済) | — | `worstcase.analyze_worst_case` | 解消済み(f7bdec4) | 唯一の能動探索層が未接続だった |
| B-1g | 過剰(接続済) | — | `equivalence.classify_divergence` | 解消済み(72a79e2) | LAYOUT／真の発散の判別が未接続だった |
| B-1h | 過剰(接続済) | — | `rollout.divergence_step_quantile` | 解消済み(2db24d6) | 完全デッドコードだったが統計的価値があり接続を選択 |
| B-1i | 過剰(点推定→上側限界) | — | `rollout`/`calibration`/`decision` 5 箇所（A-1 含む） | 解消済み(2db24d6/7057d6c/3a00c5b/39ce477/6e044f3) | 点推定の過信（小 N で偽OK）を Wilson／ブートストラップ上側限界に修正 |
| B-1j | 過剰(接続済) | — | `envelope.check_outlier_features`/`check_softmax_input` | 解消済み(067f5d5) | envelope phase が check_tensor しか呼ばず、outlier feature 検出・softmax exp-overflow 検査が未接続だった |
| B-1k | 過剰(接続済) | — | `decision.binary_margin` | 解消済み(4abeaa9) | compare_decisions にある near-tie 健全性チェックが compare_task(binary) に存在しない構造的非対称だった |
| B-1l | 過剰(接続済) | — | `occupancy.cross_vendor_occupancy` | 解消済み(d6f2457) | occupancy phase が nvidia/amd_cdna の 2 者間ギャップにハードコードされ、targets の amd_rdna が未報告だった |
| B-2 | 過剰(意図的) | — | メタツール群（`calibration.make_corpus` 等 6 件） | 維持（正当） | 検証器の校正用・テスト専用・CLI 等で facade 非接続が正しい |
| B-3 | 削除候補 | — | （該当なし） | ゼロ件 | 現時点で真のデッドコードは無い（B-1h が唯一の候補で接続済み） |

---

## A. 不足（deficiency）— 優先度順

フォーマット: `[ID][優先度] 対象 — 何が無いか / なぜ危険か / 推奨アクション`

### P0（次に着手すべき）

1. **[A-1] ✅ 解消済み(commit 6e044f3)** `python/tsugi/decision.py` `compare_task()` の
   予算判定が点推定のままだった問題
   - 何が無かったか: regression/binary/ranking タスクの BLOCK/WARN 判定が観測フリップ率
     `fr`（= k/n の点推定）を直接 `flip_budget` と比較していた。分類タスク用の
     `compare_decisions()` は Wilson 上側限界 `flip_rate_ub` で判定するよう修正済み
     （commit 39ce477）だったが、非分類 3 タスクは未修正だった。
   - 修正内容: `rollout.flip_rate_upper_bound(k, n, confidence)` を再利用し、
     `TaskReport` に `flip_rate_ub` フィールドを追加、判定をそちらに切り替えた。
     ranking の 1D 単一クエリ入力（返り値が 0.0/1.0 の決定的結果）には Wilson 拡張を
     適用せず、2D バッチ入力はクエリ数（`a_.shape[0]`）を試行数として widening する。
     tests: `test_compare_task_uses_flip_rate_ub_for_small_batch`・
     `test_compare_task_ranking_single_query_no_wilson_widening`。
     verify.py 不変条件 50 番。詳細は CHANGELOG.md の commit 6e044f3 に対応するエントリを参照。

2. **[A-2] 実機 GPU での end-to-end 検証が一度もない**
   - 何が無いか: 全検証層は CPU シミュレーション（NumPy oracle・擬似ベンダー）でのみ検証済み。
     NVIDIA/AMD 実機での動作確認はゼロ（`docs/SOCRATIC-50-improvements.md` の Q50）。
   - なぜ危険か: 「検証器が実機で正しい」という主張自体が未検証。SAFETY 係数（4.0）等の
     定数も実機ノイズでの校正待ち。
   - 推奨アクション: `tests/gpu/` ハーネス（GPU 無しでは正直に SKIP する設計）を実機で実行し、
     `audit_cross_vendor()` に実カーネルを渡す。要 GPU 環境のため優先度は高いが着手可能性は
     環境依存。

### P1（P0 の次）

3. **[A-3] 一部解消(commit efdc027)** `python/tsugi_torch/__init__.py` `_tsugi_compile()`
   が `audit_fx` しか呼ばない
   - 何が無いか: torch.compile バックエンド経路は FX グラフの静的監査
     （`fxbridge.audit_fx`: 増幅 op・モデル発散・非決定 op・dynamic shape 検出）のみ。
     `audit_runtime()` が持つ豊富な検証（worstcase 能動探索・attribution 層別診断・
     LAYOUT 判別・タスク別 decision）は torch 経路に届かない。
   - **一部解消済み**: `audit_fx` 自体は既に `nondeterministic_ops`/`requires_noise_floor`
     を計算していたが、`_tsugi_compile()` の警告メッセージにはそれが一切反映されて
     いなかった（audit_fx の戻り値がユーザー向け警告という facade に届いていない、
     という同型の未接続）。警告メッセージに non-deterministic op 情報を追加し、
     `tests/correctness/test_tsugi_torch_compile.py`（新規・torch 無し環境で
     duck-typed FX スタンドインを使う）で固定した。
   - なぜ危険か（残る部分）: 製品の想定入口は `torch.compile(model, backend="tsugi")`。
     そこから使えない機能（worstcase・attribution・LAYOUT 判別・タスク別 decision）は
     依然として大半のユーザーに存在しないのと同じ。
   - 推奨アクション: example_inputs から実行時検証に必要なデータ（代表テンソル・logits）を
     best-effort で取り出し、`audit()` の `sample=` / `ref_logits=` に相当する情報を
     警告メッセージに追加する段階的接続。フル接続は実行時出力が要るため codegen 後。

4. **[A-4] GPU codegen 未実装**
   - 何が無いか: `python/tsugi/lowering.py` は tile op → NVVM/ROCDL/SPIRV の対応表
     （データ）のみで、実際の PTX/AMDGCN 生成は無い。バックエンドは eager 素通し
     （その旨を明示的に warn する誠実な実装にはなっている）。
   - なぜ危険か: 「単一ソースで両ベンダー対応」という製品価値の中核が未達。
   - 推奨アクション: LLVM/MLIR 環境が要る Phase 4 作業。ロードマップは
     `python/tsugi_torch/__init__.py` の docstring に 5 段階で記載済み。

5. **[A-5] 一部解消(commit 425fc22)** `python/tsugi/propagation.py` が正規化層での
   scale リセットを追えない（Q11）
   - 何が無いか: `propagate()` は相対発散を op 列に沿って合成するが、LayerNorm/RMSNorm が
     活性の scale を正規化して発散の絶対量をリセットする効果をモデル化していない（amp≈1 と単純化）。
   - なぜ危険か: 深いモデルでの発散予測が過大（偽BLOCK 側）または構造誤りになりうる。
   - **一部解消済み**: `propagate()` 自体の数値モデルはまだ scale リセットを考慮しないが
     （恣意的な減衰係数を未検証のまま導入するリスクを避けるため意図的に据え置き）、
     `fxbridge.audit_fx()` に `has_normalization: bool` を追加し、正規化層があるグラフでは
     `model_divergence` が「scale リセット効果を未考慮の保守的な上界」であることを
     `_tsugi_compile()` の警告メッセージで明示するようにした。ユーザーが過大な
     WARN を額面通り受け取り過剰反応しないための透明性確保。
   - 推奨アクション（残る本体）: `GraphOp` に scale 伝播を追加し、正規化 op で発散を
     再基準化する版を検討。ただし減衰係数は理論的検証済みの値であるべき——例えば
     `residual=True` の √ 合成は実際の pre-norm transformer 数値実験で検証済みだったが、
     正規化層の scale-invariance を反映する減衰係数はまだ検証されていない。
     未検証のまま導入すると過大な dilution が偽OK の温床になりうるため、
     実際の LayerNorm/RMSNorm 実装に対する数値実験による検証が先決。

6. **[A-6] ✅ 解消済み(commit 88846ec)** facade 未接続・デッドコードの機械的
   スキャンが手動のままだった問題（Q56）
   - 何が無かったか: 「実装済みだが facade から呼ばれない関数」を検出する仕組みが
     CI に無く、毎回手動の Python ワンライナーで実行していた。過去にこの型の
     欠陥が 11 件見つかっている（下記セクション B）。
   - 修正内容: `verify._facade_disconnected_functions()` を追加し、セクション D の
     スキャン手法を `verify.py` の不変条件にした。`_FACADE_DISCONNECT_ALLOWLIST` で
     意図的な非接続（B-2）と既知の未実装ギャップ（A-12）を理由つきで明示的に除外し、
     許容リスト外の新規未接続だけを報告する。plant-and-detect のスモークテストで
     検出器自体が機能することを確認済み。verify.py 不変条件 57 番。

12. **[A-12] ✅ 一部解消** `audit()` の propagation phase が SSA の実 DAG 構造を捨てて
    線形化していた問題（Round 1: 恒等路つきフォークを接続）
   - 何が無かったか: `python/tsugi/ir.py` の `Op` は `operands: list[Value]` / `result: Value` を
     持ち、SSA 参照から本物のデータフロー DAG（フォーク・マージ）を再構築できる情報を
     既に保持している。だが `audit.py:_graph_ops()` は `module.kernels[i].body` を
     単純に線形走査するだけで、この operand/result 参照を一切見なかった——結果として
     `propagation.propagate_dag()`（フォーク→マージを表現できる一般化版・
     `merge_divergence` で相関/非相関の合成則も実装済み）は `tests/correctness/test_propagation.py`
     と `verify.py` からしか呼ばれず、`audit()` は常に線形版 `propagate()` しか使わなかった。
   - なぜ危険か: 実際の transformer は multi-head attention（並列ヘッドの fork→merge）や
     residual 接続（恒等路 + 変換路の merge）という非線形 DAG 構造を持つ。線形化は
     これらの構造を無視し、発散予測を歪める（fork/merge の合成則は correlated/非correlated
     で大きく異なる——`propagate_dag` のテストが実測発散を正しく上界することを既に検証済み）。
   - **Round 1 の修正内容（一部解消）**: `_graph_ops` を SSA use-def 対応に書き換えた。
     (1) `consumers` マップ（`Op.result.name → 消費 op index 列`）を 1 パスで構築、
     (2) `_identity_fork_merge` が「result の消費者がちょうど 2・合流点が result を直接
     operand に持つ（恒等 skip 路）・中間は一本鎖で合流の先へ漏れない」形（= residual
     y=x+f(x) と softmax の `row - reduce(row)` 再利用）を検出し、
     (3) `propagate_dag` のフォークノード `[[], branch]`（恒等路＋計算路）として出す。
     `audit()` の呼び出しを `propagate(gops)` → `propagate_dag(gops)` に切替え、
     `_iter_graphops` でフォーク混在列の葉 GraphOp を走査する（cond 実測・増幅 op 集計）。
     検出できない形は従来通り線形（保守側・fail-safe）。K ループ dot 集約は
     `_classify_ops` に括り出して直列区間とフォーク計算路の両方で再利用。
     tests: `test_graph_ops_extracts_ssa_fork_from_traced_softmax`
     （softmax の 2 reduce がフォーク計算路に入り propagate_dag が保守側に評価する）・
     `test_graph_ops_collapses_kloop_dots_into_one_matmul`（フォーク無しグラフは平坦列の
     まま・回帰なし）。verify.py 不変条件 63。commit は CHANGELOG の該当エントリ参照。
   - **Round 2 の修正内容（追加解消）**: 恒等路の *無い* 計算 2 分岐（literal
     multi-head attention のヘッド和・row を exp と reduce の 2 経路が消費し add で合流）も
     検出。`_computed_fork_merge` が各分岐を単一消費の一本鎖として辿り、共通の合流 op で
     `operands == {tailA, tailB}`（両末端ちょうど）として再合流する形を厳密検証し、
     `propagate_dag` の `[[branchA], [branchB]]`（恒等路なし）として出す。`_detect_fork` が
     Case-A→Case-B の順に試し、どちらも完全検証できなければ線形（保守側）。**偽OK 対策**:
     `audit()` の `propagate_dag` 呼び出しを `correlated=True`（合流を線形和 Σδ で合成）に
     変更した。クロスベンダー発散は系統的（相関）でありうる（`calibration.check_systematic`）
     ため、independent 仮定（`correlated=False` の √Σδ²）は並列分岐を過小評価し偽OK に
     なりうる。correlated=True で DAG 発散が線形版 `propagate` を下回らないことを保証する
     （fail-safe・過小評価しない）。test: `test_graph_ops_extracts_computed_two_branch_merge`。
     verify.py 不変条件 64。
   - **Round 3 の修正内容（A-12 解消）**: `_computed_fork_merge` の 2 分岐限定を撤廃し
     N（≥2）分岐へ一般化。全枝が単一消費の一本鎖として同一の合流 op に収束し、
     合流 op の operands が全末端ちょうど・領域が N 鎖で丁度覆われる場合のみ
     `[[A],[B],[C],…]` として受理する（それ以外は線形＝保守側）。3 operand を持つ
     `dot(a,b,acc)` での 3 分岐合流を実カーネルで検証。
     test: `test_graph_ops_extracts_three_way_computed_fork`。verify.py 不変条件 68。
   - **残（設計上の限界）**: 交差辺のある一般 DAG（重み共有・cross-attention の往復）は
     series-parallel で表現できないため線形/SP 近似に留まる（`propagate_dag` の設計前提）。
   - 将来の精緻化候補: `equivalence.simulate_vendor_matmul` は累積順序差のみを模擬する
     単純モデル。テンサーコアのビット精度（累積幅・truncation/RNE 差）を明示的にモデル化する
     手法が報告されている（docs/SOURCES.md「確度中」節・一次確認前）——`propagate_dag` の
     fork/merge 構造が入った後、ノード単位の誤差モデルをこの方向に精緻化する余地がある。

### P2（理論的ギャップ・構造改善。`docs/SOCRATIC-50-improvements.md` に詳細）

7. **[A-7] 統計判定の残り**: Q19（per-sample δ）は解消済み——`flip_bound_from_divergence` が
   グローバル RMS と per-sample RMS の max を δ_abs に使い、低スケール多数派に紛れた
   高スケール near-tie サンプルのフリップ risk を過小評価しない（偽OK方向の修正・
   `verify.py` 不変条件 61 番）。残るのは多 seed 分布報告
   （Q48: 「発散 ~2000倍」等の数字が単一 seed）。
8. **[A-8] scale 推定の精緻化**: Q13-16（`audit()` の `sample=` 引数で大枠は解消済みだが、
   dtype 別 denormal 下限・propagation→decision 橋の仮定明文化が残る）。
9. **[A-9] タスクモデルの拡張（一部解消）**: Q31（oracle がある時の accuracy 差併記）は
   解消——`audit_runtime(logits_oracle=)` が各ベンダーの判断誤り率を併記し、A↔B が一致
   （低フリップ率）でも両方 oracle 判断と食い違う task レベル shared-mode を WARN する
   （`decision.flip_rate` を oracle 相手に再利用・`verify.py` 不変条件 65）。残: Q21（代表
   logit はキャリブレーション集合から、というガイド）、Q22/Q32（beam search・温度
   サンプリング下の分布一致）。
10. **[A-10] 検証基盤の構造**: Q34（`verify.py` 単一巨大 main() の関数分割）は解消済み——
    61 セクションをテーマ別の 12 個の `_check_*()` 関数に分割し、`main()` は順に呼ぶだけの
    薄い関数にした（挙動・実行順・check 文言・件数は分割前と一字一句同一であることを
    出力 diff で確認済み）。残るのは Q33（test との重複整理の明文化）、
    Q38（カバレッジ計測）、Q43（乱数依存テストの境界余裕点検）。
11. **[A-11] ✅ 解消済み** 開発運用: Q4/Q5（envelope/decision の残り閾値の named constant
    化＋境界感度テスト）・Q42（依存ライセンス自動監査・commit 4605479）・Q46（遅延 import
    の方針明文化・`CONTRIBUTING.md`「Import 方針」節）、いずれも解消済み。
    Q4/Q5 は定数化済みだった実装に `test_envelope_thresholds_are_sensitive_to_their_constants`・
    `test_near_tie_threshold_is_sensitive_to_its_constant`（Q6 の `SAFETY` 感度テストと
    同型の境界±固定）を追加し、値が判定境界を実際に支配することを機械的に保証した。

---

## B. 過剰（excess）— 処遇つき

「過剰」= 実装コストを払ったのに価値を届けていない機能。3 つの処遇に選別する。

### B-1. 接続済み（過去に発見・解消した過剰。再発防止の参照事例）

以下の 8 件は「実装・テスト済みだが facade 未接続（または誰からも呼ばれない）」だった。
すべて解消済み。**新機能を追加する時は、この表の轍を踏まないこと**（追加した関数が
facade から実際に呼ばれるかを必ず確認する）。

| 機能 | 何が起きていたか | 解消 commit |
|---|---|---|
| `envelope.certify_from_sample` | scale=1 暗黙仮定の解消関数を追加したのに `audit()` は `certify_gemm(K,dtype,1.0)` 固定のままだった | e288b7f |
| `propagation.empirical_cond` | データ依存の条件数実測が実装済みなのに `audit()` は常に静的下界 cond=1 を使っていた | 2ed0a96 |
| `nondeterminism` の robust noise floor | 外れ値頑健な 10-90 パーセンタイル床があるのに実機入口 `audit_cross_vendor()` は脆弱な max-min 固定だった | 4d68287 |
| `decision.compare_task` | 回帰/二値/ランキングのタスク判定が `audit_runtime()` から呼ばれず、非分類ワークロード全体が対象外だった | f44f889 |
| `attribution.diagnose` | 層別診断＋責帰の集大成関数が未接続で、audit は BLOCK を出すだけで「どの層のどちらのベンダーか」を特定しなかった | fc55388 |
| `worstcase.analyze_worst_case` | 唯一の能動的最悪ケース探索が未接続で、検証がすべて受動的サンプル比較だった | f7bdec4 |
| `equivalence.classify_divergence` | レイアウト不一致（転置・値は正しい）と真の数値発散の判別が未接続で、診断の手がかりを捨てていた | 72a79e2 |
| `rollout.divergence_step_quantile` | 完全デッドコード（テストからも呼ばれない）だった。削除でなく接続を選択：初回発散の中央値は平均より系統的に小さく（幾何分布の右裾）、平均だけの報告は楽観バイアス | 2db24d6 |
| `envelope.check_outlier_features`/`check_softmax_input` | `audit_runtime()` の envelope phase が `check_tensor()` しか呼ばず、outlier feature（massive activations）検出と fp16 softmax exp-overflow 検査が実行時監査に一切届いていなかった | 067f5d5 |
| `decision.binary_margin` | `compare_decisions`（分類）は near-tie 健全性チェック（フリップは決定境界近傍に集中すべき）を持つが、`compare_task(binary)` にはこの診断ロジック自体が丸ごと存在しなかった。単なる関数未接続でなく機能の構造的非対称 | 4abeaa9 |
| `occupancy.cross_vendor_occupancy` | `audit()` の他フェーズ（portability・feasibility）は `targets` を尊重するのに、occupancy phase だけが `occupancy_gap(cfg,"nvidia","amd_cdna")` に固定され、`targets` に `amd_rdna` が含まれていても一切報告していなかった | d6f2457 |

同系統の「点推定の過信」も 5 箇所で解消済み: `rollout`（平均に加え中央値を報告・2db24d6）、
`calibration.check_systematic`（bias±ブートストラップ標準誤差の上側限界で判定・7057d6c）、
`decision.predicted_flip_bound`（Wilson 上側限界・3a00c5b）、
`decision.compare_decisions`（同・39ce477）、`decision.compare_task`（同・6e044f3）。

### B-2. 意図的に facade 非接続（正当。削除も接続も不要）

| 対象 | 役割 |
|---|---|
| `calibration.make_corpus` / `evaluate` / `roc_sweep` | 検証器**自身**を検証するメタツール（偽OK率の測定）。製品判定経路でなく開発時の校正用 |
| `equivalence.simulate_vendor_matmul`・`nondeterminism.simulate_*` | GPU 実機なしで検証層をテストするための擬似ベンダー／CPU シミュレータ。テスト専用が正当 |
| `portcheck.main` | スタンドアロン CLI のエントリポイント |
| `autotune.grid_search` | タイル構成の探索ユーティリティ（codegen 実装後に本経路に入る予定） |
| `provenance.env_fingerprint` / `fingerprint_hash` / `changed_fields` | `certify`/`is_stale` の内部部品。facade は上位関数経由で利用済み |
| `oracle_check.oracle_is_trustworthy` | `verify_oracle().ok` の便宜ラッパ。audit は verify_oracle を直接使う |

### B-3. 削除候補

**現時点でゼロ。** 全公開関数の参照スキャン（セクション D の手法）で見つかった唯一の
完全デッドコード `rollout.divergence_step_quantile` は、統計的に意味のある値（中央値）
だったため削除でなく接続を選択済み（2db24d6）。今後スキャンで新たに見つかった場合は
「統計的・診断的に意味があるなら接続、実験の残骸なら削除」を判断基準にする。

---

## C. 適正（well-covered）

以下は十分に機能しており、当面の作業対象ではない:

- **13 の独立検証層** — portability / equivalence / occupancy / tolerance / feasibility /
  propagation / envelope / decision / rollout / worstcase / attribution / blame /
  nondeterminism。各層は単独でも呼べ、facade（audit 系）が束ねる。
- **fail-safe 設計の一貫性** — 偽OK/偽BLOCK の非対称コストを全層が共有。不確実な判定は
  上側限界・保守側に倒す（B-1 の修正で徹底済み）。
- **CPU oracle と検証器の自己検証** — float64 リファレンス＋メタモルフィック検証
  （`oracle_check`）＋検証器の偽OK率測定（`calibration`）で「検証器を信じる根拠」まで検証。
- **dtype テーブル** — float16/bfloat16/float32/float64/tf32/float8_e4m3/float8_e5m2/
  mxfp4_e2m1/mxfp6_e2m3/mxfp6_e3m2 の 10+1 種を `tolerance.UNIT_ROUNDOFF`・
  `equivalence.TOLERANCE`・`envelope.DTYPE_LIMITS` の 3 表で整合管理（新 dtype は
  この 3 表に同時追加するのが規約）。NVFP4（NVIDIA 専用・AMD 非対応）は意図的に対象外
  （docs/SOURCES.md「Microscaling (MX) / NVFP4 低精度フォーマット」節）。
- **機械検証可能な不変条件 163 件**（`verify.py`）と 27 テストファイル・property test
  （10 性質 × 200 試行）。

---

## D. 引き継ぎ手順

### 作業順序

P0-1（`compare_task` の Wilson 化）→ P1-3（torch 経路への段階接続）→ P1-5（scale 伝播）
→ P1-6（スキャンの CI 化）。P0-2 と P1-4 は GPU/LLVM 環境が得られ次第。

### 1 作業の定型手順（このリポジトリの慣例）

1. 対象モジュールを修正（既存実装の再利用を優先。例: Wilson 限界は
   `rollout.flip_rate_upper_bound` を再利用し重複実装しない）
2. `tests/correctness/test_<module>.py` に「問題を再現するケース」と「回帰なし
   （大 N で挙動不変等）」の両方のテストを追加し、`main()` のテストリストにも登録する
3. `verify.py` に不変条件を追加（末尾の連番コメント形式に従う）
4. `python verify.py` と `for f in tests/correctness/test_*.py; do python "$f"; done` が全 PASS
5. `CHANGELOG.md` の Unreleased に「何が・なぜ・どう直したか・実証データ」を記録
6. commit（日本語・問題→修正→実証の順で本文を書く）して push

### facade 未接続・デッドコードのスキャン手法（P1-6 で CI 化するまでの手動手順）

```python
# python/tsugi/*.py の公開関数が (a) audit.py から (b) どこからも 呼ばれているかを照合する
import re, glob
files = glob.glob("python/tsugi/*.py") + glob.glob("python/tsugi_torch/*.py")
texts = {f: open(f, encoding="utf-8").read() for f in files}
tests = {f: open(f, encoding="utf-8").read()
         for f in glob.glob("tests/correctness/*.py") + ["verify.py"]}
for f, src in texts.items():
    for fn in re.findall(r'^def ([a-z][a-zA-Z0-9_]*)\(', src, re.M):
        in_own = src.count(fn) > 1                      # 自ファイル内の呼び出し
        in_src = any(fn in t for g, t in texts.items() if g != f)
        in_test = any(fn in t for t in tests.values())
        if not in_own and not in_src:
            print(("[tested only]" if in_test else "[FULLY DEAD]"), f, fn)
```

- `[FULLY DEAD]` = 真のデッドコード候補（B-3 の判断基準で接続か削除）
- `[tested only]` = テストからは呼ばれるが製品経路から呼ばれない。B-2 の正当理由
  （メタツール・シミュレータ・CLI）に該当しなければ facade 接続候補（A の不足）
- 出力には false positive が混ざるため人手の選別が必要。B-2 の表が除外リストの出発点
