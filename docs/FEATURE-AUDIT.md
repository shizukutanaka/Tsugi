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
- **検証基盤の規模**: `verify.py` に 128/128 の機械検証可能な不変条件。
  `tests/correctness/` に 26 テストファイル。すべて CPU で実行可能（`python verify.py`）。

---

## A. 不足（deficiency）— 優先度順

フォーマット: `[優先度] 対象 — 何が無いか / なぜ危険か / 推奨アクション`

### P0（次に着手すべき）

1. **`python/tsugi/decision.py:306` `compare_task()` の予算判定が点推定のまま**
   - 何が無いか: regression/binary/ranking タスクの BLOCK/WARN 判定が観測フリップ率
     `fr`（= k/n の点推定）を直接 `flip_budget` と比較している。分類タスク用の
     `compare_decisions()` は Wilson 上側限界 `flip_rate_ub` で判定するよう修正済み
     （commit 39ce477）だが、非分類 3 タスクは未修正。
   - なぜ危険か: 小さい評価バッチ（例 n=30）でたまたま観測フリップ 0 件でも母集団の
     真の率は予算超でありうる。0 件観測を「フリップ率 0%」と過信するのは偽OK の温床
     （rule of three: 0/n 観測でも真の p は ~3/n までありうる）。
   - 推奨アクション: `rollout.flip_rate_upper_bound(k, n, confidence)`（`python/tsugi/rollout.py`・
     Wilson 上側限界の既存実装）を再利用し、`TaskReport` に `flip_rate_ub` フィールドを追加、
     判定をそちらに切り替える。**注意**: ranking の 1D 単一クエリ入力は返り値が 0.0/1.0 の
     決定的結果（サンプリングされた推定値でない）なので Wilson 拡張を適用しないこと。
     参照実装: commit 39ce477 の `compare_decisions` への同型修正と
     `tests/correctness/test_decision.py::test_compare_decisions_uses_flip_rate_ub_for_small_batch`。

2. **実機 GPU での end-to-end 検証が一度もない**
   - 何が無いか: 全検証層は CPU シミュレーション（NumPy oracle・擬似ベンダー）でのみ検証済み。
     NVIDIA/AMD 実機での動作確認はゼロ（`docs/SOCRATIC-50-improvements.md` の Q50）。
   - なぜ危険か: 「検証器が実機で正しい」という主張自体が未検証。SAFETY 係数（4.0）等の
     定数も実機ノイズでの校正待ち。
   - 推奨アクション: `tests/gpu/` ハーネス（GPU 無しでは正直に SKIP する設計）を実機で実行し、
     `audit_cross_vendor()` に実カーネルを渡す。要 GPU 環境のため優先度は高いが着手可能性は
     環境依存。

### P1（P0 の次）

3. **`python/tsugi_torch/__init__.py` `_tsugi_compile()` が `audit_fx` しか呼ばない**
   - 何が無いか: torch.compile バックエンド経路は FX グラフの静的監査
     （`fxbridge.audit_fx`: 増幅 op・モデル発散・非決定 op・dynamic shape 検出）のみ。
     `audit_runtime()` が持つ豊富な検証（worstcase 能動探索・attribution 層別診断・
     LAYOUT 判別・タスク別 decision）は torch 経路に届かない。
   - なぜ危険か: 製品の想定入口は `torch.compile(model, backend="tsugi")`。そこから
     使えない機能は大半のユーザーに存在しないのと同じ。
   - 推奨アクション: example_inputs から実行時検証に必要なデータ（代表テンソル・logits）を
     best-effort で取り出し、`audit()` の `sample=` / `ref_logits=` に相当する情報を
     警告メッセージに追加する段階的接続。フル接続は実行時出力が要るため codegen 後。

4. **GPU codegen 未実装**
   - 何が無いか: `python/tsugi/lowering.py` は tile op → NVVM/ROCDL/SPIRV の対応表
     （データ）のみで、実際の PTX/AMDGCN 生成は無い。バックエンドは eager 素通し
     （その旨を明示的に warn する誠実な実装にはなっている）。
   - なぜ危険か: 「単一ソースで両ベンダー対応」という製品価値の中核が未達。
   - 推奨アクション: LLVM/MLIR 環境が要る Phase 4 作業。ロードマップは
     `python/tsugi_torch/__init__.py` の docstring に 5 段階で記載済み。

5. **`python/tsugi/propagation.py` が正規化層での scale リセットを追えない（Q11）**
   - 何が無いか: `propagate()` は相対発散を op 列に沿って合成するが、LayerNorm/RMSNorm が
     活性の scale を正規化して発散の絶対量をリセットする効果をモデル化していない（amp≈1 と単純化）。
   - なぜ危険か: 深いモデルでの発散予測が過大（偽BLOCK 側）または構造誤りになりうる。
   - 推奨アクション: `GraphOp` に scale 伝播を追加し、正規化 op で発散を再基準化する版を
     検討。residual（`residual=True` の √ 合成）と同様の一次近似でよい。

6. **facade 未接続・デッドコードの機械的スキャンが手動のまま（Q56）**
   - 何が無いか: 「実装済みだが facade から呼ばれない関数」を検出する仕組みが CI に無い。
     過去にこの型の欠陥が 8 件見つかっている（下記セクション B）。
   - なぜ危険か: 新機能を追加するたびに同型の欠陥（機能はあるが届かない）が再発しうる。
     実際に commit b0e7da3〜72a79e2 の間、毎回のように発見された。
   - 推奨アクション: セクション D のスキャン手法を `verify.py` の不変条件にする。
     module-private helper の false positive を除外リストで管理する保守コストと相談。

### P2（理論的ギャップ・構造改善。`docs/SOCRATIC-50-improvements.md` に詳細）

7. **統計判定の残り**: `decision.py` の per-sample δ（Q19: δ_abs=δ_rel·RMS は平均であって
   最悪ケースでない）、多 seed 分布報告（Q48: 「発散 ~2000倍」等の数字が単一 seed）。
8. **scale 推定の精緻化**: Q13-16（`audit()` の `sample=` 引数で大枠は解消済みだが、
   dtype 別 denormal 下限・propagation→decision 橋の仮定明文化が残る）。
9. **タスクモデルの拡張**: Q21（代表 logit はキャリブレーション集合から、というガイド）、
   Q22/Q32（beam search・温度サンプリング下の分布一致）、Q31（oracle がある時の accuracy 差併記）。
10. **検証基盤の構造**: Q33/Q34（`verify.py` 単一巨大 main() の関数分割・test との重複整理）、
    Q38（カバレッジ計測）、Q43（乱数依存テストの境界余裕点検）。
11. **開発運用**: Q4/Q5（残りのマジックナンバーの named constant 化）、Q42（依存ライセンス
    自動監査）、Q46（遅延 import の方針明文化）。

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

同系統の「点推定の過信」も 4 箇所で解消済み: `rollout`（平均に加え中央値を報告・2db24d6）、
`calibration.check_systematic`（bias±ブートストラップ標準誤差の上側限界で判定・7057d6c）、
`decision.predicted_flip_bound`（Wilson 上側限界・3a00c5b）、
`decision.compare_decisions`（同・39ce477）。

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
- **dtype テーブル** — float16/bfloat16/float32/float64/tf32/float8_e4m3/float8_e5m2 の
  7+1 種を `tolerance.UNIT_ROUNDOFF`・`equivalence.TOLERANCE`・`envelope.DTYPE_LIMITS` の
  3 表で整合管理（新 dtype はこの 3 表に同時追加するのが規約）。
- **機械検証可能な不変条件 128 件**（`verify.py`）と 26 テストファイル・property test
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
