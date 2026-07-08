# Changelog

Keep a Changelog 形式。SemVer。0.x は API 未凍結（MINOR で機能追加・互換変更ありうる）。

## [Unreleased]

### Added
- **tsugi_torch._tsugi_compile — nondeterministic_ops を警告に反映（FEATURE-AUDIT.md A-3 の一部）**:
  `fxbridge.audit_fx()` は `nondeterministic_ops`/`requires_noise_floor`（scatter_add 等
  atomicAdd 由来の非決定 op の静的検出）を既に計算していたが、`_tsugi_compile()` の
  ユーザー向け警告メッセージには一切反映されていなかった——audit_fx の戻り値が
  facade（実際にユーザーへ届く警告）に届いていない、このプロジェクトが繰り返し
  見つけてきた欠陥と同型。加えて `tsugi_torch._tsugi_compile` には**専用テストが
  一件も存在しなかった**ことも判明（`test_compile.py` は無関係な `tsugi.compile`
  DSL フロントエンドのテスト）。

  - `_tsugi_compile()`: `rep["requires_noise_floor"]` が真なら警告に
    `[non-deterministic: {ops} → noise floor 実測が必須]` を追加。scatter_add 等の
    op のみを含みグラフに数値 op（matmul 等）が無い場合でも警告が出るよう、
    警告発火条件を `rep["n_ops"] or rep["requires_noise_floor"]` に拡張。
  - `tests/correctness/test_tsugi_torch_compile.py`（新規）: torch 無し環境でも
    duck-typed FX スタンドインで警告文言を検証する 3 テスト。
  - verify.py: 143→144 不変条件（58番）

- **verify.py — facade 未接続の恒常検査を追加（FEATURE-AUDIT.md A-6 の本体を解消）**:
  このプロジェクトはソース参照スキャンで 11 件の「実装済みだが facade（audit 系）から
  呼ばれない」欠陥を発見・修正してきたが、そのスキャン自体は毎回手動の Python
  ワンライナーで実行していた（第18・19回等）。今回これを恒常的な不変条件にする。

  - `verify._facade_disconnected_functions()`: `python/tsugi/*.py`・
    `python/tsugi_torch/*.py` の公開関数のうち、自ファイル内でも他ソースからも
    一切呼ばれていないものを検出する。
  - `_FACADE_DISCONNECT_ALLOWLIST`: 意図的な非接続（`docs/FEATURE-AUDIT.md` B-2 の
    メタツール群・シミュレータ・CLI 等）と、既知の未実装ギャップ（A-12: `propagate_dag`
    の DAG 対応）を理由つきで明示的に除外。許容リストに無い新規の未接続だけを報告する。
  - plant-and-detect のスモークテストで検出器自体が機能することを確認済み
    （意図的に非接続の関数を仕込み、それだけを正しく検出することを確認）。
  - 現時点で許容リスト外の新規未接続はゼロ。
  - verify.py: 142→143 不変条件（57番）

- **docs/MODEL-USAGE-GUIDE.md — Claude モデル使い分け表（個人用メモ）**:
  このセッションで実際に観測された `/model` 切り替えパターン（戦略的・探索的な
  指示の直前は Fable 5、スコープの定まった継続作業の直前は Sonnet 5）を根拠に、
  Tsugi 開発における Haiku/Sonnet/Opus/Fable 5 の使い分けと、Agent（サブエージェント）・
  `/loop` スキルの活用余地をまとめた。汎用的なモデル比較でなく、このユーザーの
  実際の作業パターンに基づく実務指針。README から接続。

## [0.3.0] — 2026-07-07
「実装済みだが facade（`audit`/`audit_runtime`/`audit_cross_vendor`）から呼ばれない」
欠陥を系統的なソース参照スキャンで 11 件発見・接続（`certify_from_sample`・
`empirical_cond`・robust noise floor・`compare_task`・`attribution.diagnose`・
`worstcase.analyze_worst_case`・`classify_divergence`・`divergence_step_quantile`・
envelope の outlier/softmax 検査・`binary_margin`・occupancy の全 targets 対応）。
同様に「点推定を過信する」小標本での偽OK パターンを 5 箇所で Wilson／ブートストラップ
上側限界に修正。市販品質強化として equivalence/audit_runtime の形状不一致 silent
broadcast を排除し、orphan テスト（未登録で一度も実行されないテスト）の恒常検査を追加。
機能過不足の台帳 `docs/FEATURE-AUDIT.md` を新設し README/VERIFICATION から接続。
詳細は下記全項目・根拠 commit hash 付き。テスト 26 ファイル / verify 142 不変条件。

### Added
- **verify.py — orphan テスト（未登録テスト）の恒常検査を追加（市販品質強化）**:
  `docs/FEATURE-AUDIT.md` A-6（facade 未接続の機械的スキャンが手動のまま）を
  部分的に解消。各テストファイルは `main()` の手書き `tests = [...]` リストで
  実行対象を選ぶ構造だが、この構造は「テストを書いたがリスト登録を忘れ、一度も
  実行されない」という静かな品質劣化を招きうる——本番コードの「facade 未接続」
  （実装したが呼ばれない関数）と同型の欠陥がテスト層で起きる経路。

  - `verify._orphan_tests()`: `tests/correctness/test_*.py` の各 `test_*` 関数が
    ファイル内で def 行以外に 1 回以上出現するか（≒ リスト登録されているか）を
    正規表現で検査する。
  - 新しい plant-and-detect のスモークテスト（一時ファイルで意図的に orphan 関数を
    仕込み、検出器がそれだけを正しく検出し登録済み関数を誤検出しないことを確認）
    で検出器自体が機能することを確認済み。
  - 現時点で orphan テストはゼロ（既存 26 ファイルは全テスト登録済み）。
  - verify.py: 141→142 不変条件（56番）

### Fixed
- **equivalence.compare / audit_runtime — 形状不一致の暗黙 broadcast を排除（市販品質強化）**:
  `_compare_with()`（`compare`/`compare_gemm` の共通実装）と `audit_runtime()` の
  equivalence phase は、a/b の形状が異なる場合でも NumPy の暗黙 broadcast に頼って
  `a - b` を計算していた。broadcast 可能な形状の組（例 `(64,1)` vs `(64,64)`・スカラ vs
  行列）では、方向次第で **偽 DIVERGENT にも偽OK にもなりうる**——後者が特に危険。

  実証したシナリオ: ベンダー B が実装バグで先頭 1 行しか返さない
  （`b = a[0]`、形状 `(8,)` vs `a` の `(8,8)`）場合、旧実装は NumPy の
  broadcast で「なんとなく比較が成立」してしまい、本来 BLOCK すべき構造的バグを
  見逃しうる。`classify_divergence()` は既に要素数不一致を `DV_DIVERGENT` として
  明示的に扱っており、「形状不一致は比較不能な構造的発散」が設計意図として
  正しいことは既に確立されていた——`_compare_with`/`audit_runtime` だけがこの
  原則から外れていた。

  - `EquivalenceReport` に `shape_mismatch: bool`・`shape_a`/`shape_b: tuple` を追加。
    `_compare_with()` は形状比較を最初に行い、不一致なら broadcast する前に
    `equivalent=False, shape_mismatch=True` を返す。
  - `audit_runtime()` は `af`/`bf` 計算直後に形状を検査し、不一致なら以降の全 phase
    （decision・correctness・attribution・worstcase 等、いずれも同形状前提）に
    入らず equivalence phase のみで BLOCK を返す。broadcast 不能な形状の組
    （ValueError でクラッシュしうる）でも同様に安全に BLOCK する。
  - 既存の同形状経路は完全に無回帰（`shape_mismatch=False` のまま）。
  - tests: `test_shape_mismatch_is_not_silently_broadcast`（equivalence）・
    `test_audit_runtime_rejects_shape_mismatch_without_broadcast`（audit）
  - verify.py: 137→141 不変条件（54-55番）

### Added
- **audit() の occupancy phase — targets 全体を報告するよう修正（第28回）**:
  第27回と同じ関数参照スキャンの継続から発見。`occupancy.cross_vendor_occupancy()`
  （全ベンダーの占有率を一度に返す・実装・テスト済み）は `audit()` から一切呼ばれておらず、
  occupancy phase は `occupancy_gap(cfg, "nvidia", "amd_cdna")` という **2 者間ギャップに
  ハードコード**されていた。`audit()` の他フェーズ（portability・feasibility）はすべて
  `targets` パラメータ（既定 `("nvidia", "amd_cdna", "amd_rdna")`）を尊重するのに、
  occupancy phase だけが `targets` を無視し `amd_rdna` の占有率を一切報告していなかった。

  - `cross_vendor_occupancy(cfg, vendors=targets)` に切り替え、`targets` に含まれる
    全ベンダーの占有率を報告。最大ギャップとそのペアも明示。
  - `targets` を絞れば occupancy phase の報告もそれに追従する（ハードコードでないことを
    テストで固定）。
  - 実証: 既定 `targets` では `amd_rdna` が occupancy phase のテキストに一切現れなかった
    ことを確認した上で修正。`targets=("nvidia", "amd_rdna")` に絞ると `amd_cdna` は
    正しく消え `amd_rdna` が現れることも確認。
  - tests: `test_audit_occupancy_phase_covers_all_targets`
  - verify.py: 135→137 不変条件（53番）

- **decision.compare_task(binary) — compare_decisions と同型の near-tie 健全性チェックを接続（第27回）**:
  第26回の envelope phase の修正作業と同じ関数参照スキャンから発見。
  `compare_decisions()`（分類）はフリップが決定境界近傍（低マージン・near-tie）に
  集中するはずという健全性チェックを持ち、確信領域（高マージン）まで巻き込むフリップは
  系統的発散の兆候として WARN する。`decision.binary_margin()`（binary タスク版の margin）は
  実装・テスト済みだったが、この診断ロジック自体が `compare_task()` に一切実装されて
  いなかった——単なる関数未接続でなく、`compare_decisions` にある診断能力そのものが
  `compare_task` 側には存在しない構造的な非対称。

  - `TaskReport` に `flipped_margin_median`/`overall_margin_median` フィールドを追加
    （`DecisionReport` と同名・同役割）。
  - `compare_task(task="binary")` で `binary_margin()` を使いフリップサンプルと全体の
    マージン中央値を計算し、フリップ中央値が全体中央値の `_NEAR_TIE_MARGIN_FRAC`（0.5）倍を
    超えたら WARN（`compare_decisions` と同じ閾値定数を再利用）。
  - regression/ranking はマージンの概念が確立していないため対象外（0.0 のまま・将来課題）。
  - 実証: 閾値付近のみがフリップする正常系では警告なし、確信領域（0.95→0.05 等）まで
    フリップする異常系では正しく WARN が立つことを両方確認。
  - tests: `test_compare_task_binary_warns_when_flips_not_near_tie`
  - verify.py: 133→135 不変条件（52番）

- **audit_runtime の envelope phase — check_outlier_features・check_softmax_input を接続（第26回）**:
  A-1（compare_task）の修正作業中に行った関数参照スキャンの副産物として発見。
  `envelope.check_outlier_features()`（outlier feature／massive activations による
  単一 scale 仮定の破綻検出）と `envelope.check_softmax_input()`（fp16 softmax の
  exp-overflow 検出）はどちらも実装・テスト済みだったが、`audit_runtime()` の
  envelope phase は `check_tensor()` しか呼んでおらず、同モジュールの他の実行時検査が
  一切製品経路に届いていなかった（第11-25回で見つけた「機能は実装済みだが facade
  未接続」と同型の欠陥）。

  - envelope phase（`env is not None` の時）で両ベンダーの出力に `check_outlier_features`
    を追加実行。outlier channel（LLM の massive activations）が単一 global scale 仮定を
    破っている場合に WARN を verdict に反映する。
  - `logits_a`/`logits_b` も併せて渡された場合、両者に `check_softmax_input` を実行し、
    fp16 の exp-overflow（`ln(65504)≈11.09` 超）を BLOCK として反映する。
  - **実装中に見つけた自己バグ**: `FindingReport.ok` は `max_risk < BLOCK` を返すため
    `if not r.ok:` で判定すると WARN 相当の findings（outlier 検出等）が黙って
    落ちる。`if r.findings:`（非空判定）に修正して初めて正しく動作した——「過信した
    条件分岐で warn を握りつぶす」という、このプロジェクトが繰り返し警戒してきた
    問題を実装中に自分でも踏んで直した。
  - `env`/`logits` 未指定時は従来通りこれらの検査は現れない（後方互換）。
  - tests: `test_audit_runtime_envelope_wires_outlier_and_softmax_checks`
  - verify.py: 131→133 不変条件（51番）

- **decision.compare_task — 非分類タスクも Wilson 上側限界(flip_rate_ub)を使用（第25回）**:
  `docs/FEATURE-AUDIT.md` の P0 項目 A-1 を解消。`compare_decisions()`（分類・argmax）は
  commit 39ce477 で観測 flip_rate の点推定でなく `flip_rate_ub`（Wilson 上側限界・
  `rollout.flip_rate_upper_bound` を再利用）で予算判定するよう修正済みだったが、
  同モジュールの非分類版 `compare_task()`（regression/binary/ranking）だけが
  取り残されていた。

  - `TaskReport` に `flip_rate_ub: float` フィールドを追加。`compare_task(..., confidence=0.95)`
    の BLOCK/WARN 判定を観測 `fr`（点推定）から `fr_ub`（Wilson 上側限界）に切り替え。
  - **ranking タスクの特殊扱い**: `ranking_flip_rate()` は 1D 入力（単一クエリ）だと
    複数試行の推定比率でなく *決定的* な 0.0/1.0（集合が一致するか否かの厳密な結果）を
    返す。サンプリングされた推定値でないものに信頼区間を付けるのは統計的に無意味なため、
    この場合は widening を適用しない（`flip_rate_ub = flip_rate`）。2D バッチ入力
    （複数クエリ）はクエリ数（`a_.shape[0]`）を試行数として widening する
    （`a_.ravel().size` ではない——これは要素数であり試行数でない）。
  - **既存テストの是正**: `test_compare_task_regression_blocks_large_value_drift` は
    `flip_budget` を渡さない（既定 `0.0` = 「未来永劫ゼロフリップ」の意）まま厳密に
    同一の入力（`a`, `a.copy()`）で OK を検証していた。有限標本からは「今後も永久に
    ゼロ」は証明不能（Wilson 上限は n=1000 でも厳密な 0 を返さない）ため、他のテストと
    同じ非ゼロ budget（`0.01`）を渡すよう是正。これは緩和でなく、このプロジェクトが
    一貫して採ってきた「点推定を信じない」設計判断をテストに反映しただけ。
  - 実証: n=30・観測フリップ 0 件（regression）で旧ロジックなら OK だったのが、
    新ロジックで正しく WARN 以上になる（rule of three）。大 N の既存テストは
    無回帰で通過。1D ranking では widening が適用されないことも固定。
  - tests: `test_compare_task_uses_flip_rate_ub_for_small_batch`・
    `test_compare_task_ranking_single_query_no_wilson_widening`
  - verify.py: 128→131 不変条件（50番）

- **docs/FEATURE-AUDIT.md — 過不足一覧表とドキュメント接続（第24回）**:
  第23回で作成した台帳の内容は正確だったが、「リスト化」の核心である一覧性と、
  README/VERIFICATION からの入口リンクが欠けていた（ドキュメント版の facade 未接続）。
  - 冒頭に「過不足一覧（インデックス）」を新設: ID・分類・優先度・対象・状態・一行要約の
    6 列で全 22 項目（不足 11・過剰 11）を 1 行 1 件でスキャン可能にした。本文 A 節の
    各項目にも対応する ID（A-1〜A-11）を付与し、表から本文へ辿れるようにした。
  - `README.md` と `docs/VERIFICATION.md` に `docs/FEATURE-AUDIT.md` への入口リンクを追加
    （従来は CHANGELOG からしか参照されておらず、リポジトリを探索する読者に届かなかった）。
  - verify.py・全テストは無回帰（ドキュメントのみの変更）。

- **docs/FEATURE-AUDIT.md — 機能過不足の台帳（第23回）**:
  機能を「不足（未接続・未実装）」「過剰（実装したが価値未提供）」「適正」に選別した
  自己完結の監査ドキュメント。セッション文脈なしの読者（人間・AI）がこのファイルだけで
  作業を引き継げるよう、用語定義・全項目へのファイルパス・根拠 commit hash を付し、
  優先度順（P0: compare_task の Wilson 化と実機 e2e / P1: torch 経路接続・codegen・
  scale 伝播・スキャン CI 化 / P2: SOCRATIC 残項目）で列挙。過去に解消した
  「facade 未接続」8 件・「点推定の過信」4 件を再発防止の参照事例として表に固定し、
  意図的に非接続のメタツール群（B-2）と削除候補ゼロの判断基準（B-3）も明記。
  デッドコードスキャンのワンライナーを引き継ぎ手順（D 節）に収録。

- **decision.compare_decisions — 主判定も Wilson 上側限界(flip_rate_ub)を使用（第22回）**:
  第21回は `predicted_flip_bound`（第2ベンダー実行 *前* の予測）を修正したが、
  `compare_decisions` 自体の予算判定（第2ベンダーを実際に走らせた *後* の主判定）は
  依然として観測 `flip_rate`（点推定 k/n）を直接 `flip_budget` と比較していた。
  こちらは decision モジュールの中心的な製品判定であり、predicted_flip_bound より
  影響範囲が大きい。

  - `DecisionReport` に `flip_rate_ub: float` フィールドを追加（Wilson 上側限界・
    `rollout.flip_rate_upper_bound` を再利用）。
  - `compare_decisions(..., confidence: float = 0.95)`: BLOCK/WARN の予算判定に
    `flip_rate`（点推定）でなく `flip_rate_ub` を使うよう変更。
  - 実証: n=30 の小評価バッチで観測フリップ 0 件・真のフリップ率は予算超のケースで、
    旧ロジックなら「予算内」（偽OK）だったのが、新ロジックで正しく WARN 以上になる。
  - 既存の大 N テスト（n=2000-4000）はすべて無回帰で通過（Wilson 上限は大 N で
    点推定に相対誤差 20% 未満で収束・境界値ギリギリのテストケースも確認済み）。
  - tests: `test_compare_decisions_uses_flip_rate_ub_for_small_batch`
  - verify.py: 126→128 不変条件（49番）
  - docs/SOCRATIC-50-improvements.md に Q58 を追加。

- **decision.predicted_flip_bound — 点推定でなく Wilson 上側限界で判定（第21回）**:
  第20回の Q55/Q57 で示した「点推定でなく上側限界で判定」パターンが calibration 以外にも
  波及するかを検証した結果、`predicted_flip_bound(ref_logits, delta)` = P(margin<2δ) も
  代表 logit n 件からの単純比率（点推定）に過ぎなかったことを発見・修正。

  n が小さい代表集合（キャリブレーションセット）ではたまたま margin<2δ 該当サンプルが
  0 件でも、母集団の真の確率が 0 とは限らない（`rollout.flip_rate_upper_bound` の
  rule-of-three と同じ問題）。0 件観測を「フリップ率 0%」と過信するのは fail-safe に反する。

  - `predicted_flip_bound(ref_logits, delta, confidence=0.95)`: 観測比率 k/n でなく
    `rollout.flip_rate_upper_bound(k, n, confidence)`（Wilson 上側限界・既存実装を再利用）
    へ委譲するよう修正。n が大きければ上限は点推定にほぼ収束し挙動は変わらない。
  - `flip_bound_from_divergence(ref_logits, rel_divergence, confidence=0.95)`:
    `confidence` を透過的に伝播。
  - 実証: n=20 の代表集合で margin<2δ 該当が 0 件のケースで、旧ロジックなら bound=0.0
    （過信）だったのが、新ロジックで bound≈0.12（正しく不確実性を反映）になる。
  - 既存テスト（`test_predicted_bound_is_upper_bound` 等）・property test（10性質×200試行）
    はすべて無回帰で通過（大 N の既存テストでは Wilson 上限が点推定に収束するため）。
  - tests: `test_predicted_bound_uses_wilson_upper_bound_for_small_representative_set`
  - verify.py: 124→126 不変条件（48番）
  - docs/SOCRATIC-50-improvements.md に Q57 を追加（残る同型候補として
    regression/binary/ranking_flip_rate の小標本過信リスクを明記）。

- **calibration.check_systematic — 点推定でなく上側限界(bias+stderr)で判定（第20回）**:
  第19回の Q55 で指摘した残課題を解消。`systematic_divergence`（RMS 比バイアス）は
  N 要素からの単一点推定に過ぎず、小テンソル（N 小）ではたまたま小さい bias が出て
  真の系統誤差を見逃しうる（偽OK）。`rollout.flip_rate_upper_bound`（0 フリップ観測でも
  p=0 と過信しない）と同じ fail-safe パターンを `calibration.check_systematic` にも適用。

  - `calibration.systematic_divergence_stderr(a, b, n_boot=200, seed=0)`: 要素の
    ブートストラップ再標本化で bias 推定の標準誤差を実測する新関数。
  - `CalibrationReport` に `bias_stderr: float` フィールドと `bias_upper_bound`
    プロパティ（`= |bias| + bias_stderr`）を追加。
  - `check_systematic()`: 判定に `bias`（点推定）でなく `bias_upper_bound` を使う。
    N が大きい（典型的な GEMM 出力）場合は stderr が無視できるほど小さく、従来の
    点推定判定と実質同じ挙動（回帰なし）。
  - 実証: N=4 の小テンソルで 1 要素だけ 5% 系統摂動させると、bias 点推定は
    たまたま極小（旧ロジックなら OK 判定）になるが、bias_upper_bound は閾値を
    大きく超え正しく BLOCK になる。
  - tests: `test_systematic_check_uses_upper_bound_not_point_estimate_for_small_n`・
    `test_systematic_check_large_n_unaffected_by_upper_bound`
  - verify.py: 121→124 不変条件（47番）
  - docs/SOCRATIC-50-improvements.md の Q55 を修正済みに更新。

- **rollout: 初回発散ステップの中央値を接続（過剰実装の発見・第19回）**:
  ソクラテス式に「不足」（未接続の facade）でなく「過剰」（呼ばれない実装）の観点から
  機械スキャンし直した結果、`rollout.divergence_step_quantile()`（初回発散ステップの
  q 分位）が唯一の**完全デッドコード**（テストからも一切呼ばれない）だったことを発見。

  平均 `expected_divergence_step`（=1/p）だけの報告は、初回発散ステップが従う幾何分布の
  右裾に引っ張られ「典型的にはもっと長く保つ」と楽観視させる危険がある
  （p=0.01 で平均100 vs 中央値69）。単なる削除でなく、統計的に意味のある値だったため
  `analyze_rollout`/`RolloutReport` に接続する判断をした。

  - `RolloutReport` に `median_step: float` フィールドを追加。
  - `analyze_rollout()`: `divergence_step_quantile(p, 0.5)` を計算し `median_step` に格納、
    `to_text()`/WARN/BLOCK メッセージに「平均tok X / 中央値tok Y」の両方を表示。
  - tests: `test_median_divergence_step_is_smaller_than_mean`
    （幾何分布の解析的性質・p=0.01 での具体値・RolloutReport 接続を検証）
  - verify.py: 120→121 不変条件（rollout セクションに追加）
  - docs/SOCRATIC-50-improvements.md に G 節（Q51-56・過不足を機械的に探す手法のまとめ）を追加。

- **equivalence.classify_divergence を audit_runtime の equivalence phase に接続（第18回）**:
  第11-17回で見つけた「機能は実装済みだが facade 未接続」欠陥を手作業の調査でなく、
  audit.py のソーステキストに対する各モジュールの公開関数の機械的な参照スキャンで
  再現・拡張して探索した結果、`equivalence.classify_divergence()`
  （LAYOUT vs 真の数値発散の判別・実装済みだが未接続）を発見・修正（7件目）。

  `classify_divergence` は element-wise 不一致が「レイアウト不一致」（転置・再タイル・
  値は正しいが位置違い）か「真の数値発散」かを、値の多重集合（ソート済み全要素）が
  保存されるかで判別する。だが `audit_runtime` の equivalence phase は DIVERGENT を
  一律 BLOCK にするだけで両者を区別していなかった —— 修正すべき箇所が全く異なる
  （LAYOUT は codegen の整列問題、DIVERGENT は数値精度チューニング）ため、この
  未接続は診断上の手がかりを捨てていたことになる。

  - `audit_runtime` の equivalence phase: `verdict == "DIVERGENT"` かつ形状一致時、
    `classify_divergence(af, bf, K, dtype)` を実行し `DV_LAYOUT` なら
    「値の多重集合は一致 → レイアウト不一致の疑い」を明記する。
  - tests: `test_audit_runtime_equivalence_distinguishes_layout_from_true_divergence`
    （転置は LAYOUT タグ付き BLOCK・真のスケール発散は LAYOUT タグ無し BLOCK を確認）
  - verify.py: 118→120 不変条件（46番）

  スキャン手法（今後の回帰防止に有効）: `python/tsugi/*.py` の公開関数名（`def [a-z]...`）を
  列挙し `audit.py` のソーステキストに文字列として現れるかを確認する。多くの false
  positive（自モジュール内でのみ使われる正当な低レベルヘルパー）が出るため人手の判断は
  要るが、「本来 facade に届くべき checker 関数」の候補を安価に洗い出せる。

- **audit_runtime(fn_a=..., worst_samples=...) — worstcase.analyze_worst_case を能動探索として接続（第17回）**:
  `worstcase.analyze_worst_case()`（唯一の能動探索層・平均ケース検証の盲点を露出）は
  実装・テスト済みだが、`audit_runtime()` は受動的な代表サンプル比較しか行わず、認証
  エンベロープ内に隠れる反例を能動的に探すことは一度もしていなかった（第11-16回で
  見つけた「機能は実装済みだが facade 未接続」と同型の欠陥の6件目）。呼び出し規約の
  違い（`fn_a(x)`/`fn_b(x)` という決定論的な x→出力の callable、`run_a(seed)` とは別物）
  から接続が後回しになっていたが、`audit_runtime` の追加オプション引数として自然に収まった。

  - `audit_runtime(..., fn_a=None, fn_b=None, worst_samples=None, worst_radius=1.0,
    worst_steps=400, worst_seed=0, worst_bounds=None, worst_tol=None)`:
    `fn_a`/`fn_b`（x を取る決定論的ベンダー実装）＋ `worst_samples`（代表入力列）を渡すと
    `analyze_worst_case` を実行し、新設の「worstcase 能動探索」phase に典型/最悪発散・
    再現可能な反例 `x_worst` を含める。`worst_tol` 未指定時は既に計算済みの `eq.atol` を
    流用（worst-case 探索に別基準を課したい場合は明示的に上書き可能）。
  - `fn_a`/`fn_b`/`worst_samples` 未指定時は従来通り worstcase phase は現れない（後方互換）。
  - tests: `test_audit_runtime_worst_case_search_finds_envelope_counterexample`
    （fp16/fp32 累積精度差のある古典例で、代表サンプルは tol 内でもエンベロープ内の
    能動探索で tol 超の反例が見つかり BLOCK になることを実証）
  - verify.py: 116→118 不変条件（45番）

- **audit_runtime(layers_a=...) — attribution.diagnose を層別診断として接続（第16回）**:
  `attribution.diagnose()`（onset/spike 特定＋blame 統合の集大成関数）は実装・テスト済みだが、
  `audit_runtime()` は BLOCK 判定を出すだけで「どの層で・どちらのベンダーが」発散源かを
  一度も特定していなかった（第11-15回で見つけた「機能は実装済みだが facade 未接続」と
  同型の欠陥の5件目）。

  - `audit_runtime(..., layers_a=None, layers_b=None, layers_oracle=None, x0=None, layer_names=None)`:
    `layers_a`/`layers_b`（層 callable のシーケンス）を渡すと `attribution.diagnose` を実行し、
    新設の「attribution 層別診断」phase に onset（発散開始層）・spike（支配的増分層）・
    責帰ベンダーを含める。`layers_oracle` も渡せば spike 層でどちらのベンダーが
    正しいかまで責帰する。
  - `layers_a`/`layers_b` 未指定時は従来通り attribution phase は現れない（後方互換）。
  - tests: `test_audit_runtime_layer_diagnosis_pinpoints_divergent_layer`
    （spike 層名・責帰ベンダーがテキストに現れることを検証・未指定時の後方互換も確認）
  - verify.py: 114→116 不変条件（44番）

- **audit_runtime(task=...) — 非分類タスク（回帰/二値/ランキング）を decision 層に接続（第15回）**:
  `decision.compare_task()`（regression/binary/ranking）は実装・テスト済みだが、
  `audit_runtime()` は常に `compare_decisions()`（分類 argmax 専用）を呼んでおり、
  非分類タスクは decision 層の恩恵を受けられなかった（第11-14回で見つけた
  「機能は実装済みだが facade 未接続」と同型の欠陥の4件目）。回帰モデル（価格/物理量）・
  二値分類（診断/異常検知）・検索/推薦（ranking）は argmax を持たないため、
  この接続漏れは分類以外のワークロード全体が対象外になっていたことを意味する。

  - `audit_runtime(..., task: str = "classification", task_kwargs: dict | None = None)`:
    `task` が `"classification"` 以外（`"regression"`/`"binary"`/`"ranking"`）なら
    `decision.compare_task(task=task, **task_kwargs)` に委譲する。
  - decision phase 名に `(task)` サフィックスを付け、どちらの経路を通ったか明示。
  - rollout（自己回帰 per-token フリップ合成）は argmax トークン選択の概念に依存するため
    `task == "classification"` の場合のみ発火するよう明示的にガード。
  - `task` 未指定時は従来通り `compare_decisions`（後方互換・挙動不変）。
  - tests: `test_audit_runtime_supports_non_classification_tasks`
    （regression の 100% 乖離・binary の閾値またぎ反転をそれぞれ BLOCK 判定・既定 classification 確認）
  - verify.py: 112→114 不変条件（43番）

- **audit_cross_vendor(robust=...) — Q49 の外れ値頑健 noise floor を実機入口に接続（第14回）**:
  `nondeterminism.compare_stable` は `robust=True`（10-90 パーセンタイル幅・単発グリッチに
  頑健・SOCRATIC Q49 で修正済み）をサポートするが、GPU 実機向けの主要入口
  `audit_cross_vendor()` は常に max-min（`spread`）を使っており、Q49 の修正から漏れていた
  （第11-13回で見つけた「機能は実装済みだが facade に未接続」と同型の欠陥）。

  - `audit_cross_vendor(run_a, run_b, K, ..., robust: bool = False)`: `robust=True` で
    run-to-run / batch-invariance 床に `spread_robust` を使う（既定 False で後方互換）。
  - 実証: noise floor 測定中の 16 run に単発グリッチ（1 run だけ 5e-2 倍ノイズ）を混ぜると、
    非 robust の noise floor は robust 版の **約 30,000 倍**に膨張する。その結果、真に
    `EQUIVALENT`（系統発散 0.08%）と判定できるはずのケースが `INDISTINGUISHABLE`
    （等価判定「未定義」の WARN）に押し込められ、運用上不要な triage が発生する。
    `robust=True` は同じグリッチ下でも正しく `EQUIVALENT` を返す。
  - tests: `test_audit_cross_vendor_robust_resists_single_glitchy_run`
  - verify.py: 110→112 不変条件（42番）

- **audit(sample=...) が empirical_cond を自動実測（第13回）**:
  propagation 層は増幅 op（reduce/softmax/exp）の条件数 `cond` を静的には常に 1.0
  （well-conditioned 仮定）とし、`empirical_cond(sample, kind)` という実測関数は存在するのに
  `audit()` から一度も呼ばれていなかった（Q7/Q8/Q11 の未接続 — 関数はあるが経路が繋がっていない）。
  第12回で `certify_from_sample` を接続したのと同型の欠陥を propagation 層でも発見・修正。

  - `audit(module, cfg, ..., sample=sample)`: `sample` が与えられると、グラフ中の増幅 op
    （`is_amplifier(kind)` かつ `cond==1.0`）の `cond` を `empirical_cond(sample, kind)` で
    実測してから `propagate()` する。
  - 実証: softmax カーネル（reduce×2/exp）を trace した例で、`sample` 無しは
    `model_divergence≈1.17e-2`（cond=1 の過小評価）、`sample` 有りは `≈7.82e-2`
    （実測 cond≈2.8 を反映）——約 6.7 倍の過小評価が是正される。
  - `sample` 未指定時は従来通り「静的 cond=1 は下界」WARN を維持（後方互換・情報を隠さない）。
    `sample` 有り時は「実測済み」に文言を切替え、`audit(sample=...)` へ誘導する文言も追加。
  - tests: `test_audit_sample_auto_measures_empirical_cond`
    （sample 有無で model_divergence が変わることを固定・text の分岐を検証）
  - verify.py: 108→110 不変条件（41番）

- **audit() facade に certify_from_sample を接続（第12回）**:
  第11回で `envelope.certify_from_sample` を追加したが、製品の主要 facade である
  `tsugi.audit.audit()` は依然 `certify_gemm(K, "float16", 1.0)` の scale=1 固定のままだった
  （関数は存在するが呼び出し経路に未接続 = 実質使われない改善）。

  - `audit(module, cfg, ..., sample=None)`: `sample`（代表テンソル）を渡すと numerics phase が
    `certify_from_sample` で実 RMS scale を認証する。
  - `sample` 未指定時は `"scale=1.0 仮定"` を numerics phase のテキストに明記し、
    暗黙のデフォルトが隠れないようにする（fail-safe: 前提を必ず可視化）。
  - tests: `test_audit_numerics_uses_sample_scale_when_given`（sample あり/なし双方のテキスト検証）
  - verify.py: 106→108 不変条件（40番）

- **長所短所分析に基づく3点改善（第11回）**:
  コードベース全体調査（13 モジュール・100 不変条件・27 テストファイル）とソクラテス50問の
  残余ギャップ分析から実装。

  1. **`certify_from_sample(x, K, dtype)` — scale=1 暗黙仮定の排除（Q14）**:
     `certify_gemm(scale=1.0)` で認証後に実 RMS が 50 のテンソルを `check_tensor` に通すと
     スケール超過 BLOCK が誤発火する根本問題。代表サンプルから RMS を実測して認証する
     `certify_from_sample` を追加。LLM の未正規化活性（scale~数十）・正規化後（scale~1）・
     FP8 量子化後（scale~小）など層ごとに scale が大きく違うモデルで重要。
     - `envelope.certify_from_sample(x, K, dtype, cond, noise_floor)`: RMS 計測 → certify_gemm
     - ゼロテンソル（scale→0 で除算）の防御: `max(scale, 1e-30)`
     - tests: 3 ケース（scale=50 誤 BLOCK 解消・ゼロ除算防止・小 scale 追従）

  2. **backend 登録の冪等化（Q28）**:
     `tsugi_torch.register()` は import 副作用で自動登録するが、モジュール reload / 二重 import で
     重複登録しうる（`register_backend` はベンダーによっては既登録名でエラー）。
     `_BACKEND_REGISTERED: bool` フラグを追加し、既登録なら即 return で冪等化。

  3. **`audit_fx()` に `ref_scale` 追加（Q14/Q13 の FX 橋サイド）**:
     `ref_logits` を渡した場合に実 RMS scale を測定して `"ref_scale"` を出力に含める。
     `certify_from_sample` へ渡す scale のヒントになり、「FX 静的検査 → envelope 認証」の
     経路を data-driven に接続する。
  - tests: `test_audit_fx_ref_scale_from_logits` (scale RMS 精度 1%)
  - verify.py: 100→106 不変条件（37-39 番）

- **torch.compile dynamic shape 検出（外部調査ベース・第10回）**:
  外部調査（PyTorch 公式ドキュメント「Dynamic Shapes Core Concepts」・
  Edward Yang blog「State of torch.compile August 2025」・GitHub issue #165506）の知見に基づく。

  研究知見: torch.compile は実行時の入力形状に shape guard を立てる。guard 違反で
  再コンパイルが発火し、形状ごとに異なるカーネルを特化する。特化カーネルは
  タイル幅・縮約順序・アキュムレータ幅が変わるため、A形状で認証した等価性は
  B形状には転用できない。Inductor はバッチ閾値を超えると atomics 経路に切り替える
  ため、batch_size=32 と batch_size=64 で異なる数値特性になる場合がある。

  実装:
  - `fxbridge._node_is_symbolic(node)`: shape meta に `int()` 変換で失敗する次元（`torch.SymInt`）
    が含まれるかを検査。torch.compile(dynamic=True) や torch.export 経路で生成される
    symbolic 次元を検出する。
  - `fxbridge.audit_fx()`: `has_dynamic_shapes: bool` フィールドを出力に追加。
    グラフに symbolic 次元があると `True` → per-shape 再検証が必要のシグナル。
  - `tests/correctness/test_fxbridge.py`: `_SymInt` duck-type クラス（`int()` で TypeError）と
    `test_audit_fx_warns_dynamic_shapes` 追加（dynamic/static/meta なし の 3 ケース）。
  - `verify.py`: invariant 35-36 を追加（dynamic → True・static → False）。
    合計 100/100 不変条件に更新。

- **FP8 (OCP OFP8: E4M3 / E5M2) dtype サポート（外部調査ベース・第9回）**:
  外部調査（Qiita/Zenn の FP8 量子化記事・OCP 8-bit Floating Point Specification・
  NVIDIA Transformer Engine）の知見に基づく。FP8 は H100/MI300/B200 世代の推論で主流
  （vLLM ネイティブ・キャリブレーション不要）になったが、Tsugi の dtype テーブルに欠けていた。
  TF32 と同じ 3 テーブル拡張パターンで追加:
  - **E4M3**（4 指数・3 仮数・max=448・無限大なし）= 重み/活性の前向き。u=0.0625。
  - **E5M2**（5 指数・2 仮数・max=57344・inf/nan あり）= 勾配の後ろ向き。u=0.125（最も粗い）。
  - `tolerance.UNIT_ROUNDOFF`: `float8_e4m3=2^-4`・`float8_e5m2=2^-3`。
  - `equivalence.TOLERANCE`: `float8_e4m3=1e-1`・`float8_e5m2=2e-1`（fp16 より大幅に緩い）。
  - `envelope.DTYPE_LIMITS`: `float8_e4m3`(max=448)・`float8_e5m2`(max=57344)。
  - **クロスベンダーの真のリスクは丸めでなく per-tensor amax スケーリング係数の差**: FP8 は値域が
    狭く（E4M3 は max=448）amax 正規化が必須。各ベンダーが amax を別の縮約順序で計算するとスケールが
    ずれ、テンソル全体が系統シフトする（calibration の系統検査が効く）。docstring に明記。
  - エンベロープ検査が FP8 で特に重要: 未スケールの活性（例 1000）が E4M3 では即 overflow（fp16 なら
    範囲内）。dtype 依存の overflow リスク差を実証。
  テスト 4 件追加（test_equivalence.py +2、test_envelope.py +2）。verify.py 不変条件 4 件（98/98）。

- **atomicAdd 由来の非決定 op 静的カタログ（PyTorch 公式由来・第8回）**:
  外部調査（PyTorch 公式 randomness ドキュメント・Qiita「seed、本当に固定できた?」・
  arXiv:2408.05148 浮動小数非結合性）の知見に基づく。PyTorch は `scatter_add` / `index_add` /
  `bincount` / `embedding_bag`(backward) / `ctc_loss` / プーリング・サンプリングの backward が
  **atomicAdd を使い seed 固定でも run-to-run で揺れる**ことを明示している。Tsugi は noise floor を
  *実測* する機構（`measure_noise_floor`）を持つが、**どの op が本質的に非決定か**を事前に
  宣言する静的カタログが欠けていた。実行前に「このグラフは静的許容では不十分・noise floor 実測が
  必須」と告げられるのが価値。
  - `nondeterminism.ATOMIC_NONDET_OPS`: PyTorch 公式が atomicAdd 由来と明示する 13 op のカタログ
    （op→非決定の理由 forward/backward）。
  - `op_is_nondeterministic(name)` / `nondeterminism_reason(name)`: 前方一致で表記揺れ
    （`scatter_add_`・`aten.scatter_add.default`・`max_pool2d`・大文字）を吸収。
  - `classify_nondeterminism(op_names) -> NondetCatalogReport`: グラフの op を走査し非決定 op を
    列挙。`requires_noise_floor` で「静的許容では不十分」を宣言。
  - **FX 橋（torch.compile 経路）への結線**: `fxbridge.audit_fx` が `requires_noise_floor` と
    `nondeterministic_ops` を返すようになり、**コード生成前に**非決定 op を audit に届ける。
    `fx_call_target_names()` で生 target 名を抽出して照合。
  テスト 4 件追加（test_nondeterminism.py +3、test_fxbridge.py +1）。verify.py 不変条件 3 件（94/94）。

- **TF32 dtype サポート: NVIDIA Ampere+ の fp16 級精度 GEMM に正しい許容を適用（第7回）**:
  外部調査（Qiita/Zenn/PyTorch 公式/NVIDIA 技術ブログ）の知見に基づく修正。
  TF32（TensorFloat32）は NVIDIA Ampere+ GPU が float32 GEMM/conv に使うハイブリッド形式で、
  fp32 指数部（8 bit）+ fp16 仮数部（10 bit）= 精度は fp16 と同等。AMD ROCm は TF32 非対応のため
  NVIDIA(TF32) vs AMD(full fp32) 比較では最大 ~1e-3 の誤差が生じる（float32 許容の 1e-4 では偽BLOCK）。
  `dtype="tf32"` を明示して fp16 と同等の許容（atol/rtol=1e-2）で比較するよう 3 箇所を拡張:
  - `tolerance.UNIT_ROUNDOFF["tf32"] = 2^-11`（fp16 と同等の unit roundoff）。モジュール docstring に
    `torch.backends.cuda.matmul.allow_tf32`（PyTorch 1.12 以降デフォルト False）・
    `torch.backends.cudnn.allow_tf32`（conv 側、デフォルト True）の説明を追記。
  - `equivalence.TOLERANCE["tf32"] = {atol=1e-2, rtol=1e-2}`（fp16 と同等）。
  - `envelope.DTYPE_LIMITS["tf32"]`（fp32 と同等の overflow リスク・TF32 は指数部が fp32 と同じ）。
  テスト 3 件追加（`test_tf32_tolerance_matches_float16`・`test_tf32_unit_roundoff_matches_float16`・
  `test_tf32_dtype_limits_match_float32`）。

- **NaN/Inf 明示タグ `has_non_finite` を `EquivalenceReport` に追加（第7回）**:
  外部調査（Zenn 半精度安定性記事・HuggingFace DebugUnderflowOverflow）の知見に基づく。
  現行の `_compare_with()` は NaN を element-wise 比較の mismatch としてカウントするが、
  **「精度発散（finite な差）」と「データ破壊（NaN/Inf 伝播、overflow/除零）」は根本原因が異なる**。
  NaN はアルゴリズム精度問題でなく上流の overflow/入力破損で生じ、修正方針も異なる。
  - `EquivalenceReport.has_non_finite: bool = False` フィールド追加（デフォルト後方互換）。
  - `_compare_with()` で `np.isfinite()` による事前検出を追加。NaN/Inf を含む配列は
    `has_non_finite=True`、`equivalent=False`（BLOCK）として返す。
  - `to_text()` に `[NaN/Inf検出]` タグを追加（スタックトレースなしに根本原因を絞り込める）。
  - `abs_err.max()` を `np.nanmax()` に変更（NaN を含む場合も max_abs_err が有限値を返す）。
  テスト 1 件追加（`test_nan_in_output_flagged_as_non_finite`）。

- **`float64` を `envelope.DTYPE_LIMITS` に追加: float32 フォールバック防止（第7回）**:
  `DTYPE_LIMITS` に `"float64"` が欠落していたため、`dtype_limits("float64")` が float32 の限界値
  （max_normal=3.4e38）にフォールバックしていた。float64 の max_normal=1.8e308（float32 の 270 桁超）
  なので、float64 テンソルの値 1e100 が overflow BLOCK と誤判定される偽陽性の源。
  - `DtypeLimits("float64", max_normal=1.7976931348623157e308, min_normal=2.2250738585072014e-308, ...)`
  - `DtypeLimits("tf32", ...)` も同時追加（fp32 と同等の overflow 範囲）。
  テスト 2 件追加（`test_float64_dtype_limits_are_correct`・`test_tf32_dtype_limits_match_float32`）。
  `test_envelope.py` の型別デモ表示に `float64`・`tf32` を追加。
  verify.py 不変条件 5 件追加（91/91）。

### Fixed
- **docs/SOCRATIC-50-improvements.md の Q28 表記漏れ（第16回）**: 第11回で
  `tsugi_torch._BACKEND_REGISTERED` により冪等化済みだったが、ドキュメントの ✅ マークが
  漏れていた（コードは直っていたが記録が追いついていなかった）。表記を修正済みに更新。
- **float64 が float32 の緩い許容にフォールバックする偽OK バグを修正（外部調査ベース・第6回）**:
  Qiita/Zenn と PyTorch 公式の `torch.testing.assert_close` 調査から、dtype 別許容の標準
  （fp16=1e-3 / fp32=1e-4 / **fp64=1e-8**）を確認。Tsugi の `equivalence.TOLERANCE` と
  `tolerance.UNIT_ROUNDOFF` 両方に **float64 が欠落**しており、`compare(a, b, "float64")` が
  float32 の許容（atol=1e-4・u=5.96e-8）にフォールバックしていた。倍精度の真の精度は ~1e-16 で、
  **5〜8 桁緩い許容での偽OK**（Tsugi の最重要リスク「偽OK は致命的」）を生む欠陥。
  - `equivalence.TOLERANCE` に `float64=(atol=1e-7, rtol=1e-7)` を追加。
  - `tolerance.UNIT_ROUNDOFF` に `float64=2^-53 (≈1.11e-16)` を追加（calibration の検出限界も連動修正）。
  - 回帰テスト 2 件（`test_float64_does_not_fall_back_to_float32_tolerance` /
    `test_float64_accepts_genuine_double_precision_noise`）＋ verify.py 不変条件 2 件（86/86）。

### Changed
- **型注釈の補完（API 明確化）（第5回）**:
  1. **attribution/blame 関数の型注釈追加**: `layer_divergences` / `attribute` / `diagnose` /
     `layer_blame` / `compare_accuracy` / `accuracy_relative` のパラメータに型注釈を追加。
     `layers_a: Sequence[Callable]` / `x: np.ndarray | Sequence` のように API の引数の型を明示。
  2. **import の追加**: `collections.abc.Callable`, `Sequence` を両ファイルに追加。
  3. 効果: IDE 自動補完が機能・新ユーザーの理解が容易・型チェッカーが検証可能になる。

- **定数の統合と散在の排除（第4回）**:
  1. **blame の `ratio_threshold` を共有定数に**: `compare_accuracy` で `ratio_threshold=2.0` を
     デフォルトとしていたが、`diagnose()` が同じ値を硬コーディングしていた。
     `_RATIO_THRESHOLD_TIED = 2.0` を blame.py に定義し、`BlameReport` のデフォルトと
     `attribution.diagnose()` の判定がこれを参照するようにした（シングル・ソース・オブ・トゥルース）。
  2. **calibration の系統バイアス警告閾値定数化**: `0.5 * thresh` を `_WARN_BIAS_RATIO = 0.5` に。
     BLOCK 閾値の半分を警告ポイントとする保守的な設定の根拠を docstring に記述。

- **DRY 再発防止 + 残存マジック数の排除（第3回）**:
  1. **`_LayerProfileMixin` で `spike_name`/`onset_name` を一元化**: `AttributionReport` と
     `DiagnosisReport` が同一の層名解決ロジックを別々に実装しており、片方がメソッド・片方が
     `@property` に drift した（第2回で修正）。共有ミックスインに集約し、両クラスが継承する
     ことで構造的に再発を防止。重複 ~20 行を削減。
  2. **attribution.py の `tol * 10` を `_BLOCK_DIV_RATIO = 10.0` 定数化**: `attribute()` と
     `diagnose()` に残っていた 2 箇所のマジック数を排除。blame の `_BLOCK_DIST_RATIO` と同一論拠を
     docstring に明記し、診断チェーン全体で同じ閾値を使うことを示した。

- **5つの弱点を修正（長所短所分析→改良・第2回）**:
  1. **API 一貫性**: `AttributionReport.spike_name`/`onset_name` をメソッドから `@property` に変換。
     `DiagnosisReport` はすでに `@property` だったため、両クラスの呼び出し規約が一致した。
     呼び出し側が `()` の有無を意識する必要がなくなる。
  2. **names 正規化**: `attribute()` と `diagnose()` で渡す `names` リストの長さを `len(divs)` に
     自動整合（長すぎれば切り捨て、短すぎれば `layer[i]` で補完）。これまで names が divs より長いと
     `layer_names` と `divs` のインデックスがズレていた。
  3. **envelope の閾値定数化（Q4）**: `0.1`（overflow 近接 WARN）・`0.7`（exp-overflow WARN）・`1.5`
     （scale BLOCK）を `_OVERFLOW_WARN_FRAC`/`_EXP_WARN_FRAC`/`_SCALE_BLOCK_RATIO` に。
     根拠コメントを定数 docstring に記述。
  4. **decision の閾値定数化（Q5）**: `0.5 * overall_margin_median`（near-tie 判定）・
     `10 * flip_budget`（BLOCK 格上げ比率）・`0.01`（最小 BLOCK フリップ率）を名前付き定数に。
  5. **blame の TIED→BLOCK 格上げ（Q31）**: TIED かつ両ベンダーが `tol × _BLOCK_DIST_RATIO` 超の
     場合、WARN → BLOCK に格上げ。「両方同程度に間違っている（共有前提の誤り）」という系統的失敗は
     BLOCK 相当。テスト: `test_compare_accuracy_tied_both_far_becomes_block`。
  テスト合計: test_attribution.py 34 件、test_blame.py 20 件、84/84 不変条件。

- **3つの弱点を修正（長所短所分析→改良・第1回）**:
  1. **Oracle 失敗時の blame スキップ**: oracle_check が BLOCK のとき blame は汚染された oracle で
     走り誤ったベンダー指摘を出す可能性があった。`oracle_healthy` フラグで guard し、不健全な oracle
     では「blame はスキップ（誤指摘を防ぐ）」と明示する。
  2. **blame の `tol * 10` を `_BLOCK_DIST_RATIO = 10.0` に名前付き定数化**: tolerance.derive_tolerance
     の safety マージン込み tol のさらに 10× 超は系統的誤りとして BLOCK に格上げする、という判断根拠を
     docstring に記述。マジックナンバーを排除し感度分析を可能にした。
  3. **`diagnose()` 統合診断 API を attribution.py に追加**: これまで attribution と blame は
     孤立 API で開発者が手動で結合する必要があった。`diagnose(layers_a, layers_b, layers_oracle, x, *)`
     が1回の呼び出しで onset/spike（attribution）と spike 層での責帰（blame: spike_closer=A/B/TIED）を
     返す。`DiagnosisReport.to_text()` が "fix vendor X" を含む完全診断チェーンを出力。
  4. **audit.py docstring を 8→13 視点に更新（前コミット）**。

- **blame を `audit_runtime` の correctness 層に統合（診断チェーンを製品経路で閉じる）**:
  oracle を渡すと shared-mode 検出に加え blame（新視点13）が走り、「vendor X の実装を優先修正」を
  verdict に算入。両ベンダーが oracle 内なら「責帰不要」、片方が乖離なら修正方向を明示、
  同程度の乖離なら「両実装/oracle を疑う」。これまで `compare_accuracy` は孤立 API だったが
  製品経路（audit）から到達可能になり、attribution（どの層）→ blame（どちらのベンダー）の
  診断チェーンが audit で閉じる。audit.py docstring を 13 視点に更新。

### Added
- **新視点13: ベンダー責帰 — どちらのベンダーが oracle に近いか？（`tsugi.blame`）**: BLOCK 判定後の
  「どちらのベンダーを修正するか」問題を解決（Q1-Q5 ソクラテス連鎖）。`accuracy_relative` が
  oracle への相対距離を計算、`compare_accuracy` が dist_a / dist_b の ratio で `closer=A/B/TIED`
  と責帰方向を判定、`layer_blame` が per-layer で (dist_a, dist_b) を返す。attribution（どの層か）
  と組み合わせて "layer X の vendor Y 実装を直せ" という完全診断チェーンを完成させる。
  oracle_check（shared mode 検出）と相補的: oracle_check は "共有モード障害の有無"、blame は
  "責帰の割り当て"。tests/correctness/test_blame.py 19 テスト全通過。verify.py invariants 29-30
  追加（82/82）。
- **新視点12: 発散帰属 — 出力の不一致はどこから来るか？（`tsugi.attribution`）**: 移植失敗時の
  O(L) 手動デバッグを O(log L) に短縮する per-layer 因果特定層。`layer_divergences` が両ベンダー
  同一入力で各層出力の発散を prefix scan し、`find_onset`（汚染開始層）・`find_spike`（最大増幅層）
  を特定。`bisect_onset` が prefix-forward 構造を使い O(log L) で onset を探索。
  `attribute` が `AttributionReport` を返し onset/spike/全層プロファイルをレポート。
  onset ≠ spike（汚染開始と最大増幅が別層）を INFO で通知。`propagation.dominant`（理論予測）と
  `attribution.spike`（実測）の照合で理論検証/反証の接続点を提供。tests/correctness/test_attribution.py
  で 27 テスト全通過。verify.py で invariants 27-28 追加（77/77）。
- **新視点11: タスク多様性 — argmax ⇏ 全タスク（`tsugi.decision` 拡張）**: argmax 分類専用だった
  decision 層を非分類タスクへ拡張（Q29/Q30）。`regression_flip_rate`（値の相対/絶対許容乖離）・
  `binary_flip_rate`（sigmoid+threshold 跨ぎ）・`binary_margin`（決定境界距離）・
  `ranking_flip_rate`（top-k 集合変化・listwise）・`compare_task(task=regression/binary/ranking)`。
  バイナリ sigmoid に argmax を適用すると flip_rate = 0 に固まり何も測れていなかった（静かな誤用）。
  未知 task 種別は ValueError で弾き、静かな誤計算を防ぐ（fail-safe）。
- **新視点9: 自己回帰的等価（`tsugi.rollout`）**: per-token フリップ率 p を生成長 L へ合成。
  自己回帰生成は一度ズレたら戻らず survival=(1−p)^L で複利減衰（p=1% でも L=100 で 37%）。
  `sequence_survival`/`expected_divergence_step`/`safe_generation_length`/`analyze_rollout`/
  `rollout_from_logits`/`simulate_rollout`(Monte Carlo 確認)。`audit_runtime(gen_length=L)` で
  rollout 層を verdict に算入。propagation（per-kernel⇏per-model）の自己回帰版。
- **lowering spec を DSL 全 op に同期**: `VENDOR_LOWERING` を 6→14 op に拡張
  （`sub`/`mul`/`div`/`max`/`exp`/`sqrt`/`rsqrt`/`reduce` を追加）。softmax/attention/
  norm 系カーネルが silently `<UNSUPPORTED>` に落ちる穴を塞ぐ。`lowering.unlowered_ops(target)`
  と `tracer.EMITTABLE_OPS`（唯一の出所）で DSL↔lowering の drift を不変条件として検出。
- **verdict の provenance スタンプ**: `Audit.certificate`/`is_stale(**env)`/`stamp()`。
  `audit`/`audit_runtime`/`audit_cross_vendor` が `provenance={...}`（cuda/rocm/driver 等）で
  verdict を環境フィンガープリントに束ね、スタック更新で stale（再検証要）を自動判定。
  provenance 層を facade に統合（「一度認証=永遠に有効」を排す）。
- **docs/SPEC-verification.md**: 検証 API の規範仕様（全関数の契約・判定意味論・保証/盲点・
  検証連鎖の接地構造）。仕様化で `audit_runtime` の correctness 欠落が判明し下記を実装。
- **audit_runtime の correctness 層**: `audit_runtime(..., oracle=)`。oracle を渡すと
  oracle_check（oracle 自体の信頼性）＋ detect_shared_mode を回し、a≈b でも両方 oracle と
  不一致なら共有モード障害として BLOCK。oracle 無しは portability のみ（correctness は未確定と明示）。
- **証明書の陳腐化検出（temporal drift）**: `tsugi.provenance`。verdict は point-in-time —
  特定スタック(ROCm/CUDA/driver/compiler/numpy)で計算される。`certify(verdict, **env)` で
  verdict を環境フィンガープリントに束ね、`is_stale`/`changed_fields` でスタック更新時の
  陳腐化を検出（再検証要）。「一度認証=永遠に有効」の誤りを防ぐ。docs/PERSPECTIVE-provenance.md。
- **outlier feature 検出（単一スケール仮定の破綻）**: `envelope.check_outlier_features` /
  `channel_scale_spread`。実 LLM 活性は一部チャネルが 100–1000x 大（massive activations）。
  tolerance/envelope/floor の単一 global scale 仮定が破綻し outlier チャネルが誤許容で判定される。
  チャネル scale 広がりで検出し per-channel 検証要を WARN（実測: N(0,1) ~1 / outlier ~208）。
- **レイアウト不一致 vs 数値発散の分類**: `equivalence.classify_divergence(a, b, K)`。
  cross-vendor は同じ論理テンソルを異なるレイアウト(転置/タイル順)で書きうる。素朴な
  element-wise は転置-but-equal を BLOCK と誤判定するが、レイアウト不一致は値の多重集合を
  保存する → EQUIVALENT/LAYOUT(整列バグ・数値は正しい)/DIVERGENT(真の発散)を区別。
- **同点（tie-break 規約）診断**: `decision.tie_rate(logits, eps)`。argmax は同点で規約依存
  （np は先頭 index）。2 ベンダーが異なる規約だと数値一致でもフリップ＝ハード発散でない。
  量子化/マスクで多発（実測 29.5%）。`compare_decisions` が同点率高で WARN し flip 誤帰属を防ぐ。
- **オラクル自体の検証（無限後退を断つ）**: `oracle_check.verify_oracle()`。shared-mode 検出は
  オラクルを信頼するが、オラクル(NumPy)も実装。実装非依存のメタモルフィック関係(matmul 恒等・
  分配則・sum(ones)=n・exp(a+b)=exp(a)exp(b)・softmax が 1 に和し shift 不変・rsqrt 恒等)＋
  高精度(float64)照合で、第二オラクル無しにオラクルを *検証* する。rtol=0 で逸脱を捕える能力も担保。
- **共有モード障害の検出（一致≠正しさ）**: `calibration.detect_shared_mode(a, b, oracle, K)`。
  cross-vendor 一致は *必要条件であって十分条件でない* —— 両ベンダーが同じバグを共有すると
  A≈B で「等価」=緑だが両方誤り（convergent エラー）。cross-vendor 検証は構造的にこれに盲目で、
  oracle 照合でのみ SHARED_MODE/DIVERGENT/OK を区別できる。本番(oracle 無)では検出不能と明記。
  docs/PERSPECTIVE-shared-mode.md。
- **生成タスク向け top-k 候補集合フリップ**: `decision.topk_flip_rate(a, b, k)`。LLM 生成は
  top-k/top-p から選ぶので候補集合の一致を測る。k=1 で argmax フリップ率に一致・k で単調・
  スケール不変。決定空間の等価判定を分類(top-1)から生成(top-k)へ拡張。
  `compare_decisions(topk=k)` に統合し DecisionReport に併記（製品経路から到達可能）。
- **生成タスク向け top-p（nucleus）集合フリップ**: `decision.nucleus_flip_rate(a, b, p, temperature)`。
  nucleus は確率依存ゆえ argmax/top-k と違い *スケール不変でない*（温度がベンダー間一致に効く）。
  top-p 生成の候補安定性を測る。`compare_decisions(top_p=, temperature=)` に統合し
  DecisionReport に併記（製品経路から到達可能）。
- **torch backend の静的→タスク翻訳**: `audit_fx(gm, ref_logits=)` がモデル発散を判断
  フリップ率上界 `task_flip_bound` に翻訳。backend は example 出力を代表 logit として
  best-effort 利用し、`model_divergence` でなくユーザーに見える「予測フリップ率」を warn
  （propagation→decision を FX/製品経路で接続）。
- **propagation 残差トポロジ対応（Q11/Q12）**: `GraphOp(residual=True)` で残差ブロック
  y=x+f(x) を表現。skip 接続が δ_in を再増幅しないため δ_out=sqrt(δ_in²+(amp·local)²) と
  random-walk 希釈（~√L）になり、平坦チェーンの線形累積より小さい。pre-norm transformer
  の numpy 実測で「残差 < 平坦」を検証（test_propagation・verify 不変条件）。テスト 129 関数
  / verify 50 不変条件。

### Added
- **新視点10: 最悪ケース発散探索（`tsugi.worstcase`）**: 既存全視点は *代表データ上の率* を測る
  受動検証だが、本番入力を選ぶのはユーザー（時に敵対者）。認証エンベロープ（box 制約）内で発散を
  最大化する入力を微分フリー探索（黒箱・ランダム再開ヒルクライム）し、平均ケース等価 ⇏ 最悪ケース
  等価を露出。許容超過の反例がエンベロープ内に見つかれば BLOCK（envelope が緩い）。`x_worst` は
  seed 固定で再現する監査可能な反例。`divergence`/`search_worst_input`/`analyze_worst_case`。
  envelope（領域を定義）と閉ループ（worstcase がその内部を突く）。ML の adversarial/fuzzing の移植版。
- **propagation の DAG 対応（Q12 完了）**: `propagate_dag(nodes, *, correlated=)` ＋
  `merge_divergence(divs, *, correlated=)`。線形列だけだった伝播をフォーク→合流へ一般化し、
  attention 並列ヘッド・残差・concat の series-parallel DAG を表現。合流則は独立(√Σδ²・希釈)/
  相関(Σδ・worst-case)を選べ、非対称コスト下で保守側を選択可能。numpy の 2 ブランチ合流で
  実測発散を上界することを検証。深さ(`propagate`)に続き *幅* 方向の合成を獲得。

### Changed
- **rollout をデコード方式に整合**: `rollout_from_logits(..., decode={greedy,topk,nucleus})`。
  実 LLM はサンプリング生成が主流で、候補集合が分岐すれば argmax 同一でも生成分布は分かれる。
  greedy argmax フリップ率はサンプリング生成の per-token 発散を過小評価するため、decision 層の
  集合フリップ率を再利用して運用デコードに p を整合（集合フリップ ≥ argmax＝honest に厳しい側）。

### Fixed
- **facade の rollout fail-safe 漏れ**: `audit_runtime(gen_length>0)` の rollout 層が点推定
  `flip_rate` を使い、standalone に入れた上側信頼限界（0 フリップ観測≠p=0）をバイパスしていた。
  完全一致 logit でも巨大 L で survival=100% と過信する穴を塞ぎ、facade も `flip_rate_upper_bound`
  を通すよう修正（standalone と fail-safe を一致）。テスト 167 関数 / verify 72 不変条件。

## [0.2.0] — 2026-06-16
検証層を 8 視点 + メタ(calibration) + 基盤(nondeterminism) + 翻訳(decision) へ拡張し、
統合ファサード `audit`/`audit_runtime`/`audit_cross_vendor` で 1 判定に集約。最新研究
（batch-invariance / 構造的 FP ノイズ）を取り込み、property test・ROC・FX backend 検証・
外れ値頑健ノイズ床を追加。詳細は下記 [Unreleased] からの全項目。パッケージ metadata 修正
（`tsugi` パッケージ・numpy 依存を宣言）。テスト 127 関数 / verify 49 不変条件。

### Added
- Phase 0 完成形ファイル: SPEC / ARCHITECTURE / ADR-001..004 / README / FAQ / BENCHMARK
- Phase 1 骨格: CMake, tsugi.tile/tsugi.gpu dialect (TableGen), vendor lowering skeleton,
  torch.compile backend skeleton, CI
- **リファレンス実装（CPU/NumPy・正しさの真値）**: tsugi パッケージ本体
  - dtypes / tile namespace (load/store/dot/reduce/exp/rsqrt/...) / @tsugi.jit / grid launch
  - autotune 探索（vendor別 warp/wavefront 制約・共有メモリプルーン）
- **correctness テスト（実行可能・全通過）**: matmul(square/padded) / rmsnorm / attention
  を NumPy 真値と照合。autotune 単体テスト。計 8 テスト PASS。
- CONTRIBUTING.md
- **tracer**: @tsugi.jit カーネルを tsugi.tile IR へ具体トレース（MLIR 風テキスト出力）
- **lowering plan**: IR op → 各社 intrinsic 写像（NVVM wmma / ROCDL mfma・ADR-004 を機械可読化）
- correctness/tracer/lowering/autotune テスト
- **新視点（ソクラテス問答で発見）: クロスベンダー検証層**
  - `tsugi.portability`: traced IR から移植リスク静的解析（warp/wavefront・MMA形状・bf16・累積順序）
  - `tsugi.equivalence`: 数値等価性判定。擬似ベンダーで f16 累積発散の検出を実証
  - `python -m tsugi.portcheck [kernel.py]`: 移植性レポート CLI（ユーザーカーネル対応）
  - `tsugi.occupancy`: ベンダー別占有率推定。HW定数を一次情報源の実値に
    （H100/Hopper・MI300X/CDNA3・RX7900XTX/RDNA3。docs/SOURCES.md に出典）
  - docs/PERSPECTIVE-cross-vendor-verification.md
- **新視点2（ソクラテス問答）: 導出される許容誤差**
  - `tsugi.tolerance`: 許容を K・dtype の機械イプシロンから導出（固定 1e-2 を置換）
  - `equivalence.compare_gemm`: 導出許容で GEMM 等価性判定
  - 固定閾値の過剰検出（大K GEMM の偽陽性）を解消・真の発散は依然検出
  - docs/PERSPECTIVE-derived-tolerance.md
- 一次情報源化: occupancy HW定数を公式仕様の実値に・docs/SOURCES.md
- bf16 忠実丸め: oracle が bf16 精度損失を実際に再現（従来 f32 マップで無視していた弱点修正・tolerance の u=2^-8 と整合）
- portcheck 統合: 累積 matmul に導出許容の目安を併記（portability+occupancy+equivalence を1レポート）
- **新視点3（ソクラテス問答）: 起動可能性という上流ゲート**
  - `tsugi.feasibility`: per-block 上限（smem/LDS・threads・regs）で構成が *起動できるか* を categorical 判定
  - `portability.analyze` 修正: 起動不能（旧コードは occ=0% を性能 WARN と誤分類）を **BLOCK** に正す
  - `feasibility.first_vendor_only`: 片方でしか起動しない=単一ソース約束の破綻を抽出
  - portcheck に起動可能性セクション（`TILE_CONFIG` 対応）。同一構成 m128n128k64s4w8 が
    NVIDIA 起動可 / AMD CDNA・RDNA 起動不能（LDS 64KiB 超過）を実証
  - docs/SOURCES.md に per-block 上限の出典追加
  - docs/PERSPECTIVE-launch-feasibility.md
- **新視点4（ソクラテス問答）: 合成的等価性（per-kernel 等価 ⇏ per-model 等価）**
  - `tsugi.propagation`: ベンダー間発散を op グラフに沿って伝播（δ_out = amp·δ_in + local）
  - `propagate()` が累積発散・支配的増幅 op・素朴な per-kernel 和との乖離を返す
  - `model_tolerance()`: モデルレベルで正当に生じうる発散（per-model 許容の目安）
  - ill-conditioned op（相殺 reduction・小値除算・大 exp/softmax）を amp=条件数で扱う
  - 実証（numpy）: 累積順序違いの 2 ベンダーを matmul+rmsnorm の鎖に流すと
    発散が 1→12 層で約 2000 倍に累積。モデル許容は単一カーネル許容の 12 倍
  - docs/PERSPECTIVE-error-propagation.md
- **新視点5（ソクラテス問答）: 数値エンベロープの実行時検査（静的保証の契約化）**
  - `tsugi.envelope`: 等価性を認証した前提（scale/cond/dtype 範囲）を Envelope として明示
  - `check_tensor()`: 本番入力の逸脱（NaN/Inf・fp16 overflow・denormal/FTZ・scale 逸脱）を
    単一ベンダー・oracle 不要で検出。scale 逸脱は認証 atol を無効化＝要再認証
  - `check_softmax_input()`: fp16 で生 logit が ln(65504)≈11.09 超 → exp overflow を検出
  - dtype 別エンベロープを IEEE 754 実値で（fp16 は overflow・bf16 は precision が主リスク）
  - portcheck に「認証エンベロープ（保証が有効な前提）」を併記
  - docs/PERSPECTIVE-runtime-envelope.md
- **新視点6（ソクラテス問答）: 検証器そのものの検証（偽OK の非対称コストと検出限界）**
  - `tsugi.calibration`: 検証器自身を ground-truth コーパスで採点し偽OK率を測るメタ層
  - 偽OK（発散を等価と誤判定）はオラクル無きベンダーに silent 出荷＝致命・非対称コスト
  - `detectability_floor()`: 許容判定が見逃す誤差の下限 = safety·√K·u。K で拡大
    （fp16: 256→3.1%, 2048→8.8%, 8192→17.7%）＝視点2（導出許容）の双対コスト
  - `systematic_divergence()`/`check_systematic()`: scale/K 不変な RMS 比で系統バグを相補検出
  - `is_equivalent_combined()`: max_abs（乱雑）+ 系統（相関）の fail-safe 合成判定
  - 実証: 0.5% 系統スケール誤差を max_abs 単独は全 K で見逃す（偽OK 3/6）が合成判定は 0/6
  - docs/PERSPECTIVE-verifier-calibration.md
- **新視点7（ソクラテス問答）: 非決定実行（ベンダー出力は点でなく分布）**
  - `tsugi.nondeterminism`: GPU の atomic 非決定を擬似再現し run-to-run ノイズを実測
  - 単一 run 比較は「ベンダー内ノイズ」と「ベンダー間発散」を混同（フレーク）→ 実証
  - `measure_noise_floor()`: 複数 run で noise_floor を実測（tolerance.py の決定論仮定 noise=0 を埋める）
  - `attribute()`/`compare_stable()`: クロス差を noise/tol に対し 3 状態へ帰属。
    ノイズ未満は **INDISTINGUISHABLE**（等価判定が原理的に未定義）と正直に報告
  - 第二の床: 実効分解能 = max(数値検出限界, ノイズフロア)。ノイズ律速を警告
  - docs/PERSPECTIVE-nondeterminism.md
- **新視点8（ソクラテス問答）: タスクレベル等価性（判断は数値でなく決定で測る）**
  - `tsugi.decision`: 数値発散でなく判断フリップ率（argmax/選択トークンの変化）で等価判定
  - `flip_rate()`: スケール不変（logit 10 倍で abs 誤差 10 倍でもフリップ率は同一）を実証
  - `margin()`/`predicted_flip_bound()`: フリップ率 ≤ P(margin<2δ)。数値発散→タスク影響の橋
  - `compare_decisions()`: タスク予算（例 フリップ率<0.1%）で判定。near-tie 裾外の
    フリップ＝系統的発散の疑いを警告
  - 数値等価 ⇏ タスク等価（大きな数値発散でもマージン大ならフリップ無視可能）を実証
  - docs/PERSPECTIVE-task-equivalence.md
- **統合監査ファサード（視点が出揃ったので統合）**: `tsugi.audit`
  - traced IR ＋構成から静的層（portability/feasibility/occupancy/tolerance/envelope/
    calibration）をまとめて回し、1 つの Audit 判定に集約（深刻度を単一責任で束ねる）
  - 実機データが要る層（envelope.check_tensor/nondeterminism.compare_stable/
    decision.compare_decisions）を「実行時チェックリスト」として明示・判定からは除外
  - 検証ライフサイクル（静的→動的→メタ→基盤→翻訳）を一望できる to_text
  - portcheck.report は audit へ委譲しアドホックな統合グルーを除去（DRY）
  - **propagation を統合**: traced IR を論理 op 列へ写し（K ループの dot 群を 1 matmul に集約）、
    per-kernel 静的判定を per-model（合成的等価性）へ拡張。モデル発散 vs naive 和を併記
  - **`audit_runtime()`**: 実行時チェックリストの *実行版*。実機/実データのクロスベンダー
    出力を envelope/equivalence(+noise 3 状態)/systematic/decision で束ねて 1 判定に。
    静的 audit() の鏡像。真の発散は BLOCK・ノイズ未満は INDISTINGUISHABLE
  - **examples/audit_demo.py**: 両 facade を一望する実行可能デモ（GPU 不要・スモークテスト付）。
    静的=AMD 起動不能 BLOCK、実行時=max_abs 盲点に隠れる 0.5% 系統バグを BLOCK
  - **docs/VERIFICATION.md**: 検証層の全体マップ（8 視点＋メタ＋audit・ライフサイクル図・
    2 つの床・相補的計量の索引）。各 PERSPECTIVE doc へリンク
  - **`audit_cross_vendor()`**: 実機向け入口。各ベンダーの run-to-run ノイズを実測して
    から audit_runtime（nondeterminism と audit をつなぐ正規経路・決定論を仮定しない）
  - **tests/gpu/ ハーネス**: 実機クロスベンダー監査の実行可能な契約。GPU 無しは正直に
    SKIP し、配線（ノイズ実測→監査）は擬似 run で CPU 検証。run.py [3] に統合
  - **propagation→decision 橋** `decision.flip_bound_from_divergence`: 静的 op グラフの
    相対発散を代表 logit のタスクフリップ率上界へ翻訳（第2ベンダー実行前に予測・視点4→8）。
    audit(ref_logits=...) が propagation フェーズに併記
  - **tracer 拡張（SOCRATIC 50 の最優先 Q9/Q10 修正）**: reduce/exp/sqrt/rsqrt/maximum＋
    elementwise(sub/mul/div) を IR に記録。softmax/rmsnorm がトレース可能になり、増幅 op が
    IR と audit の propagation グラフに流れる（perspective4 の実効化）
  - **docs/SOCRATIC-50-improvements.md**: 50 問のソクラテス監査で改善点を洗い出し優先度付け
  - **propagation 相対増幅の是正＋ empirical_cond（SOCRATIC Q8 修正）**: `_AMPLIFYING` を
    *相対*誤差を増幅する op のみ（reduce/softmax/exp）に是正（div/reciprocal/add は相対
    条件数 ~1 と実測確認）。`empirical_cond()` がデータ依存 cond を実測（reduce=Σ|x|/|Σx| の
    相殺・exp=max|x|）。audit は増幅 op に静的 cond=1 を当てる時 *下界* と WARN し過小評価を隠さない
  - **バッチ不変性の取り込み（2025 研究）**: `nondeterminism.simulate_batch_variant_reduction`/
    `measure_batch_variance`。LLM 推論の支配的非決定源はバッチ不変性（出力がバッチサイズに依存）で
    あり atomic 並行性ではない（Thinking Machines 2025 / SC'24 arXiv:2408.05148）。run-to-run とは
    独立した決定論的な第三の床。実効床 = max(run-to-run, batch-invariance, 数値検出限界)。
    FP ノイズは構造的（相関）で独立ガウスでない（arXiv:2511.00025）→ calibration の系統検出を裏づけ
  - **batch-invariance 床を比較経路に合流**: `compare_stable(batch_floor=)` と
    `audit_cross_vendor(run_batch=)` が実効床 = max(run-to-run, batch-invariance) を使う。
    バッチ律速のとき WARN（バッチ不変カーネルが要る）
  - **flip-bound の系統/残差分解（arXiv:2511.00025 取り込み・Q18/Q20）**:
    `decision.decompose_divergence`/`residual_divergence_rms`。argmax 保存的な per-sample
    アフィン系統成分（スケール α・切片 c）はフリップを起こさないので、bound を *残差* で評価。
    `compare_decisions` は systematic_frac を報告し、数値大でも系統的ならタスク等価と明示
    （total δ ベースの過大評価を解消・calibration の系統検出と対）
  - **equivalence を共通 Risk 基盤へ（SOCRATIC Q44/Q47）**: `EquivalenceReport` に
    `risk`/`max_risk`/`ok`（equivalent→OK / divergent→BLOCK）。全レポートが統一インターフェース。
    スカラ計量ゆえ FindingReport は継承せず理由を明記
  - **property-based テスト（SOCRATIC Q35）**: test_properties.py に 10 性質 × 200 試行の
    ゼロ依存 fuzz 検査（K 単調性・residual≤total・flip スケール不変・残差 bound は上界・
    アフィン系統は無フリップ・empirical_cond・attribute 領域・envelope overflow）
  - **calibration ROC（SOCRATIC Q36）**: `roc_sweep(strengths, seeds)` でバグ強度を連続掃引し
    偽OK率を測る。合成判定は系統閾値超で偽OK=0・max_abs 単独は一様スケールを吸収し見逃す。
    閾値未満（~0.2%）の残存盲点も正直に露出（9 ケース corpus を ROC 化）
  - **torch backend が検証を先に届ける（SOCRATIC Q23/Q25/Q26）**: `tsugi_torch.fxbridge`
    （`fx_to_graph_ops`/`audit_fx`）。aten op 名（addmm/bmm/softmax/mean/...）を propagation
    GraphOp へ写し、codegen 前でも FX グラフに静的検証を走らせ増幅 op・モデル発散を warn
    （`verification-only (no codegen yet)`）。duck-typed・torch 不要でテスト（実 FX は torch 要）
  - **safety 定数の単一情報源化（SOCRATIC Q1/Q2）**: `tsugi.constants.SAFETY` に集約（従来は
    tolerance/calibration/propagation の 5 箇所に重複）。docstring に根拠（√K·u·scale の 1σ に
    掛ける ~4σ ヘッドルーム・実機 noise で校正すべき初期値）を明記。挙動不変
  - **AuditPhase.when を decided/pending に改名（SOCRATIC Q45）**: 「verdict に算入」(decided)と
    「実機データ待ち」(pending)の意味を名前に一致（旧 static/runtime は二重定義だった）。
    `decided_phases` プロパティ（`static_phases` は後方互換エイリアス）
  - **テストサマリの正直化（SOCRATIC Q37）**: run.py 末尾に SUMMARY（CPU PASS 件数 + SKIP 件数
    列挙 + 「緑は CPU 検証可能範囲のみ」注記）。GPU 未検証を緑に紛れさせない
  - **外れ値頑健なノイズ床（SOCRATIC Q49）**: `measure_noise_floor`/`measure_batch_variance`
    が `spread_robust`（10-90 パーセンタイル幅）も返し、`compare_stable(robust=True)` で選択。
    測定グリッチ 1 個で max-min が ~4 万倍膨張するのを防ぐ（偽BLOCK 化対策）

### Fixed
- **packaging 欠陥（SOCRATIC Q39）**: pyproject が core `tsugi` パッケージを build 対象外にし
  numpy 依存も未宣言だった → `include=["tsugi*"]` + `dependencies=["numpy>=1.22"]`、torch は
  optional-dependencies に。version を 0.2.0 に bump（Q40・CHANGELOG に版見出し）。
- **CI の正直化（SOCRATIC Q41）**: CONTRIBUTING に「GitHub Actions は無効・ローカル
  run.py/verify.py が CI 代替」を明記。stale なテスト件数記述も更新。

### Meta
- テスト計 127 関数（property 10 × 200 試行含む）/ verify 49 不変条件

### Changed
- **検証層の統合リファクタ（視点追加でなく既存層の重複排除）**: 8 検証層が各自で
  再実装していた深刻度モデル（Risk/Finding/max_risk/to_text）を `tsugi.report` に集約。
  `report.FindingReport` 基底を `portability`/`envelope` が継承し定型を排除（DRY）。
  `Risk`/`Finding` は `portability` から後方互換 re-export。
- portcheck の累積深さ K 推定をマジックナンバー（`n_dots*32`）から実タイル構成
  （`n_dots*cfg.block_k`）由来の見積りに修正。
- production docstring から開発過程ノイズ（「第Nラウンド」等）を除去（触れた 2 モジュール）。

### Note
- リファレンス層は CPU で動作・検証済み（8/8 PASS）。
- GPU バックエンド（NVPTX/AMDGPU）の correctness/性能は **未検証**（要 LLVM/MLIR + 実機）。
  GPU codegen はこのリファレンスと max abs error < 1e-2 (FP16) で一致させる。
