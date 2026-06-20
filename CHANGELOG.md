# Changelog

Keep a Changelog 形式。SemVer。0.x は API 未凍結（MINOR で機能追加・互換変更ありうる）。

## [Unreleased]

### Added
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
