# Opus 作業指示書 — Tsugi 設計判断つき中規模ラウンド

## あなたの役割
スコープは定まっているが**設計判断を含む**ラウンドを担当する（Sonnet では不安・
Fable ほどの重さは不要な中間層）。会話履歴に依存せず、台帳だけで作業を開始できる。

## 起動時に読む 3 ファイル（この順）
1. `docs/FEATURE-AUDIT.md` — 機能過不足の台帳（現在地と残作業の単一情報源）
2. `docs/ASSESSMENT.md` — 長所短所改善案（優先度と担当の割当・あなたの担当は P1）
3. `CONTRIBUTING.md` — 検証ゲート・コミット規約・Import 方針

## 担当バックログ（ASSESSMENT の P1）

### 1. A-3: torch.compile 経路の段階接続
- **現状**: `_tsugi_compile()`（`python/tsugi_torch/__init__.py`）は
  `fxbridge.audit_fx`（静的監査）のみ。worstcase/attribution/タスク別 decision が届かない。
- **やること**: example_inputs から代表テンソル/logits を best-effort で取り出し、
  `audit()` の `sample=` / `ref_logits=` 相当の情報を警告メッセージに追加する段階的接続。
- **受け入れ基準**: torch 無し環境の duck-typed FX スタンドインでテスト
  （`tests/correctness/test_tsugi_torch_compile.py` の既存パターンを踏襲）。
  実 torch.fx 結線は「未検証」と明記。フル接続は実行時出力が要るため codegen 後。

### 2. A-12 残: 一般 series-parallel fork（3 分岐以上）
- **現状**: `_computed_fork_merge`（`python/tsugi/audit.py`）は消費者ちょうど 2 の
  場合のみフォークを検出（`len(uses) != 2` は線形＝保守側にフォールバック）。
- **やること**: `len(uses) >= 3` で各消費者から単一消費の一本鎖を辿り、全鎖が同一の
  合流 op に到達し、合流 op の operands が全末端ちょうどのときだけ `[[A],[B],[C],...]`
  を出す。それ以外は線形。`propagate_dag` は N 分岐に対応済み（`merge_divergence`）。
- **受け入れ基準**: 3 分岐カーネルのテスト＋「検証できない形は線形に落ちる」テストの
  両方。`propagate_dag(correlated=True)` の発散が線形版を下回らないことを固定。

### 3. A-2: 実機 GPU ハーネス実行計画（実機入手後）
- `tests/gpu/` を実機で回し `audit_cross_vendor()` に実カーネルを渡す手順と、
  SAFETY=4.0 の実機ノイズ校正手順を設計・文書化（コード変更は最小・計画が主）。

## 設計ガードレール（違反しそうなら手を止めて人間に確認）
1. **偽OK 方向の変更は「問題を再現するテスト」を先に書く**（修正前に赤・修正後に緑）。
2. **未検証の数値係数を導入しない**（A-5 の据え置き判断が先例——検証実験が先）。
3. **不確実なら保守側**（BLOCK 寄り）に倒す。
4. **新関数は facade 接続まで含めて 1 ラウンド**（B-1 の轍: 実装済み未接続を作らない）。
5. **タグ/Release/main ブランチ操作はしない**（人間専用・`docs/RELEASING.md` §3）。

## 定型手順（全ラウンド共通・`FEATURE-AUDIT.md` §D と同一）
1. 対象モジュールを修正（既存実装の再利用を優先）
2. `tests/correctness/test_<module>.py` に「再現ケース」と「回帰なし」の両テストを追加し
   **その file の `main()` テストリストに登録**（orphan 検査＝不変条件 56 が落ちる）
3. `verify.py` に不変条件を追加（末尾の連番コメント形式）
4. `python verify.py` ＋ `python tests/correctness/run.py` ＋
   `ruff check python/ tests/ examples/` が全 PASS
5. `CHANGELOG.md` の Unreleased に「何を・なぜ・どう直したか・実証データ」
6. commit（日本語・問題→修正→実証）して push（diff ≤500 行・超過は分割）
