# Sonnet 作業指示書 — Tsugi スコープ確定済み実行ラウンド

## あなたの役割
方向性が既に定まった**反復ラウンド**（「続けて」型）を担当する。本プロジェクトの
第11〜28回（facade 接続・Wilson 上側限界化）はこの型で実施され安定して機能した実績がある。

## 起動時に読む 3 ファイル（この順）
1. `docs/FEATURE-AUDIT.md` — 機能過不足の台帳（現在地と残作業の単一情報源）
2. `docs/ASSESSMENT.md` — 長所短所改善案（あなたの担当は P2）
3. `CONTRIBUTING.md` — 検証ゲート・コミット規約・Import 方針

## 担当バックログ（ASSESSMENT の P2）
- **Q43**: 乱数依存テストの境界余裕点検（seed 固定でも判定境界に近すぎるテストを
  midpoint 化。`test_single_run_flaky` の堅牢化が先例）
- **Q48**: 「発散 ~2000倍」等の単一 seed の例示数値を複数 seed の中央値±に更新（docs 側）
- **A-10**: カバレッジ計測（coverage.py 導入・**閾値ゲートは人間承認後**）
- **facade 未接続スキャンの確認**（不変条件 57 が緑でも、新規追加関数の接続漏れを
  1 ラウンドに 1 回目視確認）
- **ドキュメント数値同期**（不変条件件数・テスト数が実態と一致しているか）

## エスカレーション基準（該当したら実装せず停止して報告——最重要）
次に該当したら手を止め、Fable/Opus に委ねる:
1. **判定閾値・許容誤差・SAFETY 係数に触れる変更**（偽OK リスク → Fable/Opus 案件）
2. **スキャン結果の解釈が曖昧**（B-2 の正当除外か真の欠陥か判断がつかない）
3. **新しい数値係数・統計モデルが必要になった**
4. **diff が 500 行を超えそう**
5. **verify.py の既存不変条件を弱める/削除する必要が出た**

## 実務注意（このリポジトリで実際に踏んだ穴）
- テスト追加時は必ずそのファイルの `main()` リストに登録（orphan 検査＝不変条件 56 が落ちる）
- dtype を足す時は 3 表同時（`tolerance.UNIT_ROUNDOFF`・`equivalence.TOLERANCE`・
  `envelope.DTYPE_LIMITS`）
- バージョンを触る時は `pyproject.toml` と `__version__` の両方（不変条件 62 が落ちる）
- コミットは日本語・問題→修正→実証の順。**タグ/Release 操作はしない**（人間専用）。

## 定型手順（`FEATURE-AUDIT.md` §D と同一）
1. 対象モジュールを修正（既存実装の再利用を優先）
2. `tests/correctness/test_<module>.py` に再現＋回帰なしテストを追加し `main()` に登録
3. `verify.py` に不変条件を追加（連番コメント形式）
4. `python verify.py` ＋ `python tests/correctness/run.py` ＋ `ruff check` が全 PASS
5. `CHANGELOG.md` の Unreleased に記録
6. commit（日本語・問題→修正→実証）して push（diff ≤500 行）
