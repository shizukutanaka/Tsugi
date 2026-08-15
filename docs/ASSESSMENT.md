# Tsugi プロダクト評価 — 長所・短所・改善案（ASSESSMENT）

`docs/FEATURE-AUDIT.md` が「機能の過不足」台帳であるのに対し、本書はプロダクト・
プロセス・運用まで含めた評価。2026-08 時点の実状（verify 174/174・CPU 23 スイート
PASS・ruff clean）に基づき、観測した事実のみを根拠にする。
改善案の表がそのまま `INSTRUCTIONS-OPUS.md` / `INSTRUCTIONS-SONNET.md` の
バックログになる（単一情報源）。

## 長所

1. **fail-safe 設計の一貫性**: 偽OK ≫ 偽BLOCK の非対称コストを全 13 検証層が共有。
   点推定→Wilson 上側限界 5 箇所・DAG マージの保守側選択（`correlated=True`）・
   検出不能な形は線形フォールバック。方針が文書だけでなくコードの隅々まで貫通している。
2. **検証基盤の厚さと「欠陥→不変条件」文化**: 機械検証可能な不変条件 174 件＋
   27 テストファイル＋property test（10 性質×200 試行）。実際に起きたバージョン
   ドリフトが即座に不変条件 62 として固定された実例あり。回帰が構造的に起きにくい。
3. **誠実な主張**: GPU 未検証を run.py の SUMMARY で毎回明示。「静的 cond=1 は下界」
   「scale=1 仮定」等をレポート出力に明記する「暗黙化しない」慣例。過大広告がない。
4. **引き継ぎ可能性**: FEATURE-AUDIT.md / SOCRATIC-50 が前提知識ゼロの読者を想定し、
   実際にセッション再開がこの 2 文書だけで成立した。
5. **台帳駆動の自己改善プロセス**: facade 未接続 11 件を系統スキャンで発見→全接続→
   スキャン自体を CI 化（不変条件 57）。プロセスが再現可能。
6. **判定が機械可読**: `Audit.to_dict()`（JSON 直列化可能）と終了コード契約
   （OK/INFO=0・WARN=1・BLOCK=2）を持ち、CI が判定をそのままゲートに使える
   （散文パースが不要）。First Principles 分析で発見した不足を解消済み（不変条件 69）。

## 短所

1. **製品価値の中核が未達**: GPU 実機検証ゼロ（A-2）・codegen 未実装（A-4）。
   「単一ソースで両ベンダー」はまだ約束であって実績ではない。
2. **検証器自身の実機校正が未了（手続きは完成）**: SAFETY=4.0 は依然 CPU
   シミュレーション内の経験値。ただし「実機の run が手に入った瞬間に機械的に校正する
   手続き」は完成し実機入口に接続済み（`calibration.calibrate_safety`・
   `docs/GPU-BRINGUP.md`）。残るのは実機そのもので、検証器側の設計課題ではなくなった。
3. **想定入口（torch.compile）が薄い**: sample 由来の scale/cond 実測は接続済みだが、
   `_tsugi_compile` は依然として静的監査＋警告のみ。worstcase/attribution/タスク別
   decision は実行時出力が要るため codegen 後（A-3 残）。
4. **リポジトリ運用の未成熟**: main ブランチ無し・タグ/Release ゼロ・Actions 実質無効。
   CI がローカル実行（run.py/verify.py）頼み。
5. **verify.py の成長限界**: 1,400 行超・実行数十秒。テーマ別 12 関数に分割済みだが
   単一ファイルの限界は近い。
6. **ドキュメント肥大**: docs 29 本。README に目的別の reading path を新設して導線は
   確保したが（「使う人／仕組み／引き継ぐ人／その他」）、文書数自体は多いまま。

## 改善案（優先度 × 推奨実行者）

| 優先度 | 改善案 | 実行者 |
|---|---|---|
| P0 | v0.4.0 タグ/Release 作成（`docs/RELEASING.md` §3）・main 新設と protected 化・Actions 有効化（`docs/ci-reference.yml` 反映） | 人間（セッション権限外） |
| P1 | ~~A-2: 実機入手後の GPU ハーネス実行計画（SAFETY 校正手順の事前設計）~~ 完了（`docs/GPU-BRINGUP.md`＋`calibrate_safety`） | Opus |
| P1 | A-2 残: 実機での Phase 1〜3 実行と記録テンプレートへの追記 | 人間（要 GPU） |
| P2 | ~~A-5: 正規化層の scale リセットの数値実験~~ 完了（3a29b94・**前提が実験で反転**し LayerNorm は増幅と判明） | Opus |

---

関連: `docs/INSTRUCTIONS-OPUS.md`（P1 の実行指示書）・`docs/INSTRUCTIONS-SONNET.md`
（P2 の実行指示書）・`docs/MODEL-USAGE-GUIDE.md`（どのモデルに何を振るか）。
