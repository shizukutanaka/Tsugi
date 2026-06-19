# 新視点: 一致 ≠ 正しさ — 共有モード障害という構造的盲点

ソクラテス式問答の続き。`tsugi.calibration.detect_shared_mode` として実装。

## 問答

**Q1.** 実行時の全層（`audit_runtime`/`audit_cross_vendor`/equivalence/decision）は A と B を
比較し、一致すれば「等価」=緑とする。だが *一致* は *正しさ* か？

**Q2.** 否。一致は「両ベンダーが同じ答えを出した」であって「正しい答え」ではない。では
**両ベンダーが同じバグを共有** したら（同一の上流ライブラリ欠陥・同じ誤った丸めモード・
同じ flawed アルゴリズム）どうなる？

**Q3.** 両者は一致 → 検証器は「等価」=緑。**cross-vendor 検証は convergent（共有モード）
エラーに構造的に盲目**。cross-vendor 一致は正しさの **必要条件であって十分条件でない**。
これは calibration が従来モデル化した「sub-floor の *divergent* バグ」とは別種の偽OK。

**Q4.** ならどうすれば見えるか？

**Q5.** 独立な **oracle**（真値・例: CPU/NumPy リファレンス）と照合して初めて見える:
- `DIVERGENT`   : A≢B → cross-vendor が捕捉する通常の発散
- `SHARED_MODE` : A≈B かつ 両方 ≢ oracle → 共有モード障害（cross-vendor は見逃す）
- `OK`          : A≈B≈oracle

## 実証（numpy）

```
両ベンダーが同じ 5% バグ（共有上流欠陥）を持つ:
  cross-vendor A vs B      → EQUIVALENT / portable（緑・見逃す）
  detect_shared_mode(...)  → SHARED_MODE（oracle 照合で暴く）
```

## 含意

- 本番（`audit_runtime`）には oracle が無い → **SHARED_MODE は原理的に検出不能**。
  cross-vendor 一致だけを根拠に「正しい」と言ってはならない。
- oracle のある経路（CI・リファレンス実装・既知の良ベンダー）でこそ oracle 照合すべき。
  Tsugi は CPU/NumPy リファレンス（`runtime_ref`）を持つので、per-kernel 正しさは
  そこに錨を下ろせる。実機の per-model 経路は共有モードに弱い —— その限界を明示する。
- これは検証器の正直さの問題: 「等価」ラベルは「ベンダー一致」であって「正しさ」でない。
  両者を区別して報告する（calibration が偽OK の二種 —— sub-floor divergent と shared-mode
  convergent —— を扱うようになった）。

## API

```python
from tsugi.calibration import detect_shared_mode, SM_OK, SM_DIVERGENT, SM_SHARED
verdict = detect_shared_mode(vendor_a, vendor_b, oracle, K, dtype)
```
