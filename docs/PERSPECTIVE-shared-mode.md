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

## 続く問答 — では誰がオラクルを検証するのか？（無限後退）

**Q.** shared-mode 検出はオラクルを真値として信頼する。だがオラクル（CPU/NumPy =
`runtime_ref`）も *実装* で、NumPy は BLAS を呼ぶ。オラクルが間違っていたら？

**A.** それは「privileged な別ベンダー」にすぎず、検証は後退する。断ち切るのは
**実装非依存の証拠** —— 任意の正しい実装が満たす **メタモルフィック関係**（matmul 恒等
A@I=A・分配則・sum(ones)=n・exp(a+b)=exp(a)exp(b)・softmax が 1 に和し shift 不変・
rsqrt(x)²x=1）と **高精度（float64）再計算** との一致。第二オラクル無しでオラクルを検証する。

`tsugi.oracle_check.verify_oracle()` がこれを実装。緑は「このプラットフォームのオラクルは
数学的性質を満たす」を意味する（病的 BLAS・ビルド不良なら赤）。`rtol=0` にすると f32 丸めが
恒等を満たさず必ず赤になる＝「常に緑」でなく実際に逸脱を捕まえる能力を持つことを担保。

これで検証の連鎖が地に足を着ける: cross-vendor 一致（portability）→ oracle 照合（correctness・
shared-mode 検出）→ メタモルフィック検証（oracle 自体の信頼性）。各段が下の段に錨を下ろす。

## さらに続く問答 — element-wise 比較は位置の対応を仮定していないか？

**Q.** 全比較は `|a[i,j] - b[i,j]|`。A と B が同じ位置に同じ論理値を持つ前提。cross-vendor で
保証されるか？

**A.** されない。ベンダーは同じ論理テンソルを異なる *レイアウト*（row/col-major・タイル順・
転置）で書きうる。素朴な element-wise は転置-but-equal を巨大発散=BLOCK と誤判定する。だが
レイアウト不一致は値の *多重集合* を保存する → `equivalence.classify_divergence` が区別:
EQUIVALENT / LAYOUT（値は正しく位置だけ違う＝整列バグ）/ DIVERGENT（multiset も崩れる＝真の発散）。
LAYOUT は transpose/再タイルで修正可能で、数値検証でなく codegen の整列問題。これも
「検証器が何を見ているか」を正直にする一歩（BLOCK の *原因* を区別する）。
