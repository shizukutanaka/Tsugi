# 新視点11: タスク多様性 — argmax ⇏ 全タスク

ソクラテス式問答の続き。`tsugi.decision` の非分類タスク拡張として実装。

## 問答

**Q1.** decision（新視点8）は argmax（多クラス分類の判断フリップ率）を測る。現実のモデルは
argmax だけを使うか？ → 否。
- **回帰**（価格予測・物理シミュレーション・埋め込み距離）: 出力は「値」。argmax に意味がない。
- **バイナリ分類**（医療診断・スパム・異常検知）: sigmoid + threshold 0.5。top-2 マージンでなく
  「境界からの距離 |y − threshold|」がマージンに相当する。
- **検索・推薦**（ランキング）: 上位 k 件の *集合* が変わるかが判断。topk_flip_rate と同思想だが
  ndim=1 の listwise スコア版（バッチ logit でなくドキュメントスコア）。

**Q2.** argmax flip_rate をこれらのタスクに適用すると何が起きるか？ → 誤計算か意味のない結果。
回帰出力に argmax を呼べば次元エラーか「どの要素が一番大きいか」という無意味な比較になる。
バイナリ sigmoid は 1 次元なので argmax は常に 0 → flip_rate = 0（まったく何も測れていない）。
**静かに誤計算する。**

**Q3.** ならば各タスクに合った「フリップ」の定義は？
- 回帰: |y_a − y_b| > atol + rtol·|y_a|（numpy allclose と整合した相対・絶対許容の組み合わせ）。
- バイナリ: (y_a ≥ threshold) ≠ (y_b ≥ threshold)（判断境界の跨ぎ）。
- ランキング: 上位 k 集合が変わる（topk_flip_rate の listwise 版）。

**Q4.** これは既存のどの視点と対をなすか？ → rollout（新視点9）: argmax を *生成長* へ拡張した。
今回は argmax を *タスク種別* へ拡張する。どちらも「argmax 専用 ⇏ 全タスク」という同型の盲点。

## 実装

```python
regression_flip_rate(a, b, *, atol=0.0, rtol=1e-3)  # 値の乖離が許容超でフリップ
binary_flip_rate(a, b, *, threshold=0.5)              # threshold 跨ぎをフリップと定義
binary_margin(a, *, threshold=0.5)                    # 決定境界からの距離（near-tie 診断）
ranking_flip_rate(scores_a, scores_b, *, k=10)        # top-k 集合の変化（listwise）
compare_task(a, b, *, task, flip_budget, ...)         # 統合ファサード（regression/binary/ranking）
```

compare_task は未知タスク種別を ValueError で弾く（静かに誤計算しない）。

## 含意

- **argmax は多クラス分類専用**。分類以外のモデルを出荷する際は `compare_task` で
  タスクに合ったフリップを測る。
- **バイナリのマージン = |y − threshold|**: 量子化（int8/fp8）や dtype 変換で threshold 付近の
  出力が揺れやすい（argmax の tie_rate と同型の問題）。
- **回帰の rtol 設定はタスク依存**: 価格予測では 0.1%、物理シミュでは 1e-5。固定 atol ではなく
  入力規模に相対的な許容を使う（tolerance 層の derive_tolerance と同じ哲学）。

## 限界（正直に）

- `compare_task` は分類 `compare_decisions` との統合 API（audit_runtime への組み込み）が
  まだ無い。audit 経路では classify/lm タスクのみ対応（rollout など）。
- 回帰の rtol 既定 1e-3 はプレースホルダ——タスクにより 3 桁変わる。ユーザーが設定すべき。
- ranking_flip_rate は集合一致のみ（NDCG/MRR の重み付き損失は未実装）。ランキング品質の
  より細かい変化（順序は変わるが上位集合は同じ）は測れない。
