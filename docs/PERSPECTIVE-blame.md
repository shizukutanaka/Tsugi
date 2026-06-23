# 新視点13: ベンダー責帰 — どちらのベンダーが oracle に近いか？

## ソクラテス式問答

**Q1. cross-vendor BLOCK が出た。どちらのベンダーの実装が間違っているか？現在の情報で分かるか？**

→ 分からない。equivalence/attribution は「A vs B の差」を測るが、「どちらが真値（oracle）に近いか」は測らない。開発者は NVIDIA と AMD の両方をデバッグするしかなく、O(2L) の作業になる。

**Q2. oracle（CPU float64 参照）との距離 dist_a / dist_b を比較すれば何が分かるか？**

```
dist_a = max|a − oracle| / (max|oracle| + ε)   (相対距離)
dist_b = max|b − oracle| / (max|oracle| + ε)

dist_a < dist_b → A が oracle に近い → vendor B の実装を優先修正
dist_b < dist_a → B が oracle に近い → vendor A の実装を優先修正
dist_a ≈ dist_b → 両方同程度に乖離 → 両実装を見直す（oracle 自身を疑う）
```

**Q3. oracle_check（視点メタ）と何が違うか？**

| 視点 | 問い | 情報 |
|------|------|------|
| oracle_check | A ≈ B ≈ oracle か？ | shared mode 障害（両者が同じバグ）を検出 |
| blame | A と B の *相対* 正確性は？ | 「どちらを修正すべきか」の方向を提供 |

相補的: oracle_check は「共有モード障害の有無」、blame は「責帰の割り当て」。

**Q4. attribution（視点12）と組み合わせると何が完成するか？**

```
attribution.spike = "layer7_attn"    → どの層か
blame.closer = "B"                   → どちらが oracle に近いか（A が culprit）
```

→ 完全な診断: **"layer 7 attention の vendor A 実装が oracle から遠い → A を直せ"**

「出力差を検出（equivalence）→ どの層か（attribution）→ どちらのベンダーか（blame）」の診断チェーンが閉じる。

**Q5. ratio（max/min 距離比）が示す情報は？**

→ ratio が大きいほど責任が一方に集中（修正方向が明確）。  
→ ratio ≈ 1 → 両方同程度に間違っている（前提の見直し推奨）。  
→ `oracle_check.verify_oracle` を先に呼び oracle 自身の健全性を担保してから使うのが理想。

## 鍵となる発見

- **診断チェーンの完成**: attribution が「どの層か」を、blame が「どちらのベンダーか」を教える。両者を組み合わせると「層 X の vendor Y 実装を直せ」という完全な修正指示になる。今まで開発者が両ベンダーの両方をデバッグしていた O(2L) の作業が、**O(1) の修正指示**になる。
- **相対距離で scale 不変**: `accuracy_relative` は max|out - oracle| / max|oracle| で計算し、テンソルのスケールに依存しない。大きな logit スケールでも誤判定しない。
- **ratio で方向の明確性を定量化**: ratio = 1 → 「両方悪い・どちらが原因か言えない」。ratio >> ratio_threshold → 「片方が明確に culprit」。開発者が確信を持って修正できる閾値を制御できる。
- **oracle の健全性前提を明示**: blame は oracle が正しいことを前提にする。oracle 自身が怪しい場合は `oracle_check.verify_oracle` で確認すること、と API docstring に明示。無限後退を防ぐ。

## 計量

```
closer = "A"   iff dist_a < dist_b and ratio ≥ ratio_threshold   → blame B
closer = "B"   iff dist_b < dist_a and ratio ≥ ratio_threshold   → blame A
closer = "TIED"  iff ratio < ratio_threshold                       → 方向不明

ratio = max(dist_a, dist_b) / min(dist_a, dist_b)   (≥ 1)
```

## 使用例

```python
import tsugi
import numpy as np

# oracle: CPU float64 参照実装
oracle_out = ref_matmul_fp64(x)

rep = tsugi.blame.compare_accuracy(
    nvidia_out, amd_out, oracle_out,
    tol=1e-4,
    ratio_threshold=2.0,
)
print(rep.to_text())
# → blame (dist_a=2.1e-5, dist_b=8.4e-3, closer=A, ratio=400.0x, tol=1.00e-04)
# → closer=vendor A (dist=2.1e-05) / farther=vendor B (dist=8.4e-03, ratio=400.0x)
#   → vendor B の実装を優先修正

# per-layer でどちらが oracle から遠いか
dists = tsugi.blame.layer_blame(layers_nv, layers_amd, layers_ref, x)
# dists[7] = (dist_a=2.1e-5, dist_b=8.4e-3) → layer 7 で AMD が大きく乖離
```

## API

| 関数 | 目的 |
|------|------|
| `accuracy_relative(out, oracle, *, eps)` | 相対距離 max\|out-oracle\| / (max\|oracle\| + ε) |
| `compare_accuracy(a, b, oracle, *, tol, ratio_threshold, relative)` | closer / ratio / risk を返す BlameReport |
| `layer_blame(layers_a, layers_b, layers_oracle, x)` | per-layer (dist_a, dist_b) リスト |

## 他視点との関係

| 視点 | 関係 |
|------|------|
| oracle_check (メタ) | oracle の健全性確認。blame の前提を検証する |
| attribution (12) | 「どの層か」を特定。blame が「どちらのベンダーか」を追加して診断完成 |
| equivalence (−) | A vs B の差。blame は A vs oracle / B vs oracle を追加する |
| calibration (6) | 検証器の偽OK 率。blame は個別 run での責帰。相補的 |

## 検証ライフサイクルでの位置

```
出力差を検出（equivalence）
    │
    ▼
どの層か（attribution.spike）
    │
    ▼
どちらのベンダーか（blame.closer）    ← 新視点13
    │
    ▼
修正（vendor X の layer Y を直す）
```
