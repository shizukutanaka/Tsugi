# 新視点12: 発散帰属 — 出力の不一致はどこから来るか？

## ソクラテス式問答

**Q1. equivalence/decision/rollout はすべて *出力層* の最終発散を測る。「移植が壊れた」とき、開発者は次に何をするか？**

→ 手動でデバッグする。どの層/op で発散が起きたか、中間テンソルを順番に印刷しながら探す。これは O(L) の作業で深いモデルでは数時間かかる。

**Q2. propagation（新視点4）は「どの op が dominant amplifier か」を *理論的に* 予測する。だが実測データでそれを確認する手段はあるか？**

→ 無い。propagation は上界推定であって、実データでの因果分析ではない。理論モデルが正しいか、実際にどこで発散が爆発するかは出力だけを見ても分からない。

**Q3. ならば中間テンソルを各層で比較すれば何が分かるか？**

→
- **onset（発散の始まり）**: threshold を超える最初の層 = ここより後は汚染されている。
- **spike（最大増幅）**: 発散の増分が最大の層 = propagation の dominant amplifier を実測で照合。
- **binary search**: 第 L 層が同じなら原因は L より後ろ。違えば前。O(log L) で絞れる。

**Q4. これは propagation の何を補うか？**

→ propagation は *理論的な上界*（保守）。attribution は *実データでの因果特定*（観測）。`propagation.dominant` が「softmax が dominant」と言ったとき、`attribution.spike` が layer=7 の softmax を示せば理論が確かめられる。示さなければ理論のモデル誤差が可視化される。**理論（propagation）と実験（attribution）の接続点。**

**Q5. これは何（ごく普通の）デバッグ実践を *体系化* するか？**

→ 開発者が「とりあえず中間層を印刷」していたものを: 定量的 API（onset/spike/全層 divergence プロファイル）にし、audit 経路から呼べるようにする（修正の証拠にもなる）。

## 鍵となる発見

- **O(L) → O(log L)**: `bisect_onset` が前提（prefix-forward が独立に計算可能な構造）を使って binary search で onset を絞る。深いモデルで数時間 → 数分。
- **理論と実験の接続**: `propagation.dominant` が予測する dominant op と `attribution.spike` が実測する最大増幅層を照合できる。一致 → 理論確認。不一致 → モデルの仮定が実データと合っていない。
- **onset ≠ spike**: onset（最初に threshold を超えた層）と spike（最も大きく発散が増幅した層）は一致しないことがある。この乖離が可視化されると「汚染開始点と増幅点は別の op」と開発者に伝わる。
- **入力差を排除**: `layer_divergences` は両ベンダーで同一 `x` を入力として使う。入力の差異でなく計算の差異だけを測る。

## 計量

```
layer_divergences(layers_a, layers_b, x):
    xa, xb = x, x  # 同一入力
    for (la, lb) in zip(layers_a, layers_b):
        xa = la(xa); xb = lb(xb)
        div_i = max|xa - xb| / (max|xa| + ε)   # relative
    return [div_0, div_1, ..., div_{L-1}]

onset   = first i where div_i > tol
spike   = argmax(div_i - div_{i-1})   # Δdiv が最大の層
```

## 使用例

```python
import tsugi
import numpy as np

# CPU 参照で各層の関数を定義
layers_vendor_a = [embed_a, attn_a, ffn_a, norm_a]
layers_vendor_b = [embed_b, attn_b, ffn_b, norm_b]

x = np.random.randn(32, 512)  # 入力（両ベンダー共通）

# per-layer 発散スキャン
rep = tsugi.attribution.attribute(
    layers_vendor_a, layers_vendor_b, x,
    tol=1e-4,
    names=["embed", "attn", "ffn", "norm"],
)

print(rep.to_text())
# → onset=attn (div=3.2e-4) | dominant spike=attn (Δ=3.1e-4) | final=3.2e-4
# → 疑うべき実装: attn

# propagation の理論予測との照合
from tsugi.propagation import GraphOp, propagate
ops = [GraphOp("matmul", K=512), GraphOp("softmax", cond=5.0), ...]
theory_dominant = propagate(ops).dominant
print(f"theory: {theory_dominant.kind}, experiment: {rep.spike_name()}")
```

## API

| 関数 | 目的 |
|------|------|
| `layer_divergences(layers_a, layers_b, x, *, relative)` | 全層の発散リストを返す（prefix scan）|
| `find_onset(divs, threshold)` | threshold を最初に超える層インデックス |
| `find_spike(divs)` | 発散増分が最大の層（dominant amplifier 実測）|
| `attribute(layers_a, layers_b, x, *, tol, names)` | AttributionReport を返す統合関数 |
| `bisect_onset(fn_prefix_a, fn_prefix_b, x, n_layers, *, tol)` | O(log L) での onset 探索 |

## 他視点との関係

| 視点 | 関係 |
|------|------|
| propagation (4) | 理論予測（上界）。attribution が実測で照合・反証する |
| equivalence (−) | 出力層のみ。attribution が「どこで」を答える |
| worstcase (10) | 最悪入力を探索。attribution がそこで「どの層が」を答える |
| envelope (5) | 入力が認証前提内か。attribution は前提内での計算差を層別に測る |
