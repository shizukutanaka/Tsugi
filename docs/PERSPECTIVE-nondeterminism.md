# 新視点7: 非決定実行 — ベンダーの出力は点でなく分布

ソクラテス式問答・第7ラウンド。`tsugi.nondeterminism` として実装。

## 問答

**Q1. これまでの全ての等価判定（視点1・2・6・伝播）は何を暗黙に仮定していたか？**
「同じカーネルを同じベンダーで走らせれば同じ結果が出る」=**決定論**。比較は A と B を
それぞれ固定点として扱う。

**Q2. その仮定は正しいか？**
否。GPU の atomic 加算（split-K の atomicAdd、reduction の並列合算）はスレッド到着順で
浮動小数の和の順序が変わり、**run-to-run で結果が揺れる**。ベンダーの出力は固定点でなく
**分布**である。tolerance.py の `noise_floor` 引数が既定 0 だったのは、この決定論仮定の化石。

**Q3. 出力が分布なら「単一 run A vs 単一 run B」の比較は何を測っているか？**
2 つの差の源を**混同**している:
- (a) ベンダー *内* の run-to-run ノイズ（同じ A でも run が違えば違う）
- (b) ベンダー *間* の真の発散

両者を分離できなければ食い違いを attribute できない。そもそも「ベンダー A の正しい答え」が
一意に定義できない（どの run が真値か？）。

**Q4. クロスベンダー差がベンダー内ノイズより小さいとき、何が言えるか？**
何も言えない。「A vs B」は「A vs A（別 run）」と**区別不能**。等価でも発散でもなく
第三の状態 **INDISTINGUISHABLE**（判定が原理的に未定義）。正直にそう報告すべき。

**Q5. これは視点6（検出限界）とどう関係するか？**
独立した**第二の床**。視点6 の検出限界（safety·√K·u）は *数値* の床。ノイズフロアは
*ハードウェア* の床。検証器の実効分解能 = **max(数値検出限界, ノイズフロア)**。
ノイズが数値許容を超えれば検証器は**ノイズ律速**になり、ノイズを発散と誤判定（偽BLOCK）
するか、許容を緩めて偽OK を増やす。

## 実証（numpy・CPU で atomic 非決定を擬似再現）

```
run-to-run ノイズフロア（atomic 非決定, K=4096 fp32）:
  spread = 3.66e-04   rel = 5.5e-06   (n=20 runs)
```

**単一 run 比較はフレークする** — 真に等価な 2 ベンダー（同じ部分和・atomic 順だけ違う）を
観測差の中間に置いた許容で比較すると、同じ対が run の引きで **EQ にも DV にもなる**:
```
verdicts over 24 trials = {EQ, DV}   ← 真に等価なのに判定が割れる
```

**ノイズを測れば正直な 3 状態判定ができる**:
```
stability [INDISTINGUISHABLE] cross=7.6e-05 noise=3.4e-04 tol=...
  [WARN] クロス差 7.6e-05 ≤ run-to-run ノイズ 3.4e-04 → 区別不能・等価判定は未定義
```

真の発散（部分和を 5% 底上げ）はノイズを超えて **DIVERGENT** と正しく検出される。
ノイズに埋もれず、ノイズを発散とも誤判定しない。

## 方法論 — 出力を分布として扱う

`compare_stable(run_a, run_b, K, dtype, n_runs)`:
1. 各ベンダーを n_runs 回走らせ **run-to-run ノイズを実測**（noise_floor）
2. noise を織り込んで許容を導出（`derive_tolerance(..., noise_floor=実測値)`・決定論仮定を排す）
3. クロス差を noise/tol に対し 3 状態へ帰属:

| クロス差の範囲 | 判定 | 意味 |
|---|---|---|
| `[0, noise]` | INDISTINGUISHABLE | ベンダー内ノイズと区別不能（未定義）|
| `(noise, tol]` | EQUIVALENT | ノイズと区別できる正当な数値差 |
| `(tol, ∞)` | DIVERGENT | 真の発散 |

## 含意

- **単一 run 比較は方法論的に不健全**。基準（reference）を分布として特徴づけずに
  比較してはならない。CI の「1 回走らせて一致」は偽の安心。
- INDISTINGUISHABLE という第三の verdict を持つことが**正直**。GPU 数値検証は
  「等価/発散」の二値でなく、ノイズ床を明示した三値。
- 視点6 の系統バイアス検出（RMS 比）は **ノイズに頑健**: ノイズは zero-mean、系統バグは
  相関ゆえ、複数 run 平均で系統信号だけが残る（ノイズ床の下の系統バグも掘り出せる）。
- 実機 GPU では `run_fn` を実カーネルにするだけで、本モジュールがそのまま noise_floor を
  実測し tolerance/calibration に供給する（CPU 実装は atomic 非決定の擬似・明示）。

## API

```python
from tsugi.nondeterminism import (
    simulate_nondeterministic_reduction,  # atomicAdd 非決定の擬似（CPU・明示）
    measure_noise_floor,                  # 複数 run で run-to-run ノイズを実測
    attribute,                            # クロス差を 3 状態へ帰属
    compare_stable,                       # 分布として扱う健全なクロスベンダー比較
    EQUIVALENT, DIVERGENT, INDISTINGUISHABLE,
)
```
