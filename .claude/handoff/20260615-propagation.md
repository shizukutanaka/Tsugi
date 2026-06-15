[SESSION HANDOFF - 2026-06-15 #6 新視点4追加]

== ソクラテス問答で発見した新視点 ==
既存検証層(equivalence/tolerance/feasibility/portability)は全て *単一カーネル* 判定。
だがユーザーは torch.compile(model) で op グラフをコンパイルする。
per-kernel 等価 ⇏ per-model 等価: 発散はグラフを伝播し、深さで累積・
ill-conditioned op(相殺reduction/小値除算/大exp)で増幅。
検証単位はカーネルでなくグラフ。docs/PERSPECTIVE-error-propagation.md に記録。

== 実装済み(CPU検証済み・GPU不要) ==
- tsugi/propagation.py: δ_out = amp·δ_in + local をグラフに沿って合成
  - GraphOp(kind, K, dtype, cond): op の局所発散と増幅率の仕様
  - propagate(ops) → 累積発散・支配的増幅op・素朴なper-kernel和との乖離
  - model_tolerance(ops) → モデルレベル許容の目安
  - 増幅opは amp=条件数(reduce/add/exp/div/rsqrt/softmax)、matmul/scaleは amp≈1
- numpy実証: 累積順序違いの2ベンダーをmatmul+rmsnormの鎖に流すと
  発散が 1層9.3e-7 → 12層1.9e-3 (約2000倍)。モデル許容1.88e-1は単一1.56e-2の12倍。
- test_propagation.py 5件追加。計56テストPASS / verify19 / ruff clean

== 検証順序(確立) ==
feasibility(動くか) → propagation(モデルで一致するか) → occupancy(速いか)。
等価性判定の単位がカーネル→モデルへ昇格。

== 次アクション(保留・実機/拡張) ==
- グラフ自動抽出: torch.fx / traced IR の複数カーネル連結から propagation レポートを直接生成
- 実機 noise_floor を local に注入し propagate でモデルレベルのノイズ予算に積み上げ
- portcheck にモデルグラフ入力時の propagation セクション(現状は単一カーネル)
