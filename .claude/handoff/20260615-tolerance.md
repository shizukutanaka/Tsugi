[SESSION HANDOFF - 2026-06-15 #8 新視点2: 導出許容]

== ソクラテス問答2で発見 ==
equivalenceの固定許容(1e-2)が恣意的。両ベンダーは累積順序違いで両方IEEE正当。
「真値に一致」でなく「数学が許す範囲に収まる」へ。
docs/PERSPECTIVE-derived-tolerance.md

== 実装(CPU検証済み) ==
- tsugi/tolerance.py: 許容を K・dtype の機械イプシロンから導出
  expected_gemm_abs_error(K,dtype,scale,safety) = safety·√K·u·scale
  derive_tolerance(): max(導出, noise_floor) を {atol,rtol}
- equivalence.compare_gemm(a,b,K,dtype): 導出許容でGEMM判定
- 実証: K=64→atol1.6e-2 / K=2048→8.8e-2(K依存)。固定1e-2の過剰検出解消
- test_tolerance.py 6件 → 計39テストPASS / verify14 / ruff clean

== 次アクション ==
- 実GPU: 各ベンダーrun-to-run非決定性を実測→noise_floorに投入(さらに原理的)
- compare_gemm を実GPU両ベンダー出力に適用
- equivalence固定TOLERANCEは汎用op用に残す(GEMM以外)。重複でない
