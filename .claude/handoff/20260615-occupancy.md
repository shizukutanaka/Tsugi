[SESSION HANDOFF - 2026-06-15 #6 占有率+CLI拡張]

== 追加実装（CPU検証済み） ==
- tsugi/occupancy.py: ベンダー別占有率推定。同一構成がNVIDIA/AMDで差(warp32/64・LDS差)
  実証: m64n64k32s3w4 → NVIDIA25% / AMD CDNA20% / AMD RDNA25%
  ※HW定数は代表値・要実機確認(明記済み)。constants上書き可能
- portability.py: cfg渡しで占有率<25%をWARN追加
- portcheck.py: ユーザー.py読込実装(契約: kernel@jit + make_args() + 任意BLOCK_DIMS)
  examples/user_kernel.py で実読込確認
- test_occupancy.py 5件 + test_portcheck.py 3件 → 計33テストPASS / verify13 / ruff clean

== 検証層の現状(新視点・4本柱完成) ==
1. portability(静的リスク・warp/MMA/bf16/累積/占有率) ✅
2. equivalence(数値等価性検出) ✅
3. occupancy(ベンダー別性能予測) ✅
4. portcheck CLI(ユーザーカーネル対応) ✅
→ GPU codegen未完でも実用的な移植診断ツールとして完成

== 次アクション ==
- occupancy HW定数を各archの実値に(NVIDIA/AMD公式仕様から・要出典)
- shared memバンクコンフリクト検出ルール
- 実GPU: equivalence compare()を両ベンダー実出力に適用
- GPU codegen本体(別軸・要LLVM/MLIR+実機)
