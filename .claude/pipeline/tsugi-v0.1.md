---
task: Tsugi v0.1 統一GPU計算層
size: XL
stage: 7 (上流完了・GPU codegen残)
gate: frontend→IR→vendor-map 完了(14テストPASS) / GPU codegen本体は要実機
blocked: false
---
[stage履歴]
0-5 受付〜完成形 完了 (2026-06-15)
7 実装(上流完了):
  - リファレンス実装(tsugi pkg・正しさの真値)
  - tracer: @tsugi.jit→tsugi.tile IR(具体トレース・MLIR風テキスト)
  - lowering plan: IR op→各社intrinsic写像(NVVM/ROCDL・ADR-004)
  - autotune探索
  - correctness/tracer/lowering/autotune 計14テストPASS・ruff clean
  ↓ 残: GPU codegen本体(tsugi.tile→gpu→NVVM/ROCDL の実MLIRパス・実コンパイル)
[次アクション]
  実機(LLVM/MLIR+NVIDIA/AMD GPU)で:
  1. src/tsugi/ir/*.td から dialect をビルド
  2. VendorLowering.cpp に lowering.py の写像を実MLIRパスとして実装
  3. GPU実行結果を tests/correctness リファレンスと max abs err<1e-2 照合
[既知の地雷]
  - R1(行列コア抽象)最難関。Phase2 DoD(cuBLAS比70-80%)未達→attention-only縮小 or ADR-004再設計
  - sandbox に GPU 無し。上流(frontend→IR→map)は全てCPU検証済み
  - lowering.py が実MLIR実装の機械可読仕様。これと実装を一致させる
