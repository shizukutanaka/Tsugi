# Tsugi FAQ

## Q. これは CUDA のクローン？

No。CUDA Runtime API を模倣しない。CUDA の堀は言語でなく**ライブラリ（cuDNN/cuBLAS）と PyTorch 統合**。ゆえに API 競合でなく `torch.compile` のバックエンドとして刺す（[ADR-003](adr/ADR-003-torch-backend-first.md)）。

## Q. ZLUDA と何が違う？

ZLUDA は **CUDA バイナリ/PTX をそのまま非NVIDIA で動かす変換層**。NVIDIA EULA の翻訳層禁止条項に抵触し、AMD が法的理由で 2024年8月に撤回。

Tsugi は **新DSL とソースレベルのみ**。バイナリ変換しない。CUDA ソースを動かす必要がそもそもない（楔は PyTorch 層）。合法を最優先（[ADR-002](adr/ADR-002-no-binary-cuda.md)）。

## Q. Triton と何が違う？ 競合？

Triton は素晴らしい先行例で、TorchInductor のデフォルトカーネル生成器。Tsugi は Triton を**競合でなく前例/貢献対象**として扱う。差別化は **3社の行列コアレイアウトを統一する autotuning 抽象層**（Triton/IREE/Mojo がまだ苦戦する箇所・[ADR-004](adr/ADR-004-tensorcore-abstraction.md)）。Triton への貢献も選択肢。

## Q. なぜ SPIR-V を共通IRにしない？

SPIR-V は NVIDIA で **CUDA 計算の first-class 入力でない**。NVIDIA driver が消費するのは PTX。SPIR-V 中核にすると NVIDIA で最高性能の経路を捨てる。OpenCL 失敗の教訓（市場リーダー上で最高性能不可なら普及しない）。SPIR-V は Intel/Vulkan の移植フォールバックに限定（[ADR-001](adr/ADR-001-mlir-over-spirv.md)）。

## Q. 性能は？ CUDA に勝てる？

「勝つ」より「実用的に近い」が目標。標準 GEMM で cuBLAS/rocBLAS 比 20-30% 以内、カーネル4系統で Triton 比 20% 以内（[BENCHMARK.md](BENCHMARK.md)）。19年の cuDNN/cuBLAS 蓄積に正面勝負せず、hot カーネルのみ生成し標準opは escape-hatch で委譲する。

## Q. ソロ開発で CUDA に挑むのは無謀では？

CUDA 全面複製は非現実的。ゆえに**縦に狭く深く**: 2バックエンド（NVIDIA+AMD）・カーネル4系統・推論優先。既存OSS（MLIR/LLVM/IREE/SPIR-V）を土台に統合層を被せる（車輪の再発明を避ける）。

## Q. なぜ AMD MLPerf の話が根拠に出る？

AMD の MLPerf Inference v5.0 SDXL（2025年4月）が **IREE + MLIR で AMDGCN 直接生成**（手書きベンダー GEMM ライブラリなし）で本番競合を達成。「MLIR コンパイラ層がベンダーライブラリなしで本番品質に届く」最強の実証。Tsugi はこの経路を踏襲。

## Q. Intel / Apple は？

v1.0 以降。Intel は SPIR-V/Level Zero（Level Zero が SPIR-V ingestion 必須なので親和性高い）、Apple は Metal/SPIR-V。v0.1 は NVIDIA+AMD に集中。

## Q. 訓練（training）は？

v1.0 まで**推論優先**。AMD は訓練で CUDA パリティ未達（RCCL<NCCL・FlashAttention3 差・SemiAnalysis 2024）。推論（特にメモリバウンド・MI300X の 192GB HBM3 が効く領域）から攻める。

## Q. GPU が無い環境でビルドできる？

コンパイラ部（MLIR パス）は CPU でビルド可。だが **correctness/性能検証は NVIDIA・AMD 実機が必要**。CPU のみの CI では SPIR-V emit 検証等に留まる。動作未検証の経路は「未検証」と明記する（主張と実装の一致）。

## Q. ライセンスは？

Apache-2.0。依存の LLVM（Apache-2.0 with LLVM exception）・IREE（Apache-2.0）と整合。GPL/AGPL は不採用。

## Q. 収益は？

OSS。寄付は Bitcoin: `bc1qjaet6jgpk08la46jelmlpgsz84luc4lc0tnwr5`。プロダクトコードに課金機能は組み込まない。
