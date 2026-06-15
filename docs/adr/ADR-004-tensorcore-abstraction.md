# ADR-004: Tensorコア抽象はMLIR intrinsic経由（Vulkan coopmat非依存）

- 日付: 2026-06-15
- 状態: Accepted（最重要リサーチベット・R1）

## 状況

行列コア（Tensorコア）抽象がAI向け最難関。GEMMがLLM実行時間を支配（prefill 87.6%/decode 76.2% @ llama3.2-1B f16, arXiv 2505.06461）。3社の行列命令（NVIDIA WMMA/HMMA・AMD MFMA/WMMA・Intel XMX/DPAS）は不透明・スレッドデータレイアウト非公開・世代間で挙動変化（HMMA.884→16816、mma.m8n8k4 が低速FMAに退化）。

候補: (a) Vulkan `VK_KHR_cooperative_matrix` を portable 行列抽象に使う、(b) MLIRで`tile.matmul`を抽象opとして持ち各社LLVM intrinsicへ直接lowering。

## 決定

**`tile.matmul` 抽象opを各社intrinsic（NVVM wmma / ROCDL mfma・wmma）へMLIR経由でlowering。Vulkan cooperative matrixには依存しない。SPIR-V coopmatは移植フォールバックのみ。**

## 根拠

1. **Vulkan KHR coopmat は compute-only かつ HW gated。** AMD RADV は RDNA3(GFX11)+ のみ露出。subgroup-scope行列が小さくGEMMにshared memoryタイリング手動必要・reduction/conversion組み込みなし（Khronos自身がcoopmat2提案で限界を列挙）。
2. **真に有用な `VK_NV_cooperative_matrix2` はNVIDIA専有。** 575ドライバ系。これに依存すると移植性が崩れる。llama.cpp Vulkanがcoopmat2+最新ドライバでのみCUDA競合/超過（NVIDIA Vulkanised 2025）。
3. **NVIDIA反復パターンに巻き込まれる。** ベンダー拡張→Khronos標準化（subset）→次世代は再びNVIDIA専有（coopmat→KHR coopmat→NV coopmat2）。Khronos追従戦略は永遠にNVIDIA最新HWを追いかける。
4. **MLIR intrinsic経路は堅牢。** NVPTXがWMMAをLLVM intrinsicとして公開→MLIRから到達可能。AMDGPUも同様（MFMA/WMMA・rocWMMA）。コンパイラ層で各社intrinsicを直叩きすれば、portable runtime extensionの不均一性を回避。
5. **IREE/Triton/Mojoが同経路。** AMD MLPerf SDXL（2025）はIREEでAMDGCN直接生成（SPIR-V coopmat非経由）。実証済み。

## 却下した代替案

- **Vulkan KHR cooperative matrix を行列抽象の主経路**: compute-only・HW gated・有用なv2はNVIDIA専有。却下（移植フォールバックには残す）。
- **各社行列命令を手書きアセンブリ**: 保守不能・世代変化追従不可。却下。
- **行列コア不使用（FMAのみ）**: 性能20倍劣化。却下。

## 結果（後日追記・make-or-break）

Phase2のDoD: FP16 GEMMが両ベンダーで行列コア経由動作・cuBLAS/rocBLAS比20-30%以内。

ここに届かない場合の分岐: スコープをattention-onlyに縮小、または行列コア抽象の設計を再検証（ソクラテス問答G4）。

**戦略を変えるベンチマーク**: KHR coopmatが3社compute経路で均一・高性能化したら、Vulkan/SPIR-V経路を再評価。
