# ADR-001: NVIDIA経路にPTX採用・SPIR-V不採用

- 日付: 2026-06-15
- 状態: Accepted

## 状況

統一GPU計算層の中間表現を決める。候補は (a) SPIR-V を全ベンダー共通IRにする、(b) MLIR + 各社LLVMバックエンド（NVIDIA=PTX, AMD=AMDGCN）。

SPIR-V は Khronos の portable IR で「全ベンダー共通」に見える。だが調査で**非対称性**が判明。

## 決定

**NVIDIA経路はPTX（NVPTX backend）を採用。SPIR-Vは移植フォールバック（Intel/Vulkan）に限定。中核IRはMLIR。**

## 根拠

1. **SPIR-VはNVIDIAでCUDA計算のfirst-class入力でない。** NVIDIA CUDA driver が消費するのは PTX。SPIR-V は NVIDIA の Vulkan グラフィックス/計算ドライバ経由でしか届かず、成熟した CUDA ランタイムには入らない。SPIR-V中核にするとNVIDIAで最高性能の経路（CUDA driver + PTX）を捨てることになる。
2. **OpenCL失敗の教訓: 市場リーダー（NVIDIA）上で最高性能を出せないと普及しない。** NVIDIAをVulkan経由の二級市民にする設計は致命的。
3. **NVPTX backend は upstream LLVM。** 誰でもLLVM IRからPTXを吐ける（`ptx_kernel` calling convention・`libdevice`）。MLIRから到達可能。
4. **MLINEは3社共通の唯一の基盤。** NVPTX・AMDGPU が upstream。MLIR中核にすれば各社へ分岐可能。
5. **SPIR-Vは依然有用。** Intel（Level Zero がSPIR-V ingestion必須）・Vulkan移植には最適。ゆえに「不採用」でなく「フォールバックに限定」。

## 却下した代替案

- **SPIR-V全ベンダー共通IR**: NVIDIAで最高性能不可（上記1,2）。却下。
- **各社ネイティブIR直書き（PTX手書き＋AMDGCN手書き）**: 共通化の利点ゼロ・保守地獄。却下。
- **新IR をゼロから設計**: C6ゼロベースだが車輪の再発明。MLIRで足りる。却下。

## 結果（後日追記）

（Phase1完了後に記録: NVPTX/AMDGPU両経路でmatmul動作確認の結果）
