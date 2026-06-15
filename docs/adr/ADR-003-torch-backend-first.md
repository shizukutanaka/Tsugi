# ADR-003: TorchInductorバックエンド優先（API競合しない）

- 日付: 2026-06-15
- 状態: Accepted

## 状況

楔の刺し方を決める。候補: (a) CUDA Runtime API相当の統一APIを公開しユーザーに直接使わせる、(b) PyTorch `torch.compile` のバックエンドとして刺す。

## 決定

**`torch.compile(backend="tsugi")` / TorchInductorバックエンドを第一の提供形態にする。独自API直接公開は二次。**

## 根拠

1. **PyTorchが新しい抽象層。** 2023-2026の決定的シフト。開発者は`torch`を叩き、CUDAを直接触らない。`torch.compile`（TorchDynamo→TorchInductor）はデフォルトでTritonカーネルを生成済み。カーネルバックエンドは既にpluggable。
2. **バックエンドになればPyTorchのエコシステム全体を継承。** API競合（CUDA Runtime APIクローン）は誰も乗り換えない。バックエンド差し替えは透過的。
3. **前例: Tritonが同じ経路で成功。** TorchInductorのデフォルトカーネル生成器がTriton。NVIDIA(PTX)+AMD(AMDGCN)を1カーネルから生成。IBM/vLLMが単一Tritonコードで両ベンダーSoTA到達。
4. **ライブラリ堀の回避にもなる。** バックエンド層なら標準op（GEMM）でcuBLAS/rocBLASにescape-hatch可能。全面再実装不要。
5. **PMF信号が明確。** 「実HuggingFace transformerが両ベンダーでe2e動作」が成功の単一指標になる。

## 却下した代替案

- **CUDA Runtime APIクローン**: OpenCL/HIP既存・差別化なし・誰も乗り換えない。却下。
- **独自DSL直接公開のみ**: Triton/Mojoと正面競合・採用障壁高い。DSLは持つがバックエンド統合を主経路にする。
- **JAX/PJRTプラグイン優先**: 有力だがPyTorchの方が市場大。v1.0以降にIREE経由で追加。

## 結果

開発の検証ベクトルが「torch.compileで実モデルが両ベンダー動作するか」に集約。Phase4のDoDがそのまま製品検証になる。
