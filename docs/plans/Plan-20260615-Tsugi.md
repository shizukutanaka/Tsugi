# Plan-20260615-Tsugi

> **Tsugi** = 継ぎ — GPU ベンダーを接合する移植検証層
> 統一GPU計算レイヤー。NVIDIA / AMD / Intel横断。CUDA牙城への楔。
> **状態: DRAFT（未承認）** — G1により承認まで実装着手不可

---

## 0. プロダクト定義（C11準拠・全判断の基準）

このプロダクトは **PyTorchで開発するML技術者** が **GPUベンダーロックイン（CUDA依存）から脱却** するために使う **`torch.compile`バックエンド兼タイルDSLコンパイラ** である。**「NVIDIA専用ライブラリ（cuDNN/cuBLAS）の全面再実装はしない」「バイナリCUDA変換はしない（法的回避）」を設計上の制約とする。**

形態3軸: 実行環境=ライブラリ＋コンパイラ / インタラクション=バッチ（AOT/JITコンパイル） / データ所在=ローカル（外部送信ゼロ）

---

## 1. 目的

CUDAの堀は**言語でなくライブラリとフレームワーク統合**。ゆえに正面突破（CUDA APIクローン）でなく、**フレームワーク層（PyTorch）に楔を打つ**。開発者は既に`torch`を叩いており`torch.compile`はTritonカーネルを生成済み。ここを横取りする。

- **1文目的**: 1つのタイルカーネルを書けば（または`torch.compile`が生成すれば）、コンパイラが各ベンダー向けに自動チューニングしてNVIDIA・AMD両対応バイナリを吐く層を作る。
- **勝ち筋**: コンパイラ/IR層（MLIR + 各社LLVMバックエンド）に集約。OpenCL失敗の教訓＝リファレンス実装先行・NVIDIA上で最高性能・強い方針性。

---

## 2. スコープ

### 含む（v0.1〜v1.0）
- タイルベースDSL（Triton型）またはTriton薄ラッパー — フロントエンド1種
- MLIR中心IR → 2バックエンド: **NVIDIA（NVPTX→PTX）/ AMD（AMDGPU→AMDGCN）**
- カーネル4系統: **GEMM（FP16/BF16）/ Fused Attention / LayerNorm・RMSNorm / Elementwise融合**
- タイルサイズ・行列コアレイアウトの自動チューニング
- **TorchInductorバックエンド**として提供（実験的）
- Tensorコア抽象: `matmul`タイルopを各社intrinsic（WMMA/MFMA）へlowering
- ベンダーライブラリへのescape-hatch（cuBLAS/rocBLASが勝つ箇所は委譲）

### 含まない（明示的除外・スコープ確定の核）
- ❌ CUDAバイナリ変換層（ZLUDA型）— NVIDIA EULA違反・法的リスク（AMD ZLUDA撤回事例）
- ❌ cuDNN/cuBLASの全面再実装 — 19年の蓄積に正面勝負しない
- ❌ 統一ランタイムAPIの自作 — **IREE HALを再利用**（cuda/hip/vulkan/metal/cpu抽象済み）
- ❌ グラフィックス機能 — 計算特化
- ❌ 学習特化最適化（v1.0までは推論優先。AMD訓練パリティ未達のため）
- ❌ Intel/Appleバックエンド（v1.0以降。SPIR-V/Level Zero/Metalは将来）
- ❌ 課金コード・PII収集（憲法C5・課金禁止）

### 再利用する既存OSS（C6ゼロベース×車輪の再発明回避の両立）
| 資産 | 用途 | 理由 |
|------|------|------|
| MLIR + LLVM (NVPTX/AMDGPU) | コード生成 | upstream・無償・3社共通基盤 |
| Triton | DSL前例 or 拡張対象 | 競合でなく貢献も検討 |
| IREE | ランタイムHAL・MLIRコンパイラ参照 | ランタイム再構築を回避 |
| SPIR-V | 移植フォールバック | Intel/Vulkan経路（v1.0以降） |

---

## 3. フェーズ + DoD（ステージゲート型）

### Phase 0: 完成形ファイル先行作成（C11/G12）
**着手前必須。** 仕様が実装を駆動する。
- DoD: 下記「§4 完成形ファイル一覧」が全て存在

### Phase 1: 基盤検証（PoC・S/M規模・定義逆算許可）
- MLIR最小パイプライン: 1つの`matmul`タイルop → NVPTX → PTX動作確認
- 同op → AMDGPU → AMDGCN動作確認
- **DoD**: 単一GEMMがNVIDIA・AMD両GPUで数値正答（correctness先行・性能は後）

### Phase 2: Tensorコア抽象（最重要リサーチベット）
- `matmul`タイルop → WMMA（NVPTX intrinsic）/ MFMA・WMMA（AMDGPU/rocWMMA）lowering
- **Vulkan cooperative matrixに依存しない**（compute-only・HW gated・有用なv2はNVIDIA専有）
- **DoD**: FP16 GEMMが両ベンダーで行列コア経由動作・cuBLAS/rocBLAS比20〜30%以内

### Phase 3: カーネル4系統 + 自動チューニング
- GEMM / Attention / Norm / Elementwise融合を実装
- タイルサイズ・レイアウトのautotuning
- **DoD**: 4系統が両ベンダー動作・固定transformer shape群でTriton比20%以内

### Phase 4: TorchInductorバックエンド統合（楔の本体）
- `torch.compile`経由でカーネル生成
- **DoD**: 実HuggingFace transformerがNVIDIA・AMD両GPUでe2e動作・数値正答（=PMF信号）

### Phase 5: 出荷（v0.1）
- 出荷チェックリスト全PASS（§6）
- **DoD**: GitHub公開・README/SECURITY/CHANGELOG完備・クリーンインストール検証

### Phase戻り規則
- Phase1でTriton比20%に届かない → スコープ縮小（attention-onlyへ）
- correctness未達 → 性能着手前に必ず修正

---

## 4. 完成形ファイル一覧（C11・実装前に作成）

| ファイル | 内容 | Phase |
|---------|------|-------|
| `docs/SPEC.md` | タイルDSL文法・IR階層・lowering規則・対応op一覧 | 0 |
| `docs/ARCHITECTURE.md` | 3層構成（IR/ランタイム/カーネル）・MLIRパイプライン図・各社バックエンド境界 | 0 |
| `docs/adr/ADR-001-mlir-over-spirv.md` | NVIDIA経路にPTX採用・SPIR-V不採用の根拠（NVIDIA非first-class問題） | 0 |
| `docs/adr/ADR-002-no-binary-cuda.md` | バイナリCUDA変換不採用・法的根拠（EULA・ZLUDA事例） | 0 |
| `docs/adr/ADR-003-torch-backend-first.md` | API競合でなくTorchInductorバックエンド優先の戦略根拠 | 0 |
| `docs/adr/ADR-004-tensorcore-abstraction.md` | 行列コアをMLIR intrinsic経由でlowering・Vulkan coopmat非依存の根拠 | 0 |
| `README.md` | 3層説明文・Features・Installation・Usage最小例・License | 0 |
| `docs/FAQ.md` | 「CUDAクローンか?」「ZLUDAと何が違う?」「性能は?」等 | 0 |
| `docs/BENCHMARK.md` | 計測対象shape・比較対象（Triton/cuBLAS/rocBLAS）・閾値定義 | 0 |
| `SECURITY.md` | 報告先・SLA・PII禁止方針 | 5 |
| `CHANGELOG.md` | Keep a Changelog形式 | 5 |
| `CONTRIBUTING.md` | 開発環境・ブランチ戦略・Conventional Commits | 5 |

---

## 5. リスク（ソクラテス問答・STRIDE由来）

### 技術リスク
| # | リスク | 対策 |
|---|--------|------|
| R1 | **Tensorコア抽象が最難関**。3社の行列命令（WMMA/MFMA/XMX）が不透明・スレッドレイアウト非公開・世代間で挙動変化（HMMA.884→16816、mma.m8n8k4が低速FMAに退化） | Phase2でPoC優先・MLIR intrinsic経路・Vulkan coopmat非依存 |
| R2 | コンパイラがhand-tuned CUTLASS比で最大20%劣化（H100の最難カーネル） | escape-hatchでベンダーライブラリ委譲・標準shapeで勝負 |
| R3 | SPIR-VがNVIDIAでCUDA計算のfirst-class入力でない | NVIDIA=PTX経路固定・SPIR-Vは移植フォールバックのみ |
| R4 | AMD訓練パリティ未達（RCCL<NCCL・FlashAttention3差） | v1.0まで推論優先・訓練は後 |

### 戦略・非技術リスク
| # | リスク | 対策 |
|---|--------|------|
| R5 | **ライブラリ堀は存在的脅威**（15年のcuDNN/cuBLAS蓄積に勝てない） | 全面再実装せず・hotカーネルのみ生成＋委譲 |
| R6 | OpenCL失敗パターン（coopetition統治・実装断片化・リファレンス不在） | リファレンス実装先行・強い方針性・NVIDIA上で最高性能 |
| R7 | NVIDIA反復パターン（ベンダー拡張→Khronos標準化→次世代は再びNVIDIA専有） | Khronos追従に依存しない・コンパイラ経路で各社intrinsic直叩き |
| R8 | ソロ/小規模でCUDA全面複製は非現実的 | 縦に狭く深く（GEMM+attention+norm+elementwise・2バックエンド） |

### 法的リスク（最優先回避）
| # | リスク | 対策 |
|---|--------|------|
| R9 | **NVIDIA EULAが翻訳層を禁止**（2021〜・CUDA11.6で明文化） | バイナリ変換を一切しない・ソースレベル/新DSLのみ・公開ドキュメント準拠 |

---

## 6. 出荷チェックリスト（v0.1・全PASS必須）

- [ ] Linter警告ゼロ（clang-tidy / ruff / mypy）
- [ ] テスト全通過（correctness: 両ベンダー数値正答 / カバレッジMVP≥50%）
- [ ] 脆弱性スキャンCRITICAL/HIGHゼロ
- [ ] CrossReview完了（AIセルフレビューStop hook代替）
- [ ] 完成形ファイル（SPEC/ARCHITECTURE/ADR×4/README/FAQ/BENCHMARK）と実装の整合性確認
- [ ] クリーンインストール検証（NVIDIA環境・AMD環境両方）
- [ ] 自己ドッグフーディング（実transformer 1本をe2e実行）
- [ ] ベンチマーク結果記録（Triton比・cuBLAS/rocBLAS比をBENCHMARK.mdへ）
- [ ] PII漏洩スキャン完了（外部送信ゼロ確認）
- [ ] **課金コード不在確認**（Stripe・寄付ボタン・有料機能ゼロ）
- [ ] ロールバック手順添付（Git revert / feature flag）
- [ ] ライセンス確定（MIT/Apache-2.0推奨・依存ライセンス監査: LLVM=Apache-2.0 with LLVM exception / IREE=Apache-2.0）

---

## 7. 成功指標（アウトカム視点・出力でなく成果）

出荷=完了でない。以下で「正しいものだったか」を測る。
- **PMF信号**: 実HuggingFace transformerが両ベンダーでe2e動作・数値正答（Phase4 DoD）
- **性能指標**: 標準transformer shapeでTriton比20%以内
- **採用指標（出荷後）**: GitHub Issue/Star・自分が継続使用するか
- 指標が動かない → 実装品質が高くても「正しいものでなかった」と判断・撤退をdecisions.mdへ記録

---

## 8. 規模判定・実行戦略

- **規模: XL**（新プロダクト立ち上げ・アーキ全体設計）
- モデル配分（サンドイッチ型）: 設計・監査=Opus / 実装=Sonnetサブエージェント / 整形=Haiku
- Plan.md: **必須**（本書）
- 品質ゲート: フルパイプライン + 付議（G8該当箇所）
- `/effort`: xhigh〜max

---

## 9. 戦略を変えるベンチマーク（リサーチ由来・判断の分岐点）

- `VK_KHR_cooperative_matrix`が3社compute経路で均一・高性能化 → Vulkan/SPIR-V経路が魅力化・各社LLVMバックエンド置換を再検討
- AMD ROCm+IREEが訓練でCUDAパリティ達成 → AMD-first強化
- NVIDIAがSPIR-VをCUDA計算first-class入力に → 移植計算が大幅簡素化

---

## ⚠️ 承認待ち（G1）

**本Plan.mdは未承認。** 実装着手には人間承認が必要。

確認してほしい論点（不可逆・G8該当を含む）:
1. **スコープの狭さ**: 「2バックエンド（NVIDIA+AMD）・カーネル4系統・推論優先」で良いか。Intel/訓練を後回しにする判断。
2. **楔の選択**: 「TorchInductorバックエンド優先（API競合しない）」戦略で良いか。
3. **プロダクト名**: `Tsugi`で良いか（別案あれば）。
4. **ライセンス**: MIT / Apache-2.0どちらか（LLVM/IREE依存はApache-2.0系のため整合性ではApache-2.0推奨）。

承認 or 修正指示をもらえれば、Phase 0（完成形ファイル作成）に着手する。
