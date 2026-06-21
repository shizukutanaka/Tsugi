<h1 align="center">Tsugi</h1>

<p align="center">
  <b>継ぎ — GPU ベンダーを接合する移植検証層</b><br>
  1つのカーネルを書けば NVIDIA でも AMD でも動く。CUDA ロックインからの脱却。
</p>

---

## これは何か

Tsugi は **PyTorch 開発者が GPU ベンダーロックイン（CUDA 依存）から脱却するための `torch.compile` バックアンド兼タイル DSL コンパイラ**。

大きなモデルを動かすたびに NVIDIA に縛られている。`cuDNN`/`cuBLAS` の壁は厚く、AMD/Intel に移れない。Tsugi は **MLIR を中核に各社 LLVM バックエンド（NVIDIA=PTX, AMD=AMDGCN）へ lowering** し、1 つのタイルカーネルから両ベンダー対応バイナリを生成する。CUDA の言語ではなく **フレームワーク層（PyTorch）に楔**を打つ。

外部送信なし・完全ローカル動作。バイナリ CUDA 変換はしない（合法・[ADR-002](docs/adr/ADR-002-no-binary-cuda.md)）。

> **状態: v0.x POC 段階。** 公開 API は未凍結。実 GPU での性能検証は進行中。

## Features

- **1ソース・2ベンダー** — 同じタイルカーネルが NVIDIA(PTX) と AMD(AMDGCN) で動く
- **torch.compile バックエンド** — `torch.compile(model, backend="tsugi")` で透過利用。codegen 前でも FX グラフに静的検証（propagation）を走らせ増幅 op・モデル発散を warn（検証だけ先に届ける）
- **Tensorコア抽象** — `tile.dot` が各社行列コア（WMMA/MFMA）へ自動 lowering
- **autotuning** — タイルサイズ・レイアウトをベンダー別に自動探索
- **escape-hatch** — 標準 GEMM は cuBLAS/rocBLAS へ委譲（性能優先）
- **移植性検証** — `tsugi.portability` がクロスベンダー移植リスクを*実行前に*告げる（GPU不要・新視点）
- **起動可能性検証** — `tsugi.feasibility` が「片方で起動すらしない」構成を*実行前に*BLOCK判定（占有率と別の上流ゲート・新視点3）
- **数値等価性保証** — `tsugi.equivalence` が両ベンダーの数値発散を検出（Triton にない保証）
- **占有率推定** — `tsugi.occupancy` が同一構成のベンダー別占有率差を計算
- **導出許容誤差** — `tsugi.tolerance` が許容を K・dtype から導出（固定値でなく数学が許す範囲）
- **合成的等価性** — `tsugi.propagation` が発散を op グラフに沿って伝播しモデルレベルで予測（per-kernel 等価 ⇏ per-model 等価・新視点4）
- **実行時エンベロープ検査** — `tsugi.envelope` が本番入力の認証前提逸脱（overflow/denormal/scale/logit）を単一ベンダー・oracle 不要で検出（新視点5）
- **検証器の較正** — `tsugi.calibration` が検証器自身の偽OK（発散を等価と誤判定）を ground-truth で測り、許容判定の検出限界（√K で拡大）の下に隠れる系統バグを相補計量で捕える（新視点6）
- **非決定性の考慮** — `tsugi.nondeterminism` が GPU の run-to-run ノイズを実測し、出力を分布として比較。クロス差がノイズ未満なら INDISTINGUISHABLE と正直に判定（単一 run 比較のフレークを排す・新視点7）
- **タスクレベル等価** — `tsugi.decision` が数値発散でなく判断フリップ率（argmax/選択トークンの変化）で等価を測る。フリップ率 ≤ P(margin<2δ)・タスク許容はマージン分布（新視点8）
- **統合監査** — `tsugi.audit` が 8 視点＋メタ＋基盤を 1 つの判定に束ね、静的層の verdict と実行時チェックリスト（実機データが要る層）をライフサイクル順に一望（運用統合）
- **verdict の鮮度保証** — `tsugi.provenance` が監査結果を環境フィンガープリント（python/numpy/driver/rocm/cuda）に束ね、スタック更新で `is_stale` を自動判定。「一度 OK＝永遠に OK」を排す（時間軸の統合）
- **permissive のみ** — 依存は全て MIT/Apache-2.0 系

## Installation

> 要 LLVM/MLIR（NVPTX + AMDGPU backend 有効）・CMake・Ninja。NVIDIA は CUDA Toolkit、AMD は ROCm。

```bash
git clone https://github.com/shizukutanaka/tsugi
cd tsugi
cmake -B build -G Ninja -DTsugi_ENABLE_NVIDIA=ON -DTsugi_ENABLE_AMD=ON
cmake --build build
pip install -e python/
```

## Usage（最小例）

```python
import torch
import tsugi_torch  # backend登録

model = MyTransformer().cuda()           # or .to("hip")
compiled = torch.compile(model, backend="tsugi")
out = compiled(x)                        # Tsugi経由・両ベンダー対応
```

タイルカーネルを直接書く:

```python
import tsugi
from tsugi import tile

@tsugi.jit
def matmul(a, b, c, M, N, K,
           BM: tsugi.constexpr, BN: tsugi.constexpr, BK: tsugi.constexpr):
    pid_m, pid_n = tsugi.program_id(0), tsugi.program_id(1)
    acc = tile.zeros((BM, BN), tsugi.float32)
    for k in range(0, K, BK):
        acc += tile.dot(tile.load(a, (pid_m, k), (BM, BK)),
                        tile.load(b, (k, pid_n), (BK, BN)))
    tile.store(c, (pid_m, pid_n), acc.to(tsugi.float16))
```

検証層をまとめて回す（静的＋実行時の両 facade・GPU 不要のデモ）:

```bash
python examples/audit_demo.py      # tsugi.audit（静的）と tsugi.audit_runtime（実データ）
python -m tsugi.portcheck k.py     # 移植性レポート CLI（audit へ委譲）
```

## アーキテクチャ

```
Tile DSL / torch.compile  →  tsugi.tile IR  →  tsugi.gpu IR  →  ┬ NVVM → PTX
                                                                └ ROCDL → AMDGCN
```

詳細: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) / 仕様: [docs/SPEC.md](docs/SPEC.md)（DSL/コンパイラ）・[docs/SPEC-verification.md](docs/SPEC-verification.md)（検証 API）/ 検証層の全体マップ: [docs/VERIFICATION.md](docs/VERIFICATION.md)


## 実装状況（正直な現在地・主張と実装の一致）

| マイルストーン | 状態 | 検証 |
|--------------|------|------|
| 完成形ファイル（仕様/ADR/README/FAQ/Benchmark/SPEC-verification） | ✅ 完了 | — |
| リファレンス実装（CPU/NumPy・正しさの真値） | ✅ 完了 | test_reference（数値真値） |
| 上流コンパイラ（DSL→tsugi.tile IR→各社intrinsic写像） | ✅ 完了 | DSL 全14opにlowering同期（drift不変条件）・tracer/compile テスト |
| 不変条件 verify | ✅ 完了 | verify.py 63/63 invariants・全22スイート147テスト関数PASS |
| 移植性検証層（portability・新視点） | ✅ 完了 | warp/MMA/bf16/累積順序 リスク検出 |
| 数値等価性層（equivalence・新視点） | ✅ 完了 | 擬似ベンダーで発散検出を実証 |
| 占有率推定（occupancy） | ✅ 完了 | 一次情報源HW値・同一構成のベンダー差 |
| 導出許容誤差（tolerance・新視点2） | ✅ 完了 | K依存・固定値の過剰検出を解消 |
| 起動可能性検証（feasibility・新視点3） | ✅ 完了 | 同一構成がNVIDIA起動可/AMD起動不能をBLOCK検出 |
| 合成的等価性（propagation・新視点4） | ✅ 完了 | 発散が深さで~2000倍累積・モデル許容は単一カーネルの12倍 |
| 実行時エンベロープ検査（envelope・新視点5） | ✅ 完了 | fp16 overflow/denormal/scale逸脱/logit>11.09 を単一ベンダーで検出 |
| 検証器の較正（calibration・新視点6） | ✅ 完了 | max_abs単独は偽OK 3/6・合成判定で偽OK 0/6・検出限界 K=2048で8.8% |
| 非決定性の考慮（nondeterminism・新視点7） | ✅ 完了 | run-to-run ノイズ実測・出力を分布として3状態判定・単一run比較のフレーク実証 |
| タスクレベル等価（decision・新視点8） | ✅ 完了 | 判断フリップ率はスケール不変・abs誤差10倍でも同一・P(margin<2δ)上界を実証 |
| oracle 健全性検査（oracle_check） | ✅ 完了 | a≈b≈oracle で共有モード障害（両ベンダー同一バグ）を検出 |
| 統合監査（audit・運用統合） | ✅ 完了 | 静的層を1判定に集約・実行時チェックリスト併記・portcheck は audit へ委譲 |
| verdict 鮮度保証（provenance・時間軸統合） | ✅ 完了 | verdict を環境fingerprintに束ね・スタック更新で is_stale 自動判定 |
| portcheck CLI（ユーザーカーネル対応） | ✅ 完了 | `python -m tsugi.portcheck k.py` |
| GPU codegen本体（MLIR→PTX/AMDGCN・実コンパイル） | ⬜ 未実装 | **要 LLVM/MLIR + 実機** |
| 両ベンダーGPU correctness/性能 | ⬜ 未検証 | **要 NVIDIA/AMD GPU** |

CPU で検証可能な範囲（frontend→IR→各社写像→数値真値）は完成・検証済み。
機械語生成と GPU 実行は実機が必要で未着手。`lowering.py` がその実装仕様。

## ZLUDA と何が違う？

ZLUDA は CUDA バイナリを翻訳する（NVIDIA EULA 抵触・AMD が撤回）。Tsugi は **新 DSL とソースレベルのみ**。バイナリ変換しない。詳細: [docs/FAQ.md](docs/FAQ.md)

## ロードマップ

| バージョン | スコープ |
|-----------|---------|
| v0.1 | NVIDIA+AMD / GEMM・Attention・Norm・Elementwise / torch backend |
| v1.0 | 推論本番品質・Intel(SPIR-V)追加 |
| v1.x | 訓練最適化・Apple Metal・JAX(PJRT) |

## License

Apache-2.0（依存の LLVM/IREE が Apache-2.0 系のため整合）。

## 設計哲学

Carmack（性能）× Martin（単一責任）× Pike（簡潔）。ゼロ/最小依存。主張と実装の一致 — 未検証の経路は「未検証」と明記する。
