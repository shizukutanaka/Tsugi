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

**✅ 今日動く（検証・GPU 不要／`pip install` して `python -m tsugi`・[QUICKSTART](docs/QUICKSTART.md)）**

- **移植性検証** — `tsugi.portability` がクロスベンダー移植リスクを*実行前に*告げる
- **起動可能性検証** — `tsugi.feasibility` が「片方で起動すらしない」構成を*実行前に*BLOCK判定（占有率と別の上流ゲート）
- **数値等価性保証** — `tsugi.equivalence` が両ベンダーの数値発散を検出（Triton にない保証。入力精度/累積順序/丸めモード＋TF32 ポリシーまで分類）
- **占有率推定** — `tsugi.occupancy` が同一構成のベンダー別占有率差を計算
- **タスク影響・実行時検証** — `tsugi.audit_runtime` が数値発散を判断フリップ率/サンプリング分布差へ翻訳
- **CI ゲート** — 判定が終了コード契約（OK/INFO=0・WARN=1・BLOCK=2）——そのまま CI 合否に
- **1ソース・2ベンダー codegen（L2）** — `tsugi.codegen` が単一 IR から実 PTX / 実 AMDGCN を生成し、
  **ベンダー自身のアセンブラ**（ptxas / llvm-mc・どちらも GPU 不要）が受理するまでを機械検証
  （[CODEGEN.md](docs/CODEGEN.md)）

**🔜 実機を要する部分（L3・未検証）**

- **生成物の実行** — 数値の正しさ・レイアウト接合（フラグメント↔タイル軸）・性能
- **torch.compile の実実行** — `torch.compile(model, backend="tsugi")`（今は静的検証だけ届き eager 素通し）
- **autotuning**・**escape-hatch**（cuBLAS/rocBLAS 委譲）

> 完成の線引きは [ASSESSMENT.md](docs/ASSESSMENT.md#プロダクト完成の線引きfirst-principles完成をless-dumbに定義) 参照——
> **検証器プロダクトは完成（使える・正しい・境界明確）**、codegen は実機を要する別軸。
- **導出許容誤差** — `tsugi.tolerance` が許容を K・dtype から導出（固定値でなく数学が許す範囲）
- **合成的等価性** — `tsugi.propagation` が発散を op グラフに沿って伝播しモデルレベルで予測（per-kernel 等価 ⇏ per-model 等価・新視点4）
- **実行時エンベロープ検査** — `tsugi.envelope` が本番入力の認証前提逸脱（overflow/denormal/scale/logit）を単一ベンダー・oracle 不要で検出（新視点5）
- **検証器の較正** — `tsugi.calibration` が検証器自身の偽OK（発散を等価と誤判定）を ground-truth で測り、許容判定の検出限界（√K で拡大）の下に隠れる系統バグを相補計量で捕える（新視点6）
- **非決定性の考慮** — `tsugi.nondeterminism` が GPU の run-to-run ノイズを実測し、出力を分布として比較。クロス差がノイズ未満なら INDISTINGUISHABLE と正直に判定（単一 run 比較のフレークを排す・新視点7）
- **タスクレベル等価** — `tsugi.decision` が数値発散でなく判断フリップ率（argmax/選択トークンの変化）で等価を測る。フリップ率 ≤ P(margin<2δ)・タスク許容はマージン分布（新視点8）
- **自己回帰的等価** — `tsugi.rollout` が per-token フリップ率を生成長へ合成。自己回帰生成では一度ズレたら戻らず survival=(1−p)^L で複利減衰（p=1% でも L=100 で 37%）。per-token 許容 ⇏ per-sequence 許容（propagation の自己回帰版・新視点9）
- **タスク多様性** — `tsugi.decision` の `compare_task` が回帰（値の相対/絶対許容）・バイナリ（sigmoid+threshold 跨ぎ）・ランキング（top-k 集合変化）のフリップ率を測る。argmax を非分類タスクに使うと flip_rate=0 に固まる静かな誤用を防ぐ（新視点11）
- **最悪ケース発散探索** — `tsugi.worstcase` が認証エンベロープ内で発散を最大化する入力を能動探索（黒箱・微分フリー）。代表データ上は良性でも、許容超過の反例がエンベロープ内に在れば BLOCK。平均ケース等価 ⇏ 最悪ケース等価（受動測定への能動反証・envelope と閉ループ・新視点10）
- **発散帰属** — `tsugi.attribution` が移植失敗時の O(L) デバッグを O(log L) に短縮。各層の出力発散を prefix scan し onset（汚染開始層）と spike（最大増幅層）を特定。`bisect_onset` で binary search により 100層モデルも 7 回の測定で onset を絞る。`propagation.dominant`（理論予測）と `attribution.spike`（実測）を照合し理論と実験の接続点を提供（新視点12）
- **ベンダー責帰** — `tsugi.blame` が oracle との相対距離（dist_a / dist_b）を比較し「どちらのベンダーを修正するか」の方向を提供。attribution（どの層か）と組み合わせて "layer X の vendor Y を直せ" という完全診断チェーンを完成させる。oracle_check（shared mode 検出）と相補的（新視点13）
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

### 60 秒で試す（GPU 不要・今すぐ動く）

Tsugi の**今日届く価値は「移植ブロッカーを実機前に告げる検証」**。1 コマンドで体験できる:

```bash
pip install -e python/
python -m tsugi                 # 自己デモ: AMD で "起動すらしない" 構成を検出して見せる
python -m tsugi my_kernel.py    # 自分のカーネル（@tsugi.jit + make_args() 契約）を検証
```

終了コードは **CI ゲート契約**（OK/INFO=0・WARN=1・BLOCK=2）——そのまま CI の合否に使える。
詳しくは [QUICKSTART.md](docs/QUICKSTART.md)。

### torch.compile バックエンド（**想定ユーザーの入口**・検証は今動く）

```python
import torch
import tsugi_torch                       # backend 登録

model = MyTransformer()
compiled = torch.compile(model, backend="tsugi")
out = compiled(x)                        # 実行は eager 素通し（後述）
```

警告として届くもの（GPU 不要・実測例）:

```
[tsugi] verification-only (codegen は L2 まで検証済み・実行は eager 素通し):
  3 numeric ops, amplifiers=['layer_norm', 'softmax'], model_divergence≈5.1e-01,
  実測フリップ 0.000%（上界 1.046%・最悪クラス f16acc・n=256・CPU 2 ベンダー模倣
  ＝実機発散の下界）, task_flip_bound≤55.1%（天井・予測ではない）
  … codegen: 呼び出し 3 件中 3 を IR へ降下し 3/3 ターゲットでアセンブル検証。
  実行は未検証（要実機）
```

**なぜ数字が 2 つあるのか**: `model_divergence` は静的伝播が出す **許容の天井**で、
どのモデルでも大きく出る（実測の 100〜1000 倍になりうる）。判断に使うのは
**実測**のほう——同じ降下 IR を CPU で「2 ベンダー」として走らせ、既知の発散クラス
（累積順序・f16 累積・TF32 入力・RTZ 丸め）ごとに測った値である。実測は既知クラス
しか含まないので**実機発散の下界**である。天井が上界なのは *スケール正規化*
（`max|Δ|/max|a|`）の尺度に限る——`equivalence.compare` と同じ **要素ごと** の相対誤差
では天井を上回りうるので、レポートは両方を出す。

**なぜ実行は eager のままか**: 生成した機械語は L2（ベンダーのアセンブラが受理・
意図どおり符号化・ローダの形）までしか検証されておらず、走らせて正しい保証が無い。
検証は今届き、実行は実機が来てから——[CODEGEN.md](docs/CODEGEN.md) の検証レベル参照。

`tsugi.verify(fx_graph)` を直接呼べば同じ判定を `Audit`（`exit_code`/`to_text`）で得られる。

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

## ドキュメント案内（目的別の読む順）

docs/ は本数が多いので、**目的から入口を選ぶ**。

**使う人**（今すぐ試したい）
[QUICKSTART.md](docs/QUICKSTART.md)（60 秒・`python -m tsugi`）→
[SPEC-verification.md](docs/SPEC-verification.md)（検証 API の仕様）→
[CODEGEN.md](docs/CODEGEN.md)（実 PTX/AMDGCN 生成と検証レベル L0-L3）→
[FAQ.md](docs/FAQ.md) → [BENCHMARK.md](docs/BENCHMARK.md)

**仕組みを知る人**（なぜこの設計か）
[ARCHITECTURE.md](docs/ARCHITECTURE.md) → [SPEC.md](docs/SPEC.md)（DSL/コンパイラ）→
[CODEGEN.md](docs/CODEGEN.md)（生成側・アセンブラを真値に使う）→
[VERIFICATION.md](docs/VERIFICATION.md)（検証層の全体マップ）→
`docs/PERSPECTIVE-*.md`（15 本・各検証層が「なぜ必要か」をソクラテス式問答で導いた記録）→
`docs/adr/`（設計判断の記録）

**開発を引き継ぐ人（人間・AI とも）** ← 会話履歴なしでここから始められる
1. [FEATURE-AUDIT.md](docs/FEATURE-AUDIT.md) — 機能過不足の台帳（**引き継ぎの起点**・現在地）
2. [ASSESSMENT.md](docs/ASSESSMENT.md) — 長所短所改善案（優先度と担当の割当）
3. [INSTRUCTIONS-OPUS.md](docs/INSTRUCTIONS-OPUS.md)（設計判断つき中規模ラウンド）/
   [INSTRUCTIONS-SONNET.md](docs/INSTRUCTIONS-SONNET.md)（スコープ確定済み反復＋エスカレーション基準）
4. [CONTRIBUTING.md](CONTRIBUTING.md)（検証ゲート・規約）/ [RELEASING.md](docs/RELEASING.md)（リリース手順）
5. [GPU-BRINGUP.md](docs/GPU-BRINGUP.md) — **実機 GPU を入手した日に上から実行する手順書**
   （ノイズ床実測 → `SAFETY` 定数の校正 → クロスベンダー検証。Phase 1/2 は GPU 1 台で完結）

**その他**: [SOCRATIC-50-improvements.md](docs/SOCRATIC-50-improvements.md)（問答による改善の全履歴）/
[SOURCES.md](docs/SOURCES.md)（外部出典）/
[MODEL-USAGE-GUIDE.md](docs/MODEL-USAGE-GUIDE.md)（Claude モデル使い分け・個人用メモ）


## 実装状況（正直な現在地・主張と実装の一致）

| マイルストーン | 状態 | 検証 |
|--------------|------|------|
| 完成形ファイル（仕様/ADR/README/FAQ/Benchmark/SPEC-verification） | ✅ 完了 | — |
| リファレンス実装（CPU/NumPy・正しさの真値） | ✅ 完了 | test_reference（数値真値） |
| 上流コンパイラ（DSL→tsugi.tile IR→各社intrinsic写像） | ✅ 完了 | DSL 全14opにlowering同期（drift不変条件）・tracer/compile テスト |
| 不変条件 verify | ✅ 完了 | `python verify.py` 全不変条件 PASS・`python check.py` が全 CPU スイートを実行 |
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
| 発散帰属（attribution・新視点12） | ✅ 完了 | onset/spike で O(L)→O(log L)・propagation 理論と実測の照合 |
| ベンダー責帰（blame・新視点13） | ✅ 完了 | dist_a/dist_b 比較で修正方向を特定・attribution と完全診断チェーンを完成 |
| GPU codegen（IR→PTX/AMDGCN 生成＋アセンブル） | ✅ 完了(L2) | ベンダーのアセンブラが受理・**実行は未検証**（codegen.py） |
| 生成物の実行・レイアウト接合（L3） | ⬜ 未検証 | **要 NVIDIA/AMD GPU** |
| 両ベンダーGPU correctness/性能 | ⬜ 未検証 | **要 NVIDIA/AMD GPU** |

CPU で検証可能な範囲（frontend→IR→各社写像→**実アセンブリ生成→アセンブル**→数値真値）は
完成・検証済み。残るのは機械語の**実行**で、それだけが実機を要する
（[CODEGEN.md](docs/CODEGEN.md) の検証レベル L0-L3 を参照）。

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
