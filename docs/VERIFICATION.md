# Tsugi 検証層 — 全体マップ

Tsugi の楔は GPU codegen でなく **「移植が正しいと証明するクロスベンダー検証層」**
（ADR・docs/PERSPECTIVE-cross-vendor-verification.md）。ソクラテス式問答で 8 つの
視点を積み上げ、`tsugi.audit` が 1 判定に束ねる。本書はその索引。

すべて **CPU で実行・検証済み**（GPU 不要）。GPU codegen 本体は実機が必要で未着手
（README の実装状況表）。

## なぜ検証が要るか

Triton は NVIDIA/AMD のカーネルを両方生成するが、**両者が同じ数値・同じ判断を出す
保証はしない**。堀の本質はライブラリ＋QA であり（SemiAnalysis: AMD の弱点は性能でなく
QA 文化）、クロスベンダー QA そのものが差別化になる。

## 検証のライフサイクル

```
                 デプロイ前（静的・traced IR ＋構成だけで判定）
  feasibility ──→ portability ──→ occupancy ──→ tolerance/envelope ──→ propagation
  (起動するか)   (移植リスク)    (速いか)       (数値許容の目安)        (per-model 発散)
                                                                            │
                 本番（動的・実データが要る）                               ▼
  envelope.check_tensor ──→ nondeterminism ──→ decision
  (入力が認証前提内)        (出力は分布・ノイズ床)  (判断フリップ=タスク影響)
                                                                            │
                 メタ（検証器自身を検証）                                    ▼
  calibration (偽OK 率・検出限界 = この検証スタックは信頼できるか)
```

`tsugi.audit(module, cfg)` が静的層を、`tsugi.audit_runtime(a, b, K, ...)` が実行時層を
それぞれ 1 つの `Audit` 判定に束ねる（examples/audit_demo.py が両方を実演）。

## 8 視点 + メタ

| # | モジュール | 問い | 鍵となる発見 | 詳細 |
|---|-----------|------|-------------|------|
| 1 | `portability` | 移植して壊れるか | warp/wavefront・MMA 形状・bf16・累積順序 | [link](PERSPECTIVE-cross-vendor-verification.md) |
| 2 | `tolerance` | 許容誤差はいくつか | 固定 1e-2 でなく √K·u·scale で *導出*（大K の偽陽性を解消）| [link](PERSPECTIVE-derived-tolerance.md) |
| 3 | `feasibility` | そもそも起動するか | 占有率 0%（起動不能）は性能 WARN でなく BLOCK（離散ゲート）| [link](PERSPECTIVE-launch-feasibility.md) |
| 4 | `propagation` | モデル全体で一致するか | per-kernel 等価 ⇏ per-model 等価。発散は深さ×条件数で増幅 | [link](PERSPECTIVE-error-propagation.md) |
| 5 | `envelope` | 本番入力は認証前提内か | 静的証明を契約化。overflow/denormal/scale/logit を単一ベンダーで検出 | [link](PERSPECTIVE-runtime-envelope.md) |
| 6 | `calibration` | 検証器自身は信頼できるか | 偽OK が致命（非対称）。検出限界 safety·√K·u 未満のバグは不可視 | [link](PERSPECTIVE-verifier-calibration.md) |
| 7 | `nondeterminism` | 出力は再現するか | atomic 非決定で出力は分布。クロス差 ≤ ノイズは INDISTINGUISHABLE | [link](PERSPECTIVE-nondeterminism.md) |
| 8 | `decision` | ユーザーに見える差は何か | 判断フリップ率（スケール不変）。タスク許容＝マージン分布 | [link](PERSPECTIVE-task-equivalence.md) |
| 9 | `rollout` | 生成 1 本で一致するか | per-token フリップ率を生成長へ合成。survival=(1−p)^L で複利減衰。per-token 許容 ⇏ per-sequence 許容 | [link](PERSPECTIVE-rollout.md) |
| 10 | `worstcase` | 最悪入力でも一致するか | 認証エンベロープ内で発散最大化入力を能動探索。平均ケース等価 ⇏ 最悪ケース等価 | [link](PERSPECTIVE-worstcase.md) |
| 11 | `decision`（拡張） | 分類以外のタスクでも一致するか | 回帰/バイナリ/ランキングのフリップ率。argmax を非分類タスクに使うと静かに誤計算する | [link](PERSPECTIVE-task-diversity.md) |
| 12 | `attribution` | 出力の不一致はどこから来るか | onset（汚染開始層）と spike（最大増幅層）を per-layer スキャンで特定。O(L)デバッグ → O(log L)。propagation の理論予測を実測で照合 | [link](PERSPECTIVE-attribution.md) |

補助: `occupancy`（ベンダー別占有率）・`equivalence`（2 出力の等価判定）・
`provenance`（verdict の鮮度）・`oracle_check`（oracle 自体の検証）・
`report`（共通の Risk/Finding/FindingReport）・`audit`（統合ファサード）。

## 2 つの床（検証器の実効分解能）

検証器が「等価」と言える分解能は 2 つの床の大きい方で決まる:

- **数値の床**（視点6）= safety·√K·u — これ未満の系統バグは max_abs に隠れる（偽OK）。
  K とともに √K で拡大するので大K GEMM ほど盲点が広い。
- **HW の床**（視点7）= run-to-run ノイズ — これ未満のクロス差は同一ベンダーの別 run と
  区別できない（INDISTINGUISHABLE）。

  実効分解能 = max(数値の床, HW の床)

そして視点8 は、この床を **タスク影響** に翻訳する: 発散 δ が判断を覆すのは
margin < 2δ のときだけ。ゆえに フリップ率 ≤ P(margin < 2δ)。

## 相補的な計量

単一の計量では足りない（視点6）:

- `max_abs / 導出許容` — 乱雑・局所的な発散に効く（dropblock 等）
- `systematic（RMS 比）` — 系統バイアス（スケール誤差等）に効く。scale/K 不変
- `decision flip rate` — タスク影響。スケール不変

`audit_runtime` はこれらを **fail-safe に合成**する（どれかが発散を示せば BLOCK）。
非対称コスト（偽OK ≫ 偽BLOCK）ゆえ不確実なら BLOCK 寄りに倒す。

## 使い方

```python
import tsugi
mod = tsugi.trace(kernel, args, {}, (0, 0))
static = tsugi.audit(mod, cfg)                       # デプロイ前
print(static.to_text())

runtime = tsugi.audit_runtime(nv_out, amd_out, K,    # 本番/実機データ
                              env=env, noise_floor=nf,
                              logits_a=la, logits_b=lb)
print(runtime.to_text())
```

CLI: `python -m tsugi.portcheck [kernel.py]`（audit へ委譲）。
デモ: `python examples/audit_demo.py`（GPU 不要）。

## 未検証（要実機）

- GPU codegen 本体（MLIR→PTX/AMDGCN）— `lowering.py` が仕様。
- 実機のクロスベンダー出力で `audit_runtime` を駆動し、`noise_floor` を実測して
  tolerance/calibration/decision へ供給する経路。
