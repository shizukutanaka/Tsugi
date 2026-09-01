# 実機 GPU 立ち上げ手順（GPU BRING-UP）

`docs/FEATURE-AUDIT.md` の **A-2「実機 GPU での end-to-end 検証が一度もない」** に対する
実行計画。**実機を持っていない読者が、実機を入手した日に上から順に実行できる**ことを
目標に書く。会話履歴への参照は置かない。

現状の正直な位置づけ: 全検証層は CPU シミュレーション（NumPy oracle・擬似ベンダー）
でのみ検証済み。本書が扱うのは「検証器を実機に当てる」手順であって、GPU codegen 本体
（A-4・要 LLVM/MLIR）ではない。**codegen が無くても Phase 1〜3 は実行できる**
——検証器は run 関数（seed を受け取り出力テンソルを返す呼び出し可能）しか要求せず、
それは PyTorch の eager カーネルでも満たせるからである。

---

## 前提と成果物

| | Phase 1: ノイズ実測 | Phase 2: SAFETY 校正 | Phase 3: クロスベンダー | Phase 4: codegen |
|---|---|---|---|---|
| 要るもの | GPU 1 台 + PyTorch | GPU 1 台 + PyTorch | NVIDIA と AMD 各 1 台 | + LLVM/MLIR |
| 検証できること | 床が実在するか・大きさ | `SAFETY=4.0` の当否 | 判定そのもの | 単一ソース約束 |
| ブロッカー | なし（今日できる） | なし | 2 ベンダー入手 | A-4 |

Phase 1/2 は **GPU 1 台**（どちらのベンダーでもよい）で完結する。「2 ベンダー揃うまで
何もできない」は誤りで、検証器の定数校正は片側だけで着手できる。

---

## Phase 1 — run-to-run ノイズ床の実測

### 1.1 ハーネスが SKIP でなく実行に入ることを確認

```bash
python tests/gpu/test_audit_runtime_contract.py
```

GPU が無ければ正直に SKIP する設計（`tests/gpu/harness.py:gpu_vendors()` が
`torch.cuda.is_available()` で判定）。実機では SKIP が消えることが第一の関門。
ベンダー判定はデバイス名の文字列一致（`amd`/`radeon`/`instinct`）なので、
**新 SKU で誤判定しないかをここで確認する**（`gpu_vendors()` の戻り値を印字）。

### 1.2 非決定 op を必ず含めること

`tsugi.nondeterminism.ATOMIC_NONDET_OPS` に載る op（`scatter_add`・`index_add`・
`bincount`・`embedding` の backward など）を含まないカーネルは run-to-run で
揺れない可能性が高く、**床がゼロに見えて「校正できた」と誤解する**。
`classify_nondeterminism(op_names).requires_noise_floor` が True になる
グラフを最低 1 つ計測対象に入れる。

```python
from tsugi.nondeterminism import collect_runs, noise_floor_from_runs
stack = collect_runs(lambda s: run_real_kernel(seed=s), n_runs=32)
print(noise_floor_from_runs(stack))   # spread / spread_robust / std / rel
```

### 1.3 記録すべきこと

`spread`（max-min）と `spread_robust`（10-90 パーセンタイル幅）の **両方**。
両者が桁で違えば測定グリッチが混入している（`robust=True` を既定にすべき証拠）。

---

## Phase 2 — `SAFETY=4.0` の校正（本書の主眼）

### 2.1 なぜ校正が要るか

`SAFETY` は許容 `atol = SAFETY·√K·u·scale` と検出限界 `rel = SAFETY·√K·u` の
**両方**を一律にスケールする。誤っていれば**全層**の判定が同じ向きに狂う:

- 大きすぎる → 許容が緩く、真の発散が検出限界の下に隠れる（**偽OK・致命的**）
- 小さすぎる → 良性ノイズを発散と誤判定（**偽BLOCK・回復可能**）

現在の 4.0 は「4σ 相当」という経験値である（`python/tsugi/constants.py`）。
**「4σ」が正しいのは σ が既知のときだけ**で、実測から σ を推定する現実では
必要係数は常にそれより大きい（§2.3）。

### 2.2 実行

```python
from tsugi.audit import audit_cross_vendor
ad = audit_cross_vendor(run_a, run_b, K=K, n_runs=32, robust=True)
print(next(p for p in ad.phases if "safety" in p.name).to_text())
```

`audit_cross_vendor` が集めた run をそのまま校正に使う（**追加の GPU run を
消費しない**）。単体で回すなら:

```python
from tsugi.calibration import calibrate_safety
from tsugi.nondeterminism import collect_runs, pair_deviations
stack = collect_runs(run_a, n_runs=32)
rep = calibrate_safety(pair_deviations(stack), K=K, dtype="float16", scale=measured_rms)
print(rep.to_text(), rep.required)     # required = SAFETY が満たすべき値（σ 単位）
```

### 2.3 何 run 回せばよいか（計画の中身）

校正の標本は「独立な run 対の差」であり、**n_runs 個の run から取れるのは
floor(n_runs/2) 対**（重ならない対を使う。共通の参照 run を使うと標本が相関し
許容限界の統計が正当化できない）。

| 目標 | 必要な対の数 | 必要な run 数 | 根拠 |
|---|---|---|---|
| 正規理論で 0.99/0.95 | 実用上 30〜100 対 | 60〜200 | k(30)=3.06・k(100)=2.68 に収束 |
| 分布仮定なしで 0.99/0.95 | **299 対** | **598** | Wilks 1941: n ≥ ln(0.05)/ln(0.99) |
| 分布仮定なしで 0.999/0.95 | 2995 対 | 5990 | 同上 |

現実的な計画: **まず 64 run（32 対）**で桁を掴み、`required` が 4.0 に近いか
桁違いかを見る。桁違いに小さければ（例 0.01σ）、その時点で「4.0 は少なくとも
不足ではない」と言える。4.0 に近ければ 598 run まで積む価値がある。

`calibrate_safety` は達成信頼度を毎回 INFO で報告する（16 対なら 14.9%）ので、
**「何 run 回したか」でなく「どこまで主張できるか」で記録する**こと。

### 2.4 結果の読み方（何をしてよくて何をしてはいけないか）

| 観測 | 意味 | してよいこと |
|---|---|---|
| `required > 4.0` | 良性ノイズが許容を超える | **`SAFETY` の引き上げ**、または `noise_floor` の実測供給 |
| `required ≪ 4.0`（run-to-run 標本） | 同一ベンダー内は静か | **何もしない**（下げてはいけない・下記） |
| `required ≪ 4.0`（cross-vendor 標本） | 真に余裕がある | 引き下げの検討（盲点が縮む） |

**run-to-run 標本で `SAFETY` を下げてはならない。** 同一ベンダー内の揺れは
*縮約順序差*だけを含み、クロスベンダー発散（タイル形状・行列コア・ライブラリ実装の
差を含む）の **下界** にすぎない。下げれば未測定のクロス成分を許容から外すことになり、
偽OK 方向に倒れる。`calibrate_safety` はこの但し書きを毎回 INFO で出し、
`source="cross_vendor"` を明示したときだけ下げ代を提示する（機械的に封じてある）。

引き下げが正当化できるのは Phase 3（2 ベンダー実機）で、**良性と分かっている**
クロス発散を標本にしたときだけである。

### 2.5 変更するときの手順

`SAFETY` は `python/tsugi/constants.py` の 1 箇所（単一情報源・verify.py 不変条件 26）。
変更したら:

1. `python verify.py` — 閾値の境界感度テスト（不変条件 71 ほか）が新しい値で通るか
2. `python tests/correctness/test_calibration.py` — 検証器の偽OK 率（corpus 評価）が
   悪化していないか。**引き上げは偽OK を増やす方向**なので、`roc_sweep` の
   `false_ok_combined` が閾値未満強度以外で 0 のままであることを確認する
3. `constants.py` の docstring に「実機校正済み（機種・run 数・達成信頼度・日付）」を
   記録する。経験値のままか実測かを後任が区別できるようにする

---

## Phase 3 — 2 ベンダー実機でのクロス検証

```python
from tests.gpu.harness import audit_two_vendors, run_vendor_kernel
rep = audit_two_vendors(
    lambda s: run_vendor_kernel("nvidia", kernel, args, seed=s),
    lambda s: run_vendor_kernel("amd",    kernel, args, seed=s),
    K=K, env=env)
assert rep.portable
```

### 3.1 先に確かめること（順序が重要）

1. **oracle の信頼性**: `oracle_check.verify_oracle()` をメタモルフィック関係で通す。
   oracle が壊れていれば以降の判定は全部無意味。
2. **shared-mode の検出可能性**: 両ベンダーが *同じ* 誤りを出す場合、A↔B 比較では
   原理的に検出できない（`docs/SPEC-verification.md` §4.1）。float64 CPU oracle を
   必ず併走させる。
3. **batch-invariance 床**: `run_batch=` を渡して `measure_batch_variance` を効かせる。
   **`VLLM_BATCH_INVARIANT=1` は NVIDIA 専用で ROCm 未対応**なので、片側だけ床が
   ゼロに近いという非対称が出うる（`docs/SOURCES.md`）。これ自体が観測対象。

### 3.2 記録すべき失敗

`audit_cross_vendor` が BLOCK を出したら、そこで止めずに診断チェーンを回す:
`attribution.diagnose`（どの層か）→ `blame`（どちらのベンダーか）→
`equivalence.classify_divergence`（レイアウト不一致か真の数値発散か）。
BLOCK の *理由* まで記録しないと、次の人が同じ実機時間を使い直すことになる。

---

## Phase 4 — codegen 後

A-4（PTX/AMDGCN 生成）完了後に `run_vendor_kernel` の `NotImplementedError` を
実装で置き換える（`python/tsugi/lowering.py` が実装仕様）。ここで初めて
「単一ソースで両ベンダー」という製品の中核主張が検証対象になる。

---

## 記録テンプレート（結果はここに追記する）

```
日付 / 機種（SKU・アーキ）/ driver・ROCm/CUDA バージョン
Phase 1: n_runs=__ spread=__ spread_robust=__ 対象 op=__
Phase 2: 対の数=__ required=__σ 達成信頼度=__% source=run_to_run|cross_vendor
         判断: SAFETY 据え置き / 引き上げ（__ → __）
Phase 3: verdict=__ BLOCK の内訳（attribution/blame/classify の出力）
```

関連: `docs/FEATURE-AUDIT.md`（A-2 の台帳エントリ）・`docs/SOURCES.md`
（許容限界の統計の出典）・`docs/PERSPECTIVE-nondeterminism.md`（床の考え方）・
`docs/PERSPECTIVE-verifier-calibration.md`（検証器を検証するという発想）。
