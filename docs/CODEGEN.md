# CODEGEN — 単一 IR から実 PTX / 実 AMDGCN を生成し、ベンダーのアセンブラで検証する

Tsugi の「単一ソースで両ベンダー」という約束の**生成側**。本書はその実体と、
**どこまで検証されていて、どこからが未検証か**を述べる。

## 要件の再定義（なぜ長らく「不可能」と書いていたか）

50 回以上のラウンドで「codegen は LLVM/MLIR + 実機が要るので不可能」と書き続けたが、
これは要件の誤りだった。工程を分けると:

| 工程 | 必要なもの | 実機が要るか |
|---|---|---|
| IR → アセンブリ**テキスト**生成 | 純関数（文字列） | 不要 |
| アセンブリ → 機械語（**アセンブル**） | `ptxas` / `llvm-mc` | 不要（**CPU ツール**） |
| 機械語の**実行** | GPU | **要る** |

必要なのは実機ではなく**アセンブラ**であり、それは CPU プログラムである。
この一行の誤りが、検証可能だったプロダクトの半分を凍結させていた。

## 検証レベル（正直な梯子）

| レベル | 意味 | 誰が保証するか |
|---|---|---|
| **L0 未対応** | その op / arch に命令列が無い（アセンブラが不受理を含む） | — |
| **L1 生成のみ** | テキストは出るがアセンブラが無く未確認 | 誰も（**合格と言わない**） |
| **L2 アセンブル検証済み** | 命令の存在・構文・arch 可用性 | **ベンダーのアセンブラ** |
| **L3 実機実行検証済み** | 数値・レイアウト・性能 | 実機（**このリポジトリでは常に空**） |

**L2 が保証しないもの**（黙らない）: データレイアウト——どのレーンがどの要素を持つか。
行列コア命令（WMMA/MFMA）とスカラーレーンのレイアウト接合は実機照合が要り、
該当 op には `layout-unstitched` 注記が付く。AMD 側は `.amdhsa_kernel` 記述子を
出力しないため、生成物は**アセンブル可能だがロード不可**である。

## 使う

```python
import tsugi
from tsugi import codegen

mod = tsugi.trace(kernel, args, {}, program_ids=(0, 0))
em, asm = codegen.verify_codegen(mod, target="amd_cdna")   # arch 既定 gfx90a
print(em.text)          # 実 AMDGCN
print(asm.level)        # L2-アセンブル検証済み / L1-生成のみ
print(em.unstitched)    # 保証しない箇所
```

`compile` からも到達する:

```python
art = tsugi.compile(kernel, args, target="nvidia", emit_machine_code=True)
print(art.asm, art.level)
```

`tsugi.verify(...)` / `python -m tsugi` の判定にも codegen phase が載る。

## アセンブラを真値に使う（本層の存在理由）

手書きの対応表では作り込めない種類の移植ブロッカーが、ツールチェインから**事実として**返る:

```python
codegen.probe_op("dot", target="nvidia", arch="sm_60").stderr
# ptxas: Feature 'WMMA with floating point types' requires .target sm_70 or higher

codegen.probe_op("dot", target="amd_cdna", arch="gfx1100", isa="amd_cdna").stderr
# llvm-mc: instruction not supported on this GPU   （MFMA は CDNA 専用）
```

`lowering.VENDOR_LOWERING` が「どの命令に落ちるか」を*主張*する層、
`codegen` がその主張を**確かめる**層、という役割分担になっている。

## アセンブラの入れ方（任意・本体の依存ではない）

無くても codegen は動く（L1-生成のみに落ちる）。L2 を得たいときだけ入れる。

| target | ツール | 入れ方 | ライセンス |
|---|---|---|---|
| nvidia | `ptxas` | `pip install nvidia-cuda-nvcc-cu12` | **NVIDIA CUDA EULA（proprietary）** |
| amd_cdna / amd_rdna | `llvm-mc` | `apt install llvm`（LLVM 同梱） | Apache-2.0 with LLVM exception |

`ptxas` は proprietary ゆえ **Apache-2.0 配布物の依存として宣言していない**
（`pyproject.toml` に入れない）。開発者が自分の判断で入れる任意ツールとして扱う。
場所は `TSUGI_PTXAS` / `TSUGI_LLVM_MC` 環境変数でも指定できる。

いずれも **GPU は不要**（アセンブラは CPU で走る）。

## 発散源としての命令選択

`codegen.BIT_EXACT_ACROSS_VENDORS` は「生成した命令自体がベンダー間でビット同一か」を
持つ。`equivalence` 層が発散の *量* を扱うのに対し、ここは発散の *出所* を名指す。

| 分類 | op | 根拠 |
|---|---|---|
| ビット同一 | add / sub / mul / max / cast / zeros / load / store | IEEE-754 が結果を一意に定める |
| **非同一** | exp | 双方とも 2^x 近似（`ex2.approx.f32` / `v_exp_f32`）で実装が異なる |
| **非同一** | div | NVIDIA `div.rn.f32` は正確丸め / AMD は `v_rcp_f32` + 乗算 |
| **非同一** | sqrt / rsqrt | NVIDIA `sqrt.rn.f32` は正確丸め・`rsqrt.approx` は近似 / AMD は双方近似 |
| **非同一** | dot / reduce | 行列コアの累積順序・入力精度、クロスレーン縮約の結合順序 |

ULP 数は持たない。ISA ドキュメント由来の**構造的分類**のみで、測定していない数値を
係数として持ち込まない（本リポジトリのガードレール）。実際の発散量は
`equivalence` / `nondeterminism` が実データから測る。

## これから（L3 へ）

実機が手に入ったときの順序は [GPU-BRINGUP.md](GPU-BRINGUP.md) と同じ枠に入る:

1. `.amdhsa_kernel` 記述子と PTX のパラメータ ABI を実ランタイムに合わせる（ロード可能化）
2. レイアウト接合（フラグメント ↔ タイル軸）を実機出力で照合し `layout-unstitched` を潰す
3. リファレンス（CPU/NumPy）と数値照合し、`equivalence` の予測と突き合わせる

そこまで到達して初めて L3 を主張できる。それまで本層は L2 までしか言わない。

---

関連: [`python/tsugi/codegen.py`](../python/tsugi/codegen.py)（実装）・
[`tests/correctness/test_codegen.py`](../tests/correctness/test_codegen.py)（真値はベンダーのツール）・
不変条件 90-92（`verify.py`）・[SOURCES.md](SOURCES.md)（ISA 出典）
