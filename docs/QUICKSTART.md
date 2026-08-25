# QUICKSTART — 60 秒で移植性検証

Tsugi の**今日届く価値**は「1 つのカーネルが NVIDIA と AMD で*同じ数値を出すか・
そもそも両方で起動するか*を、実機前に CPU だけで告げる検証」。codegen（PTX/AMDGCN 生成）は
Phase 4（要 LLVM/MLIR + 実機）だが、**検証は今すぐ動く**。

## 1. 入れる

```bash
git clone https://github.com/shizukutanaka/tsugi && cd tsugi
pip install -e python/     # 依存は numpy のみ
```

## 2. 動かす

```bash
python -m tsugi            # 自己デモ
```

出力の要点（デモは意図的に「AMD で起動しない」構成）:

```
[BLOCK] feasibility 起動可能性
    nvidia: LAUNCHABLE
    amd_cdna: NOT-LAUNCHABLE
    単一ソース約束の破綻: shared_mem required=131072 > amd_cdna 上限 65536
判定（静的層）: 移植ブロッカーあり [max_risk=BLOCK]
```

これは**性能の話ではない**——NVIDIA 前提の共有メモリ構成が AMD では*起動すらしない*
（占有率でなく launch 失敗）ことを、GPU を触る前に告げている。

## 3. CI に使う

終了コードが判定そのもの（`report.exit_code` 契約）:

| verdict | exit code | CI での意味 |
|---|---|---|
| OK / INFO | `0` | 通過 |
| WARN | `1` | 要確認（任意で失敗にできる） |
| **BLOCK** | `2` | **出荷を止める**（必ず失敗させる） |

```bash
python -m tsugi my_kernel.py || echo "exit $? — BLOCK なら 2"
```

## 4. 自分のカーネルを検証する

ファイルに `kernel`(@tsugi.jit) と `make_args()`（トレース用の引数 tuple を返す）を
定義する契約（例: [`examples/user_kernel.py`](../examples/user_kernel.py)）:

```python
import tsugi
from tsugi import tile

BLOCK_DIMS = (48,)          # 任意: 占有率/warp 解析用（48 は wavefront(64) 非倍数 → AMD で WARN）
# TILE_CONFIG = ...         # 任意: タイル構成

@tsugi.jit
def kernel(a, b, c, M, N, K, BM, BN, BK): ...

def make_args():            # 必須: kernel をトレースする引数 tuple を返す
    ...
```

```bash
python -m tsugi examples/user_kernel.py
```

## 5. PyTorch モデルを検証する（想定ユーザーの経路）

タイル DSL を書いていなくてよい。`torch.compile` が作る FX グラフをそのまま渡せる:

```python
import torch, tsugi
gm = torch.fx.symbolic_trace(model)        # or torch.compile の backend が受け取る GraphModule
ad = tsugi.verify(gm, ref_logits=logits, flip_budget=0.001)
print(ad.to_text()); raise SystemExit(ad.exit_code)   # CI ゲート契約
```

判定の考え方（fail-safe）: **静的な FX グラフだけでは等価性を認証できない**（第2ベンダーの
実出力が無い）。よって発散量に閾値を発明して BLOCK にはしない。BLOCK になるのは
**あなたが与えた `flip_budget` を予測フリップ率上界が超えたときだけ**。非決定 op や
dynamic shape は WARN として判定に載り、実機クロスベンダー照合が要ることは
pending phase で明示される。

`torch.compile(model, backend="tsugi")` を使えば codegen 前でも同じ静的検証が警告として届く
（実行は eager に素通し・嘘をつかない）。

## 6. Python から 1 行で（CLI と同じ判定）

```python
import tsugi
ad = tsugi.verify("my_kernel.py")   # パス or traced module → Audit
print(ad.exit_code)                 # CI ゲート契約（BLOCK=2）
print(ad.to_text())                 # 人間可読レポート / ad.to_dict() で JSON
```

`tsugi.verify()` は `python -m tsugi` の Python 版——`trace()`＋`audit()` を手で繋ぐ必要はない。

## 7. 実データがあるとき（タスク影響まで）

logits やテンソルの実測出力があれば、静的検証を超えてタスク影響まで測れる:

```python
import tsugi
ad = tsugi.audit_runtime(out_nvidia, out_amd, K=2048, dtype="float16",
                         logits_a=logits_nv, logits_b=logits_amd)
print(ad.to_text())         # 数値等価性・系統バイアス・判断フリップ率・サンプリング分布差…
print(ad.exit_code)         # 同じ CI ゲート契約
```

実機での使い方（GPU 入手後）は [GPU-BRINGUP.md](GPU-BRINGUP.md)。

---

**何が「今」動いて、何が「これから」か**（正直な線引き）:

| | 状態 |
|---|---|
| 移植性・起動可能性・数値等価性・タスク影響の**検証**（CPU） | ✅ 動く（このページ） |
| 実機 GPU でのクロスベンダー検証 | 手順は完成・実機待ち（[GPU-BRINGUP.md](GPU-BRINGUP.md)） |
| codegen（PTX/AMDGCN 生成・torch.compile 実行） | Phase 4・要 LLVM/MLIR + 実機 |

reading path は [README](../README.md#ドキュメント) 参照。
