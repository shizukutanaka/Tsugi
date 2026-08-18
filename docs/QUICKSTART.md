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

## 5. 実データがあるとき（Python API）

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
