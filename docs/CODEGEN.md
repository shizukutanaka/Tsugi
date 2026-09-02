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
| L2＋符号化照合 | 意図した命令が*実際にその機械語になった*こと | **ベンダーの逆アセンブラ／リンカ** |
| L2＋ロード構造 | ローダが要求する部品（記述子・メタデータ・起動情報）が実在すること | **ELF（llvm-nm / llvm-readelf / objdump）** |
| **L3 実機実行検証済み** | 数値・レイアウト・性能 | 実機（**このリポジトリでは常に空**） |

**L2 が保証しないもの**（黙らない）: データレイアウト——どのレーンがどの要素を持つか。
行列コア命令（WMMA/MFMA）とスカラーレーンのレイアウト接合は実機照合が要り、
該当 op には `layout-unstitched` 注記が付く。**そして「ロードできる形」であることは
「走らせて正しい」ことを何ら意味しない**——L3 に到達しない限り数値は未検証のままである。

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

## 符号化照合 — 「受理された」と「意図どおり符号化された」は別

アセンブラが受理しても、書いた綴りが別名として別の命令に解釈されている可能性が残る。
そこで出来上がったオブジェクトを**第二のツールに読み直させる**（`verify_encoding`）。

| target | 方法 | 根拠 |
|---|---|---|
| amd_cdna / amd_rdna | `llvm-objdump -d` で逆アセンブル → 意図したニーモニックの実在を照合 | **往復検証** |
| nvidia | cubin の ELF シンボル（カーネル名の実在）＋ `ptxas -v` の資源レポート（レジスタ数・spill） | 往復では**ない**（SASS 逆アセンブラ nvdisasm は本環境で入手手段が無い。`method` フィールドがそう自己申告する） |

**この検査は実際に欠陥を見つけた**。RDNA3 は同じ機械語を別ニーモニックで綴る
（`global_load_dword` → `global_load_b32`、`s_load_dwordx4` → `s_load_b128`）。
llvm-mc は CDNA の綴りを**別名として黙って受理**しており、アセンブル成功だけでは
気づけなかった。対処は別名表を持つことではなく（それでは「自分の表を信じる」に逆戻り）
**arch ごとに正しい綴りを出す**こと。不変条件 93 が回帰を固定する。

```python
enc = codegen.verify_encoding(mod, target="amd_rdna")
print(enc.ok, enc.method, enc.missing, enc.symbols)
```

## ロード構造の検査 — 「アセンブルできる」と「ローダが受け付ける」も別

`.text` だけのオブジェクトには起動情報（kernarg サイズ・レジスタ数・ワークグループ
上限）が無く、ROCm ローダは拒否する。そこで HSA カーネル記述子（`.amdhsa_kernel`）と
AMDGPU メタデータノートも出し、**ELF から部品の実在を確かめる**（`verify_loadable`）。

| target | 確かめる部品 |
|---|---|
| amd_cdna / amd_rdna | `<kernel>`（.text の関数）・`<kernel>.kd`（記述子）・`NT_AMDGPU_METADATA` ノート |
| nvidia | `<kernel>` シンボル・`.nv.info.<kernel>`（起動パラメタ情報）。ptxas の cubin は元よりロード可能な形 |

記述子は飾りではない: llvm-mc が**内部整合を検査する**（`.amdhsa_accum_offset` が
VGPR 総数を超えれば error、レジスタ数が範囲外なら error）。レジスタ数は追跡変数でなく
**出力テキストそのものから数える**ので、記述子と本文が食い違わない。
一方でメタデータの `.symbol` と記述子シンボルの一致は llvm-mc が見ない（実測: 不一致
でも rc=0）ため、そこは ELF シンボル表で確かめる。不変条件 94/95。

**繰り返す**: これはロードして走らせた証明ではない。構造の検査に留まる（L3 は空のまま）。

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

## 独立オラクル — LLVM 自身の命令選択と突き合わせる

ここまでの検査（受理・符号化・ロード構造）は、どれも**私が書いた命令**を前提にしている。
命令選択そのものが妥当かは、ISA 文書の読み取り＝人手の判断に依存したままだった。

LLVM の AMDGPU / NVPTX バックエンドは同じ問題を解いている**独立した実装**である。
同じ演算を LLVM IR で書いて `llc` に落とさせ、Tsugi の選択と突き合わせる
（`reference_lowering` / `cross_check_lowering`）。LLVM は Tsugi を知らないし Tsugi の
表も参照しないので、この裏づけは循環しない。

**この検査は実際に欠陥を見つけた**。LLVM の NVPTX は `add.rn.f32` / `mul.rn.f32` と
丸めモードを**明示**する。修飾なしの `add.f32` / `mul.f32` は ptxas が `fma.rn.f32` へ
contraction しうる——積和が融合されると中間丸めが消えて数値が変わる。
**ビット等価を検証する道具が contraction 可能な形を出すのは自己矛盾**だった。
LLVM に倣って `.rn` 明示へ直した（不変条件 96）。

さらに、片側だけが精緻化列を要するかが `BIT_EXACT_ACROSS_VENDORS` の裏づけになる:

| op | NVIDIA（LLVM） | AMD（LLVM） | ビット同一 |
|---|---|---|---|
| add / sub / mul / max / cast | 単一命令 | 単一命令 | ✅ |
| **div** | `div.rn.f32` 単一 | `v_rcp` + `v_div_scale/fmas/fixup` **7 命令** | ❌ |
| **sqrt** | `sqrt.rn.f32` 単一 | `v_sqrt` + 比較・FMA の精緻化 **9 命令** | ❌ |
| **exp** | `ex2.approx.f32` 単一 | `v_exp` + 非正規域のスケーリング **5 命令** | ❌ |
| rsqrt | `rsqrt.approx.f32` | `v_rsq_f32` | ❌（**理由が違う**） |

NVIDIA が単一の正確丸め命令で済ませる演算を AMD が精緻化列で実装する——これは
「AMD の単独命令は正確丸めでない」という独立した証拠になる。`rsqrt` だけは例外で、
両社とも近似命令であり LLVM もその近似をそのまま使う。**対称だがビット同一ではない**
（近似の実装が違う）。不変条件 97 がこの対応と唯一の例外を固定する。

問えない op も黙らず持つ（`NO_LLVM_REFERENCE`）: `dot`（行列コア）と `reduce`
（クロスレーン）は単一の LLVM IR 演算に対応せず、`load`/`store`/`zeros` は命令選択の
論点が無い。

## アセンブラの入れ方（任意・本体の依存ではない）

無くても codegen は動く（L1-生成のみに落ちる）。L2 を得たいときだけ入れる。

| target | ツール | 入れ方 | ライセンス |
|---|---|---|---|
| nvidia | `ptxas` | `pip install nvidia-cuda-nvcc-cu12` | **NVIDIA CUDA EULA（proprietary）** |
| amd_cdna / amd_rdna | `llvm-mc` | `apt install llvm`（LLVM 同梱） | Apache-2.0 with LLVM exception |
| 独立オラクル（両者） | `llc` | `apt install llvm`（LLVM 同梱） | Apache-2.0 with LLVM exception |

`ptxas` は proprietary ゆえ **Apache-2.0 配布物の依存として宣言していない**
（`pyproject.toml` に入れない）。開発者が自分の判断で入れる任意ツールとして扱う。
場所は `TSUGI_PTXAS` / `TSUGI_LLVM_MC` / `TSUGI_LLC` 環境変数でも指定できる。

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

1. 実ランタイム（HIP/CUDA Driver API）で実際にモジュールをロードし起動する
2. レイアウト接合（フラグメント ↔ タイル軸）を実機出力で照合し `layout-unstitched` を潰す
3. リファレンス（CPU/NumPy）と数値照合し、`equivalence` の予測と突き合わせる

そこまで到達して初めて L3 を主張できる。それまで本層は L2 までしか言わない。

---

関連: [`python/tsugi/codegen.py`](../python/tsugi/codegen.py)（実装）・
[`tests/correctness/test_codegen.py`](../tests/correctness/test_codegen.py)（真値はベンダーのツール）・
不変条件 90-97（`verify.py`）・[SOURCES.md](SOURCES.md)（ISA 出典）

## 想定ユーザー（PyTorch 開発者）の経路

codegen は長らく **tile-DSL を書く人にしか届いていなかった**。このプロダクトの楔は
`torch.compile` であり、想定ユーザーは PyTorch 開発者である。彼らの経路は論理 op 列
（発散予測用）を作るだけで `ir.Module` を一度も作らず、「単一ソースで両ベンダー」という
看板の約束が看板の想定客に届いていなかった。`tsugi_torch.fxlower.fx_to_ir` がそこを繋ぐ。

```python
import torch, tsugi
gm = torch.fx.symbolic_trace(model)
ad = tsugi.verify(gm)        # 静的監査 + codegen（FX → IR → 実機械語）
print(ad.to_text())
```

### 降下の正直さ（黙って落とさない）

| 記録先 | 何を意味するか |
|---|---|
| `unsupported` | 表せない op。module は **partial** ＝生成物は**このモデルを計算しない**（判定に WARN）。理由を `[構造: サブテンソル選択/結合]` / `[未対応の演算]` で区別する |
| `decomposed` | 恒等式で分解した（実数では厳密でも浮動小数点では丸めが変わる＝発散源） |
| `shape_only` | view/permute 等。データ移動を**未モデル化**と言える |
| `assumptions` | 静的に決まらず仮定した値（eps 等）。黙って数値を選ばない |
| `bindings` | 各 `load` が何を読むか（`input:x` / `param:0.weight`）＝**生成カーネルの引数の意味** |

`chunk`/`getitem`/`cat` のような**構造 op** も「表せない」に倒す。値をそのまま通せば
q/k/v が同一値の別名になり、生成物はモデルをまったく計算しなくなるため
（実測: アテンションブロックで検出）。

これは**実装漏れではなく設計上の境界**である。本 IR は *数値データフロー* の IR で
あって *テンソルプログラム* の IR ではない。計算済みの値をスライスするには「どのレーンが
どの要素を持つか」を IR が知っている必要があり、それはまさに `layout-unstitched` として
実機照合に回している当のもの。実機なしにレイアウトを決め打てば、**アセンブルは通るのに
意味の違う生成物**ができる。表せないと言う方が正しい。

（グラフ入力そのものを分割する場合は `load` の offset で表せるが、実モデルでは分割対象は
`linear(...)` の出力＝計算済みの値であり、その経路には効かない。効かない機構を足すのは
複雑さを増やすだけなので入れていない。）

**erf 版 gelu は「表せない」**とする。tanh 近似での置換は実測で max|Δ|≈4.6e-4 ずれ、
等価性を検証する道具が検証対象を別の関数に差し替えるのは偽OK そのものだから。
`approximate='tanh'` と明示されているときだけ分解する。

### 意味論の照合（アセンブラにも LLVM にも問えない層）

降下が *意味* を保つかは、命令の存在でも符号化でもロード構造でもない。
`tsugi.interp.evaluate` が IR に CPU リファレンス意味論を与え、torch があれば
**eager と突き合わせる**。float64 での実測（`test_fxlower.py`）:

| 経路 | max\|Δ\| |
|---|---|
| Softmax | 2.8e-17 |
| Linear（重み・bias 込み） | 2.2e-16 |
| MLP / MLP+LN+SiLU | ≤ 3.3e-16 |
| MLP+GELU+Softmax | 5.1e-14 |
| LayerNorm | 2.2e-16 |
| RMSNorm | 4.4e-16 |
| GELU(tanh) | 9.1e-13 |
| SiLU / Tanh / Sigmoid / ReLU | ≤ 4.4e-16 |

この照合で 4 件の欠陥が出た:

| 欠陥 | ずれ |
|---|---|
| layer_norm の eps 欠落 | 1.9e-5 |
| erf 版 gelu の無断置換 | 4.6e-4 |
| `linear(x,W,b)` の転置落ち（`x·W` になっていた） | — |
| `call_function` 経路で bias が落ちる | 3.8e-1 |

**どれもアセンブルは通る**。意味論は別の層で見るしかない。

重みを含むモデルを照合できるのは、IR が `load` に**束縛記述子**を持つから
（`input:x` / `param:0.weight`）。これは生成カーネルの引数が何のテンソルかという情報
そのものでもある。束縛が足りなければ `evaluate` は**落ちる**（黙って別のテンソルを
使い回さない）。

### 実モデルでの実測（アテンションブロック）

```
FX 呼び出し 18 件中 13 を IR へ降下（グラフ全体 20 ノード）
表せない op ['getitem[構造: サブテンソル選択/結合]', 'chunk[構造: ...]'] → **partial**
  → 生成物は**このモデルを計算しない**（『大部分は動く』ではない）
  分解: layer_norm / softmax / gelu(tanh)
  形状のみ（データ移動は未モデル化）: ['transpose']
  nvidia 5928 B / amd_cdna 2200 B / amd_rdna 2072 B — 3 ターゲットともアセンブル検証済み
```

「13/18 を降下し 3/3 でアセンブル成功」を **成功と読ませない**のが要点。
降下できた命令列については検証が有効だが、モデル全体としては成立していない。

なお `interp` が示すのは *IR の意味* であって *生成した機械語の意味* ではない。
レーン割当・行列コアのレイアウト・実機の丸めはここに現れない（L3・要実機）。
