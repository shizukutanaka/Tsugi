# 視点の補完 5: 数値エンベロープの実行時検査（ソクラテス問答・第5ラウンド）

> 2026-06-15。これまでの 4 視点すべてに共通する前提を問うて発見した盲点。
> すべて *デプロイ前の静的検証* で、本番の oracle 無し環境を見ていなかった。

---

## 問答の記録

**Q1（前提検証）**: portability・tolerance・feasibility・propagation はすべて
*デプロイ前*に動く。等価性は「scale はこれ・条件数はこれ・K はこれ」という前提の下で
*認証* される（tolerance は scale を仮定し derive、propagation は cond を仮定する）。
だが本番では NVIDIA も oracle も第2ベンダーも存在しない。AMD 単体で *未知の入力* が流れる。
→ デプロイ前検証は本番の正しさを保証するか?

**Q2（帰結）**: → **否。** propagation で見た通り発散は *データ依存*（条件数・スケール次第）。
認証は scale=1・well-conditioned を暗黙仮定したが、本番入力はその *エンベロープ
（認証済み動作範囲）を逸脱* しうる。逸脱すると静的保証は *静かに* 無効化される:
- **scale 逸脱**: atol は scale に比例（tolerance）。本番スケールが認証 scale_max の
  50 倍なら、保証した atol も 50 倍に膨らむ — もはや「等価」を保証していない。
- **dtype 範囲逸脱**: fp16 は max=65504 と狭い。活性値がこれを超えれば overflow→inf。
- **exp-overflow**: fp16 softmax で生 logit が ln(65504)≈11.09 を超えると exp が inf。
  片ベンダーが fp16 で exp を計算するとここで破綻し、f32 経路の他方と壊滅的に発散。
- **denormal**: 最小正規数（fp16 で 6.1e-5）未満は denormal。FTZ（flush-to-zero）の
  有無がベンダーで異なり、静かな発散源になる。

**Q3（メタ）**: 静的「証明」を一度きりで信じるのが誤り。
**認証済みエンベロープ + その前提が成り立つかを実行時に検査する契約**
（design-by-contract を数値に適用）にすべき。しかも oracle も第2ベンダーも要らない、
*単一ベンダーで計算できる安価な数値ヘルスチェック*。等価性が認証された「前提」を、
本番の各推論で安く検算する。

---

## 新視点（採用）

**等価性は前提付きで認証される。前提が本番入力で成り立つかを実行時に検査せよ。**
`tsugi.envelope`:
- `certify_gemm(K, dtype, scale, cond)` → `Envelope`（tolerance と同じ前提で発行・
  認証 atol を記録）。静的層と接続。
- `check_tensor(x, env)` → 単一ベンダー・oracle 不要で本番テンソルを検査:
  NaN/Inf・dtype overflow・denormal(FTZ)・**scale 逸脱（認証 atol の無効化）**。
- `check_softmax_input(logits, env)` → exp-overflow 閾値（fp16 で 11.09）超過を検出。
- 深刻度は portability.Risk に統一（OK/INFO/WARN/BLOCK）。

実証（IEEE 754 実値・実挙動確認済み）:

| dtype | max | 最小正規数 | exp-overflow 閾値 |
|-------|-----|-----------|-------------------|
| float16 | 6.55e4 | 6.10e-5 | \|x\|>11.09 |
| bfloat16 | 3.39e38 | 1.18e-38 | \|x\|>88.72 |
| float32 | 3.40e38 | 1.18e-38 | \|x\|>88.72 |

- `np.exp(np.float16(12.5))` は **実際に inf**（12.5>11.09）。検査は机上でなく実挙動。
- 同じ logit でも bf16/f32 は範囲が広く OK — **fp16 の主リスクは overflow、bf16 の
  主リスクは precision/denormal** という dtype 依存の差を可視化。
- scale=1 で認証した atol は、本番スケール 50 でそのまま無効（実許容は ~50 倍）→ 要再認証。

---

## なぜこれが正しい補完か（既存視点との接続）

- **検証の時間軸を拡張**: 1〜4 は *静的・デプロイ前*。本視点は *動的・本番*。
  「一度証明して信じる」を「証明 + 実行時に前提を検算する契約」へ。
- **tolerance/propagation の前提を顕在化**: それらが暗黙に置いた scale・cond の仮定を
  Envelope として明示し、逸脱を検出可能にする。静的層の「但し書き」を実行可能にした。
- **oracle 問題を回避**: 本番には真値も第2ベンダーも無い。本視点は *単一ベンダーの
  統計だけ* で危険を捕まえる（NaN/overflow/denormal/scale/logit）。安価で常時実行可能。
- **誠実**: エンベロープ内なら静的保証が有効、逸脱したら「保証対象外・要再認証」と
  正直に告げる。検査は十分条件の警告であり、ベンダー一致の証明ではない（明記）。

---

## 戦略への影響

- portcheck が静的レポートに **認証エンベロープ（保証が有効な前提）** を併記。
  「この移植性判定は scale ≤ X・|logit| ≤ Y の範囲で有効」と但し書きが実行可能に。
- 製品面: `torch.compile(model, backend="tsugi")` の推論に envelope guard を差し込めば、
  本番でエンベロープ逸脱（= 発散リスク高）を *第2ベンダー無しで* 警告できる。
- 実 GPU フェーズ: 各ベンダーの FTZ/denormal・丸めモードを実測し、DtypeLimits と
  envelope ルールを実機挙動に合わせる。

却下/保留した代替視点:
- 「本番でも両ベンダー走らせて突き合わせる」: コスト 2 倍・実運用非現実的。却下。
  本視点の単一ベンダー検査が安価な代替。
- 「全入力を f32 にする」: 性能・メモリを捨てる。エンベロープ検査で必要時のみ昇格が筋。保留。

## 追補（ソクラテス問答続行）— outlier feature と単一スケール仮定の破綻

tolerance/envelope/detectability floor は単一の scale（global RMS）を仮定する。だが実 LLM
活性は **outlier feature** を持つ —— 一部チャネルが 100–1000x 大きい（massive activations・
LLM.int8 / SmoothQuant が示す通り）。すると global scale は outlier チャネルを過小評価し、
そこでの発散が誤った許容で判定される。検証は標準正規データで較正されてきたが、本番は
そうでない。`envelope.check_outlier_features` がチャネル scale 広がり（max/median channel-RMS）で
これを検出し WARN（per-channel 許容/検証が必要）。実測: N(0,1) は広がり ~1、outlier feature
（数チャネル 200x）は ~208。これも「検証の前提（入力分布）が本番と一致するか」を正直にする一歩。
