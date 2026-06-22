# 新視点9: 自己回帰的発散 — per-token 等価 ⇏ per-sequence 等価

ソクラテス式問答の続き。`tsugi.rollout` として実装。

## 問答

**Q1.** decision（新視点8）は *1 トークンの* 判断フリップ率 p を測る。出荷される LLM は
何をするか？ → *自己回帰* 生成。トークン t+1 は t に条件づく。

**Q2.** ならば 2 ベンダーが各ステップで確率 (1−p) で一致すれば、*生成シーケンス* も
等価か？ → 否。あるトークンで判断が分かれると以降の文脈が分岐し、後続は別軌道
（無相関）になる。一度ズレたら戻らない。

**Q3.** ならば意味ある量は何か？ → per-token フリップ率でなく **シーケンス survival**：
長さ L の生成が一致する確率 = (1−p)^L。初回発散ステップの期待値 = 1/p（幾何分布）。

**Q4.** per-token で「許容」な p はシーケンスでどうなるか？ → 複利的に破綻する。
p=1% は 1 トークンでは 99% 一致だが、L=100 で survival=37%、L=1000 で 0.4%。
**per-token の許容判断は生成では破綻する。**

**Q5.** これは既存のどの視点に似るか？ → propagation（新視点4: per-kernel ⇏ per-model）の
自己回帰版。propagation は発散を op グラフの *深さ* に沿って合成した。rollout は判断
フリップ risk を生成 *長* に沿って合成する。どちらも「局所の等価は大域に合成されない」。

## 実装

- `sequence_survival(p, length)`: (1−p)^length。シーケンス完全一致の確率。
- `expected_divergence_step(p)`: 1/p。初回発散トークンの期待位置。
- `safe_generation_length(p, confidence)`: survival ≥ confidence を保てる最大長。
- `divergence_step_quantile(p, q)`: 初回発散位置の q 分位（中央値など）。
- `analyze_rollout(p, target_length, *, confidence)`: 生成長へ合成して verdict
  （safe_len 内=OK / survival≥0.5=WARN / それ未満=BLOCK）。
- `rollout_from_logits(a, b, target_length, *, conservative=True)`: 代表 logit から p を測り
  合成。既定で `flip_rate_upper_bound` を使い小標本での過信を防ぐ（下記 fail-safe）。
- `flip_rate_upper_bound(flips, n, confidence)`: 観測フリップ数からの p の片側上側信頼限界
  （Wilson）。0/n でも p≲3/n（rule of three）を計上する。
- `simulate_rollout(p, length, trials)`: Monte Carlo で survival を実測し解析式を確認。

```
analyze_rollout(0.01, L):
  L=1    : survival=99.00%  → OK
  L=10   : survival=90.44%  → WARN
  L=100  : survival=36.60%  → BLOCK（初回発散 ~tok 100）
  L=1000 : survival= 0.00%  → BLOCK
```

## 含意

- 検証の単位を「1 回の forward」から「生成 1 本」へ引き上げる。decision の per-token
  フリップ率だけ見て出荷すると、長文生成・エージェント連鎖で別物の出力になりうる。
- safe_generation_length は運用ガイドになる：「このベンダー対では N トークンまで同一
  生成が confidence で保証される」。N が目標生成長より短ければ移植は task 的に未達。
- audit_runtime に `gen_length` を渡すと rollout 層が verdict に算入される（facade 統合）。

## 限界（正直に）

- **fail-safe な p 推定**: 観測フリップ率の点推定は小標本で過小評価し、0 フリップ観測でも
  p=0 ではない。移植可を過信するのは calibration（新視点6）の偽OK と同じ致命傷ゆえ、
  `rollout_from_logits` は既定で上側信頼限界 p を使う（過信より過検出に倒す）。
- **survival は完全トークン一致の確率**＝厳格な下界。意味的に等価な別文（同義語・語順）も
  「発散」に数えるので、task 等価より厳しい。意味等価まで測るには別途タスクモデルが要る。
- **フリップ率の定常性を仮定**（位置非依存）。実際は near-tie の多寡で位置依存しうる。

これでセッションの検証連鎖に *生成長* の次元が加わる: 数値（equivalence）→ タスク
（decision）→ シーケンス（rollout）と、ユーザーに見える単位へ段階的に引き上がる。
