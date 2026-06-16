# ソクラテス式問答 50 回 — 改善点の洗い出し

現コードベース（python/tsugi 2,512 行・95 テスト・verify 38 不変条件・8 視点＋audit）に
対し、前提を 50 問で問い直し改善点を洗い出す。各問は実コードの観察に基づく。
末尾に優先度（P0 即修正 / P1 近く / P2 後）で総括。

---

## A. safety 係数とマジックナンバー（Q1–6）

**Q1.** tolerance の `safety=4.0` は全許容を決める最重要定数だが、根拠は？
→ コメントは「モデルの粗さを吸収」のみ。出典も導出も無い。**改善: safety の選定根拠
（√K ランダムウォーク仮定 + 何σ をカバーするか）を SOURCES.md に明記し、実機 noise で校正。**

**Q2.** safety は 5 箇所に重複（tolerance/calibration/propagation 等）。一元化されているか？
→ 各所で既定 4.0 を独立に持つ。**改善: `tsugi.constants` に SAFETY を集約し単一情報源化。**

**Q3.** safety=4.0 は fp16/bf16/fp32 で同じでよいか？
→ dtype で丸めの統計的性質は変わりうる。**改善: dtype 別 safety の妥当性を検討（少なくとも
「同一でよい」根拠を記す）。**

**Q4.** 検出限界 `0.1*max_normal`（overflow 近接 WARN の 10%）の 0.1 は？
→ envelope の閾値。任意。**改善: 閾値群（0.1・0.7・1.5・0.5*thresh 等）を名前付き定数化。**

**Q5.** decision の `confident_k`/裾判定 `0.5 * overall_margin_median` の 0.5 は？
→ 「near-tie 裾に集中していない」判定の閾値が任意。**改善: 根拠か感度分析を添える。**

**Q6.** これらの定数群はテストで固定されているが、値を変えた時の挙動は誰が守る？
→ 値変更で多数テストが沈黙して壊れうる。**改善: 定数の感度テスト（境界±で判定が反転する
ことの確認）。**

## B. 静的な条件数の盲目（propagation）（Q7–12）

**Q7.** propagation の `cond` 既定 1.0 は何を意味するか？
→ 全 op を well-conditioned と仮定。**改善: これは propagation が存在する理由（ill-conditioned
増幅の検出）を既定で無効化している。最低限ドキュメントで強く警告し、cond 推定の道筋を示す。**

**Q8.** audit の `_graph_ops` は実カーネルから cond をどう得るか？
→ 得ない（常に 1）。✅ **修正済**: (1) `_AMPLIFYING` を *相対*増幅する op のみ（reduce/
softmax/exp）に是正（div/reciprocal/add は相対条件数 ~1 で増幅しないと実測確認）。
(2) `empirical_cond(sample,kind)` でデータ依存 cond を実測（reduce=Σ|x|/|Σx| の相殺・
exp=max|x|）。(3) audit は増幅 op があるのに静的 cond=1 を当てる時 *下界* と WARN し
empirical_cond/audit_runtime を案内（過小評価を隠さない）。

**Q9.** 現トレーサは reduce/exp/softmax を IR に出すか？
→ 出さない（dot/load/store/cast/add/zeros のみ）。よって audit の propagation は実質常に
単一 matmul。**改善: 増幅 op を IR に表現できないと perspective4 が空回り。tracer の op 語彙拡張。**

**Q10.** つまり「発散が深さで ~2000倍」の実証は audit 経路では再現されないのでは？
→ その通り。standalone propagate では出るが、audit に流れる実グラフでは出ない。
**改善: 多 op を出すカーネル例（rmsnorm/attention）を trace して audit に通す統合テスト。**

**Q11.** model_divergence は相対だが、各 op の scale 変化（正規化層）を追えているか？
→ 追っていない（amp≈1 と単純化）。**改善: scale を伝播する版（正規化で発散がリセットされる
効果）を検討。**

**Q12.** propagation は線形 op 列のみ。分岐・残差接続（transformer）は？
→ 非対応。残差は発散を加算的に運ぶ。**改善: DAG（残差/分岐）対応を将来課題として明記。**

## C. scale=1.0 という暗黙仮定（Q13–17）

**Q13.** certify_gemm/derive_tolerance/detectability_floor の `scale=1.0` 既定は現実的か？
→ 実テンソルの RMS は 1 でない。audit も `certify_gemm(K,"float16",1.0)` と固定。
**改善: audit が代表 scale を（与えられれば）使う・無ければ「scale=1 仮定」と明示。**

**Q14.** envelope は check 時に RMS scale を測るが certify は引数 scale。整合は誰が保証？
→ 呼び出し側任せ。ズレると認証 atol が無意味。**改善: certify に代表サンプルを渡して scale を
測る補助を用意。**

**Q15.** propagation→decision 橋は `scale=RMS(logits)`。logit scale と GEMM 出力 scale は同一か？
→ 別物（logit は最終層出力）。相対発散を logit に適用する近似の妥当域を要明記。
**改善: 橋の仮定（相対発散が最終 logit にそのまま乗る）を明文化し限界を書く。**

**Q16.** bf16 と fp16 で scale の効き（denormal 域）が違うのに一律 RMS でよいか？
→ envelope は dtype 別だが scale 推定は一律。**改善: scale 推定に dtype 別の下限（denormal）考慮。**

**Q17.** 全 scale 既定 1.0 は「とりあえず動く」値。テストも scale~1 のデータばかりでは？
→ そう。**改善: scale≫1 / scale≪1 のテストを追加し、tolerance/envelope の追従を確認。**

## D. 橋の分布仮定（propagation→decision）（Q18–22）

**Q18.** flip_bound は P(margin<2δ)。これは δ が「どの向きにも等確率」を暗黙仮定。系統発散では？
→ calibration が示した通り、実発散は系統的（相関）でありうる。系統バイアスは argmax を
一方向に押す → 上界が緩すぎ/きつすぎになりうる。**改善: 系統成分（バイアス）と乱雑成分を
分け、系統分は全 logit を平行移動（argmax 不変）、乱雑分のみ flip に効く、と精密化。**

**Q19.** δ_abs = δ_rel·RMS は最悪/平均どちら？
→ 平均的スケール。最悪サンプル（小 logit）では相対誤差が増幅。**改善: per-sample の δ を使う
（一律 RMS でなく |logit| 依存）版を検討。**

**Q20.** 「margin<2δ が必要」は十分でない（向きが要る）。上界は実測の何倍緩い？
→ 実測 5.8% vs 上界 18.7%（約 3.2 倍）。**改善: 期待値版（向きを確率で割引）を併記し、上界と
推定値の両方を返す。**

**Q21.** 代表 logit 分布はどこから来る？本番分布と違えば予測は外れる。
→ 呼び出し側が渡す前提。**改善: 「代表 logit はキャリブレーション集合から」とガイドし、
分布シフト時は再評価が要ると明記。**

**Q22.** top-1 argmax だけ見るが、top-k / beam / sampling は？
→ 非対応。**改善: top-k 一致・分布距離（KL）など task 多様性への拡張余地を明記。**

## E. torch.compile の楔が検証を届けていない（Q23–28）

**Q23.** 製品の入口は `torch.compile(model, backend="tsugi")`。今そこで検証は走るか？
→ 走らない。backend は eager 素通し。**改善: codegen 前でも、FX グラフに対し portability/
propagation 静的監査を走らせ警告する（検証だけ先に届ける）。これが楔の早期価値。**

**Q24.** backend は torch 無し環境で全くテストされていない。回帰は誰が捕まえる？
→ 誰も。**改善: torch を test extra に入れ、最小 FX グラフで backend 登録と素通しを検証。**

**Q25.** backend が eager 素通しなのにユーザーは「両ベンダー対応」と思い込まないか？
→ README は明記するがコードは沈黙。**改善: backend が一度 warn ログ（「verification-only / 
no codegen yet」）を出す。**

**Q26.** FX グラフ → `_graph_ops` の橋は無い。audit は traced tile IR 専用。二重路線では？
→ そう。tile IR と FX の二系統。**改善: FX→propagation GraphOp の写像を作れば backend で
audit が即使える（B9 の op 語彙とも合流）。**

**Q27.** escape-hatch（cuBLAS/rocBLAS 委譲）時、その区間の等価性は誰が保証？
→ 委譲先はベンダー実装で発散源そのもの。**改善: escape-hatch 区間こそ audit_runtime の対象、
と設計に明記。**

**Q28.** backend 登録は import 副作用（自動 register）。テスト分離・冪等性は？
→ 二重 import で二重登録の可能性。**改善: 登録の冪等化（既登録チェック）。**

## F. タスクモデルの狭さ（decision）（Q29–32）

**Q29.** decision は分類 argmax 前提。回帰・生成・検出は？
→ argmax 固定。**改善: task 種別を抽象化（分類/回帰しきい値/生成トークン）。**

**Q30.** margin = top1−top2 は多クラス前提。2 値 sigmoid（しきい値 0.5）では？
→ 別定義（|logit−threshold|）。**改善: 2 値タスクのマージン定義を追加。**

**Q31.** flip は「正しさ」でなく「一致」を測る。両ベンダーとも同じく誤るケースは？
→ flip=0 でも両方間違いはある。**改善: oracle がある検証集合では accuracy 差も併記。**

**Q32.** 温度・top-p サンプリング下では同 logit でも出力トークンが確率的に違う。
→ 非対応。**改善: サンプリング下の「分布一致」（決定論的 argmax でなく）を将来課題に。**

## G. テスト/verify の構造（Q33–38）

**Q33.** verify.py の不変条件は test_*.py と大きく重複。二重保守では？
→ 重複多数。**改善: verify は「機械可読な主張のサマリ」と位置づけ、重複を意図的と明記するか、
test から不変条件を生成する。**

**Q34.** 単一巨大 main() の verify は失敗箇所の局所化が弱い。
→ 1 関数に全 check。**改善: セクション分割（既に番号付きだが関数化）。**

**Q35.** テストは固定 seed の単発。数値主張なのに fuzz/property test が無い。
→ ✅ **修正済**: test_properties.py に 10 性質 × 200 試行のゼロ依存 property 検査を追加
（derive_tolerance の K 単調性・residual≤total・flip_rate スケール不変・残差 bound は上界・
アフィン系統は無フリップ・attribute 領域・envelope overflow 等）。hypothesis 不使用（ゼロ依存）。

**Q36.** calibration corpus は 9 ケースの合成。「TRUSTWORTHY=偽OK 0」は統計的に弱い。
→ n が小さい。**改善: ケース数を増やし、バグ強度を連続掃引して ROC を描く。**

**Q37.** GPU 経路は全 SKIP。SKIP が緑に紛れ「検証済み」と誤読されないか？
→ run.py は明記するが集計は緑。**改善: SKIP 件数をサマリに出す（"X passed, Y skipped"）。**

**Q38.** カバレッジ計測が無い。147 関数中どれが未実行？
→ 不明。**改善: coverage を CI で測り閾値ゲート。**

## H. 再現性・依存・CI（Q39–43）

**Q39.** numpy 版が未固定。noise_floor/RNG 挙動は版で変わりうる。
→ pyproject に numpy 制約なし。**改善: numpy 下限固定・RNG は default_rng で版間安定だが明記。**

**Q40.** version は全変更を通じ 0.1.0 のまま。SemVer 運用は？
→ 据え置き。**改善: 0.x で API 変化を MINOR/PATCH に反映、CHANGELOG に版見出し。**

**Q41.** .github/workflows は除外で CI が実際には回らない。「CI」主張は願望では？
→ そう（権限制約で除外）。**改善: CI が無効である事実を CONTRIBUTING に明記、ローカル
verify.py を「CI 代替」と位置づけ。**

**Q42.** ライセンス/依存監査は手動。permissive 主張の自動チェックは？
→ 無い。**改善: 依存ライセンスの自動検査（pip-licenses 等）を verify に追加可能。**

**Q43.** 乱数依存テストの flake 耐性は？seed 固定だが境界ケースは脆い。
→ 一部境界（test_single_run_flaky は midpoint で堅牢化済）。**改善: 全乱数テストの境界余裕を点検。**

## I. API/意味の一貫性（Q44–47）

**Q44.** equivalence.EquivalenceReport は report.FindingReport を継承していない（統合の取り残し）。
→ ✅ **修正済**: 等価判定はスカラ計量で所見リスト型でないため FindingReport は継承せず（理由を
docstring に明記）、共通の判定インターフェース `risk`/`max_risk`/`ok` を備えて第一級レポート化。

**Q45.** AuditPhase.when は "static"/"runtime" だが audit_runtime は実データ層を "static" と付ける
（意味の二重定義）。
→ 「判定に算入=static」と運用しているが名前と乖離。**改善: when を "decided"/"pending" に
改名し意味を一致させる。**

**Q46.** 多くの関数が関数内 import（遅延）。一貫していない（top と function 内が混在）。
→ 循環回避のためだが基準が不明瞭。**改善: 遅延 import の方針を 1 行で明文化。**

**Q47.** report の severity は Risk(IntEnum) だが equivalence は bool(equivalent)。粒度不一致。
→ ✅ **修正済**: EquivalenceReport.risk が equivalent→OK / divergent→BLOCK を返す。全レポートが
`max_risk` を持つ統一インターフェースに。

## J. 主張の統計的厳密さ（Q48–50）

**Q48.** 「per-model 発散 ~2000倍」等の数字は単一 seed・単一構成。一般性は？
→ 例示値。**改善: 複数 seed の分布（中央値±）で示し、example でなく統計と明記。**

**Q49.** noise_floor は spread(max-min)。外れ値 1 つで過大評価では？
→ max-min は外れ値敏感。**改善: ロバスト版（percentile 幅）も併記し選択可能に。**

**Q50.** これら検証層は「実機で正しい」ことをまだ一度も確認していない。CPU 擬似のみ。
→ 最大の未検証。**改善: 実機での最小 e2e（1 カーネルを両ベンダーで走らせ audit_cross_vendor）
を最優先の次マイルストーンに据える（lowering.py 実装が前提）。**

---

## 総括（優先度）

**P0（即修正・低リスク・高価値）**
- ✅ **Q9/Q10（本パスで修正済）**: tracer が reduce/exp/sqrt/rsqrt/maximum＋elementwise
  （sub/mul/div）を IR に記録するようにした。softmax がトレース可能になり、増幅 op が IR と
  audit の propagation グラフに現れる（perspective4 の実効化）。残: Q8（cond 推定で増幅の
  *大きさ* を出す）は P1。
- Q2/Q1: safety 等の定数を `tsugi.constants` に集約＋根拠を SOURCES に明記。
- Q45: AuditPhase.when を decided/pending に改名（意味の一致）。
- Q37: テストサマリに SKIP 件数を表示（緑の誤読防止）。

**P1（近く）**
- ✅ Q8（修正済）: 相対増幅 op の是正＋ empirical_cond（data-driven）＋ 静的下界の WARN。
- Q23/Q26: torch backend に静的 audit を差し込み「検証だけ先に届ける」。FX→GraphOp 写像。
- Q18/Q20: 橋を系統/乱雑成分に分け、上界＋期待値の両方を返す。
- ✅ Q44/Q47（修正済）: equivalence に共通 risk/max_risk/ok インターフェースを付与。
- ✅ Q35（修正済）: property test 10×200 追加。残 Q36: calibration corpus を ROC へ拡充。

**P2（後・要設計/実機）**
- Q12/Q11: propagation の DAG・scale 伝播対応。
- Q29–32: decision の task 多様性（回帰/生成/サンプリング）。
- Q50: 実機 e2e（最重要だが GPU 必須）。

最大の構造的発見: **B 群（Q9/Q10）— audit に流れる実グラフは単一 matmul なので、目玉の
propagation（per-kernel⇏per-model 増幅）が統合経路では発火していなかった。** さらに調べると
tracer が reduce/exp 等を IR に *全く記録していなかった*（softmax/rmsnorm はトレース不能）
ことが根因。**本パスで tracer を拡張して修正済**（softmax がトレースでき、増幅 op が audit に
流れる）。増幅の *大きさ*（cond 推定・Q8）が次の最優先。
