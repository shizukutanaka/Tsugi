[SESSION HANDOFF - 2026-06-15 #9 新視点7 非決定実行]

== ソクラテス問答で発見した新視点 ==
既存6視点+較正メタ層は全て「同じカーネルを同じベンダーで走らせれば同じ結果」=
決定論を暗黙仮定していた盲点。GPUのatomic加算(split-K atomicAdd)はスレッド到着順で
和の順序が変わりrun-to-runで揺れる→ベンダー出力は点でなく分布。
単一run A vs 単一run Bは(a)ベンダー内ノイズと(b)ベンダー間発散を混同→食い違いを
attribute不能・「Aの正しい答え」が一意に定義できない。クロス差がノイズ未満なら
「AvsB」は「AvsA(別run)」と区別不能=第三状態INDISTINGUISHABLE(判定原理的に未定義)。
視点6の検出限界(数値の床)とは独立した第二の床(HWの床)。
実効分解能=max(数値検出限界,ノイズフロア)。tolerance.pyのnoise_floor既定0は決定論仮定の化石。

== 実装済み(CPU検証済み・GPU不要) ==
- tsugi/nondeterminism.py:
  - simulate_nondeterministic_reduction(parts,seed): atomicAdd非決定の擬似(ランダム順fp32累積・明示)
  - measure_noise_floor(run_fn,n_runs): 複数runでrun-to-runノイズ実測{spread,std,rel}
  - attribute(cross,noise,tol): 3状態帰属 [0,noise]INDIST/(noise,tol]EQ/(tol,∞)DV
  - compare_stable(run_a,run_b,K,dtype,n_runs): 分布として健全比較・noise織込みtol導出
  - StabilityReport(FindingReport継承)・noise_limited判定・EQUIVALENT/DIVERGENT/INDISTINGUISHABLE
- 実証: 真に等価な2ベンダーの単一run比較が観測差中間の許容でEQ/DVにフレーク。
  クロス差7.6e-5<ノイズ3.4e-4でINDISTINGUISHABLE。真の発散(5%底上げ)はノイズ超でDIVERGENT。
- 統合: verify不変条件3件(25→28)・portcheckに非決定性セクション・__init__公開
- test_nondeterminism.py 6件。計75テストPASS / verify28 / ruff clean

== 検証の全体像(7視点 + 1メタ) ==
静的: feasibility(動くか)→propagation(モデルで一致)→occupancy(速いか)
動的: envelope(本番入力が認証前提内か)
メタ: calibration(検証器自身は信頼できるか=偽OK率)
基盤: nondeterminism(そもそも出力は分布・比較の床はノイズ) ← New
  検証器の実効分解能 = max(数値検出限界[視点6], ノイズフロア[視点7])

== 次アクション(保留) ==
- compare_stable を calibration/equivalence に織り込む(ノイズ込みの偽OK率測定)
- run_fn を実GPUカーネルにしてnoise_floor実測→tolerance/calibrationへ供給(要実機)
- 統合ファサード tsugi.audit(前回からの保留・7層+メタを1レポートに)
- noise下の系統バイアス検出(視点6のRMS比を複数run平均で・ノイズ床下の系統バグ発掘)
