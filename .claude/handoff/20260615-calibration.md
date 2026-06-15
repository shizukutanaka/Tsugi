[SESSION HANDOFF - 2026-06-15 #8 新視点6 検証器の較正]

== ソクラテス問答で発見した新視点 ==
既存8層は全て「カーネル/モデルを検証する」が、検証器自身は未検証だった盲点。
検証器の最大の罪 = 偽OK（発散を等価と誤判定）→ オラクル無きもう片方のベンダーに
silent な誤りを出荷 = 検出不能で致命。偽BLOCK（回復可能）とコスト非対称 →
不確実なら BLOCK 寄りに倒すべき（fail-safe）。
許容ベース等価判定には検出限界 floor = safety·√K·u（相対）があり、これ未満の
系統バグは原理的に不可視。floor は K で √K 拡大（fp16: 256→3.1%/2048→8.8%/
8192→17.7%）= 視点2（導出許容）が大K偽BLOCKを消した双対コストとして偽OK盲点を拡大。
救済 = scale/K 不変な相補計量 RMS比（乱雑発散は zero-mean / 系統バグは相関）。

== 実装済み（CPU検証済み・GPU不要） ==
- tsugi/calibration.py: 検証器を検証するメタ層
  - detectability_floor(K,dtype,scale,safety): 見逃す誤差の下限 {abs,rel}・rel=safety√K·u
  - systematic_divergence(a,b): RMS比-1（scale/K不変な系統誤差指標）
  - check_systematic(a,b,K,dtype): 閾値 safety·u（K不変）超で BLOCK（fail-safe）。
    CalibrationReport は report.FindingReport を継承（前回の統合と整合）
  - is_equivalent_combined(a,b,K,dtype): max_abs(compare_gemm) + 系統 の合成・どちらか
    発散で DIVERGENT
  - Case/Confusion/make_corpus/evaluate: ground-truthコーパスで偽OK率を採点
    Confusion.trustworthy = (false_ok==0)・非対称ゆえ偽OKのみを基準
- 実証: 0.5%系統スケール誤差を max_abs単独は全Kで見逃す（偽OK 3/6=UNTRUSTWORTHY）が
  合成判定は 0/6（TRUSTWORTHY）。RMS比は全Kで0.5%を捉える
- コーパスは「等価」と「不可分なバグ(scale/dropblock)」のみ。fp16accum(lossyだが正当)は
  trustworthy主張を曖昧にするため意図的に除外
- 統合: verify.py に不変条件3件（22→25）・portcheck に検出限界セクション・__init__ 公開
- test_calibration.py 6件。計69テストPASS / verify25 / ruff clean

== 検証の全体像（6視点 + 1メタ） ==
静的: feasibility(動くか)→propagation(モデルで一致)→occupancy(速いか)
動的: envelope(本番入力が認証前提内か)
メタ: calibration(検証器自身は信頼できるか=偽OK率) ← New

== 次アクション（保留） ==
- calibration を equivalence に織り込む: compare_gemm をデフォルトで合成判定にするか検討
  （後方互換に注意・現状は別関数 is_equivalent_combined で非破壊）
- noise_floor 実測・GPU 実機で floor の妥当性を検証
- 統合ファサード tsugi.audit（前回からの保留・8層+メタを1レポートに）
