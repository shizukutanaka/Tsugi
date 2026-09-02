# Opus 作業指示書 — Tsugi 設計判断つき中規模ラウンド

## あなたの役割
スコープは定まっているが**設計判断を含む**ラウンドを担当する（Sonnet では不安・
Fable ほどの重さは不要な中間層）。会話履歴に依存せず、台帳だけで作業を開始できる。

## 起動時に読む 3 ファイル（この順）
1. `docs/FEATURE-AUDIT.md` — 機能過不足の台帳（現在地と残作業の単一情報源）
2. `docs/ASSESSMENT.md` — 長所短所改善案（優先度と担当の割当・あなたの担当は P1）
3. `CONTRIBUTING.md` — 検証ゲート・コミット規約・Import 方針

## 担当バックログ（ASSESSMENT の P1）

### 1. A-3: torch.compile 経路の段階接続
- **現状**: `_tsugi_compile()`（`python/tsugi_torch/__init__.py`）は
  `fxbridge.audit_fx`（静的監査）のみ。worstcase/attribution/タスク別 decision が届かない。
- **やること**: example_inputs から代表テンソル/logits を best-effort で取り出し、
  `audit()` の `sample=` / `ref_logits=` 相当の情報を警告メッセージに追加する段階的接続。
- **受け入れ基準**: torch 無し環境の duck-typed FX スタンドインでテスト
  （`tests/correctness/test_tsugi_torch_compile.py` の既存パターンを踏襲）。
  実 torch.fx 結線は「未検証」と明記。フル接続は実行時出力が要るため codegen 後。

### 2. A-12 残: 一般 series-parallel fork（3 分岐以上）
- **現状**: `_computed_fork_merge`（`python/tsugi/audit.py`）は消費者ちょうど 2 の
  場合のみフォークを検出（`len(uses) != 2` は線形＝保守側にフォールバック）。
- **やること**: `len(uses) >= 3` で各消費者から単一消費の一本鎖を辿り、全鎖が同一の
  合流 op に到達し、合流 op の operands が全末端ちょうどのときだけ `[[A],[B],[C],...]`
  を出す。それ以外は線形。`propagate_dag` は N 分岐に対応済み（`merge_divergence`）。
- **受け入れ基準**: 3 分岐カーネルのテスト＋「検証できない形は線形に落ちる」テストの
  両方。`propagate_dag(correlated=True)` の発散が線形版を下回らないことを固定。

### 3. ✅ A-2: 実機 GPU ハーネス実行計画（完了）
- `docs/GPU-BRINGUP.md`（4 フェーズの手順書・Phase 1/2 は GPU 1 台で完結）と
  `calibration.calibrate_safety`（実機入口 `audit_cross_vendor` に接続済み）で完了。
- 残るのは実機での実行そのもの（人間・要 GPU）。結果は同文書の記録テンプレートへ。

### 4. ✅ A-5: 正規化層の相対発散（完了・3a29b94）——**前提が実験で反転した先例**
- この指示書は当初「LN のヤコビアンは 2 方向を射影落とすので増幅は理論上 ≤1 になるはず。
  これを数値実験で確かめる」と書いていた。**射影は正しかったが結論は誤りだった**:
  残る最大特異値は `g/√(σ²+eps)` で、相対量に直すと上界は
  `RMS(x)/√(σ²+eps) = 1/√(1−(μ/RMS)²) ≥ 1` となり **1 を下回れない**。
  実測でも shift=10 で amp=10.10（零平均では 1.00）。
- **教訓**: 「scale-invariant だから安全」は *絶対* スケールの話で、*相対* 発散の増幅とは
  別物。ガードレール 2（未検証の数値係数を導入しない）が **符号すら逆の減衰係数**の
  混入を防いだ。もし「≤1 のはず」を信じて減衰を入れていれば、平均優勢入力で発散を
  過小評価する偽OK を全モデルに仕込んでいた。**指示書に書かれた理論的予想も検証対象**。
- 詳細は `FEATURE-AUDIT.md` A-5・`PERSPECTIVE-error-propagation.md` 追補2。

### 5. ✅ A-9/Q32: 温度サンプリング下の分布一致（完了・29964d0）
- `compare_task(task="sampling", temperature=T)` が出力分布の TV 距離で判定する。
  上界は `tanh(ε/T)`（大域的・実測でタイト）。指示書が出発点に挙げた係数 1/2 型は
  **実測比 2.0 で破れた**（偽OK）——A-5 に続き「係数は数値実験で確かめてから入れる」が効いた。
- **得られた知見**: softmax は shift 不変だが scale 非不変、argmax は両方に不変。この非対称の
  ため既存の `residual_divergence_rms`（純 scale で ≈0 → 偽OK）も `divergence_rms`（純 shift で
  大 → 偽BLOCK）も使えず、shift のみを除いた第 3 の量が要った。**既存関数の再利用を検討する際は
  「その関数が前提にしている不変性」が新しい用途でも成り立つかを確かめること**。
- 詳細は `PERSPECTIVE-task-equivalence.md` 追補・`SOURCES.md`「サンプリング下の分布一致」節。

### 6. ✅ A-9/Q22: beam search（完了・「静的には認証不可」と結論）
- **数値実験の結論**: per-token フリップ率の `(1−p)^L` 合成は beam に不健全。argmax フリップ→
  復元を無視して偽OK（実 survival が合成の 50〜95 倍）、frontier 集合フリップ→過度に悲観的。
  beam は累積対数尤度で並べ替えるため per-token 独立合成の前提が崩れる。
- **得られた法則**: `beam survival ≥ greedy survival`（beam の冗長性が過渡的発散を復元・全
  k/L/ε で実測）。だが証明でなく傾向なので `decode="beam"` は greedy を参考値として返しつつ
  verdict を never-OK にする（fail-safe）。真の認証は系列レベル decode（実行時）が要る。
- **教訓（3 つ目のパターン）**: A-5「前提が反転」・Q32「TV で解決」に続き、Q22 は
  **「静的検証の原理的境界を数値実験で確定し、未解決 TODO を *解決済みの設計判断* に
  格上げする」**。実装できないことの証拠を残すのも 1 ラウンドの成果。詳細は
  `PERSPECTIVE-rollout.md` 追補・`SOURCES.md`「beam search の等価性はなぜ…」節。

### 7. ✅ A-9/Q21: 代表集合の裾サポート（完了・A-9 完全解消）
- 予測橋は代表集合から `P(margin<2δ)` を推定するが、大マージンばかりの集合は決定境界を踏まず
  本番を過小評価する（偽OK・実測 3.4 倍）。Wilson では閉じない（size は扱えるが
  representativeness は扱えない）。`decision.flip_bound_support` が **超過数 k** を報告
  （裾確率の相対不確実性は n でなく k に支配・≈1/√k・極値理論の目安 30）。判定は変えず
  透明化に徹する（Wilson が既に値を保守化済み・二重計上を避ける）。
- **教訓**: 「予測の *信頼性* を、予測が実際に踏んだサンプル数で定量化する」——散文の
  「妥当域を外れうる」を機械可読なシグナルに変えた。docs 主体の指示だったが、fail-safe を
  保ちつつ定量診断まで踏み込めた（コード変更は最小のまま）。
- **A-9 は完全解消**（Q21/Q22/Q31/Q32）。残る A-9 的テーマは beam の系列レベル decode のみで、
  これは実行時活動（モデル状態が要る）として worstcase/attribution と同じ「codegen 後」軸に属す。

## 設計ガードレール（違反しそうなら手を止めて人間に確認）
1. **偽OK 方向の変更は「問題を再現するテスト」を先に書く**（修正前に赤・修正後に緑）。
2. **未検証の数値係数を導入しない**（A-5 が先例——検証実験の結果、想定と *符号が逆* の
   係数を入れかけていたことが判明した。指示書に書かれた理論的予想も検証対象に含める）。
3. **不確実なら保守側**（BLOCK 寄り）に倒す。
4. **新関数は facade 接続まで含めて 1 ラウンド**（B-1 の轍: 実装済み未接続を作らない）。
5. **タグ/Release/main ブランチ操作はしない**（人間専用・`docs/RELEASING.md` §3）。

## 定型手順（全ラウンド共通・`FEATURE-AUDIT.md` §D と同一）
1. 対象モジュールを修正（既存実装の再利用を優先）
2. `tests/correctness/test_<module>.py` に「再現ケース」と「回帰なし」の両テストを追加し
   **その file の `main()` テストリストに登録**（orphan 検査＝不変条件 56 が落ちる）
3. `verify.py` に不変条件を追加（末尾の連番コメント形式）
4. `python verify.py` ＋ `python tests/correctness/run.py` ＋
   `ruff check python/ tests/ examples/` が全 PASS
5. `CHANGELOG.md` の Unreleased に「何を・なぜ・どう直したか・実証データ」
6. commit（日本語・問題→修正→実証）して push（diff ≤500 行・超過は分割）
