# 新視点: 検証は point-in-time — 証明書の陳腐化（temporal drift）

ソクラテス式問答の続き。`tsugi.provenance` として実装。

## 問答

**Q1.** 全ての verdict（audit / equivalence 認証 / Envelope）は、特定の SW/HW スタック
（ROCm/CUDA/driver/compiler/dtype/numpy）で計算される。verdict は *いつまで* 有効か？

**Q2.** スタックが変わるまで。driver 更新・library 回帰・compiler 差は、カーネル選択や
累積順序を変え、回帰を入れうる。だが「検証済み等価」結果は *認証されたスタックに束ねられて
いない* → アップグレードで **silent に無効化** される。「一度認証＝永遠に有効」は誤り。

**Q3.** どうするか？ → verdict を **環境フィンガープリント** に束ねる。現在の環境が認証時と
違えば **stale**（再検証要）と判定し、何が変わったかを示す。

## 実装

- `env_fingerprint(**extra)`: python/numpy/platform を捕捉。実機では cuda/rocm/driver/
  compiler 版を `extra` で渡してフィンガープリントに含める。
- `certify(verdict, **extra)`: verdict ＋ フィンガープリント＋安定ハッシュの `Certificate`。
- `is_stale(cert, **extra)`: 現在の環境が認証時と違えば True（再検証要）。
- `changed_fields(cert, **extra)`: 変わったフィールドを `{field: (old, new)}` で（根拠を明示）。

```
certify("portable", rocm="6.0", driver="550.0")
  → 同一環境       : not stale
  → driver 560 に更新: STALE（再検証要）
  → changed_fields   : {"driver": ("550.0", "560.0")}
```

## 含意

- 検証は写真であって保証書でない —— 撮った瞬間のスタックでのみ正しい。CI を新しい
  ROCm/CUDA/driver で回し直すまで、古い verdict を信じてはならない。
- これは時間軸の盲点を埋める: 既存層は *ある瞬間の* 数値/タスク/レイアウトを見る。
  provenance は *その瞬間がいつのものか* を束ね、陳腐化を検出する。
- 実機 CI に組み込めば「driver 更新 → 関連証明書が全 stale → 自動再検証」が回る。

これでセッションの検証連鎖に時間次元が加わる:
portability（移植可）→ correctness（正しい・oracle）→ oracle 信頼性（metamorphic）
→ **provenance（その verdict はまだ現在のスタックで有効か）**。
