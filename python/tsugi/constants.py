"""Tsugi constants — 検証層が共有する数値定数（単一情報源）。

最重要は SAFETY: tolerance / calibration / propagation の許容・検出限界・増幅を一律に
スケールする安全係数。従来は各モジュールに `4.0` が散在しており（5 箇所）、変更時に
不整合を生むリスクがあった。ここに集約して single source of truth とする（DRY）。
"""
from __future__ import annotations

# 安全係数。導出許容 atol ≈ SAFETY·√K·u·scale、検出限界 rel = SAFETY·√K·u、
# propagation の local 発散などで一律に使う。
#
# 根拠（誠実な近似・厳密な最悪ケース境界ではない）:
#   GEMM の f16 入力量子化誤差は K 次元の累積でランダムウォーク的に広がり、絶対誤差の
#   標準偏差は ~√K·u·scale（中心極限）。SAFETY はこの 1σ 見積りに掛けるヘッドルームで、
#   (1) モデルが一次近似である粗さ と (2) 誤差分布の裾 を吸収する。
#   4.0 ≈ 4σ 相当 —— 真の発散は捕えつつ正当な順序差での偽BLOCK を避ける経験的バランス。
#   実機の run-to-run ノイズ実測（nondeterminism.measure_noise_floor）で校正すべき初期値。
#   出典・考察: docs/SOURCES.md / docs/PERSPECTIVE-derived-tolerance.md。
SAFETY: float = 4.0
