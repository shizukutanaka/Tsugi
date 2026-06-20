"""tsugi.oracle_check — 参照オラクル自体を検証する（メタモルフィック関係＋高精度照合）。

shared-mode 検出（calibration.detect_shared_mode）はオラクルを真値として信頼する。だが
オラクル（CPU/NumPy = runtime_ref の数値エンジン）も *実装* であり、NumPy は BLAS を呼ぶ。
誰がオラクルを検証するのか？ —— 検証の無限後退。

これを断つのは **実装非依存の証拠**: 任意の正しい実装が必ず満たす *メタモルフィック関係*
（恒等式・分配則・softmax が 1 に和する・shift 不変 等）と、**高精度（float64）再計算**との
一致。第二のオラクルを要さずオラクルを *主張* でなく *検証* する（metamorphic testing）。

オラクルが壊れた環境（病的 BLAS・ビルド不良・極端な FTZ 等）ならここで赤になる。緑は
「このプラットフォームのオラクルは数学的性質を満たす」を意味する。
"""
from __future__ import annotations

import numpy as np

from .report import FindingReport, Risk


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def verify_oracle(seed: int = 0, rtol: float = 1e-4) -> FindingReport:
    """参照オラクル（NumPy float32 エンジン）が実装非依存の性質を満たすか検証する。

    各関係の逸脱が rtol（恒等式は厳密近傍）を超えたら BLOCK 所見を立てる。
    所見なし＝オラクルは健全（このプラットフォームで数学的に整合）。
    """
    rng = np.random.default_rng(seed)
    rep = FindingReport()

    def fail(op: str, msg: str) -> None:
        rep.add(Risk.BLOCK, op, msg)

    # --- matmul: A@I=A（厳密）・分配則・f32 が f64 に追従 ---
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 32)).astype(np.float32)
    C = rng.standard_normal((32, 32)).astype(np.float32)
    if np.abs(A @ np.eye(32, dtype=np.float32) - A).max() > 1e-5:
        fail("matmul", "A@I ≠ A（恒等が崩れる）")
    if np.abs(A @ (B + C) - (A @ B + A @ C)).max() > 1e-3:
        fail("matmul", "分配則 A@(B+C) ≠ A@B+A@C を満たさない")
    f64 = A.astype(np.float64) @ B.astype(np.float64)
    if np.abs((A @ B).astype(np.float64) - f64).max() / (np.abs(f64).max() + 1e-30) > rtol:
        fail("matmul", "f32 結果が f64 真値から rtol 超で乖離")

    # --- reduce(sum): sum(ones)=n（厳密）・f32 が f64 に追従 ---
    if abs(float(np.sum(np.ones(1000, np.float32))) - 1000.0) > 1e-3:
        fail("reduce", "sum(ones) ≠ n")
    x = rng.standard_normal(4096).astype(np.float32)
    s64 = np.sum(x.astype(np.float64))
    if abs(float(np.sum(x)) - s64) / (abs(s64) + 1e-30) > 1e-3:
        fail("reduce", "f32 和が f64 真値から乖離")

    # --- exp: exp(0)=1・exp(a+b)=exp(a)exp(b) ---
    if abs(float(np.exp(np.float32(0.0))) - 1.0) > 1e-6:
        fail("exp", "exp(0) ≠ 1")
    a = (rng.standard_normal(100) * 2).astype(np.float32)
    b = (rng.standard_normal(100) * 2).astype(np.float32)
    if np.abs(np.exp(a + b) - np.exp(a) * np.exp(b)).max() / (np.abs(np.exp(a + b)).max()) > rtol:
        fail("exp", "exp(a+b) ≠ exp(a)·exp(b)")

    # --- softmax: 1 に和する・shift 不変 ---
    if abs(float(_softmax(a).sum()) - 1.0) > 1e-5:
        fail("softmax", "確率が 1 に和しない")
    if np.abs(_softmax(a) - _softmax(a + 5.0)).max() > rtol:
        fail("softmax", "shift 不変でない（softmax(x+c) ≠ softmax(x)）")

    # --- rsqrt: rsqrt(x)²·x = 1 ---
    xp = np.abs(rng.standard_normal(100).astype(np.float32)) + 0.1
    if np.abs((1.0 / np.sqrt(xp)) ** 2 * xp - 1.0).max() > rtol:
        fail("rsqrt", "rsqrt(x)²·x ≠ 1")

    return rep


def oracle_is_trustworthy(seed: int = 0) -> bool:
    """オラクルが全メタモルフィック関係を満たす（BLOCK 所見なし）か。"""
    return verify_oracle(seed).ok
