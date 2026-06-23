"""Tsugi worstcase — 最悪ケース発散探索（新視点10）。

ソクラテス式問答:

  Q1. これまでの全視点（equivalence/decision/rollout/propagation/calibration…）は
      何を入力に取るか？ → *代表データ* か、*与えられた* 2 出力。すべて受動的だ——
      手元のサンプル上で発散の **率/分布** を測る。

  Q2. では出荷後、その代表サンプルを誰が選ぶのか？ → 開発者だ。だが本番の入力を選ぶのは
      ユーザー（時に敵対者）。代表分布と本番分布は一致しない。

  Q3. 代表サンプルで flip率 0.01%・発散 < 許容 なら「移植可」と言えるか？ → 言えない。
      測ったのは *平均* であって、認証エンベロープ内に発散を *最大化* する希少入力
      （near-cancellation・outlier channel・境界 logit）が存在しない保証はない。
      **平均ケース等価 ⇏ 最悪ケース等価。**

  Q4. ならば測るべきは率でなく？ → 認証エンベロープ内で発散を最大化する *入力* と、その
      大きさ。受動的サンプリングから **能動的探索** へ。これは ML の adversarial
      robustness / fuzzing をベンダー移植に移したもの。

  Q5. これは既存のどの視点と対をなすか？ → envelope（運用入力が認証前提内か検査）。
      envelope は「入力が領域 R に留まれば認証は有効」と言う。worstcase は「R の *内部*
      でも発散が許容を超えうるか」を能動的に探す。超える入力が見つかれば、R（認証
      エンベロープ）が緩すぎる＝認証が不健全、という反証になる。

実装: 2 ベンダーのカーネルは（クロスベンダーゆえ）微分不能な黒箱。勾配を使えないので
微分フリー探索（ランダム再開つきヒルクライム）で、box 制約（＝エンベロープ）内の
発散最大化入力を探す。box 制約が要：無制約だと自明に overflow を見つけてしまう。
返す反例 x_worst は seed 固定で再現でき、監査可能な成果物になる。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .report import FindingReport, Risk


def divergence(fn_a, fn_b, x, *, relative: bool = True) -> float:
    """入力 x における 2 ベンダー出力の発散（既定は相対・最大絶対差／出力規模）。"""
    a = np.asarray(fn_a(x), dtype=np.float64)
    b = np.asarray(fn_b(x), dtype=np.float64)
    num = float(np.max(np.abs(a - b))) if a.size else 0.0
    if relative:
        return num / (float(np.max(np.abs(a))) + 1e-30) if a.size else 0.0
    return num


def _box(x0: np.ndarray, radius: float, bounds):
    """探索領域（＝認証エンベロープ）。bounds 明示か、無ければ x0 を中心に相対 radius。"""
    if bounds is not None:
        lo, hi = bounds
        return (np.broadcast_to(np.asarray(lo, dtype=np.float64), x0.shape).copy(),
                np.broadcast_to(np.asarray(hi, dtype=np.float64), x0.shape).copy())
    span = radius * (float(np.max(np.abs(x0))) + 1e-12)
    return x0 - span, x0 + span


def search_worst_input(fn_a, fn_b, x0, *, radius: float = 1.0, steps: int = 400,
                       seed: int = 0, restarts: int = 4, relative: bool = True,
                       bounds=None):
    """box 制約内で発散を最大化する入力を微分フリー探索（ヒルクライム＋ランダム再開）。

    黒箱（微分不能なクロスベンダーカーネル）が対象ゆえ勾配を使わない。返り値
    (x_worst, div_worst) は seed 固定で再現する反例。`bounds=(lo,hi)` で認証エンベロープを
    明示でき、無ければ x0 まわりの相対 box を使う。
    """
    x0 = np.asarray(x0, dtype=np.float64)
    lo, hi = _box(x0, radius, bounds)
    rng = np.random.default_rng(seed)
    best_x, best_d = x0.copy(), divergence(fn_a, fn_b, x0, relative=relative)
    per = max(1, steps // max(1, restarts))
    for r in range(max(1, restarts)):
        x = x0.copy() if r == 0 else rng.uniform(lo, hi)
        d = divergence(fn_a, fn_b, x, relative=relative)
        scale = (hi - lo) / 4.0
        for _ in range(per):
            cand = np.clip(x + rng.standard_normal(x.shape) * scale, lo, hi)
            dc = divergence(fn_a, fn_b, cand, relative=relative)
            if dc > d:
                x, d = cand, dc
            else:
                scale *= 0.98          # 受理されなければ歩幅を縮める（焼きなまし）
        if d > best_d:
            best_x, best_d = x, d
    return best_x, best_d


@dataclass
class WorstCaseReport(FindingReport):
    """能動探索で見つけた最悪ケース発散の所見（平均ケース検証の盲点を露出）。"""

    typical_divergence: float = 0.0
    worst_divergence: float = 0.0
    amplification: float = 1.0     # worst / typical（平均がどれだけ最悪を過小評価したか）
    tol: float = 0.0
    n_samples: int = 0
    x_worst: object = field(default=None)   # 再現可能な反例入力（監査用）

    def to_text(self) -> str:  # type: ignore[override]
        return super().to_text(
            header=(f"worst-case search (typical={self.typical_divergence:.2e}, "
                    f"worst={self.worst_divergence:.2e}, ×{self.amplification:.1f} vs typical, "
                    f"tol={self.tol:.2e}, n={self.n_samples})"),
            empty="(no in-envelope input exceeds typical divergence)")


def analyze_worst_case(fn_a, fn_b, samples, *, tol: float, radius: float = 1.0,
                       steps: int = 400, seed: int = 0, relative: bool = True,
                       bounds=None) -> WorstCaseReport:
    """代表サンプル上の *典型* 発散と、エンベロープ内の *最悪* 発散を対比する。

    最悪発散が認証許容 tol を超える入力がエンベロープ内に見つかれば BLOCK（代表データは
    良性でも、本番で踏みうる反例が存在する＝envelope が緩い）。tol 以内でも典型の何倍も
    大きければ WARN（平均が最悪を覆い隠している）。

    samples: 代表入力の列（各々 fn_a/fn_b に渡せる配列）。探索は最も発散した代表から開始。
    """
    samples = [np.asarray(s, dtype=np.float64) for s in samples]
    typ = [divergence(fn_a, fn_b, s, relative=relative) for s in samples]
    typical = float(np.median(typ)) if typ else 0.0
    start = samples[int(np.argmax(typ))] if typ else np.zeros(1)
    x_w, d_w = search_worst_input(fn_a, fn_b, start, radius=radius, steps=steps,
                                  seed=seed, relative=relative, bounds=bounds)
    d_w = max(d_w, max(typ) if typ else 0.0)
    amp = d_w / (typical + 1e-30)
    rep = WorstCaseReport(typical_divergence=typical, worst_divergence=d_w,
                          amplification=amp, tol=tol, n_samples=len(samples), x_worst=x_w)
    if d_w > tol:
        rep.add(Risk.BLOCK, "worst",
                f"エンベロープ内に許容超過の反例: 最悪発散 {d_w:.2e} > tol {tol:.2e} "
                f"(代表は {typical:.2e}・×{amp:.1f})。平均ケース検証は良性でも本番で踏みうる "
                "→ 認証エンベロープが緩い（再現可能な反例 x_worst を参照）")
    elif amp >= 10.0 and d_w > 0.0:
        rep.add(Risk.WARN, "worst",
                f"最悪発散 {d_w:.2e} は典型 {typical:.2e} の ×{amp:.1f}（tol {tol:.2e} 内）。"
                "平均が最悪を覆い隠している → 代表サンプルの代表性に注意")
    elif d_w > 0.0:
        rep.add(Risk.INFO, "worst",
                f"最悪発散 {d_w:.2e} ≤ tol {tol:.2e}・典型の ×{amp:.1f}（エンベロープ内は健全）")
    return rep
