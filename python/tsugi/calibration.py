"""Tsugi calibration — 検証器そのものを検証する（偽OK の非対称コストと検出限界）。

盲点: 既存 8 層はすべて「カーネル/モデルを検証する」。だが検証器自身は未検証だった。
検証器の最大の罪は **偽OK（false equivalence）= 発散を等価と誤判定すること**。
偽BLOCK（移植可を不可と言う）は開発者が気づける・回復可能。だが偽OK は、オラクルの
無いもう片方のベンダーに *silent な誤り* を出荷する — 検出不能で致命的。
コストは非対称ゆえ、検証器は不確実なら BLOCK 寄りに倒すべき（fail-safe）。

新視点: 許容ベースの等価判定には **検出限界（detectability floor）** がある。
正当な累積発散より小さいバグは原理的に見えない。floor は相対で safety·√K·u であり、
K とともに拡大する — つまり視点2（導出許容）が大K GEMM の偽BLOCK を消した代償として、
偽OK の盲点を √K で広げていた（どの修正にも双対のコストがある）。

実証（numpy）: K=2048/fp16 で floor≈8.8%。0.5% の系統スケール誤差は max_abs では
全 K で不可視（偽OK）。救済は **scale/K 不変な相補計量** = RMS（エネルギー）比。
乱雑な累積発散は zero-mean ゆえ RMS 比≈0、系統バグ（スケール/バイアス）は相関ゆえ
RMS 比≠0。max_abs（乱雑・局所発散）と RMS 比（微小・系統バイアス）は相補的。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import SAFETY
from .report import FindingReport, Risk
from .tolerance import unit_roundoff

# 系統バイアスの警告閾値。BLOCK 閾値（thresh）の何割を超えたら WARN にするか。
# 正確には: bias > thresh → BLOCK（系統誤り確定）/ bias > 0.5*thresh → WARN（接近中）。
# safety マージンの半分まで逃げ道を残し、超えたら即警告する保守的な設定。
_WARN_BIAS_RATIO: float = 0.5


def detectability_floor(K: int, dtype: str = "float16", scale: float = 1.0,
                        safety: float = SAFETY) -> dict[str, float]:
    """許容ベース等価判定が検出できる最小誤差（これ未満のバグは不可視＝偽OK）。

    rel = safety·√K·u（scale 非依存）。K とともに拡大する点が肝。
    """
    u = unit_roundoff(dtype)
    rel = safety * math.sqrt(max(1, K)) * u
    return {"abs": rel * scale, "rel": rel, "K": K, "dtype": dtype}


def systematic_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """系統的（相関）発散の指標 = RMS 比 - 1（scale/K 不変）。

    b が a より一様に α 倍なら α-1 を返す。乱雑な累積順序差は zero-mean ゆえ ≈0。
    """
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    ra = float(np.sqrt(np.mean(af ** 2)) + 1e-30)
    rb = float(np.sqrt(np.mean(bf ** 2)))
    return rb / ra - 1.0


def systematic_divergence_stderr(a: np.ndarray, b: np.ndarray, n_boot: int = 200,
                                 seed: int = 0) -> float:
    """systematic_divergence の推定標準誤差をブートストラップで実測する。

    盲点: bias は N 要素から計算した点推定だが、N が小さい（小テンソル）ほど
    たまたま小さい bias が出て偽OK になりうる（真の系統誤差を運悪く見逃す）。
    rollout.flip_rate_upper_bound と同じ fail-safe パターン: 点推定でなく
    *推定の不確実性込みの上側限界* で判定すべき。ここでは要素を復元抽出で
    再標本化し、bias 統計量のばらつき（標準偏差）を経験的に求める。
    N が大きければ標準誤差は無視できるほど小さくなり挙動は変わらない。
    """
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    n = af.size
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ra = np.sqrt(np.mean(af[idx] ** 2, axis=1)) + 1e-30
    rb = np.sqrt(np.mean(bf[idx] ** 2, axis=1))
    boots = rb / ra - 1.0
    return float(np.std(boots))


@dataclass
class CalibrationReport(FindingReport):
    """系統バイアス検査の所見（max_abs の盲点を埋める相補計量）。"""

    bias: float = 0.0
    bias_stderr: float = 0.0
    floor_rel: float = 0.0

    @property
    def bias_upper_bound(self) -> float:
        """推定不確実性込みの |bias| 上側限界（fail-safe 判定に使う・点推定でなくこちら）。"""
        return abs(self.bias) + self.bias_stderr

    def to_text(self) -> str:  # type: ignore[override]
        return super().to_text(
            header=f"systematic check (bias={self.bias * 100:+.3f}%±{self.bias_stderr * 100:.3f}%, "
                   f"max_abs floor={self.floor_rel * 100:.1f}%)",
            empty="(no systematic divergence)")


def check_systematic(a: np.ndarray, b: np.ndarray, K: int = 1,
                     dtype: str = "float16", safety: float = SAFETY,
                     n_boot: int = 200) -> CalibrationReport:
    """系統的スケール/バイアス誤差を検査する（K 不変の閾値 = safety·u）。

    閾値は √K を掛けない（系統誤差は累積発散と違い K で増えない）。ゆえに max_abs の
    検出限界（safety·√K·u）が見逃す微小な系統バグを、K に依らず捕まえる。

    fail-safe: 判定には点推定 bias でなく bias_upper_bound（= |bias| + ブートストラップ
    標準誤差）を使う。小テンソル（N 小）では bias の点推定がたまたま閾値未満になり
    真の系統誤差を見逃しうる（rollout.flip_rate_upper_bound と同じ「点推定でなく
    上側限界で判定する」パターン）。N が大きければ標準誤差は無視できるほど小さく、
    従来の点推定判定と実質同じ挙動になる。
    """
    floor = detectability_floor(K, dtype, 1.0, safety)
    bias = systematic_divergence(a, b)
    stderr = systematic_divergence_stderr(a, b, n_boot=n_boot)
    rep = CalibrationReport(bias=bias, bias_stderr=stderr, floor_rel=floor["rel"])
    thresh = safety * unit_roundoff(dtype)  # scale/K 不変
    ub = rep.bias_upper_bound
    if ub > thresh:
        # fail-safe: 閾値超えの系統バイアス（推定不確実性込み）は（小さくても）DIVERGENT に倒す。
        rep.add(Risk.BLOCK, "scale",
                f"系統バイアス {rep.bias * 100:+.3f}%±{stderr * 100:.3f}%（上側限界 "
                f"{ub * 100:.3f}%）> 閾値 {thresh * 100:.3f}% "
                f"→ max_abs 検出限界 {floor['rel'] * 100:.1f}% の下に隠れる系統バグ")
    elif ub > _WARN_BIAS_RATIO * thresh:
        rep.add(Risk.WARN, "scale",
                f"系統バイアス {rep.bias * 100:+.3f}%±{stderr * 100:.3f}%（上側限界 "
                f"{ub * 100:.3f}%）が閾値 {thresh * 100:.3f}% に近接")
    return rep


# --- 検証器そのものを検証するメタ層（ground-truth コーパスで偽OK率を測る） -----

@dataclass
class Case:
    label: str           # "equivalent" | "divergent"
    a: np.ndarray
    b: np.ndarray
    K: int
    kind: str            # 生成種別（order/scale/dropblock/fp16accum）


@dataclass
class Confusion:
    """非対称コストを明示した混同行列（偽OK が cardinal metric）。"""

    false_ok: int = 0      # divergent を equivalent と誤判定（致命・出荷事故）
    false_block: int = 0   # equivalent を divergent と誤判定（回復可能）
    n_divergent: int = 0
    n_equivalent: int = 0
    missed: list[str] = field(default_factory=list)  # 見逃したバグ種別

    @property
    def false_ok_rate(self) -> float:
        return self.false_ok / max(1, self.n_divergent)

    @property
    def false_block_rate(self) -> float:
        return self.false_block / max(1, self.n_equivalent)

    @property
    def trustworthy(self) -> bool:
        """検証器が信頼に足るか = 偽OK ゼロ（非対称ゆえ偽OK のみを基準にする）。"""
        return self.false_ok == 0

    def to_text(self) -> str:
        status = "TRUSTWORTHY" if self.trustworthy else "UNTRUSTWORTHY"
        lines = [f"verifier calibration ({status})",
                 f"  false-OK   (cardinal) = {self.false_ok}/{self.n_divergent} "
                 f"({self.false_ok_rate * 100:.0f}%) ← 致命: 発散を等価と誤判定",
                 f"  false-BLOCK           = {self.false_block}/{self.n_equivalent} "
                 f"({self.false_block_rate * 100:.0f}%)   回復可能"]
        if self.missed:
            lines.append(f"  見逃したバグ種別: {', '.join(sorted(set(self.missed)))}")
        return "\n".join(lines)


def make_corpus(seed: int = 0, K: int = 2048) -> list[Case]:
    """ground-truth ラベル付きの擬似ベンダー対コーパス（CPU・シミュレーション）。"""
    from .equivalence import simulate_vendor_matmul
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    # 不明瞭なケース（fp16 累積など「lossy だが正当」）は意図的に除く。コーパスは
    # 「等価」と「不可分なバグ（誤答）」のみ — trustworthy の主張を曖昧にしないため。
    for i in range(9):
        M = N = 64
        a = rng.standard_normal((M, K)).astype(np.float16)
        b = rng.standard_normal((K, N)).astype(np.float16)
        base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
        if i % 3 == 0:    # 等価: 累積順序だけ違う（両方 IEEE 正当）
            cases.append(Case("equivalent", base,
                              simulate_vendor_matmul(a, b, accum="f32", split_k=8), K, "order"))
        elif i % 3 == 1:  # バグ: 0.5% 系統スケール誤差（max_abs floor の下に隠れる）
            cases.append(Case("divergent", base, base * 1.005, K, "scale"))
        else:             # バグ: 最後の K ブロック落ち（off-by-one・大きく外れる）
            cases.append(Case("divergent", base,
                              simulate_vendor_matmul(a[:, :K - K // 8], b[:K - K // 8, :],
                                                     accum="f32", split_k=1), K, "dropblock"))
    return cases


def evaluate(corpus, predict) -> Confusion:
    """検証器 predict(a,b,K)->bool(equivalent) をコーパスで採点し偽OK率を測る。"""
    c = Confusion()
    for case in corpus:
        pred_equiv = predict(case.a, case.b, case.K)
        if case.label == "equivalent":
            c.n_equivalent += 1
            if not pred_equiv:
                c.false_block += 1
        else:
            c.n_divergent += 1
            if pred_equiv:
                c.false_ok += 1
                c.missed.append(case.kind)
    return c


def is_equivalent_combined(a: np.ndarray, b: np.ndarray, K: int,
                           dtype: str = "float16") -> bool:
    """fail-safe な合成判定: max_abs（乱雑・局所）と系統バイアスの両方を見る。

    どちらかが発散を示せば DIVERGENT に倒す（非対称コストゆえ BLOCK 寄り）。
    これで max_abs 単独の検出限界に隠れる系統バグも捕まえる。
    """
    from .equivalence import compare_gemm
    random_ok = compare_gemm(a, b, K, dtype).equivalent
    systematic_ok = check_systematic(a, b, K, dtype).ok
    return random_ok and systematic_ok


# 共有モード（convergent）障害の検出 — cross-vendor 一致の構造的盲点 ----------
SM_OK = "OK"
SM_DIVERGENT = "DIVERGENT"
SM_SHARED = "SHARED_MODE"


def detect_shared_mode(a: np.ndarray, b: np.ndarray, oracle: np.ndarray,
                       K: int, dtype: str = "float16") -> str:
    """oracle がある時、cross-vendor 検証の *構造的盲点*（共有モード障害）を検出する。

    cross-vendor 等価判定は A と B の *一致* を見るが、一致 ≠ 正しさ。両ベンダーが同じ
    バグ（同一の上流ライブラリ欠陥・同じ誤った丸め・同じ flawed アルゴリズム）を持つと
    A≈B で「等価」= 緑になるが両方とも誤り。これは cross-vendor 一致では原理的に検出不能。
    oracle（独立な真値・例: CPU/NumPy リファレンス）と照合して初めて見える:

      DIVERGENT   : A≢B            → cross-vendor が捕捉する通常の発散
      SHARED_MODE : A≈B かつ 両方 ≢ oracle → 共有モード障害（cross-vendor は見逃す）
      OK          : A≈B≈oracle

    oracle が無い本番（audit_runtime）では SHARED_MODE は検出不能 —— cross-vendor 一致は
    *必要条件であって十分条件でない*。CI/リファレンスのある経路で oracle 照合すべき。
    """
    from .equivalence import compare_gemm
    if not compare_gemm(a, b, K, dtype).equivalent:
        return SM_DIVERGENT
    a_ok = compare_gemm(a, oracle, K, dtype).equivalent
    b_ok = compare_gemm(b, oracle, K, dtype).equivalent
    return SM_OK if (a_ok and b_ok) else SM_SHARED


def roc_sweep(strengths=(0.001, 0.005, 0.02, 0.05, 0.1), K: int = 2048,
              dtype: str = "float16", seeds: int = 20) -> list[dict]:
    """バグ強度を連続掃引し検証器の偽OK率を測る（9 ケース corpus の ROC 化・Q36）。

    各強度の *系統スケールバグ*（出力を一様に (1+strength) 倍）を seeds 個生成し、
    max_abs 単独 と 合成判定（max_abs + 系統）の偽OK率を比較する。max_abs の rtol は
    一様スケールを吸収するため広い範囲で偽OK だが、合成判定は系統閾値（safety·u・K 不変）
    を超える強度で偽OK=0 に落ちる。閾値未満（~0.2%）は合成判定でも見逃す（正直な残存盲点）。
    """
    from .equivalence import simulate_vendor_matmul

    rows = []
    for st in strengths:
        fo_alone = fo_comb = 0
        for s in range(seeds):
            r = np.random.default_rng(s)
            a = r.standard_normal((64, K)).astype(np.float16)
            b = r.standard_normal((K, 64)).astype(np.float16)
            base = simulate_vendor_matmul(a, b, accum="f32", split_k=1)
            bug = base * (1.0 + st)
            from .equivalence import compare_gemm
            if compare_gemm(base, bug, K, dtype).equivalent:
                fo_alone += 1
            if is_equivalent_combined(base, bug, K, dtype):
                fo_comb += 1
        rows.append({"strength": st, "false_ok_max_abs": fo_alone / seeds,
                     "false_ok_combined": fo_comb / seeds})
    return rows
