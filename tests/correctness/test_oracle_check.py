"""参照オラクル自体の検証テスト（メタモルフィック関係＋高精度照合）。

オラクル(NumPy float32)が実装非依存の数学的性質を満たすことを確かめる。
「誰がオラクルを検証するのか」の無限後退を、第二オラクル無しで断つ。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.oracle_check import oracle_is_trustworthy, verify_oracle  # noqa: E402


def test_oracle_passes_metamorphic_relations():
    rep = verify_oracle()
    assert rep.ok                       # 健全なプラットフォームでは所見なし
    assert len(rep.findings) == 0
    assert oracle_is_trustworthy()


def test_oracle_check_stable_across_seeds():
    # 複数 seed で再現的に健全（特定入力に依存しない）
    for s in range(5):
        assert verify_oracle(seed=s).ok


def test_verify_oracle_can_flag(monkeypatch=None):
    # しきい値を病的に厳しく（rtol=0）すると f32 丸めが恒等を満たさず必ず所見が立つ
    # ＝検証が「常に緑」でなく実際に逸脱を捕まえる能力を持つことの確認。
    rep = verify_oracle(rtol=0.0)
    assert not rep.ok
    assert len(rep.findings) > 0


def main() -> int:
    ok = True
    tests = [
        test_oracle_passes_metamorphic_relations,
        test_oracle_check_stable_across_seeds,
        test_verify_oracle_can_flag,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            ok = False
    print(verify_oracle().to_text("oracle metamorphic check"))
    print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
