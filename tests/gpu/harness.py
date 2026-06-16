"""Tsugi GPU 検証ハーネス — 実機でクロスベンダー監査を駆動する契約。

GPU codegen 本体は未実装（要 LLVM/MLIR + 実機）。本ハーネスは「実機があれば何を
どう回すか」を *実行可能な契約* として固定する。GPU が無ければ正直に SKIP し、
ハーネスの配線（ノイズ実測 → audit_cross_vendor）自体は擬似 run で CPU 検証できる。

実機での使い方:
    from tests.gpu.harness import run_vendor_kernel, audit_two_vendors
    rep = audit_two_vendors(
        lambda s: run_vendor_kernel("nvidia", kernel, args, seed=s),
        lambda s: run_vendor_kernel("amd",    kernel, args, seed=s),
        K=K, env=env)
    assert rep.portable
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tsugi.audit import audit_cross_vendor  # noqa: E402


def gpu_vendors() -> list[str]:
    """実機で利用可能なベンダー（torch.cuda 経由・無ければ空）。"""
    vendors: list[str] = []
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            vendors.append("amd" if any(k in name for k in ("amd", "radeon", "instinct"))
                           else "nvidia")
    except Exception:  # noqa: BLE001
        pass
    return vendors


def run_vendor_kernel(vendor: str, kernel, args, *, seed: int = 0):
    """実 GPU でカーネルを 1 回走らせ出力テンソルを返す（codegen 完成後に実装）。

    seed は run-to-run の非決定（atomic スケジュール等）を変える手段。
    現状は未実装 —— GPU codegen（lowering.py の仕様）と実機が要る。
    """
    raise NotImplementedError(
        "GPU codegen 未実装（要 LLVM/MLIR + 実機）。lowering.py が実装仕様。")


def audit_two_vendors(run_a, run_b, K, *, env=None, n_runs=16,
                      logits_a=None, logits_b=None, flip_budget=0.001):
    """2 ベンダーの run 関数からノイズを実測し、統合監査を返す（実機/擬似 run 共通）。

    これがハーネスの本体 = nondeterminism と audit_runtime をつなぐ契約。
    実機では run_a/run_b に run_vendor_kernel を、CPU では擬似 run を渡す。
    """
    return audit_cross_vendor(run_a, run_b, K, env=env, n_runs=n_runs,
                              logits_a=logits_a, logits_b=logits_b,
                              flip_budget=flip_budget)
