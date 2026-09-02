"""Tsugi correctness エントリポイント。

2段構え:
  1. リファレンス（CPU/NumPy）correctness — GPU 不要・常に実行・正しさの真値
  2. GPU 両ベンダー数値照合 — 実機があれば実行・無ければ正直に skip

実行: python tests/correctness/run.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: 環境起因の skip を数えるパターン。テストは「実行できなかった」ことを
#: `[SKIP]` 行で自己申告する慣例（torch 無し・GPU 無し等）。
_SKIP_RE = re.compile(r"\[SKIP\]")


def _available_vendors() -> list[str]:
    vendors: list[str] = []
    try:
        import torch  # noqa: F401
        import torch.cuda
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            vendors.append("amd" if ("amd" in name or "radeon" in name
                                     or "instinct" in name) else "nvidia")
    except Exception:  # noqa: BLE001
        pass
    return vendors


def _run_suite(name: str) -> tuple[str, int, str]:
    """1 スイートを別プロセスで走らせ (名前, rc, 出力) を返す。

    出力を握って返すのは **登録順に印字する**ため。完了順に流すと実行のたびに
    行順が変わり diff できなくなる（並列化で失ってはいけない性質）。
    """
    r = subprocess.run([sys.executable, str(HERE / name)],
                       capture_output=True, text=True, check=False)
    return name, r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    rc = 0
    cpu_suites = ("test_reference.py", "test_autotune.py", "test_tracer.py", "test_lowering.py", "test_codegen.py", "test_compile.py", "test_portability.py", "test_equivalence.py", "test_occupancy.py", "test_portcheck.py", "test_tolerance.py", "test_bf16.py", "test_feasibility.py", "test_propagation.py", "test_envelope.py", "test_calibration.py", "test_nondeterminism.py", "test_decision.py", "test_rollout.py", "test_audit.py", "test_attribution.py", "test_blame.py", "test_worstcase.py", "test_properties.py", "test_fxbridge.py", "test_fxlower.py", "test_tsugi_torch_compile.py", "test_oracle_check.py", "test_provenance.py", "test_tile_ops.py")
    print("=== [1] reference correctness (CPU/NumPy oracle) ===")
    # 各スイートは独立プロセスで、ファイル書き込み・環境変数変更・chdir・入れ子
    # subprocess のいずれも持たない（検査済み）。ゆえに並列化は正当性に影響しない。
    # ここは開発サイクル時間の最大の支出（直列で全体の ~7 割）だった。
    workers = max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run_suite, cpu_suites))

    env_skips: list[str] = []
    for name, code, out in results:            # 登録順に印字（diff 可能性を保つ）
        sys.stdout.write(out)
        rc |= code
        env_skips += [ln.strip() for ln in out.splitlines() if _SKIP_RE.search(ln)]

    skipped = []
    print("\n=== [2] GPU two-vendor correctness ===")
    vendors = _available_vendors()
    if not vendors:
        print("SKIP: no GPU detected (LLVM/MLIR + NVIDIA/AMD hardware required). "
              "未検証 — 実機での実行が必要。")
        skipped.append("GPU two-vendor correctness (kernel exec vs reference)")
    else:
        print(f"detected vendors: {vendors}")
        print("kernel execution vs reference: not yet implemented (Phase1, requires GPU)")
        skipped.append("GPU two-vendor correctness (codegen not implemented)")

    print("\n=== [3] GPU cross-vendor audit contract（ハーネス配線は CPU 検証）===")
    gpu = HERE.parent / "gpu" / "test_audit_runtime_contract.py"
    rc |= subprocess.run([sys.executable, str(gpu)], check=False).returncode
    if not vendors:
        skipped.append("GPU real-kernel audit (harness wiring CPU-verified)")

    # 正直なサマリ: 緑（CPU PASS）を「全部検証済み」と誤読させない（SOCRATIC Q37/Q59）。
    status = "PASS" if rc == 0 else "FAIL"
    print(f"\n=== SUMMARY === CPU suites: {status} ({len(cpu_suites)} files, "
          f"{workers} parallel) | SKIPPED (requires hardware): {len(skipped)}")
    for s in skipped:
        print(f"  - SKIP: {s}")
    # CPU スイート *内部* の skip も数える。torch が無い環境では意味論照合と実 FX 結線が
    # 丸ごと飛ぶのに、従来のサマリは「CPU suites: PASS」としか言わなかった——
    # **緑が「全部検証した」と読まれる**（Q37 と同型の偽OK が CPU 側に残っていた・Q59）。
    print(f"  環境起因の skip（CPU スイート内）: {len(env_skips)} 件")
    for line in env_skips:
        print(f"    - {line}")
    if env_skips:
        print("  注意: 上記は *実行されなかった* 検査。緑はそれらを含まない。")
    if skipped:
        print("  注意: GPU 経路は未検証。緑は CPU 検証可能範囲のみを意味する。")
    return rc


if __name__ == "__main__":
    sys.exit(main())
