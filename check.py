#!/usr/bin/env python3
"""Tsugi 検証ゲート — ローカルと CI が共有する **単一の定義**。

    python check.py            # 全ゲートを実行（出荷前・PR 前はこれ 1 つ）
    python check.py --fast     # examples スモークを省く（内側ループ用）

なぜこのファイルがあるか（重複の削除）:
従来ゲートの一覧は `CONTRIBUTING.md`（人間向け）と `docs/ci-reference.yml`（CI 向け）に
**二重に**書かれており、実際に食い違っていた——CONTRIBUTING は `ruff check python/ tests/
examples/`（`verify.py` が抜けている）、CI は `verify.py` も lint し examples スモークまで
走らせる。この差は「ローカル緑・CI 赤」を生む（ci-reference.yml 自身が警告していた事故）。
ゲートを *実行可能な 1 定義* に畳み、両者がこれを呼ぶことで、ドリフトを検査で見つけるのでなく
**構造的に起こり得なくする**。

終了コード: 全ゲート通過で 0、1 つでも落ちれば 1（CI がそのまま失敗にできる）。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# lint 対象は 1 箇所で定義する（CONTRIBUTING / CI が別々に列挙しない）。
LINT_TARGETS = ["python/", "verify.py", "check.py", "tests/", "examples/"]
# ユーザーが最初に触る入口。壊れたら「動く例」という約束が壊れるのでゲートに含める。
SMOKE_EXAMPLES = ["examples/matmul.py", "examples/user_kernel.py", "examples/audit_demo.py"]


def _run(name: str, cmd: list[str], *, env_extra: dict[str, str] | None = None) -> tuple[str, bool, float]:
    """1 ゲートを実行し (名前, 成否, 秒) を返す。出力は失敗時のみ見せる（ノイズ削減）。"""
    import os

    env = {**os.environ, **(env_extra or {})}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    if not ok:
        sys.stdout.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
    return name, ok, dt


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    fast = "--fast" in argv
    results: list[tuple[str, bool, float]] = []

    if shutil.which("ruff"):
        results.append(_run("lint (ruff)", ["ruff", "check", *LINT_TARGETS]))
    else:
        print("[SKIP ] lint (ruff) — ruff 未インストール（pip install ruff）")

    results.append(_run("correctness suites", [sys.executable, "tests/correctness/run.py"]))
    results.append(_run("invariants (verify.py)", [sys.executable, "verify.py"]))

    if not fast:
        for ex in SMOKE_EXAMPLES:
            results.append(_run(f"smoke {ex}", [sys.executable, ex],
                                env_extra={"PYTHONPATH": str(ROOT / "python")}))

    print()
    total = 0.0
    failed = []
    for name, ok, dt in results:
        total += dt
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s)")
        if not ok:
            failed.append(name)
    print(f"\n{'ALL GATES PASS' if not failed else 'FAILED: ' + ', '.join(failed)} "
          f"({total:.1f}s{'・--fast: examples スモーク省略' if fast else ''})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
