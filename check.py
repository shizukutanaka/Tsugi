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


#: correctness スイートが報告した環境起因 skip の件数（ゲートのサマリへ透過する）。
_SKIPS: list[int] = []


def _skip_count() -> int:
    return _SKIPS[0] if _SKIPS else 0


def _run(name: str, cmd: list[str], *, env_extra: dict[str, str] | None = None) -> tuple[str, bool, float]:
    """1 ゲートを実行し (名前, 成否, 秒) を返す。出力は失敗時のみ見せる（ノイズ削減）。"""
    import os
    import re

    env = {**os.environ, **(env_extra or {})}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    if not ok:
        sys.stdout.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
    m = re.search(r"環境起因の skip（CPU スイート内）: (\d+) 件", proc.stdout or "")
    if m:
        _SKIPS.append(int(m.group(1)))
    return name, ok, dt


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    fast = "--fast" in argv

    # ゲートは **直列** に走らせる。一度 ThreadPoolExecutor で並列化してみたが、
    # 4 コア環境で 40s → **104s と大幅に悪化**した（実測）——correctness スイート自身が
    # 既に 4 ワーカーを使っており、そこへ verify.py と smoke 3 本を重ねると
    # oversubscription で全員が遅くなる。**並列化すべきは 1 段だけ**（run.py の内側）。
    # 効かない部品は入れない（Musk 第 2 段階）。
    results: list[tuple[str, bool, float]] = []
    wall0 = time.time()

    if shutil.which("ruff"):
        results.append(_run("lint (ruff)", ["ruff", "check", *LINT_TARGETS]))
    else:
        print("[SKIP ] lint (ruff) — ruff 未インストール（pip install ruff）")

    results.append(_run("correctness suites",
                        [sys.executable, "tests/correctness/run.py"]))
    results.append(_run("invariants (verify.py)", [sys.executable, "verify.py"]))

    if not fast:
        for ex in SMOKE_EXAMPLES:
            results.append(_run(f"smoke {ex}", [sys.executable, ex],
                                env_extra={"PYTHONPATH": str(ROOT / "python")}))
    wall = time.time() - wall0

    print()
    failed = []
    for name, ok, dt in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s)")
        if not ok:
            failed.append(name)
    # 「緑＝全部検証した」と読ませない: スイート側が数えた環境起因 skip を透過する。
    skips = _skip_count()
    tail = f"・環境起因 skip {skips} 件" if skips else ""
    print(f"\n{'ALL GATES PASS' if not failed else 'FAILED: ' + ', '.join(failed)} "
          f"({wall:.1f}s{tail}"
          f"{'・--fast: examples スモーク省略' if fast else ''})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
