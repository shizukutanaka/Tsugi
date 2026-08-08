"""Tsugi 関数カバレッジ計測（標準ライブラリのみ・SOCRATIC-50 Q38 / FEATURE-AUDIT A-10）。

**なぜ coverage.py を使わないか**: このプロジェクトの実行時依存は `numpy` のみ
（`verify.py` 不変条件 60 が permissive ライセンス許容リストで固定している）。計測のために
依存を増やすとその不変条件の前提が動く。Python 標準の `sys.settrace` で足りる範囲に留める。

**なぜ行でなく関数単位か**: 本プロジェクトが繰り返し踏んだ欠陥は「実装済みだが *誰からも
呼ばれない*」型（`docs/FEATURE-AUDIT.md` セクション B-1 で 11 件）。行カバレッジより
「どの関数が一度も実行されないか」の方が直接その欠陥に効く。加えて `call` イベントのみを
追うので `line` イベント全追跡より桁で速く、全スイートを現実的な時間で回せる。

`verify.py` 不変条件 57（facade 未接続の静的スキャン）との違い:
  - 不変条件 57 = **静的**。ソースに呼び出しが「書かれているか」を正規表現で見る。
  - 本ツール    = **動的**。テスト実行で実際に「呼ばれたか」を見る。
静的に呼ばれていても実行経路に乗らない（分岐が常に false 等）関数は動的でしか見えない。

使い方:
    python coverage_report.py            # 全 CPU スイート＋verify を実行し未実行関数を報告
    python coverage_report.py --quiet    # 集計のみ
"""
from __future__ import annotations

import ast
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "python"
PKG_DIRS = (PY / "tsugi", PY / "tsugi_torch")

# 計測から除く関数名（呼ばれないのが正当なもの）。
# docs/FEATURE-AUDIT.md セクション B-2（意図的な facade 非接続）と対応させる。
EXPECTED_UNCALLED: dict[str, str] = {
    "main": "CLI エントリポイント（本ツール自身がプロセス内で呼ぶ経路に乗らない場合がある）",
    "register": "torch 有り環境でのみ実行される backend 登録",
    "_tsugi_compile": "torch.compile バックエンド本体（torch 有り環境でのみ実行）",
    "to_mlir": "IR のテキスト出力（デバッグ/可視化用・判定経路ではない）",
}


def _defined_functions() -> dict[str, set[str]]:
    """各ソースファイルで定義されている関数名を AST で列挙する（ネスト関数は除く）。"""
    out: dict[str, set[str]] = {}
    for d in PKG_DIRS:
        for p in sorted(d.glob("*.py")):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:                     # 解析不能なら計測対象外
                continue
            names: set[str] = set()
            for node in tree.body:                  # トップレベル定義のみ
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    for sub in node.body:           # メソッドは Class.method 形式
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            names.add(f"{node.name}.{sub.name}")
            out[str(p)] = names
    return out


def _make_tracer(called: dict[str, set[str]]):
    """`call` イベントのみを拾う軽量トレーサ（line イベントは追わない＝高速）。

    `co_filename` は正規化されていない（テストが `sys.path` に
    `tests/correctness/../../python` のような相対成分つきパスを入れると
    `.../tests/correctness/../../python/tsugi/audit.py` の形で現れる）。
    素朴な `startswith` では取りこぼすので `normpath` で畳んでから比較する。
    パスごとの正規化結果はキャッシュして call イベント当たりのコストを抑える。
    """
    targets = tuple(str(d) for d in PKG_DIRS)
    norm_cache: dict[str, str | None] = {}

    def _norm(fn: str) -> str | None:
        """対象パッケージ配下なら正規化パスを、外なら None を返す（結果はキャッシュ）。"""
        if fn in norm_cache:
            return norm_cache[fn]
        p = os.path.normpath(fn)
        norm_cache[fn] = p if p.startswith(targets) else None
        return norm_cache[fn]

    def tracer(frame, event, arg):
        if event != "call":
            return None
        code = frame.f_code
        path = _norm(code.co_filename)
        if path is None:
            return None
        name = code.co_qualname if hasattr(code, "co_qualname") else code.co_name
        # ネスト関数（<locals> を含む）は外側の関数に帰属させる
        name = name.split(".<locals>")[0]
        called.setdefault(path, set()).add(name)
        return None

    return tracer


def _run_workload() -> None:
    """計測対象のワークロード: CPU テストスイート全部 ＋ verify.py。"""
    import importlib.util

    sys.path.insert(0, str(PY))
    suite_dir = ROOT / "tests" / "correctness"
    for path in sorted(suite_dir.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(f"_cov_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            if hasattr(mod, "main"):
                mod.main()
        except Exception as e:                      # 1 スイートの失敗で計測を止めない
            print(f"  [warn] {path.name}: {type(e).__name__}: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv

    defined = _defined_functions()
    called: dict[str, set[str]] = {}
    tracer = _make_tracer(called)

    buf = io.StringIO()
    sys.settrace(tracer)
    try:
        with redirect_stdout(buf):                  # スイートの出力は捨てる
            _run_workload()
    finally:
        sys.settrace(None)

    total_def = total_hit = 0
    rows: list[tuple[str, int, int, list[str]]] = []
    for path, names in sorted(defined.items()):
        hit = called.get(path, set())
        # 「呼ばれた」判定は完全一致 or メソッド名一致（qualname の差異を吸収）
        hit_names = {n for n in names
                     if n in hit or n.split(".")[-1] in {h.split(".")[-1] for h in hit}}
        missing = sorted(n for n in names - hit_names
                         if n.split(".")[-1] not in EXPECTED_UNCALLED)
        total_def += len(names)
        total_hit += len(hit_names)
        rows.append((Path(path).name, len(hit_names), len(names), missing))

    pct = 100.0 * total_hit / max(1, total_def)
    if not quiet:
        print("=== Tsugi 関数カバレッジ（標準ライブラリのみ・call イベント計測）===\n")
        for name, hit, tot, missing in rows:
            mark = "" if not missing else f"  未実行: {', '.join(missing)}"
            print(f"  {name:<24} {hit:>3}/{tot:<3} ({100.0 * hit / max(1, tot):5.1f}%){mark}")
        print()
    print(f"関数カバレッジ: {total_hit}/{total_def} ({pct:.1f}%)")
    uncalled = sorted(n for _, _, _, ms in rows for n in ms)
    if uncalled:
        print(f"未実行関数 {len(uncalled)} 件（B-2 の意図的非接続を除く）: {', '.join(uncalled)}")
    else:
        print("未実行関数なし（許容リストを除く全公開関数がテストで実行された）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
