"""`python -m tsugi` — 移植性検証をワンコマンドで実行する入口。

引数なし: 自己デモ（AMD で *起動すらしない* カーネルを検出して見せる）。
引数あり: ユーザーカーネルファイル（@tsugi.jit + make_args() 契約）を検証する。

    python -m tsugi                 # 自己デモ（Tsugi が何を捕まえるか）
    python -m tsugi my_kernel.py    # 自分のカーネルを検証

終了コードは CI ゲート契約（OK/INFO=0・WARN=1・BLOCK=2）。詳細は docs/QUICKSTART.md。
"""
from __future__ import annotations

from .portcheck import main

raise SystemExit(main())
