# Contributing to Tsugi

## 開発環境

### リファレンス層（GPU不要・すぐ動く）
```bash
pip install numpy ruff
python check.py          # 全ゲート（lint + correctness + 不変条件 + examples スモーク）
python check.py --fast   # 内側ループ用（examples スモークを省く）
```

`check.py` が **ゲートの単一定義**（ローカルと CI が同じものを呼ぶ）。ここに個別コマンドを
再列挙しないこと——以前ここと `docs/ci-reference.yml` にゲートが二重定義され、実際に
食い違っていた（この文書は `verify.py` を lint 対象から落としていた）。個別に走らせたい場合:
`python tests/correctness/run.py` / `python verify.py` / `ruff check ...`。

> **CI について（正直な現状）**: `.github/workflows` はこの環境の権限制約で無効化
> （`.gitignore` 除外）されており、**GitHub Actions は実際には走らない**。当面の CI 代替は
> 上記ローカルの `python check.py`（run.py の SUMMARY に CPU PASS/SKIP 件数を表示）。
> GPU 二ベンダー照合は実機が要るため常に SKIP（緑は CPU 検証可能範囲のみを意味する）。
> CI 定義の正本は追跡可能な [`docs/ci-reference.yml`](docs/ci-reference.yml)。管理者が
> GitHub UI で `.github/workflows/ci.yml` に反映する（中身は `python check.py` を呼ぶだけなので
> ローカルと一致し続ける）。

### GPU バックエンド層（要実機）
- LLVM/MLIR（NVPTX + AMDGPU target 有効化ビルド）
- NVIDIA: CUDA Toolkit / AMD: ROCm
```bash
cmake -B build -G Ninja -DMLIR_DIR=<llvm>/lib/cmake/mlir \
      -DTsugi_ENABLE_NVIDIA=ON -DTsugi_ENABLE_AMD=ON
cmake --build build
```

## ブランチ戦略
- `feature/<issue>-<説明>` / `fix/<issue>-<説明>`
- main は protected。直接 push 禁止。Squash merge + ローカル検証（run.py / verify.py）全通過
  （GitHub Actions は現状無効・上記「CI について」参照）。

## コミット規約
Conventional Commits: `feat:` `fix:` `refactor:` `docs:` `test:` `chore:`
破壊的変更: `feat!:` / `fix!:`

## リリース手順
タグ付きリリースの切り方（検証ゲート → バージョン確定 → タグ公開）は
[`docs/RELEASING.md`](docs/RELEASING.md) を参照。バージョンは `pyproject.toml` と
`tsugi.__version__` の 2 箇所を必ず一致させる（`verify.py` 不変条件 62 が強制・
過去のバージョンドリフト再発防止）。

## PR 前チェックリスト
- [ ] `python check.py` が ALL GATES PASS（lint + correctness + 不変条件 + examples スモーク）
- [ ] 新規 op はリファレンス実装 + テストを追加（正しさの真値を先に）
- [ ] GPU 経路の主張は実機検証済みか「未検証」と明記（主張と実装の一致）
- [ ] PII/課金コードを含まない
- [ ] diff 500行以内（超過は分割）

## Import 方針（SOCRATIC-50 Q46）
標準ライブラリ・`numpy` はモジュール先頭で import する。`tsugi` 内部のサブモジュール間
import は、`audit.py` のような facade 層（多数の独立サブモジュールを 1 箇所に束ねる層）
では関数内で遅延 import する——(1) 呼ばれない phase の import コストを避ける、
(2) サブモジュール同士の将来的な循環 import を予防する、の 2 点が理由。他モジュールに
依存しない葉モジュール（`report.py`・`constants.py` 等）はこの限りでなくモジュール先頭で良い。

## 設計原則
Carmack（性能）× Martin（単一責任）× Pike（簡潔）。
ゼロ/最小依存。リファレンス実装先行（OpenCL 失敗の解毒剤）。
