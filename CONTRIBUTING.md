# Contributing to Tsugi

## 開発環境

### リファレンス層（GPU不要・すぐ動く）
```bash
pip install numpy ruff
python tests/correctness/run.py   # リファレンス correctness + 検証層（CPU 全スイート）
python verify.py                  # 機械可読な不変条件（49+）
ruff check python/ tests/ examples/
```

> **CI について（正直な現状）**: `.github/workflows` はこの環境の権限制約で無効化
> （`.gitignore` 除外）されており、**GitHub Actions は実際には走らない**。当面の CI 代替は
> 上記ローカルの `run.py`（末尾 SUMMARY に CPU PASS 件数と SKIP 件数を表示）+ `verify.py`。
> GPU 二ベンダー照合は実機が要るため常に SKIP（緑は CPU 検証可能範囲のみを意味する）。
> CI 定義の正本は追跡可能な [`docs/ci-reference.yml`](docs/ci-reference.yml)。管理者が
> GitHub UI で `.github/workflows/ci.yml` に反映する（lint 範囲・examples スモークはローカルと一致）。

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
- [ ] `ruff check` clean
- [ ] `python tests/correctness/run.py` 全 PASS
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
