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

## PR 前チェックリスト
- [ ] `ruff check` clean
- [ ] `python tests/correctness/run.py` 全 PASS
- [ ] 新規 op はリファレンス実装 + テストを追加（正しさの真値を先に）
- [ ] GPU 経路の主張は実機検証済みか「未検証」と明記（主張と実装の一致）
- [ ] PII/課金コードを含まない
- [ ] diff 500行以内（超過は分割）

## 設計原則
Carmack（性能）× Martin（単一責任）× Pike（簡潔）。
ゼロ/最小依存。リファレンス実装先行（OpenCL 失敗の解毒剤）。
