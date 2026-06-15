# Contributing to Tsugi

## 開発環境

### リファレンス層（GPU不要・すぐ動く）
```bash
pip install numpy ruff
python tests/correctness/run.py   # リファレンス correctness + autotune（8テスト）
ruff check python/ tests/ examples/
```

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
- main は protected。直接 push 禁止。Squash merge + CI 全通過。

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
