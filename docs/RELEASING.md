# リリース手順（RELEASING）

Tsugi の「完成品」を GitHub に公開する（タグ付きリリースを切る）ための手順書。
このプロジェクトは `.github/workflows` が環境制約で無効（`CONTRIBUTING.md`「CI について」
参照）なため、リリースは **ローカル検証 → バージョン確定 → タグ公開** を手動で回す。

過去に「`pyproject.toml` は 0.3.0 にバンプしたのに `tsugi.__version__` が 0.2.0 の
まま取り残される」バージョンドリフトが実際に起きた（配布メタデータと実行時 API が
別バージョンを名乗る欠陥）。**この手順書と `verify.py` 不変条件 62（version 整合）が
その再発防止**である。

---

## 0. 前提

- SemVer。`0.x` は API 未凍結（MINOR で機能追加・互換変更ありうる）。
  機能追加を含むなら MINOR を上げる（例 0.3.0 → 0.4.0）。バグ修正のみなら PATCH。
- バージョン番号の**単一の真実**は 2 箇所で、**必ず一致**させる（不変条件 62 が強制）:
  - `python/pyproject.toml` の `version`
  - `python/tsugi/__init__.py` の `__version__`

## 1. 検証ゲート（壊れたものをリリースしない）

すべて PASS してからバージョンを確定する。1 つでも落ちたら中断して直す。

```bash
python check.py     # 全ゲート（lint + correctness + 不変条件 + examples スモーク）
```

ゲートの実体は `check.py`（ローカル/CI/リリースが共有する単一定義）。ここで個別コマンドを
再列挙しないこと——以前は本書・CONTRIBUTING・ci-reference.yml の 3 箇所に散らばり、
lint 対象が食い違っていた。

`run.py` の SUMMARY は「緑は CPU 検証可能範囲のみ」を意味する（GPU 二ベンダー照合は
実機が要るため常に SKIP）。GPU 経路を「検証済み」と誤読しないこと。

## 2. バージョン確定とリリースノート（1 コミット）

1. `python/pyproject.toml` の `version` と `python/tsugi/__init__.py` の `__version__` を
   新バージョンに揃える（**両方**。片方だけだと不変条件 62 が落ちる）。
2. `CHANGELOG.md` の `## [Unreleased]` を `## [X.Y.Z] — YYYY-MM-DD` に切り出し、
   空の `[Unreleased]` を新設する（Keep a Changelog 形式）。
3. `python verify.py` を再実行し、不変条件 62（version 整合）を含め全 PASS を確認。
4. コミット（日本語・問題→修正→実証の順）。例:
   `リリースX.Y.Zの切り出し(...)`

## 3. タグ公開

注釈付きタグを作り、リリースノート（CHANGELOG 該当節のハイライト）を注釈本文にする。

```bash
git tag -a vX.Y.Z -m "Tsugi vX.Y.Z — 要約..."   # または -F <notesファイル>
git push origin vX.Y.Z
```

GitHub の Tags/Releases ページにタグが公開され、ソースアーカイブも自動で付く。
リリースノート付きの **Release オブジェクト**にするには GitHub UI で当該タグを選び
"Create release from tag"（CHANGELOG 該当節を本文に貼る）。

> **権限で `git push <tag>` が 403 になる環境について**: サンドボックス/CI 等で
> git リモートがブランチスコープに限定され、タグ ref（`refs/tags/*`）への push が
> 403 で拒否される場合がある。その場合はタグを push できないため、**GitHub UI から
> リリースを作成**する（Releases → *Draft a new release* → *Choose a tag* → 新規タグ名を
> 入力 → *Create new tag on publish* → 対象コミットを選択 → ノートを貼って Publish）。
> UI 経由ならタグオブジェクトと Release メタデータが同時に作られる。

## 4. 公開確認

- タグ/リリースがリモートに見えることを確認（GitHub UI、または API/`git ls-remote --tags origin`）。
- **配布物の検証（必須）**: ソースツリー無しで動くことを確かめる（`pip install -e` は
  ソースを参照するので配布物の検証にならない）。クリーン venv で wheel を入れ、
  製品の入口が動き終了コード契約を守ることまで見る:

  ```bash
  pip wheel python/ -w /tmp/wh --no-deps          # wheel をビルド
  python -m venv /tmp/v && /tmp/v/bin/pip install /tmp/wh/tsugi_torch-*.whl
  cd /tmp && /tmp/v/bin/python -m tsugi; echo "exit=$?"   # デモは BLOCK → exit 2 が正
  /tmp/v/bin/python -c "import tsugi, tsugi_torch; print(tsugi.__version__, hasattr(tsugi,'verify'))"
  ```

  期待: `python -m tsugi` が移植レポートを出し **exit 2**（デモは AMD 起動不能 BLOCK）、
  `tsugi.verify` が公開され `tsugi_torch` も同梱されていること。

---

## チェックリスト（コピー用）

- [ ] `python verify.py` 全 PASS（不変条件 62 = version 整合を含む）
- [ ] `python tests/correctness/run.py` 全 PASS
- [ ] `ruff check python/ tests/ examples/` clean
- [ ] `pyproject.toml` の `version` と `tsugi.__version__` が一致・新バージョン
- [ ] `CHANGELOG.md` に `[X.Y.Z] — 日付` 節を切り出し・空の `[Unreleased]` を新設
- [ ] リリースコミットを作成・push
- [ ] タグ `vX.Y.Z` を作成（`git push origin vX.Y.Z` または GitHub UI から Release 作成）
- [ ] タグ/リリースがリモートに見えることを確認
