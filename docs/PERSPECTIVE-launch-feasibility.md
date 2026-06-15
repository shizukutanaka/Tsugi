# 視点の補完 3: 起動可能性という上流ゲート（ソクラテス問答・第3ラウンド）

> 2026-06-15。直前に自分が作った occupancy/portability の前提を問うて発見した盲点。
> 「占有率が低い（遅い）」と「そもそも起動できない（動かない）」を混同していた。

---

## 問答の記録

**Q1（前提検証）**: occupancy は「同じ構成でもベンダー間で占有率が違う」を示す。
これは占有率を 0–100% の *連続量* として扱う。だが `portability.analyze` は
occ < 25% を一律「性能が崩れる可能性」WARN にしていた。occ=0% も WARN だ。
→ 本当に occ=0% は「遅い」のか? **違う。occ=0% は「起動できない」= 動かない。**
連続量（速い⇔遅い）の中に、離散の崖（動く⇔動かない）が紛れ込んでいた。

**Q2（帰結）**: なぜ起動できないのか? → per-block のハード上限（共有メモリ/LDS・
threads/block・regs/thread）はベンダーで違う。とくに **smem/LDS per block は
NVIDIA Hopper 227KB vs AMD CDNA 64KiB** と 3.5 倍差。NVIDIA の広い smem を前提に
autotune した構成は AMD の LDS に *物理的に収まらず、カーネルが launch すらしない*。
→ これは「片方で遅い」ではなく「**片方で動かない**」。`1ソース・2ベンダー` の
約束そのものの破綻。占有率の議論の *前* に起きる。

**Q3（メタ）**: 検証層は何を最初に問うべきか? → 「正しく・速く動くか」の前に
**「そもそも動くか」**。feasibility は equivalence（正しさ）・occupancy（速さ）より
*上流* の categorical ゲート。最初に通すべき関門が、これまで抜けていた。

---

## 新視点（採用）

**起動可能性（feasibility）は数値等価性・占有率より上流の categorical ゲート。**
「遅い（連続）」と「動かない（離散）」を分け、後者を最高深刻度 BLOCK で告げる。

`tsugi.feasibility`:
- `check(cfg, vendor)` → per-block 上限（smem/LDS・threads・regs）に対する
  `FeasibilityReport`（`launchable: bool`・超過リソースの内訳）。
- `first_vendor_only(cfg, a, b)` → *片方でしか起動しない* = 単一ソース約束の
  破綻を抽出（「NVIDIA で動くから OK」と思った構成が AMD で起動すらしない罠）。
- `portability.analyze` に統合: 起動不能は WARN でなく **BLOCK**。
  占有率 WARN は *起動可能な場合のみ* 評価する（occ=0% の誤分類を解消）。

実証（実行確認済み・同一構成 `m128_n128_k64_s4_w8`、smem=128KB）:

| ベンダー | per-block smem 上限 | 判定 |
|---------|--------------------|------|
| NVIDIA (Hopper) | 227KB | LAUNCHABLE（occ 12% の WARN） |
| AMD CDNA (MI300X) | 64KiB | **NOT-LAUNCHABLE（BLOCK）** |
| AMD RDNA (RX7900XTX) | 64KiB | **NOT-LAUNCHABLE（BLOCK）** |

同じカーネルが NVIDIA では起動し AMD では起動しない。**Triton はこれを移植前に
告げない**（per-vendor autotune が暗黙にこの崖をまたぐ）。

---

## なぜこれが正しい補完か（既存視点との接続）

- **修正した実バグ**: 旧コードは occ=0%（起動不能）を「性能が崩れる WARN」と
  誤分類していた。本視点はそれを BLOCK に正す（回帰テストで固定）。
- **検証順序を正す**: feasibility（動くか）→ equivalence（正しいか）→
  occupancy（速いか）。最も安い・最も致命的なチェックを最初に置く。
- **単一ソース約束に直結**: equivalence/occupancy は「移植が *劣化* する」を扱う。
  feasibility は「移植が *壊れる*（起動しない）」を扱う。後者の方が約束の核心。
- **ソロ達成可能・GPU 不要**: per-block 上限と資源式（occupancy と同式）の比較だけ。
  一次情報源（docs/SOURCES.md）の HW 上限に紐付け。

---

## 戦略への影響

- `portcheck` が起動可能性を最上流に報告（構成 `TILE_CONFIG` があれば categorical 判定）。
  GPU codegen 完成前に「この構成は AMD で起動しない」を *今すぐ* 告げられる。
- autotune の含意（次の候補）: per-vendor 探索は単一ソース約束を暗黙に破りうる。
  feasibility を探索空間の *制約* として課せば「両ベンダーで起動する構成」だけに
  絞れる（co-feasible autotuning）。本視点はその土台。

却下/保留した代替視点:
- 「per-vendor で別バイナリを出せばよい」: 単一ソースの利点（1カーネル）を捨てる。
  Tsugi の楔（1ソース・2ベンダー）と矛盾。却下。
- 「LDS 上限を実機 introspect」: 価値はあるが GPU 必要・CPU 検証外。実機フェーズへ保留。
