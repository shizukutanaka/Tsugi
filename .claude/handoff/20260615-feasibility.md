[SESSION HANDOFF - 2026-06-15 #5 新視点3追加]

== ソクラテス問答で発見した新視点 ==
検証層の盲点: occ=0%(起動不能)を「性能WARN」と誤分類していた。
「占有率が低い(遅い・連続)」と「そもそも起動できない(動かない・離散)」は別物。
feasibility = equivalence(正しさ)・occupancy(速さ)より上流のcategoricalゲート。
docs/PERSPECTIVE-launch-feasibility.md に記録。

== 実装済み(CPU検証済み・GPU不要) ==
- tsugi/feasibility.py: per-block上限(smem/LDS・threads・regs)で起動可否を判定
  - check(cfg, vendor) → FeasibilityReport(launchable・超過リソース内訳)
  - first_vendor_only(): 片方でしか起動しない=単一ソース約束の破綻を抽出
  - LIMITS: NVIDIA 227KB / AMD CDNA・RDNA 64KiB smem-per-block (docs/SOURCES.md出典)
- portability.analyze 修正: 起動不能を WARN→BLOCK に正す。occ WARNは起動可能時のみ評価
- portcheck: 起動可能性セクション追加(TILE_CONFIG対応)
- 実例: 同一構成 m128n128k64s4w8 (smem=128KB) → NVIDIA起動可(occ12% WARN) /
  AMD CDNA・RDNA 起動不能(LDS 64KiB超過・BLOCK)
- test_feasibility.py 5件 + portcheck 1件追加。計51テストPASS / verify17 / ruff clean

== 戦略的含意 ==
検証順序が確立: feasibility(動くか) → equivalence(正しいか) → occupancy(速いか)。
最も安く最も致命的なチェックを最初に。Tritonはper-vendor autotuneでこの崖を暗黙にまたぐ。

== 次アクション ==
- co-feasible autotuning: feasibility を探索空間の制約に課し「両ベンダーで起動する構成」に絞る
  (autotune.SearchSpace.candidates を vendor横断 feasibility でプルーン)
- 実機: LDS/smem上限を introspect して LIMITS を実値に(現状は一次情報源の代表SKU値)
