# Tsugi 判断ログ（decisions.md）

[2026-06-15] プロダクト名: Tsugi (継ぎ — GPU ベンダーを接合する移植検証層): ソクラテス問答で「多様性・おまかせ」を多層構成と解釈。
[2026-06-15] 中核IR=MLIR / NVIDIA経路=PTX / SPIR-V=フォールバック限定: ADR-001。SPIR-VはNVIDIAでCUDA計算first-classでない。
[2026-06-15] バイナリCUDA変換=不採用: ADR-002。NVIDIA EULA・ZLUDA撤回前例。法的リスク回避最優先。
[2026-06-15] 楔=torch.compileバックエンド優先: ADR-003。API競合せずPyTorchエコシステム継承。
[2026-06-15] 行列コア抽象=MLIR intrinsic経路・Vulkan coopmat非依存: ADR-004。最重要リサーチベット(R1)。
[2026-06-15] スコープ=NVIDIA+AMD・カーネル4系統・推論優先: ソロでCUDA全面複製は非現実的。縦に狭く深く。
[2026-06-15] ライセンス=Apache-2.0: 依存(LLVM/IREE)がApache-2.0系のため整合。
[2026-06-15] 環境制約: 開発sandboxにLLVM/MLIR/torch/GPU無し。Phase0(完成形ファイル)完納。Phase1は構造骨格まで。実GPU検証は要実機(主張と実装の一致)。
[2026-06-15] リファレンス実装(CPU/NumPy)を先行実装: GPU無し環境でも「正しさの真値」を確立。OpenCL失敗の解毒剤=リファレンス先行(リサーチ由来)。GPU codegenはこれと一致させる。8テストPASS。
[2026-06-15] tracer + lowering plan 実装: @tsugi.jit→tsugi.tile IR(具体トレース)→各社intrinsic写像。MLIR風テキスト出力。frontend→IR→vendor-map が本物に。14テストPASS。GPU codegen本体(MLIR実コンパイル)のみ実機残作業。
[2026-06-15] /goal 実装完了 → CPU検証可能スコープを完成締め。tsugi.compile(上流統合dry-run)・verify.py(10不変条件)追加。17テストPASS。GPU codegen本体はCOMPLETE宣言しない(実機未検証・主張と実装の一致)。出荷チェックリストはGPU項目がblockedのため全PASS不可→COMPLETE保留。
[2026-06-15] ソクラテス問答で新視点発見: Tsugiの最強の楔はcodegenでなく「クロスベンダー検証層」。Tritonは両ベンダー生成するが数値等価性を保証しない。堀の本質=ライブラリ+QA(SemiAnalysis: AMDの弱点はQA文化)。portability.py(移植リスク静的解析・GPU不要)を即実装。同一カーネルblock=32がNVIDIA OK/AMD WARNを検出。21テストPASS。→ GPU codegen完成を待たず最初の実用価値が出る。
[2026-06-15] 新視点を深化: equivalence層(数値等価性判定)+portability累積順序ルール+portcheck CLI実装。擬似ベンダー(f32 vs f16累積)でmax_abs=3.9e-2の発散をDIVERGENT検出を実証。25テストPASS/verify12。GPU codegen未完でもportcheck CLIが今すぐ実用価値を出す。
[2026-06-15] 検証層さらに深化: occupancy推定(ベンダー別占有率・同一構成m64n64k32s3w4でNVIDIA25%/AMD CDNA20%差)+portability占有率ルール+portcheckユーザーカーネル読込実装。examples/user_kernel.py(block48)で両ベンダーWARN検出を実証。33テストPASS/verify13。
[2026-06-15] /loop: occupancy HW定数を推測値→一次情報源実値に置換(NVIDIA Hopper Tuning Guide / AMD ROCm gpu-arch-specs / GPUOpen)。docs/SOURCES.md作成。実値でNVIDIA 50%(reg制約)/AMD CDNA 25%(LDS制約)。33テストPASS維持。ループ収束判定: 残候補(バンクコンフリクト検出)は現IR粒度で低シグナル=サイズ増のみ→不採用(ハーネス: サイズ増のためだけの改善はしない)。
[2026-06-15] ソクラテス問答2で新視点: equivalenceの固定許容(1e-2)が恣意的。両ベンダーは累積順序違いで両方IEEE正当→「真値に一致」でなく「数学が許す範囲」。tolerance.py(√K·u·safety導出)+compare_gemm実装。K=2048で固定1e-2が正当なf32順序差を過剰検出していたのを解消。39テストPASS/verify14。
[2026-06-15] /loop 2反復で収束: (1)bf16忠実丸め実装(従来bf16→f32マップで精度損失を無視していた弱点修正・round_to_bf16でround-to-nearest-even・tolerance u=2^-8と整合) (2)portcheckに導出許容目安を統合(検証層を1レポートに)。45テストPASS/verify14。次候補は実機noise_floor実測=CPU不可→収束。
[2026-06-15] ソクラテス問答3で新視点: occupancy/portabilityの前提を問い、occ=0%(起動不能)を「性能WARN」と誤分類していた盲点を発見。「占有率が低い(遅い)」と「そもそも起動できない(動かない)」は連続と離散で別物。per-block上限(smem/LDS NVIDIA227KB vs AMD64KiB等)を越える構成はlaunchすらしない=単一ソース約束の破綻。feasibility.py(起動可否のcategoricalゲート)実装+portability.analyzeを修正(起動不能はWARNでなくBLOCK)+portcheckに起動可能性セクション。同一構成m128n128k64s4w8がNVIDIA起動可/AMD起動不能を実証。検証順序=feasibility(動くか)→equivalence(正しいか)→occupancy(速いか)。51テストPASS/verify17。docs/PERSPECTIVE-launch-feasibility.md。
