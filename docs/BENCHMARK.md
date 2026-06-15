# Tsugi Benchmark 仕様

> 計測しなければ「正しいものだったか」は分からない（アウトカム視点）。
> 本書は計測対象・比較対象・閾値を事前定義する。実測値は Phase 2-4 で追記。

---

## 1. 計測哲学

- correctness を性能より先に確認（数値正答が先・速度は後）
- p99 を測ってから最適化（早すぎる最適化を避ける）
- 比較対象を明示（Triton / cuBLAS / rocBLAS）・条件を固定
- 未測定は「未測定」と書く（捏造しない・主張と実装の一致）

---

## 2. 計測環境（固定）

| 項目 | NVIDIA | AMD |
|------|--------|-----|
| GPU | （実機確定後記載・例 A100/H100/RTX4090） | （例 MI250/MI300X/RX7900XTX） |
| バックエンド経路 | Tsugi→PTX→ptxas→SASS | Tsugi→AMDGCN→lld |
| 比較ライブラリ | cuBLAS / Triton | rocBLAS / Triton |
| 精度 | FP16/BF16（accum FP32） | 同 |

---

## 3. 計測対象 shape（固定・transformer由来）

### GEMM
| 名称 | M | N | K | 用途 |
|------|---|---|---|------|
| gemm-square-4k | 4096 | 4096 | 4096 | 基準 |
| gemm-llm-qkv | 4096 | 12288 | 4096 | QKV projection |
| gemm-llm-ffn | 4096 | 11008 | 4096 | FFN up |
| gemm-tall | 8192 | 1024 | 4096 | 縦長 |

### Attention（fused）
| 名称 | batch | heads | seq | head_dim |
|------|-------|-------|-----|----------|
| attn-llama-2k | 1 | 32 | 2048 | 128 |
| attn-llama-8k | 1 | 32 | 8192 | 128 |

### Norm / Elementwise
| 名称 | 形状 |
|------|------|
| rmsnorm-4k | (4096, 4096) |
| gelu-fusion | (4096, 11008) |

---

## 4. 比較指標と閾値（DoD）

| 指標 | 閾値 | Phase |
|------|------|-------|
| correctness（両ベンダー数値一致） | max abs error < 1e-2 (FP16) | 1 |
| GEMM vs cuBLAS/rocBLAS | 達成 TFLOPS が 70-80% 以上 | 2 |
| カーネル4系統 vs Triton | レイテンシ 1.2x 以内 | 3 |
| ランタイム dispatch overhead | < 1% | 4 |
| e2e transformer（実モデル）| 両ベンダーで正答・動作 | 4 |

**Phase戻り条件**: GEMM が cuBLAS 比 70% に Phase2 で届かない → スコープを attention-only に縮小、または行列コア抽象を再設計（ADR-004）。

---

## 5. 計測方法

```bash
# correctness（両ベンダー比較）
python tests/correctness/run.py --vendor nvidia --vendor amd --tol 1e-2

# 性能（warmup後 p50/p99）
python tests/perf/bench.py --kernel gemm --shapes config/shapes.yaml \
    --baseline cublas,triton --iters 100 --warmup 20
```

- warmup 20 回 → 計測 100 回 → p50/p99 報告
- スパイクは外れ値除外しない（p99 をそのまま報告）
- autotune 後のベスト構成で比較

---

## 6. 結果テンプレート（Phase完了時に追記）

```
## v0.1 GEMM 結果（YYYY-MM-DD）
| shape | Tsugi TFLOPS | cuBLAS | 比率 | Triton | 比率 |
|-------|-------------|--------|------|--------|------|
| gemm-square-4k | (未測定) | | | | |
```

---

## 7. 戦略を変えるベンチマーク（リサーチ由来）

- KHR cooperative matrix が3社 compute 経路で均一・高性能化 → Vulkan/SPIR-V 経路を再評価（行列コア lowering を切替検討）
- AMD ROCm+IREE が訓練で CUDA パリティ達成 → 訓練スコープ前倒し
- これらは BENCHMARK で監視し、decisions.md に判断を記録
