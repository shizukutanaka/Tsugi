"""Tsugi TorchInductor backend (楔の本体・ADR-003).

``torch.compile(model, backend="tsugi")`` で Tsugi 経由のカーネル生成を行う。
開発者は torch を叩くだけで NVIDIA/AMD 両対応になる。

状態: 骨格（skeleton）。実 lowering は Phase 4 で実装。
       実 GPU 検証は NVIDIA・AMD 実機が必要。未検証の経路は「未検証」と明記する。
"""
from __future__ import annotations

from typing import Any, Callable, List


def _tsugi_compile(gm: Any, example_inputs: List[Any]) -> Callable:
    """TorchDynamo から FX GraphModule を受け取り、Tsugi カーネルへ変換する。

    Phase 4 の実装計画:
      1. TorchInductor の lowering を再利用して融合機会を取得
      2. hot op (matmul/attention/norm/elementwise) を tsugi.tile IR へ変換
      3. SPEC.md §3 パイプラインでコンパイル (tsugi.tile→gpu→NVVM/ROCDL)
      4. 標準 GEMM 等は cuBLAS/rocBLAS へ escape-hatch (性能優先・R5)
      5. 残りは torch eager にフォールバック (正しさ優先)

    現状: 未実装。eager 実行へ素通し（嘘をつかない）。
    """
    # TODO(Phase4): tsugi.tile IR への変換と vendor lowering を実装。
    def _forward(*args: Any, **kwargs: Any) -> Any:
        # 未実装のため eager に委譲。性能利得なし（明示）。
        return gm.forward(*args, **kwargs)

    return _forward


def register() -> None:
    """backend="tsugi" を torch に登録する。"""
    try:
        from torch._dynamo import register_backend
    except ImportError as exc:  # torch 未導入環境
        raise RuntimeError(
            "Tsugi torch backend requires PyTorch with TorchDynamo"
        ) from exc

    register_backend(name="tsugi", compiler_fn=_tsugi_compile)


# import 時に自動登録（torch があれば）
try:
    register()
except Exception:  # noqa: BLE001 — torch 無し環境では沈黙（ライブラリとして壊さない）
    pass
