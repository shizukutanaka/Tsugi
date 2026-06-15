"""Tsugi data types and compile-time markers.

リファレンス実装（CPU/NumPy）と GPU codegen の両方が共有する型定義。
"""
from __future__ import annotations

import numpy as np


def round_to_bf16(x: np.ndarray) -> np.ndarray:
    """f32 配列を bfloat16 精度（仮数 7bit）へ round-to-nearest-even で丸める。

    NumPy に bf16 が無いため、f32 のまま下位 16bit を丸めて bf16 表現可能値にする。
    これで oracle が bf16 の精度損失を *忠実に* 再現する（tolerance の u=2^-8 と整合）。
    """
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    # round to nearest even: 下位 16bit を四捨五入してから切り捨て
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    rounded = ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)
    return rounded.copy()


class DType:
    """Tsugi dtype。NumPy dtype への写像と忠実な丸めを持つ。"""

    __slots__ = ("name", "np")

    def __init__(self, name: str, np_dtype: type) -> None:
        self.name = name
        self.np = np_dtype

    def round(self, x: np.ndarray) -> np.ndarray:
        """この dtype の精度へ忠実に丸める。bf16 は専用丸め、他は astype。"""
        if self.name == "bfloat16":
            return round_to_bf16(x)
        return np.asarray(x).astype(self.np)

    def __repr__(self) -> str:
        return f"tsugi.{self.name}"


float16 = DType("float16", np.float16)
bfloat16 = DType("bfloat16", np.float32)  # NumPy に bf16 無し → f32 で近似（リファレンス）
float32 = DType("float32", np.float32)
int32 = DType("int32", np.int32)
int8 = DType("int8", np.int8)


class constexpr:  # noqa: N801 — DSL 慣習（Triton 互換の見た目）
    """コンパイル時定数マーカー。autotune 対象。

    型注釈として使う: ``BLOCK_M: tsugi.constexpr``。
    リファレンス実行時は通常の int として渡される。
    """

    def __class_getitem__(cls, item):  # 型注釈で使えるように
        return cls
