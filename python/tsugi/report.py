"""Tsugi report — 検証層が共有する深刻度モデルと所見レポートの基底。

8 つの検証層（portability/equivalence/occupancy/tolerance/feasibility/
propagation/envelope）はそれぞれ独自の Report クラスを持ち、深刻度の尺度・
所見リスト・max_risk/to_text の定型を別々に再実装していた。本モジュールは
その共通核（Risk と Finding と FindingReport 基底）を一箇所に集約し、検証層
横断で深刻度・出力形式を統一する。新しい検証層もここを継承するだけでよい。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Risk(IntEnum):
    """検証層で共通の深刻度（高いほど移植/正しさへの脅威が大きい）。"""

    OK = 0
    INFO = 1
    WARN = 2
    BLOCK = 3  # これ以上は「移植/実行を止めるべき」境界


@dataclass
class Finding:
    """単一の所見。op は対象（op 名・"tensor"・"softmax" 等）。"""

    risk: Risk
    op: str
    message: str


@dataclass
class FindingReport:
    """Finding のリストを持つレポートの基底（max_risk/ok/to_text を提供）。

    各検証層はこれを継承し、固有のヘッダ文言だけを与える。深刻度集約・
    BLOCK 境界・整形ロジックの重複をここに吸収する。
    """

    findings: list[Finding] = field(default_factory=list)

    def add(self, risk: Risk, op: str, message: str) -> None:
        self.findings.append(Finding(risk, op, message))

    @property
    def max_risk(self) -> Risk:
        return max((f.risk for f in self.findings), default=Risk.OK)

    @property
    def ok(self) -> bool:
        """BLOCK 未満なら（注意点はあっても）通過とみなす。"""
        return self.max_risk < Risk.BLOCK

    def to_text(self, header: str = "report",
                empty: str = "(no findings)") -> str:
        lines = [f"{header} (max_risk={self.max_risk.name})"]
        for f in self.findings:
            lines.append(f"  [{f.risk.name:5s}] {f.op:7s} {f.message}")
        if not self.findings:
            lines.append(f"  {empty}")
        return "\n".join(lines)
