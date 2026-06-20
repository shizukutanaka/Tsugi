"""tsugi.provenance — 検証結果を環境フィンガープリントに束ねる（証明書の陳腐化検出）。

全ての verdict は point-in-time —— 特定の SW/HW スタック（ROCm/CUDA/driver/compiler/
dtype/numpy）で計算される。スタックが変われば（driver 更新・library 回帰・compiler 差）
検証済み等価は *silent に無効化* されうる。「一度認証＝永遠に有効」は誤り。

verdict を環境フィンガープリントに束ね、現在の環境が認証時と違えば **stale**（再検証要）と
判定する。実機では cuda/rocm/driver/compiler のバージョンを extra で渡してフィンガープリントに
含める（本環境では python/numpy/platform を捕捉・GPU フィールドは呼び出し側が供給）。
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field


def env_fingerprint(**extra: str) -> dict:
    """現在の環境フィンガープリント。extra に cuda/rocm/driver 版などを渡せる。"""
    import numpy as np
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    env.update({k: str(v) for k, v in extra.items()})
    return env


def fingerprint_hash(env: dict) -> str:
    """環境 dict の安定ハッシュ（順序非依存）。"""
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Certificate:
    """verdict ＋ それが計算された環境フィンガープリント。"""

    verdict: str
    env: dict = field(default_factory=dict)
    digest: str = ""

    def to_text(self) -> str:
        return (f"certificate[{self.verdict}] @ {self.digest} "
                f"({self.env.get('platform', '?')}, numpy {self.env.get('numpy', '?')})")


def certify(verdict: str, **extra: str) -> Certificate:
    """verdict を現在の環境フィンガープリントに束ねた証明書を発行する。"""
    env = env_fingerprint(**extra)
    return Certificate(verdict=verdict, env=env, digest=fingerprint_hash(env))


def is_stale(cert: Certificate, **extra: str) -> bool:
    """現在の環境が認証時と違えば stale（= 再検証が必要）。"""
    return fingerprint_hash(env_fingerprint(**extra)) != cert.digest


def changed_fields(cert: Certificate, **extra: str) -> dict:
    """認証時から変わったフィールドを {field: (old, new)} で返す（再検証の根拠を明示）。"""
    now = env_fingerprint(**extra)
    diff = {}
    for k in set(cert.env) | set(now):
        old, new = cert.env.get(k), now.get(k)
        if old != new:
            diff[k] = (old, new)
    return diff
