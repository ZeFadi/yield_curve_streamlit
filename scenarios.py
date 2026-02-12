from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    portfolio_pv: float
    pnl: float


def apply_parallel_shock(curve_df: pd.DataFrame, bps: float) -> pd.DataFrame:
    shocked = curve_df.copy()
    shocked["rate"] = shocked["rate"] + bps / 100.0
    return shocked


def apply_twist_shock(curve_df: pd.DataFrame, short_bps: float, long_bps: float) -> pd.DataFrame:
    shocked = curve_df.copy()
    tenors = shocked["tenor"].astype(float).values
    t_min, t_max = float(np.min(tenors)), float(np.max(tenors))
    if t_max == t_min:
        shocked["rate"] = shocked["rate"] + short_bps / 100.0
        return shocked
    weights = (tenors - t_min) / (t_max - t_min)
    bps = short_bps + (long_bps - short_bps) * weights
    shocked["rate"] = shocked["rate"] + bps / 100.0
    return shocked


def apply_key_rate_shock(curve_df: pd.DataFrame, key_bps: Dict[float, float]) -> pd.DataFrame:
    """Apply key rate shocks using triangular (tent) interpolation.

    Each shock is centred on the target tenor and decays linearly to zero
    over a width that scales with tenor (matching risk_ladder.py KRD methodology).
    """
    _widths = {2.0: 1.0, 5.0: 1.5, 10.0: 2.0, 30.0: 5.0}
    shocked = curve_df.copy()
    tenors = shocked["tenor"].astype(float).values
    bumps = np.zeros_like(tenors, dtype=float)
    for key_tenor, bps in key_bps.items():
        width = _widths.get(float(key_tenor), max(1.0, float(key_tenor) * 0.3))
        for i, t in enumerate(tenors):
            dist = abs(t - float(key_tenor))
            if dist < width:
                weight = 1.0 - dist / width
                bumps[i] += bps / 100.0 * weight
    shocked["rate"] = shocked["rate"] + bumps
    return shocked

