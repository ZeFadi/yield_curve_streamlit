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
    shocked = curve_df.copy()
    tenors = shocked["tenor"].astype(float).values
    bumps = np.zeros_like(tenors, dtype=float)
    for key_tenor, bps in key_bps.items():
        idx = int(np.argmin(np.abs(tenors - float(key_tenor))))
        bumps[idx] += bps / 100.0
    shocked["rate"] = shocked["rate"] + bumps
    return shocked

