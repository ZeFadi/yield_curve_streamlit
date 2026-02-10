"""
DV01 Ladder / Bucketed Risk Exposure
=====================================
Maps portfolio DV01 into maturity buckets to show curve positioning.

PMs use this to compare their exposure profile against benchmarks and
identify concentration risk (e.g., "80% of my DV01 is in the 5-10Y bucket").

Key Rate Duration (KRD):
    Measures sensitivity to a +1bp shock at a *specific* curve tenor while
    holding all other tenors constant.  This is computed by bumping each
    key tenor individually and repricing the portfolio -- more expensive
    than simple bucketing, but essential for hedging decisions.

Day Count Convention:
    Years-to-maturity uses ACT/365.25 for bucketing purposes.
"""

from __future__ import annotations

from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd

from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod
from pricing import price_bond, bond_risk_metrics, _resolve_curve


DEFAULT_BUCKETS = [
    ("0-2Y", 0.0, 2.0),
    ("2-5Y", 2.0, 5.0),
    ("5-10Y", 5.0, 10.0),
    ("10-30Y", 10.0, 30.0),
    ("30Y+", 30.0, float("inf")),
]


def _years_to_maturity(settlement: date, maturity: date) -> float:
    """Calculate time to maturity in years (ACT/365.25)."""
    return max((pd.Timestamp(maturity) - pd.Timestamp(settlement)).days / 365.25, 0.0)


def compute_dv01_ladder(
    positions_df: pd.DataFrame,
    curve_map: dict,
    valuation_date: date,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    buckets: List[Tuple[str, float, float]] | None = None,
    default_curve_id: str | None = None,
) -> pd.DataFrame:
    """
    Compute DV01 ladder by maturity buckets.

    Returns a DataFrame with columns: bucket, dv01, pct_of_total, count.
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    bond_dv01s = []
    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        curve = _resolve_curve(curve_map, curve_id, default_curve_id)

        settlement = getattr(row, "settlement_date", valuation_date)
        if pd.isna(settlement):
            settlement = valuation_date

        spread = float(getattr(row, "spread_bps", 0.0))

        metrics = bond_risk_metrics(
            curve=curve,
            settlement_date=settlement,
            maturity_date=getattr(row, "maturity_date"),
            coupon_rate=float(getattr(row, "coupon_rate")),
            coupon_freq=int(getattr(row, "coupon_freq")),
            notional=float(getattr(row, "notional")),
            spread_bps=spread,
            method=method,
        )

        ytm = _years_to_maturity(settlement, getattr(row, "maturity_date"))

        bond_dv01s.append(
            {
                "id": getattr(row, "id"),
                "issuer": getattr(row, "issuer"),
                "type": getattr(row, "type"),
                "years_to_maturity": ytm,
                "dv01": metrics.dv01,
                "pv": metrics.pv,
            }
        )

    bonds = pd.DataFrame(bond_dv01s)
    if bonds.empty:
        return pd.DataFrame(columns=["bucket", "dv01", "pct_of_total", "count"])

    total_dv01 = bonds["dv01"].sum()
    ladder_rows = []

    for bucket_name, lower, upper in buckets:
        mask = (bonds["years_to_maturity"] >= lower) & (bonds["years_to_maturity"] < upper)
        bucket_dv01 = bonds.loc[mask, "dv01"].sum()
        bucket_count = mask.sum()
        pct = (bucket_dv01 / total_dv01 * 100) if total_dv01 != 0 else 0.0

        ladder_rows.append(
            {
                "bucket": bucket_name,
                "dv01": bucket_dv01,
                "pct_of_total": pct,
                "count": int(bucket_count),
            }
        )

    ladder_df = pd.DataFrame(ladder_rows)
    ladder_df["cumulative_pct"] = ladder_df["pct_of_total"].cumsum()

    return ladder_df


def _bump_curve_at_tenor(
    curve: YieldCurveAnalyzer,
    target_tenor: float,
    bump_bps: float = 1.0,
    width: float = 1.0,
) -> YieldCurveAnalyzer:
    """
    Create a new curve with a localized bump around *target_tenor*.

    The bump is triangular: full bump_bps at target_tenor, linearly
    decaying to zero at target_tenor +/- width.  This isolates the
    sensitivity to a specific part of the curve.

    Note: rates in YieldCurveAnalyzer are stored as percentages (e.g. 4.5
    means 4.5%), so we convert the bps bump to percentage points (1bp = 0.01%).
    """
    tenors = curve.tenors.copy()
    rates = curve.rates.copy()

    for i, t in enumerate(tenors):
        dist = abs(t - target_tenor)
        if dist < width:
            weight = 1.0 - dist / width
            rates[i] += bump_bps / 100.0 * weight  # 1bp = 0.01 percentage points

    bumped_data = pd.DataFrame({"tenor": tenors, "rate": rates})
    return YieldCurveAnalyzer(bumped_data, extrapolate=curve.extrapolate)


def portfolio_key_rate_durations(
    positions_df: pd.DataFrame,
    curve_map: dict,
    valuation_date: date,
    key_tenors: List[float] | None = None,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    default_curve_id: str | None = None,
) -> pd.DataFrame:
    """
    Compute true Key Rate Durations (KRD) via per-tenor curve bumps.

    For each key tenor, we bump the curve by +1bp at that tenor (with
    triangular decay over +-1Y width) and reprice the entire portfolio.
    KRD_k = (PV_base - PV_bumped_k) for a +1bp bump.

    This is the correct methodology for hedging: KRD tells you how much
    notional of each key-tenor instrument you need to offset your exposure.

    Returns DataFrame with columns: tenor, krd.
    """
    if key_tenors is None:
        key_tenors = [2.0, 5.0, 10.0, 30.0]

    # Width of the triangular bump: narrower for short tenors
    widths = {2.0: 1.0, 5.0: 1.5, 10.0: 2.0, 30.0: 5.0}

    # Base PV
    base_pvs = {}
    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        curve = _resolve_curve(curve_map, curve_id, default_curve_id)

        settlement = getattr(row, "settlement_date", valuation_date)
        if pd.isna(settlement):
            settlement = valuation_date

        spread = float(getattr(row, "spread_bps", 0.0))

        pv, _, _, _ = price_bond(
            curve, settlement,
            getattr(row, "maturity_date"),
            float(getattr(row, "coupon_rate")),
            int(getattr(row, "coupon_freq")),
            float(getattr(row, "notional")),
            spread_bps=spread, method=method,
        )
        bond_id = getattr(row, "id")
        base_pvs[bond_id] = {
            "pv": pv,
            "curve_id": curve_id,
            "settlement": settlement,
            "maturity_date": getattr(row, "maturity_date"),
            "coupon_rate": float(getattr(row, "coupon_rate")),
            "coupon_freq": int(getattr(row, "coupon_freq")),
            "notional": float(getattr(row, "notional")),
            "spread_bps": spread,
        }

    total_base_pv = sum(v["pv"] for v in base_pvs.values())

    # Bump each key tenor and reprice
    krd_results = []
    for tenor in key_tenors:
        width = widths.get(tenor, 2.0)

        # Build bumped curve for each curve_id
        bumped_curves = {}
        for curve_id, curve in curve_map.items():
            bumped_curves[curve_id] = _bump_curve_at_tenor(curve, tenor, bump_bps=1.0, width=width)

        # Reprice portfolio with bumped curves
        bumped_total_pv = 0.0
        for bond_id, info in base_pvs.items():
            b_curve_id = info["curve_id"]
            b_curve = bumped_curves.get(b_curve_id, curve_map.get(b_curve_id))
            if b_curve is None:
                bumped_total_pv += info["pv"]
                continue

            pv_bumped, _, _, _ = price_bond(
                b_curve, info["settlement"],
                info["maturity_date"],
                info["coupon_rate"],
                info["coupon_freq"],
                info["notional"],
                spread_bps=info["spread_bps"],
                method=method,
            )
            bumped_total_pv += pv_bumped

        # KRD = PV change for +1bp bump at this tenor
        krd = total_base_pv - bumped_total_pv
        krd_results.append({"tenor": tenor, "krd": krd})

    return pd.DataFrame(krd_results)
