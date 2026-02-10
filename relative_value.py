"""
Relative Value Analysis
========================
Z-Spread calculation and rich/cheap screening for fixed income portfolios.

Z-Spread:   Constant spread over the zero curve that reprices a bond to its
             observed market price.  Solved via Brent root-finding.

Rich/Cheap:  Residual from a fitted Z-spread vs duration regression.
             Positive residual  -> CHEAP (trades wider than model).
             Negative residual  -> RICH  (trades tighter than model).

    The regression can be run:
      - On the full portfolio (single model)
      - Per rating bucket (IG vs HY) for more accurate peers comparison

Day Count Convention:
    Years-to-maturity uses ACT/365.25 for duration-like metrics.
"""

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod
from pricing import price_bond, bond_risk_metrics, _resolve_curve


def compute_z_spread(
    curve: YieldCurveAnalyzer,
    settlement_date: date,
    maturity_date: date,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
    clean_price_pct: float,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
) -> float:
    """
    Compute Z-spread in basis points.

    clean_price_pct is the market clean price as a percentage of par
    (e.g. 101.5 means 101.5 % of face value).

    Search range: -1000bps to +5000bps to handle both premium bonds
    (negative spread) and deeply distressed credits (HY > 3000bps).
    """
    # Get accrued interest (independent of spread)
    _, _, _, accrued = price_bond(
        curve, settlement_date, maturity_date,
        coupon_rate, coupon_freq, notional,
        spread_bps=0.0, method=method,
    )
    target_pv = notional * clean_price_pct / 100.0 + accrued

    def objective(spread_bps: float) -> float:
        pv, _, _, _ = price_bond(
            curve, settlement_date, maturity_date,
            coupon_rate, coupon_freq, notional,
            spread_bps=spread_bps, method=method,
        )
        return pv - target_pv

    try:
        return brentq(objective, -1000, 5000, xtol=0.01, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")


def portfolio_z_spreads(
    positions_df: pd.DataFrame,
    curve_map: dict,
    valuation_date: date,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    default_curve_id: str | None = None,
    by_rating_bucket: bool = True,
) -> pd.DataFrame:
    """
    Compute Z-spread and duration for every bond that has a market price.

    If by_rating_bucket is True, the rich/cheap regression is fitted
    separately for sovereign and corporate bonds, giving more accurate
    peer-relative signals.
    """
    rows = []
    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        curve = _resolve_curve(curve_map, curve_id, default_curve_id)

        settlement = getattr(row, "settlement_date", valuation_date)
        if pd.isna(settlement):
            settlement = valuation_date

        spread = float(getattr(row, "spread_bps", 0.0))

        # Risk metrics using modified duration (better for credit comparisons)
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

        market_price = getattr(row, "clean_price", None)
        z_spread = float("nan")
        if market_price is not None and not pd.isna(market_price):
            z_spread = compute_z_spread(
                curve=curve,
                settlement_date=settlement,
                maturity_date=getattr(row, "maturity_date"),
                coupon_rate=float(getattr(row, "coupon_rate")),
                coupon_freq=int(getattr(row, "coupon_freq")),
                notional=float(getattr(row, "notional")),
                clean_price_pct=float(market_price),
                method=method,
            )

        maturity_ts = pd.Timestamp(getattr(row, "maturity_date"))
        settle_ts = pd.Timestamp(settlement)
        years_to_mat = max((maturity_ts - settle_ts).days / 365.25, 0.0)

        rows.append(
            {
                "id": getattr(row, "id"),
                "issuer": getattr(row, "issuer"),
                "type": getattr(row, "type"),
                "duration": metrics.macaulay_duration,
                "years_to_maturity": years_to_mat,
                "model_spread_bps": spread,
                "z_spread_bps": z_spread,
                "clean_price": market_price if market_price is not None else float("nan"),
                "pv": metrics.pv,
            }
        )

    df = pd.DataFrame(rows)

    if by_rating_bucket:
        _fit_rich_cheap_by_bucket(df)
    else:
        _fit_rich_cheap_all(df)

    return df


def _fit_rich_cheap_all(df: pd.DataFrame) -> None:
    """Fit a single regression across all bonds."""
    valid = df.dropna(subset=["z_spread_bps", "duration"])
    if len(valid) >= 2:
        coeffs = np.polyfit(valid["duration"].values, valid["z_spread_bps"].values, 1)
        df["fitted_spread"] = coeffs[0] * df["duration"] + coeffs[1]
        df["residual_bps"] = df["z_spread_bps"] - df["fitted_spread"]
        _assign_signal(df)
    else:
        df["fitted_spread"] = float("nan")
        df["residual_bps"] = float("nan")
        df["signal"] = ""


def _fit_rich_cheap_by_bucket(df: pd.DataFrame) -> None:
    """Fit separate regressions per rating bucket (sovereign vs corporate)."""
    df["fitted_spread"] = float("nan")
    df["residual_bps"] = float("nan")
    df["signal"] = ""

    for bucket in df["type"].unique():
        mask = df["type"] == bucket
        subset = df.loc[mask]
        valid = subset.dropna(subset=["z_spread_bps", "duration"])

        if len(valid) >= 2:
            coeffs = np.polyfit(valid["duration"].values, valid["z_spread_bps"].values, 1)
            df.loc[mask, "fitted_spread"] = coeffs[0] * df.loc[mask, "duration"] + coeffs[1]
            df.loc[mask, "residual_bps"] = df.loc[mask, "z_spread_bps"] - df.loc[mask, "fitted_spread"]

    _assign_signal(df)


def _assign_signal(df: pd.DataFrame) -> None:
    """Assign RICH / CHEAP / FAIR signals based on residual."""
    df["signal"] = df["residual_bps"].apply(
        lambda x: "CHEAP" if x > 15 else ("RICH" if x < -15 else "FAIR")
        if not pd.isna(x)
        else ""
    )
