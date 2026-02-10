"""
Carry & Roll-Down Analysis
===========================
Primary return attribution tool for fixed income portfolio managers.

Carry:     Income earned from holding the bond minus cost of financing,
           netted for accrued interest changes between settlement and horizon.
Roll-Down: Price appreciation from the bond aging along an unchanged yield curve,
           measured on a clean price basis to isolate the pure curve effect.
Total Horizon Return: Carry + Roll-Down (assuming no market moves).

Day Count Convention:
    This module uses ACT/365 for year fractions and average month length
    (30.44 days) for horizon date computation. These are simplifications;
    production systems should use bond-specific conventions (30/360,
    ACT/ACT ISDA, etc.).
"""

from datetime import date, timedelta

import pandas as pd

from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod
from pricing import price_bond, _resolve_curve


def _coupons_in_period(
    settlement_date: date,
    horizon_date: date,
    maturity_date: date,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
) -> float:
    """Sum of coupon payments received strictly within (settlement, horizon]."""
    settle_ts = pd.Timestamp(settlement_date)
    horizon_ts = pd.Timestamp(horizon_date)
    mat_ts = pd.Timestamp(maturity_date)

    months = int(12 / coupon_freq)
    coupon_amount = notional * (coupon_rate / 100.0) / coupon_freq

    pay_dates = []
    current = mat_ts
    while current > settle_ts:
        pay_dates.append(current)
        current = current - pd.DateOffset(months=months)

    total = 0.0
    for d in pay_dates:
        if settle_ts < d <= horizon_ts:
            total += coupon_amount
    return total


def compute_carry_rolldown(
    curve: YieldCurveAnalyzer,
    settlement_date: date,
    maturity_date: date,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
    spread_bps: float = 0.0,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    horizon_months: int = 3,
    funding_rate_pct: float = 0.0,
) -> dict:
    """
    Compute carry and roll-down for a single bond.

    Carry is calculated net of accrued interest changes:
        carry = coupons_received + (accrued_t1 - accrued_t0) - funding_cost

    Roll-down is measured on a clean price basis:
        rolldown = clean_price_t1 - clean_price_t0

    This avoids double-counting accrued interest in carry.

    Returns dict with absolute values and annualised bps metrics.
    """
    horizon_days = int(horizon_months * 30.44)  # average days per month
    horizon_date = settlement_date + timedelta(days=horizon_days)
    horizon_years = horizon_months / 12.0

    mat_date = (
        pd.Timestamp(maturity_date).date()
        if not isinstance(maturity_date, date)
        else maturity_date
    )

    # Current pricing: PV (dirty), clean, accrued at t0
    pv_now, dirty_t0, clean_t0, accrued_t0 = price_bond(
        curve, settlement_date, mat_date,
        coupon_rate, coupon_freq, notional,
        spread_bps=spread_bps, method=method,
    )

    if pv_now == 0:
        return {k: 0.0 for k in [
            "pv_now", "pv_horizon", "clean_t0", "clean_t1",
            "accrued_t0", "accrued_t1",
            "coupons_received", "funding_cost",
            "carry", "rolldown", "total_return",
            "carry_ann_bps", "rolldown_ann_bps", "total_return_ann_bps",
        ]}

    # Bond matures before horizon
    if horizon_date >= mat_date:
        coupons = _coupons_in_period(
            settlement_date, mat_date, mat_date,
            coupon_rate, coupon_freq, notional,
        )
        ttm = (pd.Timestamp(mat_date) - pd.Timestamp(settlement_date)).days / 365.0
        funding_cost = pv_now * (funding_rate_pct / 100.0) * ttm
        # At maturity: clean = par, accrued = 0
        carry = coupons - accrued_t0 + notional - pv_now - funding_cost
        return {
            "pv_now": pv_now,
            "pv_horizon": notional,
            "clean_t0": clean_t0,
            "clean_t1": notional,
            "accrued_t0": accrued_t0,
            "accrued_t1": 0.0,
            "coupons_received": coupons + notional,
            "funding_cost": funding_cost,
            "carry": carry,
            "rolldown": 0.0,
            "total_return": carry,
            "carry_ann_bps": (carry / pv_now / max(ttm, 1e-6) * 10000),
            "rolldown_ann_bps": 0.0,
            "total_return_ann_bps": (carry / pv_now / max(ttm, 1e-6) * 10000),
        }

    # Pricing at horizon: bond has aged, curve unchanged
    pv_horizon, dirty_t1, clean_t1, accrued_t1 = price_bond(
        curve, horizon_date, mat_date,
        coupon_rate, coupon_freq, notional,
        spread_bps=spread_bps, method=method,
    )

    # Coupons received during the period
    coupons = _coupons_in_period(
        settlement_date, horizon_date, mat_date,
        coupon_rate, coupon_freq, notional,
    )

    # Funding cost
    funding_cost = pv_now * (funding_rate_pct / 100.0) * horizon_years

    # ---- CARRY (net of accrued interest) ----
    # The investor pays accrued_t0 at settlement. Over the horizon they
    # receive coupons and build new accrued_t1. The *net* income is:
    #   net_income = coupons + accrued_t1 - accrued_t0
    # Carry = net_income - funding_cost
    carry = coupons + (accrued_t1 - accrued_t0) - funding_cost

    # ---- ROLL-DOWN (clean price basis) ----
    # Isolate pure curve-aging effect by comparing clean prices
    rolldown = clean_t1 - clean_t0

    total_return = carry + rolldown

    carry_ann = (carry / pv_now / horizon_years * 10000) if horizon_years > 0 else 0.0
    roll_ann = (rolldown / pv_now / horizon_years * 10000) if horizon_years > 0 else 0.0

    return {
        "pv_now": pv_now,
        "pv_horizon": pv_horizon,
        "clean_t0": clean_t0,
        "clean_t1": clean_t1,
        "accrued_t0": accrued_t0,
        "accrued_t1": accrued_t1,
        "coupons_received": coupons,
        "funding_cost": funding_cost,
        "carry": carry,
        "rolldown": rolldown,
        "total_return": total_return,
        "carry_ann_bps": carry_ann,
        "rolldown_ann_bps": roll_ann,
        "total_return_ann_bps": carry_ann + roll_ann,
    }


def portfolio_carry_rolldown(
    positions_df: pd.DataFrame,
    curve_map: dict,
    valuation_date: date,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    horizon_months: int = 3,
    funding_rate_pct: float = 0.0,
    default_curve_id: str | None = None,
) -> pd.DataFrame:
    """Compute carry and roll-down for every position in a portfolio."""
    rows = []
    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        curve = _resolve_curve(curve_map, curve_id, default_curve_id)

        settlement = getattr(row, "settlement_date", valuation_date)
        if pd.isna(settlement):
            settlement = valuation_date

        spread = float(getattr(row, "spread_bps", 0.0))

        result = compute_carry_rolldown(
            curve=curve,
            settlement_date=settlement,
            maturity_date=getattr(row, "maturity_date"),
            coupon_rate=float(getattr(row, "coupon_rate")),
            coupon_freq=int(getattr(row, "coupon_freq")),
            notional=float(getattr(row, "notional")),
            spread_bps=spread,
            method=method,
            horizon_months=horizon_months,
            funding_rate_pct=funding_rate_pct,
        )
        result["id"] = getattr(row, "id")
        result["issuer"] = getattr(row, "issuer")
        result["type"] = getattr(row, "type")
        result["notional"] = float(getattr(row, "notional"))
        rows.append(result)

    return pd.DataFrame(rows)
