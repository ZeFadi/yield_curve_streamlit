from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd

from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod


def _resolve_curve(curve_map: dict, curve_id: str | None, default_curve_id: str | None) -> YieldCurveAnalyzer:
    curve = curve_map.get(curve_id)
    if curve is not None:
        return curve
    if default_curve_id is not None:
        curve = curve_map.get(default_curve_id)
        if curve is not None:
            return curve
    available = ", ".join(sorted(map(str, curve_map.keys())))
    raise ValueError(
        f"No curve found for curve_id='{curve_id}' and default_curve_id='{default_curve_id}'. "
        f"Available curve IDs: [{available}]"
    )


@dataclass(frozen=True)
class BondPricingResult:
    pv: float
    dirty_price: float
    clean_price: float
    accrued_interest: float
    macaulay_duration: float
    convexity: float
    dv01: float
    cs01: float


def year_fraction_act365(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).days / 365.0


def _generate_cashflows(
    settlement_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp | None, pd.Timestamp | None]:
    months = int(12 / coupon_freq)
    pay_dates = []
    current = maturity_date
    while current > settlement_date:
        pay_dates.append(current)
        current = current - pd.DateOffset(months=months)

    pay_dates = sorted(pay_dates)
    last_coupon = current if current != maturity_date else None
    next_coupon = pay_dates[0] if pay_dates else None

    coupon_amount = notional * (coupon_rate / 100.0) / coupon_freq
    cashflows = np.full(len(pay_dates), coupon_amount)
    if cashflows.size > 0:
        cashflows[-1] += notional

    times = np.array(
        [year_fraction_act365(settlement_date, pd.Timestamp(d)) for d in pay_dates],
        dtype=float,
    )
    return times, cashflows, last_coupon, next_coupon


def _accrued_interest(
    settlement_date: pd.Timestamp,
    last_coupon: pd.Timestamp | None,
    next_coupon: pd.Timestamp | None,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
) -> float:
    if last_coupon is None or next_coupon is None:
        return 0.0
    period_days = (next_coupon - last_coupon).days
    if period_days <= 0:
        return 0.0
    accrued_days = (settlement_date - last_coupon).days
    if accrued_days < 0:
        return 0.0
    coupon_amount = notional * (coupon_rate / 100.0) / coupon_freq
    return coupon_amount * (accrued_days / period_days)


def price_bond(
    curve: YieldCurveAnalyzer,
    settlement_date: date,
    maturity_date: date,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
    spread_bps: float = 0.0,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    curve_bump_bps: float = 0.0,
    spread_bump_bps: float = 0.0,
) -> Tuple[float, float, float, float]:
    settlement_ts = pd.Timestamp(settlement_date)
    maturity_ts = pd.Timestamp(maturity_date)

    times, cashflows, last_coupon, next_coupon = _generate_cashflows(
        settlement_ts, maturity_ts, coupon_rate, coupon_freq, notional
    )
    if times.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    zero_rates = curve.get_zero_rate(times, method=method)
    total_bps = curve_bump_bps + spread_bps + spread_bump_bps
    discount_rate = (zero_rates + total_bps / 100.0) / 100.0
    dfs = np.exp(-discount_rate * times)
    pv = float(np.sum(cashflows * dfs))

    accrued = _accrued_interest(
        settlement_ts, last_coupon, next_coupon, coupon_rate, coupon_freq, notional
    )
    dirty_price = pv
    clean_price = pv - accrued
    return pv, dirty_price, clean_price, accrued


def bond_risk_metrics(
    curve: YieldCurveAnalyzer,
    settlement_date: date,
    maturity_date: date,
    coupon_rate: float,
    coupon_freq: int,
    notional: float,
    spread_bps: float = 0.0,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    bump_bps: float = 1.0,
) -> BondPricingResult:
    pv, dirty, clean, accrued = price_bond(
        curve,
        settlement_date,
        maturity_date,
        coupon_rate,
        coupon_freq,
        notional,
        spread_bps=spread_bps,
        method=method,
    )

    pv_up, _, _, _ = price_bond(
        curve,
        settlement_date,
        maturity_date,
        coupon_rate,
        coupon_freq,
        notional,
        spread_bps=spread_bps,
        method=method,
        curve_bump_bps=+bump_bps,
    )
    pv_down, _, _, _ = price_bond(
        curve,
        settlement_date,
        maturity_date,
        coupon_rate,
        coupon_freq,
        notional,
        spread_bps=spread_bps,
        method=method,
        curve_bump_bps=-bump_bps,
    )

    dv01 = (pv_down - pv_up) / 2.0
    duration = 0.0
    if pv != 0:
        duration = dv01 / (pv * (bump_bps / 10000.0))

    convexity = 0.0
    if pv != 0:
        delta = bump_bps / 10000.0
        convexity = (pv_up + pv_down - 2 * pv) / (pv * delta * delta)

    pv_spread_up, _, _, _ = price_bond(
        curve,
        settlement_date,
        maturity_date,
        coupon_rate,
        coupon_freq,
        notional,
        spread_bps=spread_bps,
        method=method,
        spread_bump_bps=+bump_bps,
    )
    pv_spread_down, _, _, _ = price_bond(
        curve,
        settlement_date,
        maturity_date,
        coupon_rate,
        coupon_freq,
        notional,
        spread_bps=spread_bps,
        method=method,
        spread_bump_bps=-bump_bps,
    )
    cs01 = (pv_spread_down - pv_spread_up) / 2.0

    return BondPricingResult(
        pv=pv,
        dirty_price=dirty,
        clean_price=clean,
        accrued_interest=accrued,
        macaulay_duration=duration,
        convexity=convexity,
        dv01=dv01,
        cs01=cs01,
    )


def price_portfolio_df(
    positions_df: pd.DataFrame,
    curve_map: dict,
    valuation_date: date,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    spread_shock_bps: float = 0.0,
    default_curve_id: str | None = None,
) -> Tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    rows = []
    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        if curve_id not in curve_map:
            warnings.append(f"Missing curve '{curve_id}', using '{default_curve_id}'.")
        curve = _resolve_curve(curve_map, curve_id, default_curve_id)

        settlement = getattr(row, "settlement_date", valuation_date)
        if pd.isna(settlement):
            settlement = valuation_date

        spread = float(getattr(row, "spread_bps", 0.0))
        if getattr(row, "type", "corporate") == "corporate":
            spread += spread_shock_bps

        result = bond_risk_metrics(
            curve=curve,
            settlement_date=settlement,
            maturity_date=getattr(row, "maturity_date"),
            coupon_rate=float(getattr(row, "coupon_rate")),
            coupon_freq=int(getattr(row, "coupon_freq")),
            notional=float(getattr(row, "notional")),
            spread_bps=spread,
            method=method,
        )

        rows.append(
            {
                "id": getattr(row, "id"),
                "issuer": getattr(row, "issuer"),
                "type": getattr(row, "type"),
                "currency": getattr(row, "currency"),
                "curve_id": curve_id,
                "notional": float(getattr(row, "notional")),
                "coupon_rate": float(getattr(row, "coupon_rate")),
                "coupon_freq": int(getattr(row, "coupon_freq")),
                "maturity_date": getattr(row, "maturity_date"),
                "spread_bps": float(getattr(row, "spread_bps", 0.0)),
                "pv": result.pv,
                "dirty_price": result.dirty_price,
                "clean_price": result.clean_price,
                "accrued_interest": result.accrued_interest,
                "duration": result.macaulay_duration,
                "convexity": result.convexity,
                "dv01": result.dv01,
                "cs01": result.cs01,
            }
        )

    return pd.DataFrame(rows), warnings


def portfolio_pv(
    positions_df: pd.DataFrame,
    curve_map: dict,
    valuation_date: date,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    spread_shock_bps: float = 0.0,
    default_curve_id: str | None = None,
) -> float:
    total = 0.0
    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        curve = _resolve_curve(curve_map, curve_id, default_curve_id)

        settlement = getattr(row, "settlement_date", valuation_date)
        if pd.isna(settlement):
            settlement = valuation_date

        spread = float(getattr(row, "spread_bps", 0.0))
        if getattr(row, "type", "corporate") == "corporate":
            spread += spread_shock_bps

        pv, _, _, _ = price_bond(
            curve=curve,
            settlement_date=settlement,
            maturity_date=getattr(row, "maturity_date"),
            coupon_rate=float(getattr(row, "coupon_rate")),
            coupon_freq=int(getattr(row, "coupon_freq")),
            notional=float(getattr(row, "notional")),
            spread_bps=spread,
            method=method,
        )
        total += pv
    return total
