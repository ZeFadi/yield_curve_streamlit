"""
P&L Attribution
===============
Decompose period-over-period portfolio P&L into:
  - Carry (coupon income net of accrued changes)
  - Roll-down (price appreciation from aging)
  - Rate moves (base curve shifts)
  - Spread moves (credit spread changes)
  - Residual (cross-effects, unexplained)

This is what PMs present to their CIO / risk committees.

Methodology:
    1. PV(t0, curve_t0, spread_t0) = starting value
    2. PV(t1, curve_t0, spread_t0) = roll-down value (aging, no market move)
    3. PV(t1, curve_t1, spread_t0) = rate-move value (curves changed, spreads didn't)
    4. PV(t1, curve_t1, spread_t1) = ending value (all effects)

    Carry     = coupons - accrued change - funding
    Roll-down = (2) - (1)
    Rate move = (3) - (2)
    Spread move = (4) - (3)
    Residual  = Total - (Carry + Roll + Rate + Spread)
"""

from datetime import date

import pandas as pd

from yield_curve_analyzer import InterpolationMethod
from pricing import price_bond, portfolio_pv, _resolve_curve
from carry_rolldown import _coupons_in_period


def compute_pnl_attribution(
    positions_df: pd.DataFrame,
    curve_map_t0: dict,
    curve_map_t1: dict,
    date_t0: date,
    date_t1: date,
    spread_change_bps: float = 0.0,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
    funding_rate_pct: float = 0.0,
    default_curve_id: str | None = None,
) -> dict:
    """
    Attribute P&L from date_t0 to date_t1 with separate rate and spread effects.

    spread_change_bps: the change in credit spreads over the period.
        Applied to corporate bonds only. Positive = spreads widened (P&L loss).
    """
    horizon_years = max((date_t1 - date_t0).days / 365.0, 0.0)

    # --- Step 1: PV at t0 ---
    pv_t0 = portfolio_pv(
        positions_df=positions_df,
        curve_map=curve_map_t0,
        valuation_date=date_t0,
        method=method,
        default_curve_id=default_curve_id,
    )

    # --- Step 2: Carry (coupons - accrued change - funding) ---
    total_coupons = 0.0
    total_accrued_change = 0.0

    for row in positions_df.itertuples(index=False):
        curve_id = getattr(row, "curve_id", default_curve_id) or default_curve_id
        curve = _resolve_curve(curve_map_t0, curve_id, default_curve_id)
        spread = float(getattr(row, "spread_bps", 0.0))

        # Attribution is measured over [date_t0, date_t1], independent of trade settlement.
        coupons = _coupons_in_period(
            date_t0, date_t1,
            getattr(row, "maturity_date"),
            float(getattr(row, "coupon_rate")),
            int(getattr(row, "coupon_freq")),
            float(getattr(row, "notional")),
        )
        total_coupons += coupons

        # Accrued at t0
        _, _, _, accrued_t0 = price_bond(
            curve, date_t0,
            getattr(row, "maturity_date"),
            float(getattr(row, "coupon_rate")),
            int(getattr(row, "coupon_freq")),
            float(getattr(row, "notional")),
            spread_bps=spread, method=method,
        )

        # Accrued at t1 (on same curve)
        _, _, _, accrued_t1 = price_bond(
            curve, date_t1,
            getattr(row, "maturity_date"),
            float(getattr(row, "coupon_rate")),
            int(getattr(row, "coupon_freq")),
            float(getattr(row, "notional")),
            spread_bps=spread, method=method,
        )

        total_accrued_change += (accrued_t1 - accrued_t0)

    funding_cost = pv_t0 * (funding_rate_pct / 100.0) * horizon_years
    carry = total_coupons + total_accrued_change - funding_cost

    # --- Step 3: PV at t1, t0 curves (roll-down) ---
    pv_t1_old_curves = portfolio_pv(
        positions_df=positions_df,
        curve_map=curve_map_t0,
        valuation_date=date_t1,
        method=method,
        default_curve_id=default_curve_id,
    )
    rolldown = pv_t1_old_curves - pv_t0

    # --- Step 4: PV at t1, t1 curves, old spreads (rate move only) ---
    pv_t1_new_curves_old_spread = portfolio_pv(
        positions_df=positions_df,
        curve_map=curve_map_t1,
        valuation_date=date_t1,
        method=method,
        default_curve_id=default_curve_id,
    )
    rate_move = pv_t1_new_curves_old_spread - pv_t1_old_curves

    # --- Step 5: PV at t1, t1 curves, new spreads (spread move) ---
    pv_t1_final = portfolio_pv(
        positions_df=positions_df,
        curve_map=curve_map_t1,
        valuation_date=date_t1,
        method=method,
        spread_shock_bps=spread_change_bps,
        default_curve_id=default_curve_id,
    )
    spread_move = pv_t1_final - pv_t1_new_curves_old_spread

    # --- Total and residual ---
    total_pnl = pv_t1_final - pv_t0
    residual = total_pnl - (carry + rolldown + rate_move + spread_move)

    return {
        "pv_t0": pv_t0,
        "pv_t1": pv_t1_final,
        "total_pnl": total_pnl,
        "carry": carry,
        "rolldown": rolldown,
        "rate_move": rate_move,
        "spread_move": spread_move,
        "residual": residual,
        "coupons_received": total_coupons,
        "funding_cost": funding_cost,
    }


def format_pnl_report(attribution: dict) -> pd.DataFrame:
    """Format P&L attribution as a clean table for CIO reporting."""
    total = attribution["total_pnl"]

    rows = [
        {"Component": "Starting PV", "Value": attribution["pv_t0"], "% of Total": None},
        {"Component": "Carry (Net Income)", "Value": attribution["carry"], "% of Total": _pct(attribution["carry"], total)},
        {"Component": "Roll-Down", "Value": attribution["rolldown"], "% of Total": _pct(attribution["rolldown"], total)},
        {"Component": "Rate Moves", "Value": attribution["rate_move"], "% of Total": _pct(attribution["rate_move"], total)},
        {"Component": "Spread Moves", "Value": attribution["spread_move"], "% of Total": _pct(attribution["spread_move"], total)},
        {"Component": "Residual (Cross-Effects)", "Value": attribution["residual"], "% of Total": _pct(attribution["residual"], total)},
        {"Component": "Total P&L", "Value": total, "% of Total": 100.0},
        {"Component": "Ending PV", "Value": attribution["pv_t1"], "% of Total": None},
    ]

    return pd.DataFrame(rows)


def _pct(value: float, total: float) -> float:
    """Helper to compute percentage contribution."""
    return (value / total * 100) if total != 0 else 0.0
