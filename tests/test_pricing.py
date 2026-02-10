import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod
from portfolio import standardize_portfolio_df
from pricing import price_bond, bond_risk_metrics


def _flat_curve(rate: float = 5.0) -> YieldCurveAnalyzer:
    data = pd.DataFrame({"tenor": [1, 2, 5, 10], "rate": [rate, rate, rate, rate]})
    return YieldCurveAnalyzer(data)


def test_zero_coupon_price_matches_discount_factor():
    curve = _flat_curve(5.0)
    pv, dirty, clean, accrued = price_bond(
        curve,
        settlement_date=date(2026, 1, 1),
        maturity_date=date(2027, 1, 1),
        coupon_rate=0.0,
        coupon_freq=1,
        notional=100.0,
        method=InterpolationMethod.PCHIP,
    )
    expected = 100.0 * np.exp(-0.05 * 1.0)
    assert abs(pv - expected) < 1e-2
    assert abs(dirty - pv) < 1e-6
    assert abs(clean - pv) < 1e-6
    assert abs(accrued) < 1e-6


def test_clean_plus_accrued_equals_dirty():
    curve = _flat_curve(4.0)
    pv, dirty, clean, accrued = price_bond(
        curve,
        settlement_date=date(2026, 1, 1),
        maturity_date=date(2028, 1, 1),
        coupon_rate=6.0,
        coupon_freq=2,
        notional=100.0,
        method=InterpolationMethod.PCHIP,
    )
    assert abs((clean + accrued) - dirty) < 1e-6
    assert abs(pv - dirty) < 1e-6


def test_dv01_positive_for_bond():
    curve = _flat_curve(3.5)
    metrics = bond_risk_metrics(
        curve,
        settlement_date=date(2026, 1, 1),
        maturity_date=date(2031, 1, 1),
        coupon_rate=5.0,
        coupon_freq=2,
        notional=1_000_000.0,
        spread_bps=100.0,
        method=InterpolationMethod.PCHIP,
    )
    assert metrics.dv01 > 0


def test_portfolio_standardize_defaults_blank_curve_id_to_base():
    raw = pd.DataFrame(
        {
            "id": ["B1"],
            "issuer": ["Issuer"],
            "type": ["corporate"],
            "currency": ["USD"],
            "notional": [1000],
            "coupon_rate": [5.0],
            "coupon_freq": [2],
            "maturity_date": ["2028-01-01"],
            "curve_id": [""],
            "spread_bps": [100.0],
        }
    )
    cleaned, _ = standardize_portfolio_df(raw, date(2026, 1, 1))
    assert cleaned.loc[0, "curve_id"] == "BASE"


def test_portfolio_pv_raises_clear_error_when_curve_missing():
    curve = _flat_curve(4.0)
    positions = pd.DataFrame(
        {
            "id": ["B1"],
            "issuer": ["Issuer"],
            "type": ["corporate"],
            "currency": ["USD"],
            "curve_id": ["MISSING"],
            "notional": [1000],
            "coupon_rate": [5.0],
            "coupon_freq": [2],
            "maturity_date": [date(2028, 1, 1)],
            "settlement_date": [date(2026, 1, 1)],
            "spread_bps": [100.0],
        }
    )
    from pricing import portfolio_pv

    try:
        portfolio_pv(positions, {"UST": curve}, date(2026, 1, 1), default_curve_id="BASE")
        assert False, "Expected ValueError for missing curve resolution"
    except ValueError as exc:
        assert "No curve found" in str(exc)
