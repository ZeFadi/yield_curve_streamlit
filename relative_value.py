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
import warnings
try:
    from scipy.optimize import brentq
except Exception:
    def brentq(func, a, b, xtol=0.01, maxiter=200):
        """Fallback bisection solver when scipy is unavailable."""
        fa = func(a)
        fb = func(b)
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb > 0:
            raise ValueError("Root not bracketed")
        lo, hi = a, b
        flo, fhi = fa, fb
        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            fmid = func(mid)
            if abs(fmid) <= xtol or abs(hi - lo) <= xtol:
                return mid
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi)

from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod
from pricing import price_bond, bond_risk_metrics, _resolve_curve


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _extract_metrics(metrics_obj) -> tuple[float, float]:
    """Return (modified_duration, pv) from multiple metric return shapes."""
    # Current shape: dataclass/object with attributes
    if hasattr(metrics_obj, "modified_duration") and hasattr(metrics_obj, "pv"):
        return _safe_float(metrics_obj.modified_duration), _safe_float(metrics_obj.pv)

    # Namedtuple-like
    if hasattr(metrics_obj, "_asdict"):
        d = metrics_obj._asdict()
        return _safe_float(d.get("modified_duration", d.get("duration", 0.0))), _safe_float(d.get("pv", 0.0))

    # Alternate shape: dict-like
    if isinstance(metrics_obj, dict):
        md = metrics_obj.get("modified_duration", metrics_obj.get("duration", 0.0))
        pv = metrics_obj.get("pv", 0.0)
        return _safe_float(md), _safe_float(pv)

    # Pandas Series-like
    if isinstance(metrics_obj, pd.Series):
        return _safe_float(metrics_obj.get("modified_duration", metrics_obj.get("duration", 0.0))), _safe_float(metrics_obj.get("pv", 0.0))

    # Legacy shape: tuple/list positional output
    if isinstance(metrics_obj, (tuple, list)):
        # Expected ordering in legacy variants:
        # (pv, dirty, clean, accrued, modified_duration, convexity, dv01, cs01)
        if len(metrics_obj) >= 5:
            return _safe_float(metrics_obj[4]), _safe_float(metrics_obj[0])
        if len(metrics_obj) >= 1:
            return 0.0, _safe_float(metrics_obj[0])

    # Last resort: try scalar PV fallback and continue without duration.
    if np.isscalar(metrics_obj):
        return 0.0, _safe_float(metrics_obj)

    warnings.warn(
        f"Unsupported metrics object type '{type(metrics_obj).__name__}' in portfolio_z_spreads; "
        "defaulting duration and pv to 0 for this row.",
        RuntimeWarning,
    )
    return 0.0, 0.0


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
        mod_duration, pv_value = _extract_metrics(metrics)

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
                "duration": mod_duration,
                "years_to_maturity": years_to_mat,
                "model_spread_bps": spread,
                "z_spread_bps": z_spread,
                "clean_price": market_price if market_price is not None else float("nan"),
                "pv": pv_value,
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
        try:
            coeffs = np.polyfit(valid["duration"].values, valid["z_spread_bps"].values, 1)
            df["fitted_spread"] = coeffs[0] * df["duration"] + coeffs[1]
            df["residual_bps"] = df["z_spread_bps"] - df["fitted_spread"]
            _assign_signal(df)
        except np.linalg.LinAlgError:
            # Fallback if SVD fails (e.g. collinear points or other numerical issues)
            df["fitted_spread"] = float("nan")
            df["residual_bps"] = float("nan")
            df["signal"] = ""
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
            try:
                coeffs = np.polyfit(valid["duration"].values, valid["z_spread_bps"].values, 1)
                df.loc[mask, "fitted_spread"] = coeffs[0] * df.loc[mask, "duration"] + coeffs[1]
                df.loc[mask, "residual_bps"] = df.loc[mask, "z_spread_bps"] - df.loc[mask, "fitted_spread"]
            except np.linalg.LinAlgError:
                # Fallback if SVD fails (e.g. collinear points or other numerical issues)
                pass

    _assign_signal(df)


def _assign_signal(df: pd.DataFrame) -> None:
    """Assign RICH / CHEAP / FAIR signals based on residual."""
    df["signal"] = df["residual_bps"].apply(
        lambda x: "CHEAP" if x > 15 else ("RICH" if x < -15 else "FAIR")
        if not pd.isna(x)
        else ""
    )
