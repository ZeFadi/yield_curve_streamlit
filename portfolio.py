from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class BondPosition:
    id: str
    issuer: str
    currency: str
    type: str  # "sovereign" or "corporate"
    notional: float
    coupon_rate: float  # annual %, fixed
    coupon_freq: int  # 1, 2, or 4
    maturity_date: pd.Timestamp
    settlement_date: pd.Timestamp
    clean_price: float | None
    spread_bps: float
    curve_id: str


REQUIRED_COLS = [
    "id",
    "issuer",
    "type",
    "currency",
    "notional",
    "coupon_rate",
    "coupon_freq",
    "maturity_date",
]

OPTIONAL_COLS = ["clean_price", "settlement_date", "curve_id", "spread_bps"]


def sample_portfolio_df() -> pd.DataFrame:
    """Expanded sample portfolio with 12 bonds across sovereigns, IG, and BBB corporates."""
    return pd.DataFrame(
        [
            # --- Sovereign bonds ---
            {
                "id": "UST_2Y",
                "issuer": "US Treasury",
                "type": "sovereign",
                "currency": "USD",
                "notional": 10_000_000,
                "coupon_rate": 4.50,
                "coupon_freq": 2,
                "maturity_date": "2028-02-15",
                "curve_id": "BASE",
                "spread_bps": 0.0,
                "clean_price": 100.25,
            },
            {
                "id": "UST_5Y",
                "issuer": "US Treasury",
                "type": "sovereign",
                "currency": "USD",
                "notional": 5_000_000,
                "coupon_rate": 4.25,
                "coupon_freq": 2,
                "maturity_date": "2031-02-15",
                "curve_id": "BASE",
                "spread_bps": 0.0,
                "clean_price": 99.50,
            },
            {
                "id": "UST_10Y",
                "issuer": "US Treasury",
                "type": "sovereign",
                "currency": "USD",
                "notional": 8_000_000,
                "coupon_rate": 4.00,
                "coupon_freq": 2,
                "maturity_date": "2036-08-15",
                "curve_id": "BASE",
                "spread_bps": 0.0,
                "clean_price": 97.80,
            },
            {
                "id": "UST_30Y",
                "issuer": "US Treasury",
                "type": "sovereign",
                "currency": "USD",
                "notional": 3_000_000,
                "coupon_rate": 4.50,
                "coupon_freq": 2,
                "maturity_date": "2056-11-15",
                "curve_id": "BASE",
                "spread_bps": 0.0,
                "clean_price": 98.00,
            },
            # --- Investment Grade corporates ---
            {
                "id": "AAPL_28",
                "issuer": "Apple Inc",
                "type": "corporate",
                "currency": "USD",
                "notional": 4_000_000,
                "coupon_rate": 3.85,
                "coupon_freq": 2,
                "maturity_date": "2028-05-15",
                "curve_id": "BASE",
                "spread_bps": 55.0,
                "clean_price": 99.10,
            },
            {
                "id": "MSFT_30",
                "issuer": "Microsoft Corp",
                "type": "corporate",
                "currency": "USD",
                "notional": 3_500_000,
                "coupon_rate": 4.10,
                "coupon_freq": 2,
                "maturity_date": "2031-11-15",
                "curve_id": "BASE",
                "spread_bps": 60.0,
                "clean_price": 98.75,
            },
            {
                "id": "JPM_29",
                "issuer": "JPMorgan Chase",
                "type": "corporate",
                "currency": "USD",
                "notional": 5_000_000,
                "coupon_rate": 5.25,
                "coupon_freq": 2,
                "maturity_date": "2029-03-15",
                "curve_id": "BASE",
                "spread_bps": 95.0,
                "clean_price": 101.20,
            },
            {
                "id": "GS_32",
                "issuer": "Goldman Sachs",
                "type": "corporate",
                "currency": "USD",
                "notional": 3_000_000,
                "coupon_rate": 5.50,
                "coupon_freq": 2,
                "maturity_date": "2032-07-15",
                "curve_id": "BASE",
                "spread_bps": 110.0,
                "clean_price": 100.50,
            },
            # --- BBB / Higher-Spread corporates ---
            {
                "id": "ACME_28",
                "issuer": "ACME Corp",
                "type": "corporate",
                "currency": "USD",
                "notional": 3_000_000,
                "coupon_rate": 5.75,
                "coupon_freq": 2,
                "maturity_date": "2028-08-15",
                "curve_id": "BASE",
                "spread_bps": 180.0,
                "clean_price": 101.80,
            },
            {
                "id": "MEGA_32",
                "issuer": "Mega Industries",
                "type": "corporate",
                "currency": "USD",
                "notional": 2_500_000,
                "coupon_rate": 6.15,
                "coupon_freq": 2,
                "maturity_date": "2032-05-15",
                "curve_id": "BASE",
                "spread_bps": 240.0,
                "clean_price": 102.30,
            },
            {
                "id": "XYZ_27",
                "issuer": "XYZ Holdings",
                "type": "corporate",
                "currency": "USD",
                "notional": 2_000_000,
                "coupon_rate": 6.50,
                "coupon_freq": 2,
                "maturity_date": "2027-09-15",
                "curve_id": "BASE",
                "spread_bps": 320.0,
                "clean_price": 102.90,
            },
            {
                "id": "ENRG_35",
                "issuer": "Energy Partners",
                "type": "corporate",
                "currency": "USD",
                "notional": 4_000_000,
                "coupon_rate": 5.90,
                "coupon_freq": 2,
                "maturity_date": "2035-03-15",
                "curve_id": "BASE",
                "spread_bps": 200.0,
                "clean_price": 99.80,
            },
        ]
    )


def standardize_portfolio_df(
    df: pd.DataFrame, valuation_date: date
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    col_map = {c.lower(): c for c in df.columns}
    for col in REQUIRED_COLS:
        if col not in col_map:
            raise ValueError(f"Missing required column: {col}")

    rename_map = {col_map[col]: col for col in REQUIRED_COLS if col in col_map}
    for opt in OPTIONAL_COLS:
        if opt in col_map:
            rename_map[col_map[opt]] = opt

    df = df.rename(columns=rename_map)
    df = df[[c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]].copy()

    df["id"] = df["id"].astype(str)
    df["issuer"] = df["issuer"].astype(str)
    df["currency"] = df["currency"].astype(str)
    df["type"] = df["type"].astype(str).str.lower()
    df.loc[~df["type"].isin(["sovereign", "corporate"]), "type"] = "corporate"

    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["coupon_rate"] = pd.to_numeric(df["coupon_rate"], errors="coerce")
    df["coupon_freq"] = pd.to_numeric(df["coupon_freq"], errors="coerce").astype("Int64")
    if "spread_bps" in df.columns:
        df["spread_bps"] = pd.to_numeric(df["spread_bps"], errors="coerce")
    else:
        warnings.append("Missing spread_bps column; defaulting to 0.")
        df["spread_bps"] = 0.0

    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    else:
        df["settlement_date"] = pd.NaT

    if "clean_price" in df.columns:
        df["clean_price"] = pd.to_numeric(df["clean_price"], errors="coerce")
    else:
        df["clean_price"] = pd.NA

    before = len(df)
    df = df.dropna(subset=["notional", "coupon_rate", "coupon_freq", "spread_bps", "maturity_date"])
    if len(df) < before:
        warnings.append("Dropped rows with missing or invalid numeric fields.")

    df["settlement_date"] = df["settlement_date"].fillna(pd.Timestamp(valuation_date))
    df["coupon_freq"] = df["coupon_freq"].clip(lower=1).astype(int)
    df["spread_bps"] = df["spread_bps"].fillna(0.0)

    invalid_notional = df["notional"] <= 0
    if invalid_notional.any():
        warnings.append("Removed rows with non-positive notional.")
        df = df.loc[~invalid_notional].copy()

    invalid_maturity = df["maturity_date"] <= df["settlement_date"]
    if invalid_maturity.any():
        warnings.append("Removed rows with maturity before settlement.")
        df = df.loc[~invalid_maturity].copy()

    if "curve_id" not in df.columns:
        warnings.append("Missing curve_id column; defaulting to BASE.")
        df["curve_id"] = "BASE"
    else:
        curve_id_series = df["curve_id"]
        curve_id_series = curve_id_series.where(curve_id_series.notna(), "BASE")
        curve_id_series = curve_id_series.astype(str).str.strip()
        curve_id_series = curve_id_series.replace({"": "BASE", "nan": "BASE", "None": "BASE"})
        df["curve_id"] = curve_id_series
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid portfolio rows found after cleaning.")

    return df, warnings
