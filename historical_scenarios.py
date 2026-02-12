"""
Historical Scenario Replay
===========================
Apply real historical rate moves to current portfolio for stress testing.

These scenarios are calibrated from actual market moves and capture
realistic correlations across the curve (vs. arbitrary parallel shifts).
"""

from typing import Dict, List
import numpy as np
import pandas as pd


# Historical curve moves (in basis points) by maturity bucket
#  Format: {scenario_name: {lower_tenor: bps_shift, upper_tenor: bps_shift}}
#  Linear interpolation applied between tenors

HISTORICAL_SCENARIOS = {
    "COVID_Mar2020": {
        "name": "COVID Crash (Mar 2020)",
        "description": "Fed cuts to zero, flight to quality, massive curve flattening",
        "moves": {0.25: -150, 2: -145, 5: -120, 10: -80, 30: -60},
        "spread_shock_bps": 300,  # IG credit spreads widened ~300bps
    },
    "FedHike_2022": {
        "name": "Fed Hiking Cycle (2022)",
        "description": "Inflation shock, aggressive Fed tightening, curve inversion",
        "moves": {0.25: +250, 2: +230, 5: +200, 10: +100, 30: +50},
        "spread_shock_bps": 50,  # IG spreads widened modestly
    },
    "Lehman_2008": {
        "name": "Lehman Collapse (Sep 2008)",
        "description": "Credit crisis, policy panic, extreme volatility",
        "moves": {0.25: -200, 2: -150, 5: -100, 10: -50, 30: -30},
        "spread_shock_bps": 500,  # HY spreads blew out to 2000bps+
    },
    "TaperTantrum_2013": {
        "name": "Taper Tantrum (May-Jun 2013)",
        "description": "Fed signals QE tapering, sell-off in Treasuries",
        "moves": {0.25: +20, 2: +40, 5: +80, 10: +100, 30: +110},
        "spread_shock_bps": 30,
    },
    "Brexit_2016": {
        "name": "Brexit Vote (Jun 2016)",
        "description": "UK referendum shock, flight to safety",
        "moves": {0.25: -10, 2: -20, 5: -35, 10: -40, 30: -45},
        "spread_shock_bps": 25,
    },
    "SVB_2023": {
        "name": "SVB / Regional Banks (Mar 2023)",
        "description": "Bank run contagion, rates rally on recession fears, IG spreads widen",
        "moves": {0.25: -50, 2: -100, 5: -60, 10: -40, 30: -20},
        "spread_shock_bps": 40,
    },
}


def interpolate_scenario_shifts(scenario_name: str, curve_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply historical scenario shifts to a curve DataFrame.

    Uses linear interpolation between defined tenor points.
    Returns a new DataFrame with shifted rates.
    """
    if scenario_name not in HISTORICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(HISTORICAL_SCENARIOS.keys())}")

    scenario = HISTORICAL_SCENARIOS[scenario_name]
    moves = scenario["moves"]

    # Sort tenor points
    tenor_points = sorted(moves.keys())
    shift_points = [moves[t] for t in tenor_points]

    shocked_df = curve_df.copy()
    tenors = shocked_df["tenor"].astype(float).values

    # Interpolate shifts for each tenor
    shifts = np.interp(tenors, tenor_points, shift_points)
    shocked_df["rate"] = shocked_df["rate"] + shifts / 100.0  # Convert bps to %

    return shocked_df


def get_scenario_description(scenario_name: str) -> Dict[str, str]:
    """Return metadata for a given scenario."""
    if scenario_name not in HISTORICAL_SCENARIOS:
        return {"name": scenario_name, "description": "Unknown scenario", "spread_shock_bps": 0}
    return HISTORICAL_SCENARIOS[scenario_name]


def list_available_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(HISTORICAL_SCENARIOS.keys())
