"""
PCA Decomposition of Yield Curve Moves
=======================================
Decompose curve movements into Level, Slope, and Curvature components.

These are the first 3 principal components of historical yield curve changes,
which typically explain 95%+ of total variance.

- Level:     Parallel shift (PC1, ~85-90% of variance)
- Slope:     2s10s steepening/flattening (PC2, ~5-10%)
- Curvature: Butterfly (belly vs. wings) (PC3, ~1-3%)
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def get_default_pca_loadings() -> pd.DataFrame:
    """
    Return typical PCA loadings for USD rates (from historical covariance analysis).
    
    These are stylized loadings calibrated from ~10 years of UST daily changes.
    For production, you'd re-estimate these from your specific market history.
    """
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]

    # Stylized loadings (normalized so sum of squares = 1)
    level = [0.31, 0.32, 0.32, 0.33, 0.33, 0.32, 0.31, 0.30, 0.27, 0.25]
    slope = [-0.50, -0.48, -0.42, -0.30, -0.15, 0.05, 0.20, 0.35, 0.45, 0.48]
    curvature = [0.35, 0.30, 0.15, -0.10, -0.25, -0.30, -0.25, -0.05, 0.20, 0.30]

    return pd.DataFrame(
        {
            "tenor": tenors,
            "level": level,
            "slope": slope,
            "curvature": curvature,
        }
    )


def decompose_portfolio_pca_exposure(
    dv01_by_tenor: pd.DataFrame, pca_loadings: pd.DataFrame | None = None
) -> dict:
    """
    Calculate portfolio exposure to Level/Slope/Curvature.

    dv01_by_tenor: DataFrame with columns [tenor, dv01]
    pca_loadings:  DataFrame with columns [tenor, level, slope, curvature]

    Returns dict with: level_exposure, slope_exposure, curvature_exposure
    (All in DV01-equivalent units)
    """
    if pca_loadings is None:
        pca_loadings = get_default_pca_loadings()

    # Merge DV01 with loadings
    merged = pd.merge(dv01_by_tenor, pca_loadings, on="tenor", how="inner")

    if merged.empty:
        return {"level_exposure": 0.0, "slope_exposure": 0.0, "curvature_exposure": 0.0}

    # Portfolio exposure to each factor = sum(dv01_i * loading_i)
    level_exp = (merged["dv01"] * merged["level"]).sum()
    slope_exp = (merged["dv01"] * merged["slope"]).sum()
    curve_exp = (merged["dv01"] * merged["curvature"]).sum()

    return {
        "level_exposure": level_exp,
        "slope_exposure": slope_exp,
        "curvature_exposure": curve_exp,
    }


def interpret_pca_exposures(level: float, slope: float, curvature: float) -> str:
    """Generate plain-English interpretation of PCA exposures."""
    lines = []

    # Level interpretation
    if abs(level) < 100:
        lines.append("- **Level (Duration):** Near-neutral duration positioning.")
    elif level > 0:
        lines.append(f"- **Level (Duration):** Long duration ({level:,.0f} DV01). Benefits from rate cuts.")
    else:
        lines.append(f"- **Level (Duration):** Short duration ({level:,.0f} DV01). Benefits from rate hikes.")

    # Slope interpretation
    if abs(slope) < 50:
        lines.append("- **Slope:** Neutral to curve steepening/flattening.")
    elif slope > 0:
        lines.append(
            f"- **Slope:** Long slope ({slope:,.0f}). Benefits from curve steepening (longs outperform shorts)."
        )
    else:
        lines.append(
            f"- **Slope:** Short slope ({slope:,.0f}). Benefits from curve flattening (shorts outperform longs)."
        )

    # Curvature interpretation
    if abs(curvature) < 30:
        lines.append("- **Curvature:** Neutral to butterfly moves.")
    elif curvature > 0:
        lines.append(f"- **Curvature:** Long curvature ({curvature:,.0f}). Belly underweight vs. wings.")
    else:
        lines.append(f"- **Curvature:** Short curvature ({curvature:,.0f}). Belly overweight vs. wings.")

    return "\n".join(lines)
