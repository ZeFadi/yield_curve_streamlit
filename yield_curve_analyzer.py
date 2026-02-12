"""
Yield Curve Analyzer - Fixed Income Quantitative Tool
======================================================

A professional-grade Python tool for yield curve construction, analysis, 
and comparison using cubic spline and shape-preserving interpolation methods.

Author: Senior Quant Developer
Domain: Fixed Income / Asset Management
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

    class _LinearInterpolator:
        """Fallback interpolator when scipy is unavailable."""

        def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            bc_type: str | None = None,
            extrapolate: bool = True,
        ) -> None:
            self.x = np.asarray(x, dtype=float)
            self.y = np.asarray(y, dtype=float)
            self.extrapolate = extrapolate
            self._slopes = np.gradient(self.y, self.x)

        def __call__(self, x_new: Union[float, np.ndarray], nu: int = 0) -> Union[float, np.ndarray]:
            x_arr = np.asarray(x_new, dtype=float)
            x_eval = x_arr.copy()
            if not self.extrapolate:
                x_eval = np.clip(x_eval, self.x.min(), self.x.max())

            if nu == 0:
                out = np.interp(x_eval, self.x, self.y)
            elif nu == 1:
                out = np.interp(x_eval, self.x, self._slopes)
            else:
                out = np.zeros_like(x_eval, dtype=float)

            return float(out) if np.isscalar(x_new) else out

    CubicSpline = _LinearInterpolator
    PchipInterpolator = _LinearInterpolator


class InterpolationMethod(Enum):
    """Enumeration of supported interpolation methods."""
    CUBIC_SPLINE = "cubic_spline"
    PCHIP = "pchip"


@dataclass
class MarketData:
    """Container for yield curve market data."""
    tenors: np.ndarray  # Maturities in years
    rates: np.ndarray   # Rates in percentage (e.g., 2.5 for 2.5%)


class YieldCurveAnalyzer:
    """
    Yield Curve Construction and Analysis Engine.
    
    This class provides comprehensive functionality for building, analyzing,
    and comparing yield curves using two distinct interpolation methodologies:
    
    1. Natural Cubic Spline (C2 continuity):
       - Ensures smoothness through continuous second derivatives
       - Global sensitivity: changes at one point affect the entire curve
       - May introduce spurious oscillations in forward rates
    
    2. PCHIP / Shape-Preserving (C1 continuity):
       - Preserves monotonicity of input data (when data is monotone)
       - Local behavior: changes at one point have limited propagation
       - More stable forward rate curves, preferred for risk management
    
    Financial Context:
        Zero Rate R(t): The continuously compounded spot rate for maturity t
        Forward Rate f(t): The instantaneous forward rate, derived as f(t) = R(t) + t * R'(t)

    Important Note:
        This class assumes the input rates are already zero rates (spot rates)
        expressed in percentage terms and continuously compounded. If your
        inputs are par yields, swap rates, or simple-compounded rates, you
        should bootstrap zero rates first.
    
    Attributes:
        tenors: Array of market maturities in years
        rates: Array of market rates in decimal form
        cubic_spline: CubicSpline interpolator object
        pchip: PchipInterpolator object
    
    Example:
        >>> data = pd.DataFrame({'tenor': [1, 2, 5, 10, 30], 'rate': [3.0, 3.2, 3.5, 3.8, 4.2]})
        >>> analyzer = YieldCurveAnalyzer(data)
        >>> analyzer.get_zero_rate(7.5, method=InterpolationMethod.PCHIP)
    """
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, List[Dict[str, float]]],
        tenor_column: str = "tenor",
        rate_column: str = "rate",
        extrapolate: bool = True
    ) -> None:
        """
        Initialize the YieldCurveAnalyzer with market data.
        
        Args:
            data: Market data as DataFrame or list of dictionaries.
                  Must contain tenor (years) and rate (percentage) columns.
            tenor_column: Name of the column containing maturities.
            rate_column: Name of the column containing rates.
        
        Raises:
            ValueError: If data is empty or contains invalid values.
            TypeError: If data format is not supported.
        """
        self.extrapolate = extrapolate
        self._validate_and_parse_input(data, tenor_column, rate_column)
        self._build_interpolators()
    
    def _validate_and_parse_input(
        self,
        data: Union[pd.DataFrame, List[Dict[str, float]]],
        tenor_column: str,
        rate_column: str
    ) -> None:
        """Parse and validate input market data."""
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Data must be DataFrame or List[Dict], got {type(data).__name__}"
            )
        
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        if tenor_column not in data.columns or rate_column not in data.columns:
            raise ValueError(
                f"Required columns '{tenor_column}' and '{rate_column}' not found. "
                f"Available columns: {list(data.columns)}"
            )
        
        # Sort by tenor and extract arrays
        data_sorted = data.sort_values(by=tenor_column).reset_index(drop=True)
        self.tenors: np.ndarray = data_sorted[tenor_column].values.astype(float)
        self.rates: np.ndarray = data_sorted[rate_column].values.astype(float)
        
        # Validate data quality
        if not np.all(np.isfinite(self.tenors)) or not np.all(np.isfinite(self.rates)):
            raise ValueError("Tenors and rates must be finite numbers")

        if np.any(self.tenors <= 0):
            raise ValueError("All tenors must be strictly positive")
        
        if len(self.tenors) < 2:
            raise ValueError("At least 2 data points required for interpolation")
        
        if len(np.unique(self.tenors)) != len(self.tenors):
            raise ValueError("Duplicate tenors detected - each maturity must be unique")
        
        # Store market data container
        self.market_data = MarketData(tenors=self.tenors.copy(), rates=self.rates.copy())
    
    def _build_interpolators(self) -> None:
        """
        Construct interpolation engines using scipy.
        
        Natural Cubic Spline:
            Uses 'natural' boundary conditions (zero second derivative at endpoints).
            This is standard practice for yield curve construction.
        
        PCHIP (Piecewise Cubic Hermite Interpolating Polynomial):
            Preserves monotonicity of the data. If rates are increasing,
            the interpolated curve will also be monotonically increasing.
            This is shape-preserving but is not the same as the Hagan-West
            monotone convex algorithm used in some production curve engines.
        """
        if not _SCIPY_AVAILABLE:
            warnings.warn(
                "scipy is not available; using linear interpolation fallback. "
                "Install scipy for cubic spline and PCHIP behavior.",
                RuntimeWarning,
            )
        self.cubic_spline: CubicSpline = CubicSpline(
            self.tenors, 
            self.rates, 
            bc_type='natural',
            extrapolate=self.extrapolate
        )
        self.pchip: PchipInterpolator = PchipInterpolator(
            self.tenors,
            self.rates,
            extrapolate=self.extrapolate
        )
    
    def _get_interpolator(self, method: Union[InterpolationMethod, str]) -> Union[CubicSpline, PchipInterpolator]:
        """Return the appropriate interpolator based on method selection."""
        key = method
        if hasattr(method, "value"):
            key = getattr(method, "value")
        if isinstance(key, str):
            key_norm = key.strip().lower().replace(" ", "_")
        else:
            key_norm = str(key).strip().lower().replace(" ", "_")

        if key_norm in {"cubic_spline", "cubic"}:
            return self.cubic_spline
        if key_norm in {"pchip", "shape_preserving", "shape-preserving"}:
            return self.pchip

        raise ValueError(f"Unknown interpolation method: {method}")
    
    def _validate_maturity(self, t: Union[float, np.ndarray]) -> None:
        """Validate that maturity is within acceptable bounds."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if not np.all(np.isfinite(t_arr)):
            raise ValueError(f"Maturity must be finite, got {t}")
        if np.any(t_arr <= 0):
            raise ValueError(f"Maturity must be strictly positive, got {t}")
        if np.any(t_arr > self.tenors.max() * 1.5):
            # Warning for significant extrapolation
            warnings.warn(
                f"Extrapolating beyond 150% of maximum tenor ({self.tenors.max()}Y). "
                "Results may be unreliable.",
                UserWarning
            )
    
    def get_zero_rate(
        self, 
        t: Union[float, np.ndarray], 
        method: InterpolationMethod = InterpolationMethod.PCHIP
    ) -> Union[float, np.ndarray]:
        """
        Extract the Zero-Coupon Rate R(t) at maturity t.
        
        The zero rate is the yield to maturity of a theoretical zero-coupon bond.
        It represents the continuously compounded return for investing from today
        until time t.
        
        Args:
            t: Maturity in years (scalar or array).
            method: Interpolation method to use.
        
        Returns:
            Zero rate(s) in percentage terms.
        
        Raises:
            ValueError: If maturity is non-positive.
        
        Example:
            >>> analyzer.get_zero_rate(5.0, InterpolationMethod.CUBIC_SPLINE)
            3.45  # 3.45%
        """
        t_arr = np.asarray(t, dtype=float)
        self._validate_maturity(t_arr)
        interpolator = self._get_interpolator(method)
        result = interpolator(t_arr)
        return float(result) if np.isscalar(t) else result

    def get_discount_factor(
        self,
        t: Union[float, np.ndarray],
        method: InterpolationMethod = InterpolationMethod.PCHIP,
        spread_bps: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """
        Compute discount factors using continuously-compounded zero rates.

        df(t) = exp(-(R(t) + s) * t) with rates in decimal form.
        """
        t_arr = np.asarray(t, dtype=float)
        self._validate_maturity(t_arr)
        zero_rates = self.get_zero_rate(t_arr, method=method)
        total_rate = (zero_rates + spread_bps / 100.0) / 100.0
        df = np.exp(-total_rate * t_arr)
        return float(df) if np.isscalar(t) else df
    
    def get_forward_rate(
        self, 
        t: Union[float, np.ndarray], 
        method: InterpolationMethod = InterpolationMethod.PCHIP
    ) -> Union[float, np.ndarray]:
        """
        Calculate the Instantaneous Forward Rate f(t) at maturity t.
        
        Mathematical Derivation:
            The discount factor is: P(t) = exp(-R(t) * t)
            The instantaneous forward rate is: f(t) = -d/dt[ln P(t)]
            
            Expanding: f(t) = d/dt[R(t) * t] = R(t) + t * R'(t)
            
            Where R'(t) is the first derivative of the zero rate curve.
        
        This method uses the analytical derivative provided by scipy interpolators
        for maximum numerical precision.
        
        Args:
            t: Maturity in years (scalar or array).
            method: Interpolation method to use.
        
        Returns:
            Instantaneous forward rate(s) in percentage terms.
        
        Financial Interpretation:
            f(t) represents the market's expectation of the short rate at time t,
            under the risk-neutral measure. It's crucial for derivative pricing
            and curve arbitrage detection.
        """
        t_arr = np.asarray(t, dtype=float)
        self._validate_maturity(t_arr)
        interpolator = self._get_interpolator(method)

        # R(t): zero rate
        R_t = interpolator(t_arr)

        # R'(t): first derivative of zero rate (analytical, not numerical)
        R_prime_t = interpolator(t_arr, nu=1)  # nu=1 for first derivative

        # f(t) = R(t) + t * R'(t)
        f_t = R_t + t_arr * R_prime_t

        return float(f_t) if np.isscalar(t) else f_t
    
    def get_forward_rate_term(
        self,
        t1: float,
        t2: float,
        method: InterpolationMethod = InterpolationMethod.PCHIP
    ) -> float:
        """
        Calculate the Term Forward Rate between t1 and t2.
        
        This is the forward rate agreed today for a loan starting at t1
        and maturing at t2. Used extensively in swap pricing and 
        forward rate agreement (FRA) valuation.
        
        Formula: F(t1, t2) = [R(t2)*t2 - R(t1)*t1] / (t2 - t1)
        
        Args:
            t1: Start of forward period (years).
            t2: End of forward period (years).
            method: Interpolation method.
        
        Returns:
            Term forward rate in percentage.
        
        Example:
            >>> analyzer.get_forward_rate_term(5.0, 10.0, InterpolationMethod.PCHIP)
            4.15  # The 5Y5Y forward rate
        """
        if t1 >= t2:
            raise ValueError(f"t1 ({t1}) must be less than t2 ({t2})")
        
        self._validate_maturity(t1)
        self._validate_maturity(t2)
        
        R_t1 = self.get_zero_rate(t1, method)
        R_t2 = self.get_zero_rate(t2, method)
        
        return (R_t2 * t2 - R_t1 * t1) / (t2 - t1)
    
    def plot_curves(
        self,
        resolution: int = 200,
        figsize: Tuple[int, int] = (14, 10),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate professional visualization comparing both interpolation methods.
        
        Creates a two-panel figure:
            1. Zero-Coupon Rate Curves: R(t) for both methods with market data overlay
            2. Instantaneous Forward Rate Curves: f(t) = R(t) + t*R'(t)
        
        The visualization highlights:
            - Differences between cubic spline and PCHIP near inflection points
            - Potential oscillations in forward rates (cubic spline pathology)
            - Locality of PCHIP interpolation
        
        Args:
            resolution: Number of interpolation points for smooth curves.
            figsize: Figure dimensions (width, height) in inches.
            title: Optional main title for the figure.
            save_path: Optional path to save the figure.
        
        Returns:
            Matplotlib Figure object for further customization.
        """
        # Create dense time grid for smooth curves
        t_min, t_max = self.tenors.min(), self.tenors.max()
        t_grid = np.linspace(t_min * 0.95, t_max * 1.02, resolution)
        
        # Calculate curves for both methods
        zero_cubic = self.cubic_spline(t_grid)
        zero_pchip = self.pchip(t_grid)

        # Vectorized forward rate computation for speed
        forward_cubic = zero_cubic + t_grid * self.cubic_spline(t_grid, nu=1)
        forward_pchip = zero_pchip + t_grid * self.pchip(t_grid, nu=1)
        
        # Professional styling
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        colors = {
            'cubic': '#1f77b4',    # Blue
            'pchip': '#d62728',    # Red
            'market': '#2ca02c'    # Green
        }
        
        # Panel 1: Zero-Coupon Rates
        ax1 = axes[0]
        ax1.plot(t_grid, zero_cubic, color=colors['cubic'], linewidth=2, 
                 label='Natural Cubic Spline (C2)', linestyle='-')
        ax1.plot(t_grid, zero_pchip, color=colors['pchip'], linewidth=2,
                 label='PCHIP / Shape-Preserving (C1)', linestyle='--')
        ax1.scatter(self.tenors, self.rates, color=colors['market'], s=100, 
                    zorder=5, label='Market Data', edgecolors='black', linewidth=1)
        
        ax1.set_ylabel('Zero Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Zero-Coupon Rate Curve Comparison', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(t_grid.min(), t_grid.max())
        
        # Add minor gridlines
        ax1.minorticks_on()
        ax1.grid(True, which='minor', alpha=0.1)
        
        # Panel 2: Forward Rates
        ax2 = axes[1]
        ax2.plot(t_grid, forward_cubic, color=colors['cubic'], linewidth=2, 
                 label='Cubic Spline Forwards', linestyle='-')
        ax2.plot(t_grid, forward_pchip, color=colors['pchip'], linewidth=2, 
                 label='PCHIP Forwards', linestyle='--')
        
        # Mark market data tenors on forward curve
        forward_market_cubic = self.cubic_spline(self.tenors) + self.tenors * self.cubic_spline(self.tenors, nu=1)
        ax2.scatter(self.tenors, forward_market_cubic, color=colors['market'], 
                    s=60, zorder=5, alpha=0.7, marker='x', linewidths=2)
        
        ax2.set_xlabel('Maturity (Years)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Forward Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Instantaneous Forward Rate Curve: f(t) = R(t) + t·R\'(t)', 
                      fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.minorticks_on()
        ax2.grid(True, which='minor', alpha=0.1)
        
        # Main title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def stress_test(
        self,
        shock_maturity: float,
        shock_bps: float,
        display_results: bool = True,
        figsize: Tuple[int, int] = (14, 8),
        resolution: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply a parallel shock to a specific maturity point and analyze propagation.
        
        This stress test demonstrates the fundamental difference between
        cubic spline (global) and PCHIP (local) interpolation:
        
        - Cubic Spline: A shock at one point propagates across the entire curve
          due to the global nature of the spline basis functions.
          
        - PCHIP: A shock at one point only affects neighboring intervals,
          making it more suitable for risk management and hedging.
        
        Use Case:
            Asset managers use this to understand curve risk and ensure their
            interpolation method doesn't introduce spurious hedging requirements.
        
        Args:
            shock_maturity: The tenor (in years) to shock.
            shock_bps: Shock magnitude in basis points (1 bp = 0.01%).
            display_results: Whether to print and plot results.
            figsize: Figure size for the delta plot.
        
        Returns:
            Dictionary containing:
                - 'base_rates': DataFrame of pre-shock zero rates
                - 'shocked_rates': DataFrame of post-shock zero rates
                - 'deltas': DataFrame of changes (Delta = Shocked - Base)
        
        Example:
            >>> results = analyzer.stress_test(shock_maturity=10.0, shock_bps=10)
            Applying +10 bps shock at 10Y maturity...
        """
        # Find the closest market point to shock
        shock_idx = np.argmin(np.abs(self.tenors - shock_maturity))
        actual_shock_tenor = self.tenors[shock_idx]
        
        if display_results:
            print(f"\n{'='*70}")
            print(f"STRESS TEST: {shock_bps:+.1f} bps shock at {actual_shock_tenor}Y")
            print(f"{'='*70}")
        
        # Create shocked data
        shocked_rates = self.rates.copy()
        shocked_rates[shock_idx] += shock_bps / 100  # Convert bps to percentage
        
        # Build stressed interpolators
        shocked_cubic = CubicSpline(self.tenors, shocked_rates, bc_type='natural')
        shocked_pchip = PchipInterpolator(self.tenors, shocked_rates)
        
        # Evaluation grid (focus on neighboring points)
        t_eval = np.linspace(self.tenors.min(), self.tenors.max(), resolution)
        
        # Calculate base and shocked curves
        base_cubic = self.cubic_spline(t_eval)
        base_pchip = self.pchip(t_eval)
        stress_cubic = shocked_cubic(t_eval)
        stress_pchip = shocked_pchip(t_eval)
        
        # Calculate deltas
        delta_cubic = stress_cubic - base_cubic
        delta_pchip = stress_pchip - base_pchip
        
        # Prepare results DataFrames
        results = {
            'base_rates': pd.DataFrame({
                'Maturity': t_eval,
                'Cubic_Spline': base_cubic,
                'PCHIP': base_pchip
            }),
            'shocked_rates': pd.DataFrame({
                'Maturity': t_eval,
                'Cubic_Spline': stress_cubic,
                'PCHIP': stress_pchip
            }),
            'deltas': pd.DataFrame({
                'Maturity': t_eval,
                'Delta_Cubic_bps': delta_cubic * 100,  # Convert to bps
                'Delta_PCHIP_bps': delta_pchip * 100
            })
        }
        
        if display_results:
            # Print impact at market tenors
            print(f"\nImpact at Market Tenors (in bps):")
            print(f"{'Tenor':<10} {'Cubic Spline':<15} {'PCHIP':<15} {'Difference':<15}")
            print("-" * 55)
            
            for tenor in self.tenors:
                delta_c = (shocked_cubic(tenor) - self.cubic_spline(tenor)) * 100
                delta_p = (shocked_pchip(tenor) - self.pchip(tenor)) * 100
                print(f"{tenor:<10.1f} {delta_c:<15.3f} {delta_p:<15.3f} {abs(delta_c - delta_p):<15.3f}")
            
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Left panel: Delta comparison
            ax1 = axes[0]
            ax1.plot(t_eval, delta_cubic * 100, 'b-', linewidth=2, label='Cubic Spline Delta')
            ax1.plot(t_eval, delta_pchip * 100, 'r--', linewidth=2, label='PCHIP Delta')
            ax1.axvline(actual_shock_tenor, color='green', linestyle=':', linewidth=2, 
                       label=f'Shock Point ({actual_shock_tenor}Y)')
            ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax1.fill_between(t_eval, delta_cubic * 100, delta_pchip * 100, alpha=0.2, color='purple')
            
            ax1.set_xlabel('Maturity (Years)', fontsize=12)
            ax1.set_ylabel('Delta (bps)', fontsize=12)
            ax1.set_title(f'Curve Response to {shock_bps:+.0f}bps Shock at {actual_shock_tenor}Y', 
                         fontsize=12, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Mark market tenors
            for t in self.tenors:
                ax1.axvline(t, color='gray', linestyle=':', alpha=0.3)
            
            # Right panel: Locality demonstration
            ax2 = axes[1]
            locality_diff = np.abs(delta_cubic - delta_pchip) * 100
            ax2.fill_between(t_eval, 0, locality_diff, alpha=0.5, color='purple', 
                            label='|Cubic - PCHIP| Delta')
            ax2.axvline(actual_shock_tenor, color='green', linestyle=':', linewidth=2)
            
            ax2.set_xlabel('Maturity (Years)', fontsize=12)
            ax2.set_ylabel('Absolute Difference (bps)', fontsize=12)
            ax2.set_title('Non-Locality of Cubic Spline vs PCHIP', fontsize=12, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            # Annotate
            ax2.annotate(
                'PCHIP: Local effect\nCubic: Global propagation',
                xy=(actual_shock_tenor, locality_diff.max() * 0.8),
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nKey Insight:")
            print(f"  - Cubic Spline shows non-zero delta even far from the shock point")
            print(f"  - PCHIP maintains delta near zero outside the immediate neighborhood")
            print(f"  - Max non-local propagation (Cubic): {np.max(locality_diff):.3f} bps")
        
        return results
    
    def summary(self) -> None:
        """Print a summary of the yield curve data and key rates."""
        print("\n" + "="*60)
        print("YIELD CURVE SUMMARY")
        print("="*60)
        print(f"\nMarket Data Points: {len(self.tenors)}")
        print(f"Tenor Range: {self.tenors.min():.1f}Y to {self.tenors.max():.1f}Y")
        print(f"Rate Range: {self.rates.min():.3f}% to {self.rates.max():.3f}%")
        
        print("\n" + "-"*60)
        print("Market Data:")
        print("-"*60)
        for t, r in zip(self.tenors, self.rates):
            print(f"  {t:>6.1f}Y : {r:>7.3f}%")
        
        # Key forward rates
        print("\n" + "-"*60)
        print("Key Forward Rates (PCHIP Method):")
        print("-"*60)
        
        forwards = [
            ("1Y1Y (1yr forward, 1yr term)", 1, 2),
            ("2Y3Y (2yr forward, 3yr term)", 2, 5),
            ("5Y5Y (5yr forward, 5yr term)", 5, 10),
        ]
        
        for label, t1, t2 in forwards:
            if t2 <= self.tenors.max():
                fwd = self.get_forward_rate_term(t1, t2, InterpolationMethod.PCHIP)
                print(f"  {label}: {fwd:.3f}%")


def bootstrap_par_to_zero(
    par_tenors: np.ndarray,
    par_yields: np.ndarray,
    coupon_freq: int = 2,
) -> tuple:
    """
    Bootstrap continuously compounded zero rates from par yields.

    For T-bills (tenor ≤ 1/coupon_freq): par yield ≈ zero rate (single cashflow).
    For coupon-bearing bonds: iterative stripping using previously bootstrapped
    zero rates, with linear interpolation for intermediate coupon dates.

    Args:
        par_tenors: Maturities in years.
        par_yields: Par yields in percentage (e.g. 4.5 means 4.5%).
        coupon_freq: Coupon frequency (2 = semi-annual for UST).

    Returns:
        Tuple of (tenors, zero_rates) as numpy arrays, both sorted by tenor.
    """
    sorted_idx = np.argsort(par_tenors)
    tenors = np.array(par_tenors, dtype=float)[sorted_idx]
    yields_pct = np.array(par_yields, dtype=float)[sorted_idx]

    zero_rates = np.zeros(len(tenors), dtype=float)
    period = 1.0 / coupon_freq

    for i, (T, y) in enumerate(zip(tenors, yields_pct)):
        if T <= period + 1e-9:
            # Single cashflow: zero rate = par yield
            zero_rates[i] = y
            continue

        c_decimal = y / 100.0
        coupon_per_period = c_decimal / coupon_freq

        # Generate coupon times from period up to T
        coupon_times = np.arange(period, T + 1e-9, period)
        # Ensure the last element is exactly T
        if len(coupon_times) == 0 or abs(coupon_times[-1] - T) > 1e-9:
            coupon_times = np.append(coupon_times, T)
        else:
            coupon_times[-1] = T

        intermediate_times = coupon_times[:-1]

        # Discount intermediate coupons using already-bootstrapped zero rates
        pv_intermediate = 0.0
        if len(intermediate_times) > 0:
            known_tenors = tenors[:i]
            known_zeros = zero_rates[:i]
            if len(known_tenors) >= 1:
                z_interp = np.interp(intermediate_times, known_tenors, known_zeros)
            else:
                z_interp = np.full_like(intermediate_times, y)
            dfs = np.exp(-z_interp / 100.0 * intermediate_times)
            pv_intermediate = np.sum(coupon_per_period * 100.0 * dfs)

        # Solve for zero rate at T:
        # 100 = pv_intermediate + (coupon + 100) * exp(-z(T) * T)
        final_cf = coupon_per_period * 100.0 + 100.0
        remaining = 100.0 - pv_intermediate

        if remaining > 0 and final_cf > 0:
            df_T = remaining / final_cf
            if df_T > 0:
                zero_rates[i] = -np.log(df_T) / T * 100.0
            else:
                zero_rates[i] = y
        else:
            zero_rates[i] = y

    return tenors, zero_rates


def create_sample_ois_curve() -> pd.DataFrame:
    """
    Create realistic OIS (Overnight Index Swap) curve data.
    
    This simulates a typical Euro or USD OIS curve with
    standard market tenors. Data reflects a normal/upward
    sloping yield curve environment.
    """
    return pd.DataFrame({
        'tenor': [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30],
        'rate': [3.85, 3.90, 3.95, 3.75, 3.60, 3.45, 3.50, 3.65, 3.85, 4.00, 4.15]
    })


def create_sample_govt_curve() -> pd.DataFrame:
    """
    Create realistic Government Bond curve data.
    
    Simulates a G7 sovereign curve with typical tenors
    and a moderate steepening bias.
    """
    return pd.DataFrame({
        'tenor': [1, 2, 3, 5, 7, 10, 20, 30],
        'rate': [4.50, 4.35, 4.25, 4.15, 4.20, 4.30, 4.55, 4.70]
    })


if __name__ == "__main__":
    """
    Demonstration Script: Yield Curve Analysis for Asset Management
    ================================================================
    
    This script showcases the YieldCurveAnalyzer capabilities:
    1. Curve construction with both interpolation methods
    2. Forward rate extraction and comparison
    3. Stress testing to demonstrate locality properties
    """
    
    print("\n" + "="*70)
    print(" YIELD CURVE ANALYZER - FIXED INCOME QUANTITATIVE TOOL")
    print(" Interpolation Methods: Natural Cubic Spline vs PCHIP")
    print("="*70)
    
    # =========================================================================
    # Section 1: Initialize with OIS Market Data
    # =========================================================================
    print("\n[1] LOADING MARKET DATA...")
    
    market_data = create_sample_ois_curve()
    print("\nOIS Curve Market Data:")
    print(market_data.to_string(index=False))
    
    # Initialize analyzer
    analyzer = YieldCurveAnalyzer(market_data)
    analyzer.summary()
    
    # =========================================================================
    # Section 2: Compare Forward Rates - The 5Y5Y Trade
    # =========================================================================
    print("\n" + "="*70)
    print("[2] FORWARD RATE COMPARISON: The 5Y5Y Forward")
    print("="*70)
    print("\nThe 5Y5Y forward rate is a key indicator watched by macro traders.")
    print("It represents the market's expectation of the 5-year rate, 5 years from now.\n")
    
    fwd_5y5y_cubic = analyzer.get_forward_rate_term(5, 10, InterpolationMethod.CUBIC_SPLINE)
    fwd_5y5y_pchip = analyzer.get_forward_rate_term(5, 10, InterpolationMethod.PCHIP)
    
    print(f"5Y5Y Forward Rate (Cubic Spline): {fwd_5y5y_cubic:.4f}%")
    print(f"5Y5Y Forward Rate (PCHIP):        {fwd_5y5y_pchip:.4f}%")
    print(f"Difference:                       {abs(fwd_5y5y_cubic - fwd_5y5y_pchip)*100:.2f} bps")
    
    # Compare instantaneous forwards at key points
    print("\nInstantaneous Forward Rates f(t) at Key Maturities:")
    print(f"{'Maturity':<12} {'Cubic Spline':<15} {'PCHIP':<15} {'Delta (bps)':<12}")
    print("-" * 55)
    
    for t in [2, 5, 7, 10, 15, 20]:
        if t <= analyzer.tenors.max():
            f_cubic = analyzer.get_forward_rate(t, InterpolationMethod.CUBIC_SPLINE)
            f_pchip = analyzer.get_forward_rate(t, InterpolationMethod.PCHIP)
            delta = (f_cubic - f_pchip) * 100
            print(f"{t:<12.0f} {f_cubic:<15.4f} {f_pchip:<15.4f} {delta:<12.2f}")
    
    # =========================================================================
    # Section 3: Visualization
    # =========================================================================
    print("\n" + "="*70)
    print("[3] GENERATING COMPARISON CHARTS...")
    print("="*70)
    
    fig = analyzer.plot_curves(
        resolution=250,
        title="OIS Curve Analysis: Cubic Spline vs PCHIP (Shape-Preserving)",
        save_path="ois_curve_analysis.png"
    )
    # plt.show()
    
    # =========================================================================
    # Section 4: Stress Test - Locality Demonstration
    # =========================================================================
    print("\n" + "="*70)
    print("[4] STRESS TEST: +10 bps Shock at 10Y")
    print("="*70)
    print("\nThis test demonstrates the 'locality' property:")
    print("- Cubic Spline: Global basis functions cause ripple effects")
    print("- PCHIP: Local construction limits shock propagation")
    
    stress_results = analyzer.stress_test(
        shock_maturity=10.0,
        shock_bps=10.0,
        display_results=True
    )
    
    # =========================================================================
    # Section 5: Practical Application - Curve Risk Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("[5] CURVE RISK ANALYSIS")
    print("="*70)
    
    print("\nZero Rate Sensitivity Analysis:")
    print("Calculating DV01 impact of +1bp parallel shift at each tenor...\n")
    
    print(f"{'Shock Point':<15} {'2Y Impact':<12} {'5Y Impact':<12} {'10Y Impact':<12}")
    print("-" * 55)
    
    for shock_tenor in [2, 5, 10]:
        if shock_tenor in analyzer.tenors:
            results = analyzer.stress_test(shock_tenor, 1.0, display_results=False)
            deltas = results['deltas']
            
            # Get impacts at key tenors
            impact_2y = deltas[deltas['Maturity'].between(1.9, 2.1)]['Delta_PCHIP_bps'].mean()
            impact_5y = deltas[deltas['Maturity'].between(4.9, 5.1)]['Delta_PCHIP_bps'].mean()
            impact_10y = deltas[deltas['Maturity'].between(9.9, 10.1)]['Delta_PCHIP_bps'].mean()
            
            print(f"{shock_tenor}Y shock      {impact_2y:>10.3f}   {impact_5y:>10.3f}   {impact_10y:>10.3f}")
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. PCHIP (Shape-Preserving) provides more stable forward rates")
    print("2. Cubic Spline may introduce spurious oscillations")
    print("3. For risk management, PCHIP's locality is generally preferred")
    print("4. For derivative pricing requiring C2 smoothness, use Cubic Spline with care")
    print("\n")
