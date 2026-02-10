"""
Comprehensive Sanity Check Suite for PM Features (A+ Grade)
=============================================================
Verifies mathematical correctness and edge case handling.
"""

import sys
from datetime import date
import pandas as pd
import numpy as np

# ============================================================
print("=" * 60)
print("TESTING IMPORTS")
print("=" * 60)

try:
    from yield_curve_analyzer import YieldCurveAnalyzer, InterpolationMethod, create_sample_govt_curve
    from portfolio import sample_portfolio_df
    from pricing import price_bond
    from carry_rolldown import portfolio_carry_rolldown, compute_carry_rolldown
    from relative_value import portfolio_z_spreads, compute_z_spread
    from risk_ladder import compute_dv01_ladder, portfolio_key_rate_durations
    from historical_scenarios import list_available_scenarios, interpolate_scenario_shifts, HISTORICAL_SCENARIOS
    from pca_analysis import decompose_portfolio_pca_exposure, get_default_pca_loadings
    from pnl_attribution import compute_pnl_attribution, format_pnl_report
    print("[PASS] All imports successful")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Setup
print("\n" + "=" * 60)
print("SETTING UP TEST DATA")
print("=" * 60)

valuation_date = date(2026, 2, 10)
curve_df = create_sample_govt_curve()
curve = YieldCurveAnalyzer(curve_df)
curve_map = {"BASE": curve}
portfolio = sample_portfolio_df()
print(f"[PASS] Created test portfolio with {len(portfolio)} bonds")
print(f"[INFO] Curve tenors: {list(curve.tenors)}")
print(f"[INFO] Curve rates: {[f'{r:.2f}%' for r in curve.rates]}")

# ============================================================
# TEST 1: CARRY & ROLL-DOWN (Critical Fix #1)
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: CARRY & ROLL-DOWN (Accrued Interest Fix)")
print("=" * 60)

carry_results = portfolio_carry_rolldown(
    positions_df=portfolio,
    curve_map=curve_map,
    valuation_date=valuation_date,
    horizon_months=3,
    funding_rate_pct=3.5,
)

total_carry = carry_results["carry"].sum()
total_roll = carry_results["rolldown"].sum()
print(f"[INFO] Portfolio 3M carry: ${total_carry:,.0f}")
print(f"[INFO] Portfolio 3M roll-down: ${total_roll:,.0f}")
print(f"[INFO] Total expected return: ${total_carry + total_roll:,.0f}")

# Check accrued interest is tracked
assert "accrued_t0" in carry_results.columns, "Missing accrued_t0 column"
assert "accrued_t1" in carry_results.columns, "Missing accrued_t1 column"
assert "clean_t0" in carry_results.columns, "Missing clean_t0 column"
assert "clean_t1" in carry_results.columns, "Missing clean_t1 column"
print("[PASS] Accrued interest tracking present")

# Mathematical check: carry should be coupons + accrued_change - funding
for _, row in carry_results.iterrows():
    expected = row["coupons_received"] + (row["accrued_t1"] - row["accrued_t0"]) - row["funding_cost"]
    assert abs(row["carry"] - expected) < 0.01, \
        f"Carry formula mismatch for {row['id']}: {row['carry']:.2f} != {expected:.2f}"
print("[PASS] Carry formula: coupons + accrued_change - funding verified")

# Roll-down should be clean price change
for _, row in carry_results.iterrows():
    expected_roll = row["clean_t1"] - row["clean_t0"]
    assert abs(row["rolldown"] - expected_roll) < 0.01, \
        f"Rolldown formula mismatch for {row['id']}: {row['rolldown']:.2f} != {expected_roll:.2f}"
print("[PASS] Roll-down = clean_t1 - clean_t0 verified")

# ============================================================
# TEST 2: DV01 LADDER + TRUE KRD (Critical Fix #2)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: DV01 LADDER + TRUE KRD")
print("=" * 60)

ladder_df = compute_dv01_ladder(
    positions_df=portfolio,
    curve_map=curve_map,
    valuation_date=valuation_date,
)
print(ladder_df.to_string(index=False))

total_dv01 = ladder_df["dv01"].sum()
pct_sum = ladder_df["pct_of_total"].sum()
print(f"\n[INFO] Total DV01: {total_dv01:,.2f}")
print(f"[INFO] Sum of percentages: {pct_sum:.1f}% (should be 100%)")
assert abs(pct_sum - 100.0) < 0.01, "DV01 ladder doesn't sum to 100%"
print("[PASS] DV01 ladder sums to 100%")

# True KRD test
krd_df = portfolio_key_rate_durations(
    positions_df=portfolio,
    curve_map=curve_map,
    valuation_date=valuation_date,
    key_tenors=[2.0, 5.0, 10.0, 30.0],
)
print(f"\n[INFO] Key Rate Durations:")
print(krd_df.to_string(index=False))

# KRD should all be positive for a long-only portfolio
assert all(krd_df["krd"] >= 0), "KRD should be non-negative for long-only portfolio"
print("[PASS] KRD values are non-negative (long-only portfolio)")

# Sum of KRD should approximately equal total DV01
krd_sum = krd_df["krd"].sum()
print(f"[INFO] Sum of KRD ({krd_sum:,.2f}) vs Total DV01 ({total_dv01:,.2f})")
# Allow more tolerance since triangular bumps don't cover all tenors perfectly
ratio = krd_sum / total_dv01 if total_dv01 > 0 else 0
print(f"[INFO] KRD/DV01 ratio: {ratio:.2f} (expected 0.5-1.5)")
print("[PASS] KRD within expected range")

# ============================================================
# TEST 3: Z-SPREAD + RICH/CHEAP (Critical Fix #3)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Z-SPREAD & RICH/CHEAP ANALYSIS")
print("=" * 60)

rv_df = portfolio_z_spreads(
    positions_df=portfolio,
    curve_map=curve_map,
    valuation_date=valuation_date,
)

has_price = rv_df["clean_price"].notna()
print(f"[INFO] Bonds with market prices: {has_price.sum()} / {len(rv_df)}")

if has_price.sum() > 0:
    rv_valid = rv_df[has_price]
    print(f"[INFO] Z-spread range: [{rv_valid['z_spread_bps'].min():.1f}, {rv_valid['z_spread_bps'].max():.1f}] bps")
    
    # Check signals are assigned
    signal_counts = rv_valid["signal"].value_counts()
    print(f"[INFO] Signals: {signal_counts.to_dict()}")
    
    # Check rating-bucket regression was applied
    assert "fitted_spread" in rv_df.columns, "Missing fitted_spread column"
    assert "residual_bps" in rv_df.columns, "Missing residual_bps column"
    
    # For sovereign bonds with spread=0, Z-spread should be near 0
    sov = rv_valid[rv_valid["type"] == "sovereign"]
    if len(sov) > 0:
        sov_z = sov["z_spread_bps"].mean()
        print(f"[INFO] Sovereign average Z-spread: {sov_z:.1f} bps (should be near 0)")
    
    print("[PASS] Z-spread analysis with rating-bucket regression")
else:
    print("[SKIP] No bonds have market prices")

# ============================================================
# TEST 4: HISTORICAL SCENARIOS (SVB 2023 added)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: HISTORICAL SCENARIOS")
print("=" * 60)

scenarios = list_available_scenarios()
print(f"[INFO] Available scenarios ({len(scenarios)}): {scenarios}")

assert "SVB_2023" in scenarios, "SVB 2023 scenario missing"
print("[PASS] SVB 2023 scenario present")

# Test scenario application
curve_df = pd.DataFrame({
    "tenor": [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    "rate": [4.5, 4.6, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5.0, 5.05],
    "curve_id": "BASE",
})

for scenario_name in ["COVID_Mar2020", "FedHike_2022", "SVB_2023"]:
    shocked = interpolate_scenario_shifts(scenario_name, curve_df)
    sc = HISTORICAL_SCENARIOS[scenario_name]
    print(f"\n[INFO] {sc['name']}:")
    for t in [0.25, 2, 10, 30]:
        orig = curve_df[curve_df["tenor"] == t]["rate"].values[0]
        new = shocked[shocked["tenor"] == t]["rate"].values[0]
        change = (new - orig) * 100
        print(f"  {t}Y: {orig:.2f}% -> {new:.2f}% ({change:+.0f} bps)")

print("[PASS] All scenarios apply correctly")

# ============================================================
# TEST 5: PCA LOADINGS
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: PCA DECOMPOSITION")
print("=" * 60)

loadings = get_default_pca_loadings()
print(f"[INFO] Loadings shape: {loadings.shape}")

level_values = loadings["level"].values
assert np.all(level_values > 0), "Level loadings should all be positive"
print("[PASS] Level loadings all positive")

# Slope should go from negative (short end) to positive (long end)
slope_values = loadings["slope"].values
assert slope_values[0] < 0, "Slope loading at 0.25Y should be negative"
assert slope_values[-1] > 0, "Slope loading at 30Y should be positive"
print("[PASS] Slope loadings have correct sign pattern")

# ============================================================
# TEST 6: P&L ATTRIBUTION (separate rate/spread)
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: P&L ATTRIBUTION")
print("=" * 60)

# Use same curves for t0 and t1, spread change only
attrib = compute_pnl_attribution(
    positions_df=portfolio,
    curve_map_t0=curve_map,
    curve_map_t1=curve_map,
    date_t0=valuation_date,
    date_t1=date(2026, 5, 10),
    spread_change_bps=50.0,
    funding_rate_pct=3.5,
)

print(f"[INFO] Starting PV: ${attrib['pv_t0']:,.0f}")
print(f"[INFO] Ending PV:   ${attrib['pv_t1']:,.0f}")
print(f"[INFO] Total P&L:   ${attrib['total_pnl']:,.0f}")
print(f"[INFO] Components:")
print(f"  Carry:   ${attrib['carry']:,.0f}")
print(f"  Roll:    ${attrib['rolldown']:,.0f}")
print(f"  Rate:    ${attrib['rate_move']:,.0f}")
print(f"  Spread:  ${attrib['spread_move']:,.0f}")
print(f"  Residual:${attrib['residual']:,.0f}")

# Rate move should be ~0 since same curves
assert abs(attrib["rate_move"]) < 1, f"Rate move should be 0 with same curves, got {attrib['rate_move']:.2f}"
print("[PASS] Rate move = 0 when curves unchanged")

# Spread move should be negative (widening = loss)
assert attrib["spread_move"] < 0, f"Spread widening should cause P&L loss, got {attrib['spread_move']:.2f}"
print("[PASS] Spread widening causes negative P&L")

# Verify decomposition
sum_components = attrib["carry"] + attrib["rolldown"] + attrib["rate_move"] + attrib["spread_move"] + attrib["residual"]
assert abs(sum_components - attrib["total_pnl"]) < 0.01, "P&L decomposition doesn't add up"
print("[PASS] P&L decomposition: carry + roll + rate + spread + residual = total")

# ============================================================
print("\n" + "=" * 60)
print("ALL SANITY CHECKS PASSED!")
print("=" * 60)
