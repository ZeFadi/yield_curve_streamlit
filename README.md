# Yield Curve & Portfolio Manager

A research-oriented and production-minded Streamlit platform for yield curve analytics, fixed-income portfolio valuation, risk decomposition, and scenario-driven decision support.

**Contact:** [fadi.hafiane@outlook.fr](mailto:fadi.hafiane@outlook.fr)

---

## Abstract

This application implements a coherent fixed-income analytics stack centered on:
- zero-curve construction and interpolation,
- bond-level valuation under continuous compounding,
- first- and second-order risk estimation (DV01, CS01, convexity),
- carry and roll-down decomposition,
- key-rate and principal-component exposure analysis,
- relative-value diagnostics via Z-spread and residual screening,
- deterministic and historical stress testing,
- period P&L attribution.

The design objective is **internal consistency**: formulas, variable naming, and tab outputs follow the same modeling conventions across the full app.

---

## Mathematical Foundations

### Notation and Units

- `R(t)`: zero rate at maturity `t` in **percent**.
- `spread_bps`, `curve_bump_bps`, `spread_bump_bps`: in **basis points**.
- `t`: time in years (ACT/365 or ACT/365.25 depending on module).
- Conversion reminders:
  - `1 bp = 0.01%`
  - Decimal rate = `percent / 100`

### Discounting and Forward Rates

Discount factor under continuous compounding:

$$
DF(t)=\exp\left(-\left(\frac{R(t)+s_{bps}/100}{100}\right)t\right)
$$

Instantaneous forward:

$$
f(t)=R(t)+tR'(t)
$$

Term forward:

$$
F(t_1,t_2)=\frac{R(t_2)t_2-R(t_1)t_1}{t_2-t_1},\quad t_1<t_2
$$

### Bond PV, Clean/Dirty, and Risk

Bond present value (implementation in `price_bond`):

$$
PV=\sum_j CF_j\exp\left(-\left(\frac{R(t_j)+\texttt{total\_bps}/100}{100}\right)t_j\right)
$$

with `total_bps = curve_bump_bps + spread_bps + spread_bump_bps`.

Clean/dirty relation:

$$
\texttt{dirty\_price}=PV,\qquad \texttt{clean\_price}=PV-\texttt{accrued\_interest}
$$

Finite-difference sensitivities:

$$
\texttt{dv01}=\frac{PV_{down}-PV_{up}}{2}
$$

$$
\texttt{modified\_duration}=\frac{\texttt{dv01}}{PV(\texttt{bump\_bps}/10000)}
$$

$$
\texttt{convexity}=\frac{PV_{up}+PV_{down}-2PV}{PV\Delta^2},\quad \Delta=\texttt{bump\_bps}/10000
$$

$$
\texttt{cs01}=\frac{PV^{spread\ down}-PV^{spread\ up}}{2}
$$

---

## Page-by-Page Model Specification

The app tabs in `app.py` are listed below with their governing formulas.

### 1. Curves

- Interpolation: Natural cubic spline and PCHIP.
- Stress response comparison via shocked tenor knot and post-interpolation delta curves.
- Outputs: zero curve, instantaneous forward curve, shock propagation diagnostics.

### 2. Portfolio

- Bond-level valuation through discounted cashflows.
- Accrued-interest computation and clean/dirty split.
- Bond and portfolio aggregates for `pv`, `dv01`, `cs01`, duration, convexity.

### 3. Carry & Roll-Down

Per bond:

$$
\texttt{carry}=\texttt{coupons\_received}+(\texttt{accrued\_t1}-\texttt{accrued\_t0})-\texttt{funding\_cost}
$$

$$
\texttt{rolldown}=\texttt{clean\_t1}-\texttt{clean\_t0}
$$

$$
\texttt{total\_return}=\texttt{carry}+\texttt{rolldown}
$$

Annualized return metrics are reported in bps relative to `pv_now`.

### 4. DV01 Ladder

Bucket risk:

$$
\texttt{bucket\_dv01}_b=\sum_{i\in b}\texttt{dv01}_i
$$

$$
\texttt{pct\_of\_total}_b=\frac{\texttt{bucket\_dv01}_b}{\sum_k\texttt{bucket\_dv01}_k}\cdot100
$$

Key-rate duration (KRD): portfolio repriced under localized tenor bumps.

### 5. PCA Analysis

Portfolio exposure to stylized level/slope/curvature loadings:

$$
\texttt{level\_exposure}=\sum_i dv01_i\cdot level_i
$$

$$
\texttt{slope\_exposure}=\sum_i dv01_i\cdot slope_i
$$

$$
\texttt{curvature\_exposure}=\sum_i dv01_i\cdot curvature_i
$$

### 6. Relative Value

Z-spread (`compute_z_spread`) solves:

$$
\texttt{objective}(z)=PV(z)-\texttt{target\_pv}=0
$$

Rich/cheap residual:

$$
\texttt{residual\_bps}=\texttt{z\_spread\_bps}-\widehat{z}(\texttt{duration})
$$

Signal thresholds: CHEAP (> +15 bps), RICH (< -15 bps), FAIR otherwise.

### 7. Historical Stress

Historical scenario anchor moves are linearly interpolated by tenor and applied to the current curve.

$$
\texttt{pnl}=\texttt{stressed\_pv}-\texttt{base\_pv}
$$

### 8. Scenarios

Supported transformations:
- Parallel shift.
- Twist (short-end vs long-end linear blend).
- Key-rate shock (triangular local bumps).
- Spread shock on corporates only.

### 9. P&L Attribution

Sequential decomposition:

$$
\texttt{total\_pnl}=\texttt{carry}+\texttt{rolldown}+\texttt{rate\_move}+\texttt{spread\_move}+\texttt{residual}
$$

where each component is computed from staged repricing (`t0` curves/spreads to `t1` curves/spreads).

### 10. Documentation (in-app)

The app now includes a dedicated `Documentation` tab summarizing formulas, assumptions, and portfolio-management interpretation notes.

---

## Practical User Guide

### Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Typical Workflow

1. Load market curve data (sample, Treasury, CSV, or DB).
2. Choose interpolation method and stress controls in sidebar.
3. Load/edit portfolio in `Portfolio`.
4. Review risk and analytics tabs (`Carry & Roll-Down`, `DV01 Ladder`, `PCA Analysis`, `Relative Value`).
5. Run `Historical Stress` and custom `Scenarios`.
6. Reconcile period economics in `P&L Attribution`.
7. Use `Documentation` tab to verify formulas/interpretation with stakeholders.

### Input Data Contracts

Curve data requires:
- `tenor` (years),
- `rate` (percent),
- optional `curve_id` for multi-curve setups.

Portfolio data requires standard fixed-income fields (ID, issuer, notional, coupon terms, maturity, etc.) and supports optional `clean_price`, `curve_id`, and `spread_bps`.

---

## Financial Interpretation Notes

This app is suitable for senior PM-style workflows:
- **Positioning:** duration, KRD, and PCA jointly characterize directional and shape risk.
- **Relative value:** Z-spread residuals provide transparent rich/cheap signals with explicit model caveats.
- **Attribution:** carry/roll/rates/spreads decomposition supports committee-grade explainability.

CFA Level III-aligned perspective:
- distinguish expected carry/roll from realized market effects,
- separate structural factor exposure from idiosyncratic spread behavior,
- control model risk via residual monitoring and scenario triangulation.

---

## Validation and Consistency Controls

Before production use or release:

1. Verify `%` vs `bps` conversions on any new module.
2. Confirm carry/accrued sign conventions match `carry_rolldown.py` and `pnl_attribution.py`.
3. Keep key-rate shock geometry aligned between `risk_ladder.py` and `scenarios.py`.
4. Ensure new analytics pages include matching math notes in both this README and the in-app `Documentation` tab.

---

## Limitations and Model Risk

- Interpolation is deterministic and does not estimate parameter uncertainty.
- Day-count assumptions are simplified in some modules (ACT/365 or ACT/365.25).
- Credit modeling is reduced-form (`spread_bps`, `z_spread`) and omits liquidity and optionality adjustments.
- Historical stresses are scenario replays, not probabilistic forecasts.

---

## Project Structure

- `app.py`: Streamlit application and tab orchestration.
- `yield_curve_analyzer.py`: curve construction/interpolation/forwards/stress response.
- `pricing.py`: pricing engine and risk metrics.
- `carry_rolldown.py`: expected return decomposition.
- `risk_ladder.py`: DV01 bucketing and KRD.
- `pca_analysis.py`: level/slope/curvature decomposition.
- `relative_value.py`: Z-spread and rich/cheap framework.
- `historical_scenarios.py`, `scenarios.py`: stress engines.
- `pnl_attribution.py`: period P&L decomposition.

---

## Disclaimer

This software is for analytics, research, and decision support. It is not investment advice. Production deployment should include independent model validation, governance, and controls.

