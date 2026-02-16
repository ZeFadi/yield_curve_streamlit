from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from fund_manager import ConstraintConfig, PortfolioManager, parse_prospectus_constraints


st.title("Fund Management & Rebalancing Engine")
st.caption("Prospectus constraints, benchmark tracking, and monthly rebalance decision support.")


def _sample_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Ticker": "BOND_A", "Amount": 100000, "Market Value": 1_200_000, "Issuer": "Issuer_A", "Sector": "Financials", "Yield": 3.9},
            {"Ticker": "BOND_B", "Amount": 80000, "Market Value": 900_000, "Issuer": "Issuer_B", "Sector": "Industrials", "Yield": 4.2},
            {"Ticker": "BOND_C", "Amount": 50000, "Market Value": 700_000, "Issuer": "Issuer_A", "Sector": "Financials", "Yield": 4.0},
            {"Ticker": "BOND_D", "Amount": 70000, "Market Value": 1_000_000, "Issuer": "Issuer_C", "Sector": "Utilities", "Yield": 3.6},
        ]
    )


def _sample_benchmark() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Ticker": "BOND_A", "Weight": 0.22},
            {"Ticker": "BOND_B", "Weight": 0.28},
            {"Ticker": "BOND_C", "Weight": 0.18},
            {"Ticker": "BOND_D", "Weight": 0.32},
        ]
    )


st.markdown("### Inputs")
col_left, col_right = st.columns(2)

with col_left:
    p_src = st.radio("Portfolio input", ["Sample", "Upload CSV"], horizontal=True)
    p_df = _sample_portfolio()
    if p_src == "Upload CSV":
        f = st.file_uploader("Upload portfolio CSV", type=["csv"], key="fm_portfolio")
        if f is not None:
            p_df = pd.read_csv(f)

with col_right:
    b_src = st.radio("Benchmark input", ["Sample", "Upload CSV"], horizontal=True)
    b_df = _sample_benchmark()
    if b_src == "Upload CSV":
        f = st.file_uploader("Upload benchmark CSV", type=["csv"], key="fm_benchmark")
        if f is not None:
            b_df = pd.read_csv(f)

prospectus_text = st.text_area(
    "Prospectus text (optional, for auto-parsing limits)",
    value="",
    height=120,
    help="Paste prospectus excerpts here to auto-detect hard limits (issuer/cash).",
)

parsed_cfg = parse_prospectus_constraints(prospectus_text)

ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
    issuer_limit = st.number_input("Issuer Limit (%)", min_value=0.0, max_value=100.0, value=float(parsed_cfg.issuer_limit_pct), step=0.5)
with ctrl2:
    min_cash_pct = st.number_input("Minimum Cash (%)", min_value=0.0, max_value=100.0, value=float(parsed_cfg.min_cash_pct), step=0.5)
with ctrl3:
    cash_available = st.number_input("Cash Available", min_value=0.0, value=250_000.0, step=25_000.0)

cfg = ConstraintConfig(issuer_limit_pct=float(issuer_limit), min_cash_pct=float(min_cash_pct))
pm = PortfolioManager(portfolio_df=p_df, benchmark_df=b_df, cash_available=float(cash_available))

st.markdown("### Portfolio & Benchmark Snapshot")
show1, show2 = st.columns(2)
with show1:
    st.dataframe(p_df, width="stretch")
with show2:
    st.dataframe(b_df, width="stretch")

constraints_df = pm.compliance_table(cfg)
active_df = pm.active_exposure()
te_proxy = pm.tracking_error_proxy() * 100.0

is_ok = (constraints_df["Status"] == "OK").all()
status_text = "COMPLIANT" if is_ok else "VIOLATION"

st.markdown("### KPI Row")
k1, k2, k3 = st.columns(3)
k1.metric("Compliance Status", status_text)
k2.metric("Cash Available", f"${cash_available:,.0f}")
k3.metric("Tracking Error (proxy)", f"{te_proxy:.2f}%")

st.markdown("### Constraint Table")
st.dataframe(constraints_df, width="stretch")

st.markdown("### Benchmark Tracking")
st.dataframe(
    active_df[["ticker", "portfolio_weight", "benchmark_weight", "active_weight"]].style.format(
        {
            "portfolio_weight": "{:.2%}",
            "benchmark_weight": "{:.2%}",
            "active_weight": "{:+.2%}",
        }
    ),
    width="stretch",
)

chart_df = active_df[["ticker", "active_weight"]].set_index("ticker")
st.bar_chart(chart_df)

st.markdown("### Rebalancing Simulator")
rb1, rb2, rb3 = st.columns(3)
with rb1:
    cash_to_invest = st.number_input("Cash to invest", min_value=0.0, value=200_000.0, step=25_000.0)
with rb2:
    mode = st.selectbox("Objective", ["Minimize Tracking Error", "Maximize Yield"])
with rb3:
    rebalance_flag = st.checkbox("Rebalance to benchmark", value=True)

run = st.button("Generate Proposed Trade List", type="primary")
if run:
    trades = pm.propose_trades(
        cash_to_invest=float(cash_to_invest),
        rebalance_to_benchmark=bool(rebalance_flag),
        maximize_yield=(mode == "Maximize Yield"),
        cfg=cfg,
    )

    st.markdown("#### Proposed Trade List")
    if trades.empty:
        st.info("No trades required under current constraints and objective.")
    else:
        st.dataframe(
            trades.style.format({"Trade Value": "${:,.0f}"}),
            width="stretch",
        )

st.caption(f"As-of: {date.today().isoformat()} | This page is isolated from your core yield-curve workflow.")
