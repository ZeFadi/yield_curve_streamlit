import os
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from yield_curve_analyzer import (
    YieldCurveAnalyzer,
    InterpolationMethod,
    create_sample_ois_curve,
    create_sample_govt_curve,
    bootstrap_par_to_zero,
)
from portfolio import sample_portfolio_df, standardize_portfolio_df
from pricing import price_portfolio_df, portfolio_pv, bond_risk_metrics, _resolve_curve
from scenarios import apply_parallel_shock, apply_twist_shock, apply_key_rate_shock

from carry_rolldown import portfolio_carry_rolldown
from relative_value import portfolio_z_spreads
from risk_ladder import compute_dv01_ladder, portfolio_key_rate_durations, _years_to_maturity
from historical_scenarios import (
    interpolate_scenario_shifts,
    list_available_scenarios,
    get_scenario_description,
    HISTORICAL_SCENARIOS,
)
from pca_analysis import (
    decompose_portfolio_pca_exposure,
    get_default_pca_loadings,
    interpret_pca_exposures,
)
from pnl_attribution import compute_pnl_attribution, format_pnl_report

# =============================================================================
# DESIGN SYSTEM — Finance-grade data visualization
# =============================================================================
_PALETTE = {
    "navy":       "#1B2A4A",
    "blue":       "#2E5090",
    "teal":       "#1A7A6E",
    "slate":      "#5A6D7E",
    "orange":     "#C27230",
    "red":        "#B5443A",
    "green":      "#2A7752",
    "light_gray": "#E8ECF1",
    "mid_gray":   "#A0AEBF",
    "bg":         "#FFFFFF",
}
_FONT_FAMILY = "Inter"

# Set matplotlib defaults globally so every chart inherits Inter
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
_FONT_TITLE = 11
_FONT_LABEL = 9.5
_FONT_TICK = 8.5
_FONT_LEGEND = 8.5
_FONT_ANNOTATION = 8
_LINEWIDTH = 1.4
_MARKER_SIZE = 4


def _apply_style(ax, title=None, xlabel=None, ylabel=None, legend=True, grid_axis="both"):
    """Apply unified academic finance styling to a matplotlib Axes."""
    ax.set_facecolor(_PALETTE["bg"])
    ax.figure.set_facecolor(_PALETTE["bg"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(_PALETTE["mid_gray"])
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis="both", which="both", labelsize=_FONT_TICK, colors=_PALETTE["slate"],
                   length=3, width=0.5)
    ax.grid(True, axis=grid_axis, color=_PALETTE["light_gray"], linewidth=0.5, linestyle="-")
    if title:
        ax.set_title(title, fontsize=_FONT_TITLE, fontweight="600", color=_PALETTE["navy"],
                     fontfamily=_FONT_FAMILY, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_FONT_LABEL, color=_PALETTE["slate"], fontfamily=_FONT_FAMILY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_FONT_LABEL, color=_PALETTE["slate"], fontfamily=_FONT_FAMILY)
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=_FONT_LEGEND, frameon=True, facecolor=_PALETTE["bg"],
                  edgecolor=_PALETTE["light_gray"], framealpha=0.95, loc="best")


def _new_fig(rows=1, cols=1, figsize=(10, 4.5), sharex=False):
    """Create a figure with the finance theme pre-applied."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex,
                             facecolor=_PALETTE["bg"])
    return fig, axes


# =============================================================================
# TREASURY DATA CONFIG
# =============================================================================
TREASURY_XML_BASE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
TREASURY_DATA_NOMINAL = "daily_treasury_yield_curve"
TREASURY_FIELD_MAP = {
    "BC_6WK": 6 / 52,
    "BC_1MONTH": 1 / 12,
    "BC_1_5MONTH": 1.5 / 12,
    "BC_2MONTH": 2 / 12,
    "BC_3MONTH": 3 / 12,
    "BC_4MONTH": 4 / 12,
    "BC_6MONTH": 6 / 12,
    "BC_1YEAR": 1,
    "BC_2YEAR": 2,
    "BC_3YEAR": 3,
    "BC_5YEAR": 5,
    "BC_7YEAR": 7,
    "BC_10YEAR": 10,
    "BC_20YEAR": 20,
    "BC_30YEAR": 30,
}

st.set_page_config(layout="wide", page_title="Yield Curve & Portfolio Manager")

# Inject app typography while preserving Streamlit Material icon ligatures.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --app-font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    font-family: var(--app-font-family);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--app-font-family);
    font-weight: 600;
}

.stMarkdown, .stMetric, .stDataFrame, .stTable, .stCaption, .stAlert,
[data-baseweb="tab"], [data-baseweb="select"], [data-baseweb="input"],
[data-baseweb="radio"], [data-baseweb="slider"] {
    font-family: var(--app-font-family);
}

.stDataFrame, .stDataFrame input {
    font-variant-numeric: tabular-nums;
}

[data-testid="stIconMaterial"],
.material-icons,
.material-symbols-rounded,
[data-testid="stExpander"] summary span:first-child {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-style: normal;
    font-weight: 400;
    letter-spacing: normal;
    line-height: 1;
    text-transform: none;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# --- UI Elements ---
st.title("Yield Curve & Portfolio Manager")
st.markdown("""
This application provides a professional interface for yield curve construction, portfolio pricing,
and deterministic stress testing across sovereign and corporate bonds.
Use the sidebar to upload curves or connect to a database, then load portfolios in the Portfolio tab.
""")
st.info(
    "Assumes inputs are continuously compounded zero rates (in %). "
    "If you have par yields or swap rates, bootstrap zero rates first. "
    "Treasury data is automatically bootstrapped."
)

st.sidebar.title("Controls")
st.sidebar.header("1. Market Data")


def _standardize_curve_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    col_map = {c.lower(): c for c in df.columns}
    if "tenor" not in col_map or "rate" not in col_map:
        raise ValueError("Missing required columns: tenor, rate")

    rename_map = {col_map["tenor"]: "tenor", col_map["rate"]: "rate"}
    if "curve_id" in col_map:
        rename_map[col_map["curve_id"]] = "curve_id"
    if "as_of" in col_map:
        rename_map[col_map["as_of"]] = "as_of"

    df = df.rename(columns=rename_map)
    keep_cols = ["tenor", "rate"]
    if "curve_id" in df.columns:
        keep_cols.append("curve_id")
    if "as_of" in df.columns:
        keep_cols.append("as_of")
    df = df[keep_cols].copy()

    df["tenor"] = pd.to_numeric(df["tenor"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    before = len(df)
    df = df.dropna()
    if len(df) < before:
        warnings.append("Dropped rows with missing or non-numeric values.")

    non_positive = df["tenor"] <= 0
    if non_positive.any():
        warnings.append("Removed non-positive tenors.")
        df = df.loc[~non_positive].copy()

    if "curve_id" not in df.columns:
        df["curve_id"] = "BASE"
    if df.duplicated(subset=["curve_id", "tenor"]).any():
        warnings.append("Duplicate tenors found; keeping the latest row per curve.")
        df = df.sort_values(["curve_id", "tenor"]).drop_duplicates(["curve_id", "tenor"], keep="last")
    else:
        df = df.sort_values(["curve_id", "tenor"])

    df = df.reset_index(drop=True)
    if df.groupby("curve_id").size().min() < 2:
        raise ValueError("At least 2 valid data points are required.")

    return df, warnings


def _month_key(d: date) -> str:
    return f"{d.year}{d.month:02d}"


def _shift_month(d: date, delta_months: int) -> date:
    y = d.year + (d.month - 1 + delta_months) // 12
    m = (d.month - 1 + delta_months) % 12 + 1
    return date(y, m, 1)


def _parse_treasury_xml(xml_bytes: bytes) -> List[Tuple[date, dict]]:
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }
    root = ET.fromstring(xml_bytes)
    entries = root.findall("a:entry", ns)
    parsed: List[Tuple[date, dict]] = []
    for entry in entries:
        props = entry.find("a:content/m:properties", ns)
        if props is None:
            continue
        obs_date: Optional[date] = None
        rates: dict = {}
        for child in list(props):
            tag = child.tag.split("}")[-1]
            text = (child.text or "").strip()
            if not text:
                continue
            if tag.upper().endswith("DATE") and obs_date is None:
                try:
                    obs_date = datetime.fromisoformat(text[:10]).date()
                except ValueError:
                    pass
            if tag in TREASURY_FIELD_MAP:
                try:
                    rates[TREASURY_FIELD_MAP[tag]] = float(text)
                except ValueError:
                    continue
        if obs_date and rates:
            parsed.append((obs_date, rates))
    return parsed


@st.cache_data(ttl=3600)
def _load_treasury_daily_curve(target_date: Optional[date]) -> pd.DataFrame:
    months_to_try = []
    if target_date:
        months_to_try.append(_shift_month(target_date, 0))
    else:
        today = date.today()
        for delta in [0, -1, -2]:
            months_to_try.append(_shift_month(today, delta))

    all_entries: List[Tuple[date, dict]] = []
    for month_start in months_to_try:
        params = {
            "data": TREASURY_DATA_NOMINAL,
            "field_tdr_date_value_month": _month_key(month_start),
        }
        url = f"{TREASURY_XML_BASE}?{urlencode(params)}"
        with urlopen(url, timeout=20) as resp:
            xml_bytes = resp.read()
        all_entries.extend(_parse_treasury_xml(xml_bytes))

    if not all_entries:
        raise ValueError("No Treasury yield curve data returned.")

    if target_date:
        matches = [r for d, r in all_entries if d == target_date]
        if not matches:
            raise ValueError(f"No Treasury data for {target_date.isoformat()}.")
        rates = matches[0]
        as_of = target_date
    else:
        as_of, rates = max(all_entries, key=lambda x: x[0])

    df = pd.DataFrame(
        {"tenor": list(rates.keys()), "rate": list(rates.values())}
    ).sort_values("tenor")
    df["as_of"] = as_of
    df["curve_id"] = "UST"
    return df.reset_index(drop=True)


@st.cache_data(ttl=300)
def _load_curve_from_db(conn_str: str, sql: str) -> pd.DataFrame:
    import sqlalchemy as sa

    engine = sa.create_engine(conn_str)
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn)


@st.cache_resource
def _build_analyzer(data: pd.DataFrame, extrapolate: bool) -> YieldCurveAnalyzer:
    return YieldCurveAnalyzer(data, extrapolate=extrapolate)


def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except StreamlitSecretNotFoundError:
        return default
    return default

# --- Data Input ---
data_source = st.sidebar.radio(
    "Data Source",
    options=["Sample: OIS", "Sample: Govt", "US Treasury (Daily)", "Upload CSV", "Database"],
    horizontal=False,
)

uploaded_file = None
db_data = None
as_of_date = None
needs_bootstrap = False

if data_source == "Sample: OIS":
    data = create_sample_ois_curve()
elif data_source == "Sample: Govt":
    data = create_sample_govt_curve()
elif data_source == "US Treasury (Daily)":
    st.sidebar.caption("Official U.S. Treasury daily par yield curve (nominal).")
    st.sidebar.info(
        "Par yields are automatically bootstrapped to zero rates "
        "using iterative stripping (semi-annual coupon frequency)."
    )
    use_latest = st.sidebar.checkbox("Use latest available", value=True)
    target_date = None
    if not use_latest:
        target_date = st.sidebar.date_input("As-of date", value=date.today())
    if st.sidebar.button("Load Treasury Curve"):
        try:
            data = _load_treasury_daily_curve(target_date)
            needs_bootstrap = True
            st.sidebar.success("Treasury curve loaded.")
        except Exception as e:
            st.sidebar.error(f"Treasury load failed: {e}")
            data = create_sample_ois_curve()
    else:
        data = create_sample_ois_curve()
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your own data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            data = create_sample_ois_curve()
    else:
        data = create_sample_ois_curve()
else:
    st.sidebar.caption("Connect to your curve database (SQLAlchemy URL required).")
    default_conn = os.getenv("YC_DB_URL", "")
    default_conn = _get_secret("YC_DB_URL", default_conn)
    conn_str = st.sidebar.text_input("DB Connection URL", value=default_conn)
    sql = st.sidebar.text_area(
        "SQL Query",
        value="SELECT tenor, rate FROM curves WHERE curve_id = 'USD_OIS' ORDER BY tenor",
        height=120,
    )
    if st.sidebar.button("Load from DB"):
        try:
            db_data = _load_curve_from_db(conn_str, sql)
            st.sidebar.success("Database data loaded.")
        except Exception as e:
            st.sidebar.error(f"Database load failed: {e}")
            db_data = None
    data = db_data if db_data is not None else create_sample_ois_curve()

st.sidebar.markdown("#### Edit Market Data")
with st.sidebar.expander("Show/Edit Data", expanded=False):
    edited_data = st.data_editor(data, key="data_editor", width="stretch")

try:
    cleaned_data, data_warnings = _standardize_curve_data(edited_data)
    for w in data_warnings:
        st.warning(w)
except (ValueError, TypeError) as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# --- Bootstrap par yields to zero rates for Treasury data ---
if needs_bootstrap:
    for curve_id in cleaned_data["curve_id"].unique():
        mask = cleaned_data["curve_id"] == curve_id
        tenors = cleaned_data.loc[mask, "tenor"].values
        par_yields = cleaned_data.loc[mask, "rate"].values
        _, zeros = bootstrap_par_to_zero(tenors, par_yields, coupon_freq=2)
        cleaned_data.loc[mask, "rate"] = zeros
    st.success("✅ Par yields bootstrapped to continuously compounded zero rates.")

extrapolate = st.sidebar.checkbox(
    "Allow extrapolation beyond max tenor",
    value=True,
    help="Disable to prevent pricing outside your market data range.",
)

curve_map: Dict[str, YieldCurveAnalyzer] = {}
try:
    for curve_id, group in cleaned_data.groupby("curve_id"):
        curve_map[curve_id] = _build_analyzer(group[["tenor", "rate"]], extrapolate=extrapolate)
except (ValueError, TypeError) as e:
    st.error(f"Error building curve: {e}")
    st.stop()

curve_ids = sorted(curve_map.keys())
selected_curve_id = st.sidebar.selectbox("Curve to display", curve_ids, index=0)
analyzer = curve_map[selected_curve_id]

as_of_date = None
if "as_of" in cleaned_data.columns:
    as_of_series = cleaned_data.loc[cleaned_data["curve_id"] == selected_curve_id, "as_of"]
    if not as_of_series.empty:
        as_of_date = as_of_series.iloc[0]

if as_of_date is not None:
    st.caption(f"Data as of: {as_of_date}")


# --- Stress Test Controls ---
st.sidebar.header("2. Stress Test Simulation")

available_tenors = analyzer.tenors
shock_maturity = st.sidebar.select_slider(
    "Shock Maturity (Years)",
    options=available_tenors,
    value=available_tenors[len(available_tenors) // 2]
)

shock_bps = st.sidebar.slider(
    "Shock Size (Basis Points)",
    min_value=-100,
    max_value=100,
    value=10,
    step=5,
    help="A basis point (bp) is 0.01%. Positive values are hikes, negative values are cuts."
)

st.sidebar.header("3. Plotting Options")
interpolation_method_str = st.sidebar.selectbox(
    "Interpolation Method",
    options=[e.value for e in InterpolationMethod],
    index=1,
    format_func=lambda x: x.replace("_", " ").title()
)
interpolation_method = InterpolationMethod(interpolation_method_str)
plot_resolution = st.sidebar.slider("Plot Resolution", min_value=100, max_value=600, value=240, step=20)

# --- Main App Logic & Plotting ---

tabs = st.tabs([
    "Curves",
    "Portfolio",
    "Carry & Roll-Down",
    "DV01 Ladder",
    "PCA Analysis",
    "Relative Value",
    "Historical Stress",
    "Scenarios",
    "P&L Attribution",
    "Documentation",
])

# =============================================================================
# TAB 0: CURVES
# =============================================================================
with tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Yield Curve: Zero vs. Forward Rates")

        fig1, (ax1, ax2) = _new_fig(rows=2, cols=1, figsize=(10, 7), sharex=True)

        t_grid = np.linspace(analyzer.tenors.min(), analyzer.tenors.max(), plot_resolution)

        # Zero curve
        zero_rates = analyzer.get_zero_rate(t_grid, method=interpolation_method)
        ax1.plot(t_grid, zero_rates, color=_PALETTE["blue"], linewidth=_LINEWIDTH,
                 label=f"{interpolation_method.value.title()} Zero Curve")
        ax1.scatter(analyzer.tenors, analyzer.rates, color=_PALETTE["navy"], zorder=5,
                    label="Market Data", marker="o", s=_MARKER_SIZE**2, linewidths=0.5, edgecolors="white")
        _apply_style(ax1, title=f"Zero Coupon Curve — {interpolation_method.value.title()}",
                     ylabel="Rate (%)", grid_axis="y")

        # Forward curve
        forward_rates = analyzer.get_forward_rate(t_grid, method=interpolation_method)
        ax2.plot(t_grid, forward_rates, color=_PALETTE["orange"], linewidth=_LINEWIDTH,
                 label=f"{interpolation_method.value.title()} Forward Curve")
        _apply_style(ax2, title=f"Instantaneous Forward Curve — {interpolation_method.value.title()}",
                     xlabel="Maturity (Years)", ylabel="Rate (%)", grid_axis="y")

        fig1.tight_layout(h_pad=2.5)
        st.pyplot(fig1)

    with col2:
        st.subheader(f"Stress Test: {shock_bps:+.0f} bps at {shock_maturity}Y")

        stress_results = analyzer.stress_test(
            shock_maturity, shock_bps, display_results=False,
            resolution=max(120, plot_resolution // 2),
        )
        deltas_df = stress_results["deltas"]

        fig2, ax = _new_fig(figsize=(10, 4))
        ax.plot(deltas_df["Maturity"], deltas_df["Delta_Cubic_bps"],
                color=_PALETTE["slate"], linewidth=1.0, linestyle="--",
                label="Cubic Spline Δ", alpha=0.7)
        ax.plot(deltas_df["Maturity"], deltas_df["Delta_PCHIP_bps"],
                color=_PALETTE["teal"], linewidth=_LINEWIDTH, label="PCHIP Δ")
        ax.axvline(shock_maturity, color=_PALETTE["red"], linewidth=0.8, linestyle=":",
                   label=f"Shock Point ({shock_maturity}Y)", alpha=0.6)
        ax.axhline(0, color=_PALETTE["mid_gray"], linewidth=0.4)
        ax.fill_between(deltas_df["Maturity"], deltas_df["Delta_PCHIP_bps"], 0,
                        color=_PALETTE["teal"], alpha=0.08)
        _apply_style(ax, title="Curve Impact of Shock (Δ Zero Rate)",
                     xlabel="Maturity (Years)", ylabel="Impact (bps)", grid_axis="y")
        st.pyplot(fig2)

        st.markdown("""
        **PCHIP** localises the shock around the target tenor — preferred for risk management.  
        **Cubic Spline** propagates globally — may misrepresent hedging needs at distant key rates.
        """)

    st.subheader("Key Rate Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown("**Original Market Data**")
        st.dataframe(cleaned_data.style.format({"rate": "{:.3f}%"}), width="stretch")

    with summary_col2:
        st.markdown("**Key Forward Rates (PCHIP)**")
        forwards = {
            "1Y1Y": (1, 2),
            "2Y3Y": (2, 5),
            "5Y5Y": (5, 10),
        }
        fwd_data = {}
        for label, (t1, t2) in forwards.items():
            if t2 <= analyzer.tenors.max():
                try:
                    fwd_rate = analyzer.get_forward_rate_term(t1, t2, InterpolationMethod.PCHIP)
                    fwd_data[label] = f"{fwd_rate:.3f}%"
                except ValueError:
                    fwd_data[label] = "N/A"
        st.table(pd.Series(fwd_data, name="Forward Rate"))

# =============================================================================
# TAB 1: PORTFOLIO
# =============================================================================
with tabs[1]:
    st.subheader("Portfolio & Risk Metrics")
    valuation_date = st.date_input("Valuation Date", value=date.today())
    st.session_state["valuation_date"] = valuation_date
    st.caption(f"Available curve IDs: {', '.join(curve_ids)} (default: {selected_curve_id})")
    portfolio_source = st.radio(
        "Portfolio Source",
        options=["Sample", "Upload CSV", "Database"],
        horizontal=True,
    )

    portfolio_df = None
    if portfolio_source == "Sample":
        portfolio_df = sample_portfolio_df()
    elif portfolio_source == "Upload CSV":
        portfolio_file = st.file_uploader("Upload portfolio CSV", type=["csv"])
        if portfolio_file is not None:
            portfolio_df = pd.read_csv(portfolio_file)
    else:
        st.caption("Load portfolio positions from DB (SQLAlchemy URL required).")
        default_conn = os.getenv("YC_DB_URL", "")
        default_conn = _get_secret("YC_DB_URL", default_conn)
        port_conn = st.text_input("DB Connection URL (Portfolio)", value=default_conn)
        port_sql = st.text_area(
            "SQL Query (Portfolio)",
            value="SELECT * FROM portfolio_positions ORDER BY issuer",
            height=120,
        )
        if st.button("Load Portfolio from DB"):
            try:
                portfolio_df = _load_curve_from_db(port_conn, port_sql)
                st.success("Portfolio loaded.")
            except Exception as e:
                st.error(f"Portfolio load failed: {e}")

    if portfolio_df is None:
        st.info("Upload or load a portfolio to see pricing and risk.")
    else:
        st.markdown("#### Edit Portfolio Data")
        edited_portfolio = st.data_editor(portfolio_df, key="portfolio_editor", width="stretch")
        try:
            cleaned_portfolio, port_warnings = standardize_portfolio_df(edited_portfolio, valuation_date)
            for w in port_warnings:
                st.warning(w)
        except Exception as e:
            st.error(f"Portfolio error: {e}")
            cleaned_portfolio = None

        if cleaned_portfolio is not None:
            st.session_state["portfolio_df"] = cleaned_portfolio
            results_df, curve_warnings = price_portfolio_df(
                cleaned_portfolio,
                curve_map,
                valuation_date,
                method=interpolation_method,
                default_curve_id=selected_curve_id,
            )
            for w in curve_warnings:
                st.warning(w)

            st.markdown("#### Bond-Level Metrics")
            st.dataframe(
                results_df.style.format(
                    {
                        "pv": "{:,.2f}",
                        "dirty_price": "{:,.2f}",
                        "clean_price": "{:,.2f}",
                        "accrued_interest": "{:,.2f}",
                        "dirty_price_pct": "{:.3f}",
                        "clean_price_pct": "{:.3f}",
                        "modified_duration": "{:.3f}",
                        "convexity": "{:.3f}",
                        "dv01": "{:,.2f}",
                        "cs01": "{:,.2f}",
                    }
                ),
                width="stretch",
            )

            totals = results_df[["pv", "dv01", "cs01"]].sum().to_dict()
            st.markdown("#### Portfolio Totals")
            st.write(
                {
                    "Total PV": f"{totals['pv']:,.2f}",
                    "Total DV01": f"{totals['dv01']:,.2f}",
                    "Total CS01": f"{totals['cs01']:,.2f}",
                }
            )


# =============================================================================
# TAB 2: CARRY & ROLL-DOWN
# =============================================================================
with tabs[2]:
    st.subheader("Carry & Roll-Down Analysis")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to analyze carry and roll-down.")
    else:
        valuation_date = st.session_state.get("valuation_date", date.today())

        col1, col2 = st.columns(2)
        with col1:
            horizon_months = st.selectbox("Investment Horizon", [3, 6, 12], index=0)
        with col2:
            funding_rate = st.number_input("Funding Rate (%)", value=3.5, step=0.1, format="%.2f")

        carry_results = portfolio_carry_rolldown(
            positions_df=portfolio_state,
            curve_map=curve_map,
            valuation_date=valuation_date,
            method=interpolation_method,
            horizon_months=horizon_months,
            funding_rate_pct=funding_rate,
            default_curve_id=selected_curve_id,
        )

        st.markdown(f"#### Expected Returns — {horizon_months}M Horizon (Annualized, bps)")
        st.dataframe(
            carry_results[[
                "id", "issuer", "type", "notional",
                "carry_ann_bps", "rolldown_ann_bps", "total_return_ann_bps"
            ]].style.format({
                "notional": "{:,.0f}",
                "carry_ann_bps": "{:+.1f}",
                "rolldown_ann_bps": "{:+.1f}",
                "total_return_ann_bps": "{:+.1f}",
            }),
            width="stretch",
        )

        total_carry = carry_results["carry"].sum()
        total_roll = carry_results["rolldown"].sum()
        total_expected = total_carry + total_roll

        st.markdown("#### Portfolio Totals")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Total Carry", f"${total_carry:,.0f}")
        summary_cols[1].metric("Total Roll-Down", f"${total_roll:,.0f}")
        summary_cols[2].metric("Expected Return", f"${total_expected:,.0f}")
        pv_sum = carry_results["pv_now"].sum()
        ann_return_bps = (
            total_expected / pv_sum / (horizon_months / 12) * 10000
            if pv_sum != 0 and horizon_months > 0
            else 0.0
        )
        summary_cols[3].metric("Ann. Return (bps)",
                              f"{ann_return_bps:,.0f}")

        # Chart
        fig, ax = _new_fig(figsize=(10, 5))
        carry_results_sorted = carry_results.sort_values("total_return_ann_bps", ascending=True)
        y_pos = np.arange(len(carry_results_sorted))

        ax.barh(y_pos, carry_results_sorted["carry_ann_bps"],
                height=0.6, label="Carry", color=_PALETTE["blue"], alpha=0.85)
        ax.barh(y_pos, carry_results_sorted["rolldown_ann_bps"],
                left=carry_results_sorted["carry_ann_bps"],
                height=0.6, label="Roll-Down", color=_PALETTE["orange"], alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(carry_results_sorted["id"], fontsize=_FONT_TICK)
        ax.axvline(0, color=_PALETTE["mid_gray"], linewidth=0.5)
        _apply_style(ax, title=f"{horizon_months}M Carry + Roll-Down by Bond",
                     xlabel="Annualized Return (bps)", grid_axis="x")
        fig.tight_layout()
        st.pyplot(fig)

# =============================================================================
# TAB 3: DV01 LADDER
# =============================================================================
with tabs[3]:
    st.subheader("DV01 Ladder & Curve Exposure")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to see DV01 bucketing.")
    else:
        valuation_date = st.session_state.get("valuation_date", date.today())

        ladder_df = compute_dv01_ladder(
            positions_df=portfolio_state,
            curve_map=curve_map,
            valuation_date=valuation_date,
            method=interpolation_method,
            default_curve_id=selected_curve_id,
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### DV01 by Maturity Bucket")
            st.dataframe(
                ladder_df.style.format({
                    "dv01": "{:,.2f}",
                    "pct_of_total": "{:.1f}%",
                    "cumulative_pct": "{:.1f}%",
                }),
                width="stretch",
            )

        with col2:
            st.markdown("#### Key Rate Durations")
            krd_df = portfolio_key_rate_durations(
                positions_df=portfolio_state,
                curve_map=curve_map,
                valuation_date=valuation_date,
                key_tenors=[2.0, 5.0, 10.0, 30.0],
                method=interpolation_method,
                default_curve_id=selected_curve_id,
            )
            st.dataframe(
                krd_df.style.format({"tenor": "{:.0f}Y", "krd": "{:,.2f}"}),
                width="stretch",
            )

        # Chart
        fig, ax = _new_fig(figsize=(10, 4.5))
        bars = ax.barh(ladder_df["bucket"], ladder_df["pct_of_total"],
                       height=0.55, color=_PALETTE["navy"], alpha=0.85)
        for i, (bucket, pct) in enumerate(zip(ladder_df["bucket"], ladder_df["pct_of_total"])):
            ax.text(pct + 0.8, i, f"{pct:.1f}%", va="center", fontsize=_FONT_ANNOTATION,
                    color=_PALETTE["slate"])
        _apply_style(ax, title="Portfolio Curve Exposure (DV01 Distribution)",
                     xlabel="% of Total DV01", grid_axis="x")
        fig.tight_layout()
        st.pyplot(fig)

# =============================================================================
# TAB 4: PCA ANALYSIS
# =============================================================================
with tabs[4]:
    st.subheader("PCA Decomposition: Level / Slope / Curvature")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to see PCA decomposition.")
    else:
        valuation_date = st.session_state.get("valuation_date", date.today())

        pca_loadings = get_default_pca_loadings()
        pca_tenors = pca_loadings["tenor"].tolist()

        dv01_by_tenor = {t: 0.0 for t in pca_tenors}
        for row in portfolio_state.itertuples(index=False):
            curve_id = getattr(row, "curve_id", selected_curve_id) or selected_curve_id
            try:
                curve = _resolve_curve(curve_map, curve_id, selected_curve_id)
            except ValueError:
                continue

            settlement = getattr(row, "settlement_date", valuation_date)
            if pd.isna(settlement):
                settlement = valuation_date

            spread = float(getattr(row, "spread_bps", 0.0))
            metrics = bond_risk_metrics(
                curve=curve,
                settlement_date=settlement,
                maturity_date=getattr(row, "maturity_date"),
                coupon_rate=float(getattr(row, "coupon_rate")),
                coupon_freq=int(getattr(row, "coupon_freq")),
                notional=float(getattr(row, "notional")),
                spread_bps=spread,
                method=interpolation_method,
            )

            ytm = _years_to_maturity(settlement, getattr(row, "maturity_date"))
            nearest_tenor = min(pca_tenors, key=lambda t: abs(t - ytm))
            dv01_by_tenor[nearest_tenor] += metrics.dv01

        dv01_df = pd.DataFrame({
            "tenor": pca_tenors,
            "dv01": [dv01_by_tenor[t] for t in pca_tenors],
        })

        exposures = decompose_portfolio_pca_exposure(dv01_df, pca_loadings)
        interpretation = interpret_pca_exposures(
            exposures["level_exposure"],
            exposures["slope_exposure"],
            exposures["curvature_exposure"],
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Factor Exposures (DV01-Weighted)")
            exp_df = pd.DataFrame([
                {"Factor": "Level (PC1)", "Exposure ($)": exposures["level_exposure"]},
                {"Factor": "Slope (PC2)", "Exposure ($)": exposures["slope_exposure"]},
                {"Factor": "Curvature (PC3)", "Exposure ($)": exposures["curvature_exposure"]},
            ])
            st.dataframe(
                exp_df.style.format({"Exposure ($)": "{:,.0f}"}),
                width="stretch",
            )

        with col2:
            st.markdown("#### Interpretation")
            st.markdown(interpretation)

        # DV01 by tenor
        fig, ax = _new_fig(figsize=(10, 4.5))
        tenor_labels = [f"{t}Y" for t in pca_tenors]
        ax.bar(tenor_labels, [dv01_by_tenor[t] for t in pca_tenors],
               color=_PALETTE["blue"], alpha=0.85, width=0.6)
        _apply_style(ax, title="Portfolio DV01 Mapped to PCA Tenor Buckets",
                     xlabel="Tenor", ylabel="DV01 ($)", grid_axis="y")
        fig.tight_layout()
        st.pyplot(fig)

        # PCA loadings
        fig2, ax2 = _new_fig(figsize=(10, 4.5))
        ax2.plot(tenor_labels, pca_loadings["level"], color=_PALETTE["navy"],
                 marker="o", markersize=_MARKER_SIZE, linewidth=_LINEWIDTH, label="Level (PC1)")
        ax2.plot(tenor_labels, pca_loadings["slope"], color=_PALETTE["red"],
                 marker="s", markersize=_MARKER_SIZE, linewidth=_LINEWIDTH, label="Slope (PC2)")
        ax2.plot(tenor_labels, pca_loadings["curvature"], color=_PALETTE["teal"],
                 marker="^", markersize=_MARKER_SIZE, linewidth=_LINEWIDTH, label="Curvature (PC3)")
        ax2.axhline(0, color=_PALETTE["mid_gray"], linewidth=0.4)
        _apply_style(ax2, title="PCA Factor Loadings — Stylized USD Rates",
                     xlabel="Tenor", ylabel="Loading")
        fig2.tight_layout()
        st.pyplot(fig2)

# =============================================================================
# TAB 5: RELATIVE VALUE
# =============================================================================
with tabs[5]:
    st.subheader("Relative Value: Z-Spread & Rich/Cheap Screening")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to perform relative value analysis.")
    else:
        valuation_date = st.session_state.get("valuation_date", date.today())

        rv_df = portfolio_z_spreads(
            positions_df=portfolio_state,
            curve_map=curve_map,
            valuation_date=valuation_date,
            method=interpolation_method,
            default_curve_id=selected_curve_id,
        )

        has_price = rv_df["clean_price"].notna()
        if has_price.sum() == 0:
            st.warning("No bonds have market prices. Add `clean_price` column to portfolio data.")
        else:
            rv_display = rv_df[has_price].copy()

            st.markdown("#### Z-Spread Analysis")
            st.dataframe(
                rv_display[[
                    "id", "issuer", "type", "duration", "model_spread_bps",
                    "z_spread_bps", "fitted_spread", "residual_bps", "signal"
                ]].style.format({
                    "duration": "{:.2f}",
                    "model_spread_bps": "{:.1f}",
                    "z_spread_bps": "{:.1f}",
                    "fitted_spread": "{:.1f}",
                    "residual_bps": "{:+.1f}",
                }).map(
                    lambda val: "background-color: lightgreen" if val == "CHEAP" else
                               ("background-color: lightcoral" if val == "RICH" else ""),
                    subset=["signal"]
                ),
                width="stretch",
            )

            # Scatter plot
            fig, ax = _new_fig(figsize=(10, 5))

            cheap = rv_display[rv_display["signal"] == "CHEAP"]
            fair = rv_display[rv_display["signal"] == "FAIR"]
            rich = rv_display[rv_display["signal"] == "RICH"]

            if len(cheap) > 0:
                ax.scatter(cheap["duration"], cheap["z_spread_bps"],
                          color=_PALETTE["green"], s=60, label="CHEAP", alpha=0.85,
                          edgecolors="white", linewidths=0.6, zorder=3)
            if len(fair) > 0:
                ax.scatter(fair["duration"], fair["z_spread_bps"],
                          color=_PALETTE["slate"], s=60, label="FAIR", alpha=0.7,
                          edgecolors="white", linewidths=0.6, zorder=3)
            if len(rich) > 0:
                ax.scatter(rich["duration"], rich["z_spread_bps"],
                          color=_PALETTE["red"], s=60, label="RICH", alpha=0.85,
                          edgecolors="white", linewidths=0.6, zorder=3)

            valid_for_fit = rv_display.dropna(subset=["z_spread_bps", "fitted_spread"])
            if len(valid_for_fit) >= 2:
                dur_sorted = np.sort(valid_for_fit["duration"].values)
                fitted_sorted = valid_for_fit.set_index("duration").loc[dur_sorted, "fitted_spread"].values
                ax.plot(dur_sorted, fitted_sorted, color=_PALETTE["navy"],
                        linewidth=1.2, linestyle="--", label="Fitted Spread (OLS)", alpha=0.7)

            _apply_style(ax, title="Z-Spread vs. Duration — Rich/Cheap Signals",
                         xlabel="Modified Duration", ylabel="Z-Spread (bps)")
            fig.tight_layout()
            st.pyplot(fig)

# =============================================================================
# TAB 6: HISTORICAL STRESS
# =============================================================================
with tabs[6]:
    st.subheader("Historical Scenario Stress Testing")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to run historical scenarios.")
    else:
        valuation_date = st.session_state.get("valuation_date", date.today())

        scenario_list = list_available_scenarios()
        selected_scenario = st.selectbox(
            "Select Historical Scenario",
            scenario_list,
            format_func=lambda x: HISTORICAL_SCENARIOS[x]["name"]
        )

        scenario_meta = get_scenario_description(selected_scenario)
        st.info(f"**{scenario_meta['name']}**: {scenario_meta['description']}")

        shocked_curve_df = interpolate_scenario_shifts(selected_scenario, cleaned_data)
        shocked_curve_map = {
            cid: _build_analyzer(group[["tenor", "rate"]], extrapolate=extrapolate)
            for cid, group in shocked_curve_df.groupby("curve_id")
        }

        base_pv = portfolio_pv(
            portfolio_state, curve_map, valuation_date=valuation_date,
            method=interpolation_method, default_curve_id=selected_curve_id,
        )
        stressed_pv = portfolio_pv(
            portfolio_state, shocked_curve_map, valuation_date=valuation_date,
            method=interpolation_method,
            spread_shock_bps=scenario_meta.get("spread_shock_bps", 0),
            default_curve_id=selected_curve_id,
        )

        pnl = stressed_pv - base_pv
        pnl_pct = (pnl / base_pv * 100) if base_pv != 0 else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Base PV", f"${base_pv:,.0f}")
        col2.metric("Stressed PV", f"${stressed_pv:,.0f}")
        col3.metric("P&L", f"${pnl:,.0f}", f"{pnl_pct:+.2f}%")

        st.markdown("#### Scenario Details")
        shock_summary = pd.DataFrame([
            {"Tenor": f"{t}Y", "Rate Shift (bps)": s}
            for t, s in scenario_meta["moves"].items()
        ])
        st.dataframe(shock_summary, width="stretch")

# =============================================================================
# TAB 7: SCENARIOS
# =============================================================================
with tabs[7]:
    st.subheader("Scenario Analysis")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to run scenarios.")
    else:
        scenario_types = st.multiselect(
            "Select Scenarios",
            ["Parallel Shift", "Twist", "Key Rate", "Spread Shock"],
            default=["Parallel Shift"],
        )
        valuation_date = st.session_state.get("valuation_date", date.today())

        scenarios = []
        if "Parallel Shift" in scenario_types:
            par_bps = st.slider("Parallel Shift (bps)", -200, 200, 50, step=5)
            scenarios.append(("Parallel", {"bps": par_bps}))

        if "Twist" in scenario_types:
            short_bps = st.slider("Short-End Shift (bps)", -200, 200, 25, step=5)
            long_bps = st.slider("Long-End Shift (bps)", -200, 200, -25, step=5)
            scenarios.append(("Twist", {"short_bps": short_bps, "long_bps": long_bps}))

        if "Key Rate" in scenario_types:
            key_input = st.text_input(
                "Key Rate Shocks (tenor:bps, comma-separated)",
                value="2:25,5:-10,10:15",
            )
            key_bps = {}
            for item in key_input.split(","):
                if ":" in item:
                    tenor_str, bps_str = item.split(":", 1)
                    try:
                        key_bps[float(tenor_str.strip())] = float(bps_str.strip())
                    except ValueError:
                        continue
            scenarios.append(("KeyRate", {"key_bps": key_bps}))

        if "Spread Shock" in scenario_types:
            spread_bps = st.slider("Spread Shock (bps, corporate only)", -200, 200, 25, step=5)
            scenarios.append(("Spread", {"bps": spread_bps}))

        def _build_shocked_curve_map_for_scenario(name: str, params: dict) -> Optional[dict]:
            shockers = {
                "Parallel": lambda df: apply_parallel_shock(df, params["bps"]),
                "Twist": lambda df: apply_twist_shock(df, params["short_bps"], params["long_bps"]),
                "KeyRate": lambda df: apply_key_rate_shock(df, params["key_bps"]),
            }
            shocker = shockers.get(name)
            if shocker is None:
                return None
            shocked_curve_df = cleaned_data.groupby("curve_id", group_keys=False).apply(shocker)
            return {
                cid: _build_analyzer(group[["tenor", "rate"]], extrapolate=extrapolate)
                for cid, group in shocked_curve_df.groupby("curve_id")
            }

        base_pv = portfolio_pv(
            portfolio_state, curve_map, valuation_date=valuation_date,
            method=interpolation_method, default_curve_id=selected_curve_id,
        )

        results = []
        for name, params in scenarios:
            curve_map_shocked = _build_shocked_curve_map_for_scenario(name, params)
            if curve_map_shocked is not None:
                pv = portfolio_pv(
                    portfolio_state, curve_map_shocked, valuation_date=valuation_date,
                    method=interpolation_method, default_curve_id=selected_curve_id,
                )
            else:
                pv = portfolio_pv(
                    portfolio_state, curve_map, valuation_date=valuation_date,
                    method=interpolation_method,
                    spread_shock_bps=params["bps"],
                    default_curve_id=selected_curve_id,
                )

            results.append({"Scenario": name, "Portfolio PV": pv, "P&L": pv - base_pv})

        if results:
            st.markdown("#### Scenario Results")
            results_df = pd.DataFrame(results)
            st.dataframe(
                results_df.style.format({"Portfolio PV": "{:,.2f}", "P&L": "{:,.2f}"}),
                width="stretch",
            )

# =============================================================================
# TAB 8: P&L ATTRIBUTION
# =============================================================================
with tabs[8]:
    st.subheader("P&L Attribution")
    portfolio_state = st.session_state.get("portfolio_df")
    if portfolio_state is None:
        st.info("Load a portfolio in the Portfolio tab to run P&L attribution.")
    else:
        valuation_date = st.session_state.get("valuation_date", date.today())

        col1, col2 = st.columns(2)
        with col1:
            pnl_t0 = st.date_input("Start Date (T0)", value=valuation_date, key="pnl_t0")
            rate_change_bps = st.number_input(
                "Rate Change (bps, parallel shift)", value=0, step=5, key="pnl_rate_change"
            )
        with col2:
            default_t1 = pd.Timestamp(valuation_date) + pd.DateOffset(months=3)
            pnl_t1 = st.date_input("End Date (T1)", value=default_t1.date(), key="pnl_t1")
            spread_change_bps_input = st.number_input(
                "Spread Change (bps, corporate only)", value=0, step=5, key="pnl_spread_change"
            )

        pnl_funding = st.number_input(
            "Funding Rate (%)", value=3.5, step=0.1, format="%.2f", key="pnl_funding"
        )

        t1_curve_df = (
            cleaned_data.groupby("curve_id", group_keys=False)
            .apply(lambda df: apply_parallel_shock(df, rate_change_bps))
        )
        t1_curve_map = {
            cid: _build_analyzer(group[["tenor", "rate"]], extrapolate=extrapolate)
            for cid, group in t1_curve_df.groupby("curve_id")
        }

        attrib = compute_pnl_attribution(
            positions_df=portfolio_state,
            curve_map_t0=curve_map,
            curve_map_t1=t1_curve_map,
            date_t0=pnl_t0,
            date_t1=pnl_t1,
            spread_change_bps=float(spread_change_bps_input),
            method=interpolation_method,
            funding_rate_pct=float(pnl_funding),
            default_curve_id=selected_curve_id,
        )

        cols = st.columns(3)
        cols[0].metric("Starting PV", f"${attrib['pv_t0']:,.0f}")
        cols[1].metric("Ending PV", f"${attrib['pv_t1']:,.0f}")
        pnl_pct_str = (
            f"{attrib['total_pnl'] / attrib['pv_t0'] * 100:+.2f}%"
            if attrib["pv_t0"] != 0 else ""
        )
        cols[2].metric("Total P&L", f"${attrib['total_pnl']:,.0f}", pnl_pct_str)

        report_df = format_pnl_report(attrib)
        st.markdown("#### P&L Decomposition")
        st.dataframe(
            report_df.style.format(
                {"Value": "{:,.0f}", "% of Total": "{:.1f}"},
                na_rep="—",
            ),
            width="stretch",
        )

        # Waterfall chart
        st.markdown("#### Waterfall")
        components = ["Carry", "Roll-Down", "Rate Moves", "Spread Moves", "Residual"]
        values = [
            attrib["carry"], attrib["rolldown"], attrib["rate_move"],
            attrib["spread_move"], attrib["residual"],
        ]

        fig, ax = _new_fig(figsize=(10, 5))
        cumulative = np.cumsum([0.0] + values[:-1])
        bar_colors = [_PALETTE["green"] if v >= 0 else _PALETTE["red"] for v in values]

        ax.bar(components, values, bottom=cumulative, color=bar_colors,
               width=0.55, alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color=_PALETTE["mid_gray"], linewidth=0.4)

        # Total bar
        total_val = sum(values)
        total_color = _PALETTE["navy"]
        ax.bar("Total", total_val, color=total_color, width=0.55, alpha=0.9,
               edgecolor="white", linewidth=0.6)

        for i, val in enumerate(values):
            ax.text(i, cumulative[i] + val / 2, f"${val:,.0f}",
                    ha="center", va="center", fontsize=_FONT_ANNOTATION, color="white",
                    fontweight="600")
        ax.text(len(components), total_val / 2, f"${total_val:,.0f}",
                ha="center", va="center", fontsize=_FONT_ANNOTATION, color="white",
                fontweight="bold")

        _apply_style(ax, title="P&L Attribution Waterfall", ylabel="P&L ($)", grid_axis="y")
        fig.tight_layout()
        st.pyplot(fig)

# =============================================================================
# TAB 9: DOCUMENTATION
# =============================================================================
with tabs[9]:
    st.subheader("Documentation: Mathematical Framework & Portfolio Interpretation")

    st.markdown("""
    ### 1) Core Pricing Identities

    For each cashflow at time `t_j`:

    $$
    PV = \\sum_j CF_j \\cdot \\exp\\left(-\\left(\\frac{R(t_j) + b_{\\mathrm{total}}/100}{100}\\right)t_j\\right)
    $$

    where \\(b_{\\mathrm{total}}\\) is the sum of curve and spread bumps (in bps).

    Clean/dirty decomposition:

    $$
    P_{\\mathrm{dirty}}=PV, \\quad P_{\\mathrm{clean}}=PV-AI
    $$

    **Financial comment:** separating clean from dirty is required for consistent relative-value comparisons and period attribution.

    ### 2) Risk Measures Used in the App

    Symmetric 1bp finite differences:

    $$
    DV01=\\frac{PV_{down}-PV_{up}}{2}
    $$

    $$
    D_{\\mathrm{mod}}=\\frac{DV01}{PV \\cdot (\\Delta bps/10000)}
    $$

    $$
    \\mathrm{Convexity}=\\frac{PV_{up}+PV_{down}-2PV}{PV\\cdot\\Delta^2}, \\quad \\Delta=\\Delta bps/10000
    $$

    $$
    CS01=\\frac{PV^{spread\\ down}-PV^{spread\\ up}}{2}
    $$

    **Financial comment:** DV01 controls first-order rate beta; convexity matters under large shocks and non-linear re-pricing.

    ### 3) Carry & Roll-Down

    $$
    \\mathrm{Carry}=\\mathrm{Coupons}_{t_0\\to t_1} + (AI_{t_1}-AI_{t_0}) - \\mathrm{FundingCost}
    $$

    $$
    \\mathrm{RollDown}=P_{\\mathrm{clean},t_1}-P_{\\mathrm{clean},t_0}
    $$

    $$
    \\mathrm{TotalReturn}=\\mathrm{Carry}+\\mathrm{RollDown}
    $$

    **Financial comment:** this decomposition is the desk-standard pre-positioning lens before introducing macro spread/rate views.

    ### 4) Curve Structure and Scenario Logic

    Instantaneous forward:

    $$
    f(t)=R(t)+tR'(t)
    $$

    Term forward:

    $$
    F(t_1,t_2)=\\frac{R(t_2)t_2-R(t_1)t_1}{t_2-t_1}
    $$

    Scenario examples:
    - Parallel: $R'(t)=R(t)+\\texttt{bps}/100$
    - Twist: tenor-weighted interpolation between `short_bps` and `long_bps`
    - Key-rate: triangular local shocks by tenor

    **Financial comment:** local key-rate shocks are better aligned with hedge implementation than global polynomial distortions.

    ### 5) Relative Value & P&L Attribution

    Z-spread root:

    $$
    f(z)=PV(z)-PV_{\\mathrm{target}}=0
    $$

    OLS rich/cheap residual:

    $$
    \\mathrm{Residual}_{bps}=z_{bps}-\\widehat{z}(D_{\\mathrm{mod}})
    $$

    Period attribution:

    $$
    \\mathrm{Total}\\;P\\&L=\\mathrm{Carry}+\\mathrm{RollDown}+\\mathrm{RateMove}+\\mathrm{SpreadMove}+\\mathrm{Residual}
    $$

    **Financial comment:** residual should remain small over stable horizons; persistent residual drift indicates model misspecification or un-modeled optionality.
    """)

    st.info(
        "For full page-by-page formulas and variable naming conventions, "
        "see README.md (GitHub-ready specification)."
    )

# --- Footer ---
st.caption("⚠️ Still under development — contact fadi.hafiane@outlook.fr")
