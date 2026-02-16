from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstraintConfig:
    issuer_limit_pct: float = 10.0
    min_cash_pct: float = 0.0


def _to_snake(c: str) -> str:
    return c.strip().lower().replace(" ", "_")


def parse_prospectus_constraints(text: str) -> ConstraintConfig:
    """Best-effort parser for hard limits from free text prospectus snippets."""
    if not text or not text.strip():
        return ConstraintConfig()

    issuer_limit = None
    min_cash = None

    txt = text.lower().replace("\n", " ")

    issuer_patterns = [
        r"(?:issuer|emetteur|émetteur)[^%]{0,60}?(\d+(?:[\.,]\d+)?)\s*%",
        r"(?:single\s+issuer|par\s+émetteur)[^%]{0,60}?(\d+(?:[\.,]\d+)?)\s*%",
    ]
    cash_patterns = [
        r"(?:cash|liquidit(?:e|é)s?)[^%]{0,60}?(?:minimum|min)[^%]{0,20}?(\d+(?:[\.,]\d+)?)\s*%",
        r"(?:minimum|min)[^%]{0,30}?(?:cash|liquidit(?:e|é)s?)[^%]{0,20}?(\d+(?:[\.,]\d+)?)\s*%",
    ]

    for pat in issuer_patterns:
        m = re.search(pat, txt)
        if m:
            issuer_limit = float(m.group(1).replace(",", "."))
            break

    for pat in cash_patterns:
        m = re.search(pat, txt)
        if m:
            min_cash = float(m.group(1).replace(",", "."))
            break

    return ConstraintConfig(
        issuer_limit_pct=issuer_limit if issuer_limit is not None else 10.0,
        min_cash_pct=min_cash if min_cash is not None else 0.0,
    )


class PortfolioManager:
    def __init__(
        self,
        portfolio_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        cash_available: float = 0.0,
    ) -> None:
        self.portfolio_df = self._normalize_portfolio(portfolio_df)
        self.benchmark_df = self._normalize_benchmark(benchmark_df)
        self.cash_available = float(cash_available)

    @staticmethod
    def _normalize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["ticker", "amount", "market_value", "issuer", "sector", "yield"])

        out = df.copy()
        out.columns = [_to_snake(c) for c in out.columns]

        alias = {
            "symbol": "ticker",
            "isin": "ticker",
            "mv": "market_value",
            "marketvalue": "market_value",
            "position": "amount",
            "quantity": "amount",
            "yld": "yield",
            "yield_to_maturity": "yield",
        }
        for src, dst in alias.items():
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]

        required = ["ticker", "market_value", "issuer", "sector"]
        for c in required:
            if c not in out.columns:
                out[c] = np.nan

        if "amount" not in out.columns:
            out["amount"] = 0.0
        if "yield" not in out.columns:
            out["yield"] = np.nan

        out["ticker"] = out["ticker"].astype(str).str.strip()
        out["issuer"] = out["issuer"].astype(str).str.strip()
        out["sector"] = out["sector"].astype(str).str.strip()
        out["market_value"] = pd.to_numeric(out["market_value"], errors="coerce").fillna(0.0)
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
        out["yield"] = pd.to_numeric(out["yield"], errors="coerce")

        out = out[out["ticker"].ne("")].copy()
        return out.reset_index(drop=True)

    @staticmethod
    def _normalize_benchmark(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        out = df.copy()
        out.columns = [_to_snake(c) for c in out.columns]
        if "symbol" in out.columns and "ticker" not in out.columns:
            out["ticker"] = out["symbol"]
        if "benchmark_weight" in out.columns and "weight" not in out.columns:
            out["weight"] = out["benchmark_weight"]

        if "ticker" not in out.columns:
            out["ticker"] = ""
        if "weight" not in out.columns:
            out["weight"] = 0.0

        out["ticker"] = out["ticker"].astype(str).str.strip()
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)

        if out["weight"].max() > 1.5:
            out["weight"] = out["weight"] / 100.0

        s = out["weight"].sum()
        if s > 0:
            out["weight"] = out["weight"] / s

        return out[["ticker", "weight"]].reset_index(drop=True)

    def nav(self) -> float:
        return float(self.portfolio_df["market_value"].sum() + self.cash_available)

    def portfolio_weights(self) -> pd.DataFrame:
        nav = self.nav()
        out = self.portfolio_df.copy()
        out["weight"] = out["market_value"] / nav if nav > 0 else 0.0
        return out

    def compliance_table(self, cfg: ConstraintConfig) -> pd.DataFrame:
        w = self.portfolio_weights()
        issuer_w = w.groupby("issuer", dropna=False)["weight"].sum().sort_values(ascending=False)
        top_issuer = issuer_w.index[0] if len(issuer_w) else "N/A"
        top_issuer_w = float(issuer_w.iloc[0] * 100) if len(issuer_w) else 0.0

        cash_pct = (self.cash_available / self.nav() * 100) if self.nav() > 0 else 0.0

        rows = [
            {
                "Rule": "Max single issuer exposure",
                "Limit": f"<= {cfg.issuer_limit_pct:.2f}%",
                "Current Value": f"{top_issuer}: {top_issuer_w:.2f}%",
                "Status": "OK" if top_issuer_w <= cfg.issuer_limit_pct + 1e-9 else "VIOLATION",
            },
            {
                "Rule": "Minimum cash",
                "Limit": f">= {cfg.min_cash_pct:.2f}%",
                "Current Value": f"{cash_pct:.2f}%",
                "Status": "OK" if cash_pct + 1e-9 >= cfg.min_cash_pct else "VIOLATION",
            },
        ]
        return pd.DataFrame(rows)

    def active_exposure(self) -> pd.DataFrame:
        p = self.portfolio_weights().groupby("ticker", as_index=False)["weight"].sum()
        b = self.benchmark_df.copy()
        out = p.merge(b, on="ticker", how="outer", suffixes=("_portfolio", "_benchmark")).fillna(0.0)
        out = out.rename(columns={"weight_portfolio": "portfolio_weight", "weight_benchmark": "benchmark_weight"})
        out["active_weight"] = out["portfolio_weight"] - out["benchmark_weight"]
        return out.sort_values("active_weight", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    def tracking_error_proxy(self) -> float:
        ae = self.active_exposure()
        return float(np.sqrt((ae["active_weight"] ** 2).sum()))

    def _enforce_issuer_cap(self, holdings: pd.DataFrame, issuer_limit_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        trades = []
        nav_target = holdings["market_value"].sum() + self.cash_available
        cap_val = nav_target * issuer_limit_pct / 100.0
        proceeds = 0.0

        for issuer, grp in holdings.groupby("issuer"):
            issuer_val = float(grp["market_value"].sum())
            excess = max(0.0, issuer_val - cap_val)
            if excess <= 0:
                continue
            shares = grp[["ticker", "market_value"]].copy()
            shares["w"] = shares["market_value"] / shares["market_value"].sum()
            for _, r in shares.iterrows():
                sell_val = float(excess * r["w"])
                if sell_val <= 0:
                    continue
                idx = holdings.index[holdings["ticker"] == r["ticker"]][0]
                holdings.at[idx, "market_value"] = max(0.0, holdings.at[idx, "market_value"] - sell_val)
                proceeds += sell_val
                trades.append(
                    {
                        "Ticker": r["ticker"],
                        "Issuer": issuer,
                        "Action": "SELL",
                        "Trade Value": round(sell_val, 2),
                        "Reason": "Issuer limit normalization",
                    }
                )

        return holdings, pd.DataFrame(trades), proceeds

    def propose_trades(
        self,
        cash_to_invest: float,
        rebalance_to_benchmark: bool,
        maximize_yield: bool,
        cfg: ConstraintConfig,
    ) -> pd.DataFrame:
        holdings = self.portfolio_df[["ticker", "issuer", "market_value", "yield"]].copy()
        holdings = holdings.groupby(["ticker", "issuer", "yield"], as_index=False)["market_value"].sum()

        holdings, cap_trades, cap_proceeds = self._enforce_issuer_cap(holdings, cfg.issuer_limit_pct)

        available_cash = float(max(0.0, cash_to_invest) + self.cash_available + cap_proceeds)
        trades = []
        if not cap_trades.empty:
            trades.extend(cap_trades.to_dict("records"))

        if rebalance_to_benchmark and not self.benchmark_df.empty:
            nav_target = holdings["market_value"].sum() + available_cash
            bm = self.benchmark_df.set_index("ticker")["weight"]

            current_map = holdings.set_index("ticker")
            tickers = sorted(set(current_map.index).union(set(bm.index)))

            # Sells first
            for t in tickers:
                current_val = float(current_map.loc[t, "market_value"]) if t in current_map.index else 0.0
                target_val = float(nav_target * bm.get(t, 0.0))
                delta = target_val - current_val
                if delta < 0:
                    sell_val = min(current_val, -delta)
                    if sell_val > 0:
                        issuer = str(current_map.loc[t, "issuer"]) if t in current_map.index else "N/A"
                        trades.append({"Ticker": t, "Issuer": issuer, "Action": "SELL", "Trade Value": round(sell_val, 2), "Reason": "Rebalance to benchmark"})
                        available_cash += sell_val
                        if t in current_map.index:
                            current_map.at[t, "market_value"] = current_val - sell_val

            # Buys second
            for t in tickers:
                current_val = float(current_map.loc[t, "market_value"]) if t in current_map.index else 0.0
                target_val = float(nav_target * bm.get(t, 0.0))
                delta = target_val - current_val
                if delta > 0 and available_cash > 0:
                    buy_val = min(delta, available_cash)
                    if buy_val > 0:
                        issuer = str(current_map.loc[t, "issuer"]) if t in current_map.index else "UNKNOWN"
                        trades.append({"Ticker": t, "Issuer": issuer, "Action": "BUY", "Trade Value": round(buy_val, 2), "Reason": "Rebalance to benchmark"})
                        available_cash -= buy_val
                        if t in current_map.index:
                            current_map.at[t, "market_value"] = current_val + buy_val

        elif maximize_yield:
            ranked = holdings.sort_values("yield", ascending=False).copy()
            if ranked["yield"].notna().sum() == 0:
                return pd.DataFrame(trades)

            nav_target = holdings["market_value"].sum() + available_cash
            issuer_vals = holdings.groupby("issuer")["market_value"].sum().to_dict()
            issuer_cap = nav_target * cfg.issuer_limit_pct / 100.0

            for _, r in ranked.iterrows():
                if available_cash <= 0:
                    break
                issuer = str(r["issuer"])
                room = max(0.0, issuer_cap - float(issuer_vals.get(issuer, 0.0)))
                if room <= 0:
                    continue
                buy_val = min(room, available_cash)
                if buy_val <= 0:
                    continue
                trades.append({"Ticker": r["ticker"], "Issuer": issuer, "Action": "BUY", "Trade Value": round(buy_val, 2), "Reason": "Maximize yield"})
                issuer_vals[issuer] = float(issuer_vals.get(issuer, 0.0)) + buy_val
                available_cash -= buy_val

        return pd.DataFrame(trades)
