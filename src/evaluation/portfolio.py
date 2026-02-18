"""
Portfolio simulator — backtests 3 ML trading strategies + MACD baseline + DCA on test data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioSimulator:
    """
    Simulates 3 ML trading strategies, a MACD baseline, and DCA on the test set.

    ML strategies buy at close on Day N and sell at close on Day N+1
    (hold period = horizon_days). Position size is derived from available cash.

    Strategy 1: Invest 100% of cash when prob > 0.50
    Strategy 2: Invest 100% of cash when prob >= confidence_threshold
    Strategy 3: Invest a fraction of cash scaled by probability bin:
                prob >= 0.875 → 100%, >= 0.750 → 75%,
                >= 0.625 → 50%, >= 0.500 → 25%, else no trade
    MACD:       Invest 100% of cash when MACD_hist > 0, sell 1 day later
    DCA:        Split balance into equal portions, invest one portion at each
                interval (weekly/monthly), buy and hold until end
    """

    def __init__(self, config):
        portfolio_cfg = config.get("portfolio") or {}
        self.initial_balance = portfolio_cfg.get("initial_balance", 10000)
        self.confidence_threshold = portfolio_cfg.get("confidence_threshold", 0.6)
        self.horizon = config.get("target.horizon_days") or 1

    def _shares_for_strategy(self, strategy: int, prob: float, balance: float, price: float) -> int:
        """Determine number of shares to buy based on strategy, probability, and available cash."""
        if price <= 0 or balance <= 0:
            return 0
        if strategy == 1:
            fraction = 1.0 if prob > 0.5 else 0.0
        elif strategy == 2:
            fraction = 1.0 if prob >= self.confidence_threshold else 0.0
        elif strategy == 3:
            if prob >= 0.875:
                fraction = 1.0
            elif prob >= 0.75:
                fraction = 0.75
            elif prob >= 0.625:
                fraction = 0.50
            elif prob >= 0.5:
                fraction = 0.25
            else:
                fraction = 0.0
        else:
            fraction = 0.0
        return int(balance * fraction / price)

    def _simulate_strategy(
        self,
        prices: pd.Series,
        probs: np.ndarray,
        strategy: int,
    ) -> Tuple[List[float], List[dict]]:
        """
        Simulate a single strategy (1-day hold).

        Returns:
            (portfolio_values list aligned to prices.index, trade_log list)
        """
        balance = float(self.initial_balance)
        portfolio_values = [balance]
        trade_log = []
        n = len(prices)
        price_arr = prices.values
        dates = prices.index

        open_positions = []

        for i in range(n):
            closed_positions = []
            for pos in open_positions:
                days_held = i - pos["entry_day"]
                if days_held >= self.horizon:
                    current_price = price_arr[i]
                    entry_price = pos["entry_price"]
                    pct_change = (current_price - entry_price) / entry_price * 100
                    proceeds = pos["shares"] * current_price
                    balance += proceeds
                    trade_log.append({
                        "entry_date": str(dates[pos["entry_day"]].date()),
                        "exit_date": str(dates[i].date()),
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(current_price, 2),
                        "shares": pos["shares"],
                        "pnl": round(proceeds - pos["cost"], 2),
                        "pct_return": round(pct_change, 2),
                        "strategy": strategy,
                    })
                    closed_positions.append(pos)

            open_positions = [p for p in open_positions if p not in closed_positions]

            if i < n - self.horizon:
                prob = probs[i]
                shares = self._shares_for_strategy(strategy, prob, balance, price_arr[i])
                if shares > 0:
                    cost = shares * price_arr[i]
                    balance -= cost
                    open_positions.append({
                        "entry_day": i,
                        "entry_price": price_arr[i],
                        "shares": shares,
                        "cost": cost,
                    })

            open_value = sum(p["shares"] * price_arr[i] for p in open_positions)
            portfolio_values.append(balance + open_value)

        for pos in open_positions:
            proceeds = pos["shares"] * price_arr[-1]
            balance += proceeds
            trade_log.append({
                "entry_date": str(dates[pos["entry_day"]].date()),
                "exit_date": str(dates[-1].date()),
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(price_arr[-1], 2),
                "shares": pos["shares"],
                "pnl": round(proceeds - pos["cost"], 2),
                "pct_return": round((price_arr[-1] - pos["entry_price"]) / pos["entry_price"] * 100, 2),
                "reason": "end_of_test",
                "strategy": strategy,
            })

        portfolio_values = portfolio_values[:n]
        return portfolio_values, trade_log

    def _simulate_dca(
        self,
        prices: pd.Series,
        frequency: str,
    ) -> Tuple[List[float], List[dict]]:
        """
        Dollar-cost averaging: split initial balance into equal portions and invest
        one portion at each interval, buying and holding until end of test period.

        Args:
            prices: Close price series
            frequency: 'weekly' (~5 trading days) or 'monthly' (~21 trading days)
        """
        period = 5 if frequency == "weekly" else 21
        price_arr = prices.values
        dates = prices.index
        n = len(price_arr)

        # Determine number of buy events and equal portion per event
        buy_days = list(range(0, n, period))
        n_intervals = len(buy_days)
        portion = self.initial_balance / n_intervals

        remaining_cash = float(self.initial_balance)
        shares_held = 0
        portfolio_values = []
        trade_log = []

        for i in range(n):
            # Buy at each DCA interval
            if i in buy_days:
                invest_amount = min(portion, remaining_cash)
                shares_to_buy = int(invest_amount / price_arr[i])
                if shares_to_buy > 0:
                    cost = shares_to_buy * price_arr[i]
                    remaining_cash -= cost
                    shares_held += shares_to_buy
                    trade_log.append({
                        "entry_date": str(dates[i].date()),
                        "entry_price": round(price_arr[i], 2),
                        "shares": shares_to_buy,
                        "cost": round(cost, 2),
                    })

            portfolio_values.append(remaining_cash + shares_held * price_arr[i])

        # Finalize trade log with exit values
        final_price = price_arr[-1]
        for t in trade_log:
            t["exit_date"] = str(dates[-1].date())
            t["exit_price"] = round(final_price, 2)
            proceeds = t["shares"] * final_price
            t["pnl"] = round(proceeds - t["cost"], 2)
            t["pct_return"] = round(
                (final_price - t["entry_price"]) / t["entry_price"] * 100, 2
            )

        return portfolio_values, trade_log

    def _compute_sharpe(self, portfolio_values: List[float], risk_free_rate: float = 0.02) -> float:
        """Compute annualized Sharpe ratio from portfolio value series."""
        values = pd.Series(portfolio_values)
        daily_returns = values.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0
        excess_returns = daily_returns - risk_free_rate / 252
        return round(float(excess_returns.mean() / excess_returns.std() * np.sqrt(252)), 4)

    def _compute_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Compute max drawdown percentage."""
        values = pd.Series(portfolio_values)
        rolling_max = values.cummax()
        drawdown = (values - rolling_max) / rolling_max * 100
        return round(float(drawdown.min()), 2)

    def _summarize(self, portfolio_values: List[float], trade_log: List[dict]) -> dict:
        """Compute summary statistics for a strategy."""
        final_value = portfolio_values[-1]
        total_return_pct = (final_value - self.initial_balance) / self.initial_balance * 100
        sharpe = self._compute_sharpe(portfolio_values)
        max_dd = self._compute_max_drawdown(portfolio_values)

        n_trades = len(trade_log)
        wins = sum(1 for t in trade_log if t.get("pnl", 0) > 0)
        win_rate = round(wins / n_trades * 100, 2) if n_trades > 0 else 0.0

        return {
            "initial_balance": self.initial_balance,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "n_trades": n_trades,
            "win_rate_pct": win_rate,
        }

    def simulate(
        self,
        test_df: pd.DataFrame,
        y_prob: np.ndarray,
        macd_hist: np.ndarray = None,
    ) -> Dict[str, dict]:
        """
        Run all strategies on test data.

        Args:
            test_df: test DataFrame with 'Close' column
            y_prob: predicted probabilities (class 1)
            macd_hist: raw MACD histogram values (buy when > 0)

        Returns:
            dict with results for each strategy
        """
        prices = test_df["Close"]
        results = {}

        strategy_names = {
            1: "strategy_1_threshold_0.5",
            2: f"strategy_2_threshold_{self.confidence_threshold}",
            3: "strategy_3_variable_shares",
        }

        for s_id, s_name in strategy_names.items():
            print(f"  Simulating {s_name}...")
            pv, tl = self._simulate_strategy(prices, y_prob, s_id)
            results[s_name] = {
                "portfolio_values": pv,
                "trade_log": tl,
                "summary": self._summarize(pv, tl),
            }

        # MACD baseline — same 1-day hold, buy when MACD_hist > 0
        if macd_hist is not None:
            print("  Simulating macd_strategy...")
            macd_probs = (macd_hist > 0).astype(float)
            pv, tl = self._simulate_strategy(prices, macd_probs, 1)
            results["macd_strategy"] = {
                "portfolio_values": pv,
                "trade_log": tl,
                "summary": self._summarize(pv, tl),
            }

        # DCA — periodic buy-and-hold (weekly + monthly)
        for freq in ["weekly", "monthly"]:
            print(f"  Simulating dca_{freq}...")
            pv, tl = self._simulate_dca(prices, freq)
            results[f"dca_{freq}"] = {
                "portfolio_values": pv,
                "trade_log": tl,
                "summary": self._summarize(pv, tl),
            }

        return results

    def plot_portfolio_values(
        self,
        test_df: pd.DataFrame,
        results: Dict[str, dict],
        title: str = "Portfolio Value Over Time",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot all strategy portfolio values over time."""
        dates = test_df.index

        colors = {
            "strategy_1_threshold_0.5": "royalblue",
            f"strategy_2_threshold_{self.confidence_threshold}": "green",
            "strategy_3_variable_shares": "orange",
            "macd_strategy": "red",
            "dca_weekly": "purple",
            "dca_monthly": "magenta",
        }

        fig, ax = plt.subplots(figsize=(14, 7))
        for name, res in results.items():
            values = res["portfolio_values"]
            n = min(len(dates), len(values))
            label = name.replace("_", " ").title()
            ax.plot(
                dates[:n],
                values[:n],
                label=f"{label} (${res['summary']['final_value']:,.0f})",
                color=colors.get(name, "black"),
                linewidth=2,
            )

        ax.axhline(y=self.initial_balance, color="black", linestyle=":", alpha=0.5, label="Initial Balance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig

    def comparison_table(self, results: Dict[str, dict]) -> pd.DataFrame:
        """Create a comparison table of all strategy summaries."""
        rows = []
        for name, res in results.items():
            row = {"Strategy": name.replace("_", " ").title()}
            row.update(res["summary"])
            rows.append(row)
        return pd.DataFrame(rows).set_index("Strategy")
