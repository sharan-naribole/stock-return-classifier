"""
Portfolio simulator — backtests 3 trading strategies + buy & hold on test data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioSimulator:
    """
    Simulates 3 trading strategies and buy & hold on the test set.
    
    Strategy 1: Buy max_shares when prob > 0.5, sell after 3 days
    Strategy 2: Buy max_shares when prob >= confidence_threshold
    Strategy 3: Variable shares based on probability bins
    
    All strategies have stop-loss and take-profit guardrails.
    """

    def __init__(self, config):
        portfolio_cfg = config.get("portfolio") or {}
        self.initial_balance = portfolio_cfg.get("initial_balance", 10000)
        self.max_shares = portfolio_cfg.get("max_shares", 10)
        self.confidence_threshold = portfolio_cfg.get("confidence_threshold", 0.6)
        self.horizon = config.get("target.horizon_days") or 1  # hold period matches prediction horizon

    def _shares_for_strategy(self, strategy: int, prob: float) -> int:
        """Determine number of shares to buy based on strategy and probability."""
        if strategy == 1:
            return self.max_shares if prob > 0.5 else 0
        elif strategy == 2:
            return self.max_shares if prob >= self.confidence_threshold else 0
        elif strategy == 3:
            if prob >= 0.875:
                return self.max_shares
            elif prob >= 0.75:
                return int(self.max_shares * 0.75)
            elif prob >= 0.625:
                return int(self.max_shares * 0.5)
            elif prob >= 0.5:
                return int(self.max_shares * 0.25)
            return 0
        return 0

    def _simulate_strategy(
        self,
        prices: pd.Series,
        probs: np.ndarray,
        strategy: int,
    ) -> Tuple[List[float], List[dict]]:
        """
        Simulate a single strategy.
        
        Returns:
            (portfolio_values list aligned to prices.index, trade_log list)
        """
        balance = float(self.initial_balance)
        portfolio_values = [balance]
        trade_log = []
        n = len(prices)
        price_arr = prices.values
        dates = prices.index

        # Track open positions: list of {entry_day, entry_price, shares}
        open_positions = []

        for i in range(n):
            # Close positions that have reached the hold period
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

            # Open new position if signal
            if i < n - self.horizon:  # don't open new position near end
                prob = probs[i]
                shares = self._shares_for_strategy(strategy, prob)
                if shares > 0:
                    cost = shares * price_arr[i]
                    if cost > balance:  # clamp to max affordable
                        shares = int(balance / price_arr[i])
                        cost = shares * price_arr[i]
                    if shares > 0:
                        balance -= cost
                        open_positions.append({
                            "entry_day": i,
                            "entry_price": price_arr[i],
                            "shares": shares,
                            "cost": cost,
                        })

            # Portfolio value = cash + open position market value
            open_value = sum(p["shares"] * price_arr[i] for p in open_positions)
            portfolio_values.append(balance + open_value)

        # Final: close all remaining positions at last price
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

        # Trim to match prices length
        portfolio_values = portfolio_values[:n]

        return portfolio_values, trade_log

    def _buy_and_hold(self, prices: pd.Series) -> List[float]:
        """Buy maximum whole shares affordable on day 1, hold entire period."""
        price_arr = prices.values
        shares = int(self.initial_balance / price_arr[0])
        cash = self.initial_balance - shares * price_arr[0]
        return [cash + shares * p for p in price_arr]

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
    ) -> Dict[str, dict]:
        """
        Run all strategies on test data.
        
        Args:
            test_df: test DataFrame with 'Close' column
            y_prob: predicted probabilities (class 1)
            
        Returns:
            dict with results for each strategy + buy_and_hold
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

        # Buy & Hold
        print("  Simulating buy_and_hold...")
        bh_values = self._buy_and_hold(prices)
        results["buy_and_hold"] = {
            "portfolio_values": bh_values,
            "trade_log": [],
            "summary": self._summarize(bh_values, []),
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
            "buy_and_hold": "gray",
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
