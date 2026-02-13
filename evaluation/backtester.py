"""Framework de backtesting pour evaluer toutes les strategies.

Usage :
    backtester = Backtester(initial_capital=10000)
    metrics = backtester.run(strategy, df)
    backtester.compare_all(save_path="evaluation/results.json")
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from training.baseline.strategies import Action, BaseStrategy


@dataclass
class BacktestMetrics:
    """Metriques completes d'un backtest.

    Attributes:
        strategy_name: Nom de la strategie.
        total_return_pct: Rendement total en %.
        final_balance: Solde final.
        max_drawdown_pct: Maximum drawdown en % (negatif).
        sharpe_ratio: Ratio de Sharpe simplifie.
        calmar_ratio: Return / |max drawdown|.
        profit_factor: Gains bruts / pertes brutes.
        win_rate_pct: % de trades gagnants.
        n_trades: Nombre total de trades.
        n_buy, n_sell, n_hold: Decomposition des actions.
        start_date, end_date: Periode du backtest.
        n_candles: Nombre de bougies.
    """

    strategy_name: str
    total_return_pct: float = 0.0
    final_balance: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    n_trades: int = 0
    n_buy: int = 0
    n_sell: int = 0
    n_hold: int = 0
    start_date: str = ""
    end_date: str = ""
    n_candles: int = 0

    def to_dict(self) -> dict:
        """Convertit en dictionnaire serialisable."""
        return asdict(self)

    def print_summary(self) -> None:
        """Affiche un resume lisible des metriques."""
        logger.info(
            f"\n{'=' * 50}\n"
            f"  {self.strategy_name}\n"
            f"{'=' * 50}\n"
            f"  Return total    : {self.total_return_pct:+.2f}%\n"
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%\n"
            f"  Sharpe Ratio    : {self.sharpe_ratio:.3f}\n"
            f"  Calmar Ratio    : {self.calmar_ratio:.3f}\n"
            f"  Profit Factor   : {self.profit_factor:.3f}\n"
            f"  Win Rate        : {self.win_rate_pct:.1f}%\n"
            f"  Nb Trades       : {self.n_trades}\n"
            f"{'=' * 50}"
        )


class Backtester:
    """Backtester universel pour toutes les strategies.

    Simule l'execution de signaux BUY/SELL/HOLD avec couts de transaction
    et slippage. Supporte long et short.

    Attributes:
        initial_capital: Capital initial.
        transaction_cost: Cout en ratio (0.0002 = 2 pips).
        slippage: Slippage en ratio.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.0002,
        slippage: float = 0.0001,
    ) -> None:
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self._results: dict[str, dict] = {}

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        periods_per_year: int = 26280,
    ) -> float:
        """Calcule le Sharpe ratio annualise.

        Args:
            returns: Serie de rendements periodiques.
            periods_per_year: 252j * 24h * 4.35 bougies/h ~ 26280.

        Returns:
            Sharpe ratio.
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calcule le maximum drawdown en %.

        Args:
            equity_curve: Serie de la valeur du portefeuille.

        Returns:
            Max drawdown (valeur negative).
        """
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        return float(drawdown.min())

    def _calculate_profit_factor(self, trade_returns: pd.Series) -> float:
        """Calcule le profit factor.

        Args:
            trade_returns: Serie des PnL par trade.

        Returns:
            gross_profit / gross_loss. inf si pas de perte.
        """
        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        split_name: str = "test",
    ) -> BacktestMetrics:
        """Execute le backtest complet.

        Args:
            strategy: Strategie a tester.
            df: DataFrame M15 avec features, indexe par timestamp.
            split_name: Nom du split pour les logs.

        Returns:
            BacktestMetrics avec toutes les metriques.
        """
        logger.info(
            f"Backtest {strategy.name} sur {split_name} "
            f"({len(df)} bougies)..."
        )

        signals = strategy.generate_signals(df)

        balance = self.initial_capital
        position = 0  # -1=short, 0=flat, 1=long
        entry_price = 0.0

        equity_curve: list[float] = []
        trade_returns: list[float] = []

        closes = df["close"].values

        for i in range(len(df)):
            price = closes[i]
            action = int(signals.iloc[i])

            if action == Action.BUY and position != 1:
                exec_price = price * (1 + self.slippage)
                cost = exec_price * self.transaction_cost

                # Ferme short
                if position == -1:
                    pnl = (entry_price - exec_price) * balance / entry_price
                    balance += pnl - cost
                    trade_returns.append((pnl - cost) / self.initial_capital)
                else:
                    balance -= cost

                position = 1
                entry_price = exec_price

            elif action == Action.SELL and position != -1:
                exec_price = price * (1 - self.slippage)
                cost = exec_price * self.transaction_cost

                # Ferme long
                if position == 1:
                    pnl = (exec_price - entry_price) * balance / entry_price
                    balance += pnl - cost
                    trade_returns.append((pnl - cost) / self.initial_capital)
                else:
                    balance -= cost

                position = -1
                entry_price = exec_price

            # Mark-to-market pour equity curve
            if position == 1:
                unrealized = (price - entry_price) * balance / entry_price
            elif position == -1:
                unrealized = (entry_price - price) * balance / entry_price
            else:
                unrealized = 0.0
            equity_curve.append(balance + unrealized)

        # Ferme position finale
        if position != 0:
            final_price = closes[-1]
            cost = final_price * self.transaction_cost
            if position == 1:
                pnl = (final_price - entry_price) * balance / entry_price
            else:
                pnl = (entry_price - final_price) * balance / entry_price
            balance += pnl - cost
            trade_returns.append((pnl - cost) / self.initial_capital)
            equity_curve[-1] = balance

        # Series
        equity_series = pd.Series(equity_curve, index=df.index)
        trade_returns_series = (
            pd.Series(trade_returns) if trade_returns else pd.Series([0.0])
        )
        periodic_returns = equity_series.pct_change().dropna()

        # Metriques
        total_return = (balance - self.initial_capital) / self.initial_capital * 100
        max_dd = self._calculate_max_drawdown(equity_series)
        sharpe = self._calculate_sharpe(periodic_returns)
        pf = self._calculate_profit_factor(trade_returns_series)
        win_rate = (
            float((trade_returns_series > 0).sum() / len(trade_returns_series) * 100)
            if len(trade_returns_series) > 0
            else 0.0
        )
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

        metrics = BacktestMetrics(
            strategy_name=strategy.name,
            total_return_pct=round(total_return, 4),
            final_balance=round(balance, 2),
            max_drawdown_pct=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4),
            calmar_ratio=round(calmar, 4),
            profit_factor=round(pf, 4),
            win_rate_pct=round(win_rate, 2),
            n_trades=len(trade_returns),
            n_buy=int((signals == Action.BUY).sum()),
            n_sell=int((signals == Action.SELL).sum()),
            n_hold=int((signals == Action.HOLD).sum()),
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            n_candles=len(df),
        )

        self._results[strategy.name] = {
            "metrics": metrics,
            "equity_curve": equity_series,
            "signals": signals,
        }

        metrics.print_summary()
        return metrics

    def plot_equity_curve(
        self,
        strategy_names: Optional[list[str]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Plot les equity curves des strategies testees.

        Args:
            strategy_names: Strategies a afficher (None = toutes).
            save_path: Chemin HTML pour sauvegarder.

        Returns:
            Figure Plotly.
        """
        if not self._results:
            raise ValueError("Aucun backtest execute.")

        names = strategy_names or list(self._results.keys())

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Equity Curves", "Drawdown (%)"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        colors = ["#00ff88", "#ff4444", "#4488ff", "#ffaa00", "#ff88ff", "#88ffff"]

        for i, name in enumerate(names):
            if name not in self._results:
                continue

            equity = self._results[name]["equity_curve"]
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    name=name,
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=1,
            )

            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name=f"{name} DD",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig.add_hline(
            y=self.initial_capital,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            row=1,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            title="Comparaison Equity Curves",
            height=700,
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.success(f"Equity curve sauvegardee : {save_path}")

        return fig

    def compare_all(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Tableau comparatif de toutes les strategies testees.

        Args:
            save_path: Chemin JSON pour sauvegarder.

        Returns:
            DataFrame avec metriques de toutes les strategies.
        """
        rows = []
        for result in self._results.values():
            rows.append(result["metrics"].to_dict())

        df_compare = pd.DataFrame(rows).set_index("strategy_name")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # Serialiser proprement
            out = {}
            for _, row in df_compare.iterrows():
                out[row.name] = {
                    k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in row.to_dict().items()
                }
            Path(save_path).write_text(
                json.dumps(out, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.success(f"Comparaison sauvegardee : {save_path}")

        return df_compare
