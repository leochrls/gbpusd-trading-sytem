"""Tests unitaires pour le backtester et les strategies baseline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.backtester import Backtester, BacktestMetrics
from training.baseline.strategies import (
    Action,
    BuyAndHoldStrategy,
    RandomStrategy,
    RuleBasedStrategy,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame synthetique avec features minimales."""
    n = 300
    rng = np.random.default_rng(42)

    prices = 1.25 + np.cumsum(rng.normal(0.00001, 0.0002, n))

    df = pd.DataFrame(
        {
            "open": prices + rng.normal(0, 0.0001, n),
            "high": prices + np.abs(rng.normal(0, 0.0003, n)),
            "low": prices - np.abs(rng.normal(0, 0.0003, n)),
            "close": prices,
            "volume": rng.integers(100, 1000, n).astype(float),
            "ema_20": pd.Series(prices).ewm(span=20).mean().values,
            "ema_50": pd.Series(prices).ewm(span=50).mean().values,
            "rsi_14": np.clip(50 + rng.normal(0, 15, n), 0, 100),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="15min"),
    )

    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def test_backtester_returns_metrics(sample_df: pd.DataFrame) -> None:
    """run() retourne des metriques valides."""
    backtester = Backtester(initial_capital=10000)
    metrics = backtester.run(RandomStrategy(seed=42), sample_df)

    assert isinstance(metrics, BacktestMetrics)
    assert metrics.n_candles == len(sample_df)
    assert metrics.final_balance > 0


def test_transaction_costs_reduce_returns(sample_df: pd.DataFrame) -> None:
    """Les couts de transaction reduisent les returns."""
    bt_free = Backtester(initial_capital=10000, transaction_cost=0.0, slippage=0.0)
    bt_cost = Backtester(initial_capital=10000, transaction_cost=0.001, slippage=0.0)

    strategy = RandomStrategy(seed=42)
    m_free = bt_free.run(strategy, sample_df.copy())
    m_cost = bt_cost.run(strategy, sample_df.copy())

    assert m_free.final_balance >= m_cost.final_balance


def test_buy_hold_single_trade(sample_df: pd.DataFrame) -> None:
    """Buy & Hold ne fait qu'un seul trade."""
    backtester = Backtester(initial_capital=10000)
    metrics = backtester.run(BuyAndHoldStrategy(), sample_df)

    assert metrics.n_buy == 1
    assert metrics.n_sell == 0


def test_random_strategy_balanced(sample_df: pd.DataFrame) -> None:
    """Random a environ 1/3 de chaque action."""
    signals = RandomStrategy(seed=42).generate_signals(sample_df)
    n = len(signals)
    tolerance = 0.15

    for action in [Action.BUY, Action.SELL, Action.HOLD]:
        ratio = (signals == action).sum() / n
        assert abs(ratio - 1 / 3) < tolerance, (
            f"Action {action} : ratio {ratio:.2f} hors tolerance"
        )


def test_max_drawdown_negative(sample_df: pd.DataFrame) -> None:
    """Max drawdown est negatif ou nul."""
    backtester = Backtester(initial_capital=10000)
    metrics = backtester.run(RandomStrategy(seed=42), sample_df)
    assert metrics.max_drawdown_pct <= 0


def test_sharpe_finite(sample_df: pd.DataFrame) -> None:
    """Sharpe ratio est un nombre fini."""
    backtester = Backtester(initial_capital=10000)
    metrics = backtester.run(BuyAndHoldStrategy(), sample_df)
    assert np.isfinite(metrics.sharpe_ratio)


def test_win_rate_bounds(sample_df: pd.DataFrame) -> None:
    """Win rate entre 0 et 100."""
    backtester = Backtester(initial_capital=10000)
    metrics = backtester.run(RuleBasedStrategy(), sample_df)
    assert 0 <= metrics.win_rate_pct <= 100


def test_compare_all_returns_dataframe(sample_df: pd.DataFrame) -> None:
    """compare_all() retourne un DataFrame avec toutes les strategies."""
    backtester = Backtester(initial_capital=10000)
    for strat in [RandomStrategy(), RuleBasedStrategy(), BuyAndHoldStrategy()]:
        backtester.run(strat, sample_df)

    df_compare = backtester.compare_all()
    assert isinstance(df_compare, pd.DataFrame)
    assert len(df_compare) == 3


def test_rulebased_signals_valid(sample_df: pd.DataFrame) -> None:
    """RuleBased ne genere que des actions valides."""
    signals = RuleBasedStrategy().generate_signals(sample_df)
    valid = {Action.BUY, Action.SELL, Action.HOLD}
    assert set(signals.unique()).issubset(valid)
