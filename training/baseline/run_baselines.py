"""Script principal : execute et compare toutes les baselines."""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.backtester import Backtester
from training.baseline.strategies import (
    BuyAndHoldStrategy,
    RandomStrategy,
    RuleBasedStrategy,
)


def main() -> None:
    """Execute les 3 baselines sur val et test."""
    logger.info("=== DEBUT BASELINES ===")

    # Chargement donnees avec features
    splits_dir = PROJECT_ROOT / "data" / "splits"
    df_val = pd.read_parquet(splits_dir / "val_features.parquet")
    df_test = pd.read_parquet(splits_dir / "test_features.parquet")

    # Index timestamp
    for df in [df_val, df_test]:
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("timestamp", inplace=True)

    logger.info(f"Val  : {len(df_val):,} bougies")
    logger.info(f"Test : {len(df_test):,} bougies")

    # Strategies
    strategies = [
        RandomStrategy(seed=42),
        RuleBasedStrategy(rsi_oversold=35, rsi_overbought=65),
        BuyAndHoldStrategy(),
    ]

    # Backtester
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.0002,
        slippage=0.0001,
    )

    eval_dir = PROJECT_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # === VALIDATION (2023) ===
    logger.info("\nBACKTEST SUR VALIDATION (2023)")
    for strategy in strategies:
        backtester.run(strategy, df_val, split_name="val")

    backtester.plot_equity_curve(
        save_path=str(eval_dir / "baselines_equity_val.html")
    )
    df_val_compare = backtester.compare_all(
        save_path=str(eval_dir / "baseline_val_results.json")
    )

    # Reset pour test
    backtester._results = {}

    # === TEST FINAL (2024) ===
    logger.info("\nBACKTEST SUR TEST FINAL (2024)")
    for strategy in strategies:
        backtester.run(strategy, df_test, split_name="test")

    backtester.plot_equity_curve(
        save_path=str(eval_dir / "baselines_equity_test.html")
    )
    df_test_compare = backtester.compare_all(
        save_path=str(eval_dir / "baseline_test_results.json")
    )

    # Resume
    logger.info("\nRESULTATS VALIDATION :")
    logger.info(
        f"\n{df_val_compare[['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'n_trades']].to_string()}"
    )
    logger.info("\nRESULTATS TEST :")
    logger.info(
        f"\n{df_test_compare[['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'n_trades']].to_string()}"
    )

    logger.success("=== BASELINES TERMINEES ===")
    return backtester


if __name__ == "__main__":
    main()
