"""
Script principal : lance toute l'evaluation finale.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from evaluation.compare_all import (
    plot_drawdown_comparison,
    plot_equity_curves,
    plot_metrics_comparison,
    plot_monthly_returns,
    run_comparison,
)
from evaluation.generate_report import generate_final_report


def main() -> None:
    logger.info("=== EVALUATION FINALE COMPLETE ===")

    # 1. Backtest toutes strategies
    df_compare, backtester = run_comparison()

    # 2. Visualisations
    logger.info("Generation des visualisations...")
    plot_equity_curves(backtester)
    plot_metrics_comparison(df_compare)
    plot_monthly_returns(backtester)
    plot_drawdown_comparison(backtester)

    # 3. Rapport HTML
    logger.info("Generation du rapport HTML...")
    generate_final_report()

    logger.success(
        "\nEVALUATION TERMINEE\n"
        "Fichiers generes :\n"
        "  evaluation/final_report.html\n"
        "  evaluation/final_metrics.json\n"
        "  evaluation/final_equity_curves.html\n"
        "  evaluation/final_metrics_comparison.html\n"
        "  evaluation/final_monthly_returns.html\n"
        "  evaluation/final_drawdown.html"
    )


if __name__ == "__main__":
    main()
