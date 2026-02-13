"""
Evaluation comparative finale de toutes les strategies sur test 2024.
Random | RuleBased | BuyAndHold | ML (best) | RL v1 (best) | RL v2 D3QN (best)
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from evaluation.backtester import Backtester
from features.pipeline import FEATURE_COLUMNS
from training.baseline.strategies import (
    Action,
    BaseStrategy,
    BuyAndHoldStrategy,
    RandomStrategy,
    RuleBasedStrategy,
)


def _safe_write_image(fig: go.Figure, path: str, **kwargs) -> None:
    """Ecrit une image PNG si kaleido est installe, sinon warning."""
    try:
        fig.write_image(path, **kwargs)
    except (ImportError, ValueError) as e:
        logger.warning(
            f"write_image impossible ({e}). "
            f"Installez kaleido : pip install kaleido"
        )


# ============================================================
# CHARGEMENT MODELES
# ============================================================

def load_best_ml_model():
    """Charge le meilleur modele ML."""
    path = Path('models/v1/ml_best.pkl')
    if not path.exists():
        logger.warning("ml_best.pkl non trouve, tentative ml_lightgbm.pkl")
        path = Path('models/v1/ml_lightgbm.pkl')

    with open(path, 'rb') as f:
        model = pickle.load(f)

    logger.success(f"Modele ML charge : {path}")
    return model


def load_best_rl_agent():
    """Charge le meilleur agent RL v1."""
    from training.rl.agent import DQNAgent
    path = 'models/v1/rl_best.pth'

    if not Path(path).exists():
        logger.warning(f"{path} non trouve, RL v1 ignore")
        return None

    agent = DQNAgent.load(path)
    agent.epsilon = 0.0
    return agent


def load_best_rl_v2_agent():
    """Charge le meilleur agent RL v2 (D3QN)."""
    from training.rl_v2.agent_v2 import D3QNAgent
    path = 'models/v2/rl_v2_best.pth'

    if not Path(path).exists():
        logger.warning(f"{path} non trouve, RL v2 ignore")
        return None

    agent = D3QNAgent.load(path)
    agent.epsilon = 0.0
    return agent


# ============================================================
# STRATEGIES ML ET RL
# ============================================================

class MLStrategy(BaseStrategy):
    """Strategie basee sur modele ML."""

    def __init__(self, model, threshold: float = 0.5) -> None:
        super().__init__(name="ML_Best")
        self.model = model
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genere signaux depuis les probabilites du modele ML."""
        X = df[FEATURE_COLUMNS].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        signals = pd.Series(Action.HOLD, index=df.index, name='signal')
        signals[proba >= self.threshold] = Action.BUY
        signals[proba < (1 - self.threshold)] = Action.SELL

        return signals


class RLStrategy(BaseStrategy):
    """Strategie basee sur agent RL."""

    def __init__(self, agent, df: pd.DataFrame) -> None:
        super().__init__(name="RL_DQN")
        self.agent = agent
        self.df = df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genere signaux en mode evaluation (epsilon=0)."""
        from training.rl.environment import TradingEnv

        env = TradingEnv(df, use_risk_adjusted_reward=False)
        signals: list[int] = []
        obs, _ = env.reset()

        for _ in range(len(df)):
            action = self.agent.select_action(obs, training=False)
            signals.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        while len(signals) < len(df):
            signals.append(1)

        action_map = {0: Action.SELL, 1: Action.HOLD, 2: Action.BUY}
        return pd.Series(
            [action_map[a] for a in signals[:len(df)]],
            index=df.index,
            name='signal',
        )


# ============================================================
# COMPARAISON PRINCIPALE
# ============================================================

def run_comparison() -> Tuple[pd.DataFrame, Backtester]:
    """
    Lance le backtest de toutes les strategies sur test 2024.

    Returns:
        Tuple (DataFrame comparatif, Backtester avec resultats)
    """
    logger.info("=== EVALUATION COMPARATIVE FINALE ===")

    # Chargement test 2024
    df_test = pd.read_parquet('data/splits/test_features.parquet')
    if 'timestamp' in df_test.columns:
        df_test = df_test.set_index('timestamp')

    logger.info(f"Test 2024 : {len(df_test)} bougies")

    # Backtester
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.0002,
        slippage=0.0001,
    )

    # === STRATEGIES ===
    strategies: list[BaseStrategy] = [
        RandomStrategy(seed=42),
        RuleBasedStrategy(),
        BuyAndHoldStrategy(),
    ]

    # ML
    try:
        ml_model = load_best_ml_model()
        strategies.append(MLStrategy(ml_model))
        logger.success("Strategie ML ajoutee")
    except Exception as e:
        logger.warning(f"ML ignore : {e}")

    # RL v1
    try:
        rl_agent = load_best_rl_agent()
        if rl_agent is not None:
            strategies.append(RLStrategy(rl_agent, df_test))
            logger.success("Strategie RL v1 ajoutee")
    except Exception as e:
        logger.warning(f"RL v1 ignore : {e}")

    # RL v2 (D3QN + PER)
    try:
        rl_v2_agent = load_best_rl_v2_agent()
        if rl_v2_agent is not None:
            from training.rl_v2.train_rl_v2 import RLv2Strategy
            strategies.append(RLv2Strategy(rl_v2_agent))
            logger.success("Strategie RL v2 (D3QN) ajoutee")
    except Exception as e:
        logger.warning(f"RL v2 ignore : {e}")

    # === BACKTEST TOUTES STRATEGIES ===
    logger.info(f"Backtest de {len(strategies)} strategies sur TEST 2024...")

    for strategy in strategies:
        backtester.run(strategy, df_test, split_name="test_final")

    # === TABLEAU COMPARATIF ===
    df_compare = backtester.compare_all(
        save_path='evaluation/final_metrics.json'
    )

    # Colonnes cles pour affichage
    cols_display = [
        'total_return_pct',
        'max_drawdown_pct',
        'sharpe_ratio',
        'calmar_ratio',
        'profit_factor',
        'win_rate_pct',
        'n_trades',
    ]

    logger.info("\n" + "=" * 80)
    logger.info("RESULTATS FINAUX - TEST 2024")
    logger.info("=" * 80)
    logger.info(f"\n{df_compare[cols_display].to_string()}")
    logger.info("=" * 80)

    # Meilleure strategie
    best_strategy = df_compare['sharpe_ratio'].idxmax()
    best_sharpe = df_compare.loc[best_strategy, 'sharpe_ratio']
    logger.success(
        f"\nMeilleure strategie (Sharpe) : {best_strategy} "
        f"({best_sharpe:.4f})"
    )

    return df_compare, backtester


# ============================================================
# VISUALISATIONS
# ============================================================

def plot_equity_curves(backtester: Backtester) -> None:
    """Plot equity curves comparatives."""
    fig = backtester.plot_equity_curve(
        save_path='evaluation/final_equity_curves.html'
    )
    #_safe_write_image(fig, 'evaluation/final_equity_curves.png', width=1400, height=700)
    logger.success("Equity curves sauvegardees")


def plot_metrics_comparison(df_compare: pd.DataFrame) -> None:
    """Bar charts comparatifs des metriques cles."""

    metrics_to_plot = {
        'total_return_pct': 'Return Total (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown_pct': 'Max Drawdown (%)',
        'profit_factor': 'Profit Factor',
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metrics_to_plot.values()),
        vertical_spacing=0.2,
        horizontal_spacing=0.15,
    )

    colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#ff88ff', '#88ffff']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    strategies = df_compare.index.tolist()

    for idx, (metric, title) in enumerate(metrics_to_plot.items()):
        row, col = positions[idx]

        values = df_compare[metric].values
        bar_colors = [
            colors[i % len(colors)] for i in range(len(strategies))
        ]

        fig.add_trace(
            go.Bar(
                x=strategies,
                y=values,
                marker_color=bar_colors,
                name=title,
                showlegend=False,
                text=[f"{v:.3f}" for v in values],
                textposition='outside',
            ),
            row=row, col=col,
        )

    fig.update_layout(
        template='plotly_dark',
        title='Comparaison Metriques - Test 2024',
        height=700,
    )

    fig.write_html('evaluation/final_metrics_comparison.html')
    #_safe_write_image(
    #    fig,
    #    'evaluation/final_metrics_comparison.png',
    #    width=1400, height=700,
    #)
    logger.success("Comparaison metriques sauvegardee")


def plot_monthly_returns(backtester: Backtester) -> None:
    """Heatmap des returns mensuels par strategie."""

    fig = make_subplots(
        rows=len(backtester._results),
        cols=1,
        subplot_titles=list(backtester._results.keys()),
        vertical_spacing=0.08,
    )

    for i, (name, result) in enumerate(backtester._results.items(), 1):
        equity = result['equity_curve']

        # Returns mensuels
        monthly = equity.resample('M').last().pct_change().dropna() * 100

        if len(monthly) == 0:
            continue

        fig.add_trace(
            go.Bar(
                x=monthly.index.strftime('%Y-%m'),
                y=monthly.values,
                name=name,
                marker_color=[
                    '#00ff88' if v >= 0 else '#ff4444'
                    for v in monthly.values
                ],
                showlegend=False,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        template='plotly_dark',
        title='Returns Mensuels 2024 par Strategie',
        height=300 * len(backtester._results),
    )

    fig.write_html('evaluation/final_monthly_returns.html')
    #_safe_write_image(
    #    fig,
    #    'evaluation/final_monthly_returns.png',
    #    width=1200,
    #    height=300 * len(backtester._results),
    #)
    logger.success("Returns mensuels sauvegardes")


def plot_drawdown_comparison(backtester: Backtester) -> None:
    """Compare les drawdowns de toutes les strategies."""

    fig = go.Figure()
    colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#ff88ff', '#88ffff']
    # Versions transparentes pour le fill (rgba)
    fill_colors = [
        'rgba(0,255,136,0.1)',
        'rgba(255,68,68,0.1)',
        'rgba(68,136,255,0.1)',
        'rgba(255,170,0,0.1)',
        'rgba(255,136,255,0.1)',
        'rgba(136,255,255,0.1)',
    ]

    for i, (name, result) in enumerate(backtester._results.items()):
        equity = result['equity_curve']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy',
            fillcolor=fill_colors[i % len(fill_colors)],
        ))

    fig.update_layout(
        template='plotly_dark',
        title='Drawdown Comparatif - Test 2024',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=500,
    )

    fig.write_html('evaluation/final_drawdown.html')
    #_safe_write_image(
    #    fig,
    #    'evaluation/final_drawdown.png',
    #    width=1400, height=500,
    #)
    logger.success("Drawdown comparatif sauvegarde")
