"""
Boucle d'entrainement DQN.
Train sur 2022, validation Sharpe sur 2023.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger

from evaluation.backtester import Backtester
from features.pipeline import FEATURE_COLUMNS
from training.baseline.strategies import Action, BaseStrategy
from training.rl.agent import DQNAgent
from training.rl.environment import TradingEnv


# ============================================================
# STRATEGIE RL POUR BACKTESTER
# ============================================================

class RLStrategy(BaseStrategy):
    """Wrappe l'agent DQN pour le backtester.

    Attributes:
        agent: Agent DQN entraine.
        env: Environnement de trading pour generer les observations.
    """

    def __init__(self, agent: DQNAgent, env: TradingEnv) -> None:
        super().__init__(name="RL_DQN")
        self.agent = agent
        self.env = env

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genere signaux en mode evaluation (epsilon=0)."""
        signals: list[int] = []
        obs, _ = self.env.reset()

        for _ in range(len(df)):
            action = self.agent.select_action(obs, training=False)
            signals.append(action)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break

        # Complete si episode termine avant la fin
        while len(signals) < len(df):
            signals.append(1)  # HOLD

        action_map = {0: Action.SELL, 1: Action.HOLD, 2: Action.BUY}
        return pd.Series(
            [action_map[a] for a in signals[:len(df)]],
            index=df.index,
            name='signal',
        )


# ============================================================
# EVALUATION VALIDATION (SHARPE)
# ============================================================

def evaluate_on_val(
    agent: DQNAgent,
    df_val: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evalue l'agent sur validation (2023).
    Retourne Sharpe ratio + metriques financieres.
    """
    env_val = TradingEnv(df_val, use_risk_adjusted_reward=False)
    strategy = RLStrategy(agent, env_val)

    backtester = Backtester(initial_capital=10000, transaction_cost=0.0002)
    metrics = backtester.run(strategy, df_val, split_name="val_rl")

    return {
        'sharpe': metrics.sharpe_ratio,
        'total_return': metrics.total_return_pct,
        'max_drawdown': metrics.max_drawdown_pct,
        'n_trades': metrics.n_trades,
    }


# ============================================================
# BOUCLE D'ENTRAINEMENT
# ============================================================

def train(
    max_episodes: int = 50,
    early_stopping_patience: int = 10,
    val_freq: int = 5,
    checkpoint_freq: int = 10,
    max_steps_per_episode: int = 2000,
    seed: int = 42,
) -> DQNAgent:
    """
    Boucle d'entrainement DQN complete.

    Args:
        max_episodes: Nombre max d'episodes
        early_stopping_patience: Stop si pas d'amelioration
        val_freq: Frequence validation (en episodes)
        checkpoint_freq: Frequence checkpoint
        max_steps_per_episode: Bougies par episode (sous-ensemble aleatoire)
        seed: Seed pour reproductibilite

    Returns:
        Meilleur agent (selon Sharpe validation)
    """
    logger.info("=== DEBUT ENTRAINEMENT RL ===")

    # Chargement donnees
    df_train = pd.read_parquet('data/splits/train_features.parquet')
    df_val = pd.read_parquet('data/splits/val_features.parquet')

    if 'timestamp' in df_train.columns:
        df_train = df_train.set_index('timestamp')
        df_val = df_val.set_index('timestamp')

    logger.info(f"Train : {len(df_train)} bougies | Val : {len(df_val)} bougies")

    # Environnement train (sous-ensemble aleatoire par episode)
    env_train = TradingEnv(
        df_train,
        initial_balance=10000,
        transaction_cost=0.0002,
        max_drawdown_limit=0.20,
        use_risk_adjusted_reward=True,
        max_steps_per_episode=max_steps_per_episode,
    )

    # Agent
    agent = DQNAgent(
        state_dim=env_train.n_state,
        n_actions=3,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        buffer_size=10_000,
        target_update_freq=100,
        seed=seed,
    )

    logger.info(f"State dim : {env_train.n_state}")
    logger.info(agent.get_vram_usage())

    # Tracking
    best_sharpe = -np.inf
    best_episode = 0
    patience_counter = 0
    training_log: List[Dict] = []

    Path('models/v1').mkdir(parents=True, exist_ok=True)
    Path('evaluation').mkdir(parents=True, exist_ok=True)

    # ============================================================
    # BOUCLE PRINCIPALE
    # ============================================================
    for episode in range(1, max_episodes + 1):

        obs, _ = env_train.reset()
        episode_reward = 0.0
        episode_losses: list[float] = []
        done = False

        while not done:
            # Action epsilon-greedy
            action = agent.select_action(obs, training=True)

            # Step environnement
            next_obs, reward, terminated, truncated, info = env_train.step(action)
            done = terminated or truncated

            # Stockage transition
            agent.store_transition(obs, action, reward, next_obs, done)

            # Optimisation
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            obs = next_obs

        # Decroissance epsilon
        agent.decay_epsilon()

        # Stats episode
        ep_stats = env_train.get_episode_stats()
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        # Log tous les 5 episodes
        if episode % 5 == 0:
            logger.info(
                f"Episode {episode:4d}/{max_episodes} | "
                f"Return: {ep_stats['total_return']:+.2%} | "
                f"Trades: {ep_stats['n_trades']:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"e: {agent.epsilon:.3f}"
            )
            logger.debug(agent.get_vram_usage())

        # ============================================================
        # VALIDATION
        # ============================================================
        if episode % val_freq == 0:
            val_metrics = evaluate_on_val(agent, df_val)

            logger.info(
                f"[VAL] Episode {episode} | "
                f"Sharpe: {val_metrics['sharpe']:.4f} | "
                f"Return: {val_metrics['total_return']:+.2f}% | "
                f"MaxDD: {val_metrics['max_drawdown']:.2f}%"
            )

            training_log.append({
                'episode': episode,
                'train_return': ep_stats['total_return'],
                'val_sharpe': val_metrics['sharpe'],
                'val_return': val_metrics['total_return'],
                'epsilon': agent.epsilon,
                'avg_loss': avg_loss,
            })

            # Meilleur modele ?
            if val_metrics['sharpe'] > best_sharpe:
                best_sharpe = val_metrics['sharpe']
                best_episode = episode
                patience_counter = 0

                agent.save('models/v1/rl_best.pth', episode=episode)
                logger.success(
                    f"Nouveau meilleur modele ! "
                    f"Sharpe val = {best_sharpe:.4f} "
                    f"(episode {episode})"
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience // val_freq:
                logger.warning(
                    f"Early stopping a l'episode {episode} "
                    f"(pas d'amelioration depuis "
                    f"{patience_counter * val_freq} episodes)"
                )
                break

        # Checkpoint regulier
        if episode % checkpoint_freq == 0:
            agent.save(
                f'models/v1/rl_checkpoint_ep{episode}.pth',
                episode=episode,
            )

        # Libere VRAM periodiquement
        if episode % 10 == 0 and agent.device.type == 'cuda':
            torch.cuda.empty_cache()

    # Sauvegarde log entrainement
    with open('evaluation/rl_training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    logger.success(
        f"=== ENTRAINEMENT TERMINE ===\n"
        f"Meilleur episode : {best_episode} | "
        f"Meilleur Sharpe val : {best_sharpe:.4f}"
    )

    return agent


# ============================================================
# EVALUATION FINALE
# ============================================================

def evaluate_final(best_model_path: str = 'models/v1/rl_best.pth') -> Dict:
    """
    Evaluation FINALE sur test 2024.
    A lancer UNE SEULE FOIS apres entrainement complet.
    """
    logger.info("=== EVALUATION FINALE RL SUR TEST 2024 ===")

    df_test = pd.read_parquet('data/splits/test_features.parquet')
    if 'timestamp' in df_test.columns:
        df_test = df_test.set_index('timestamp')

    # Charge meilleur agent
    agent = DQNAgent.load(best_model_path)
    agent.epsilon = 0.0  # Pas d'exploration en eval

    env_test = TradingEnv(df_test, use_risk_adjusted_reward=False)
    strategy = RLStrategy(agent, env_test)

    backtester = Backtester(initial_capital=10000, transaction_cost=0.0002)
    metrics = backtester.run(strategy, df_test, split_name="TEST_FINAL_2024")

    # Equity curve
    backtester.plot_equity_curve(
        save_path='evaluation/rl_equity_test.html'
    )

    metrics.print_summary()

    # Sauvegarde
    results = metrics.to_dict()
    with open('evaluation/rl_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.success("Resultats RL test sauvegardes")
    return results


# ============================================================
# VISUALISATION COURBE D'ENTRAINEMENT
# ============================================================

def plot_training_curves() -> None:
    """Plot les courbes d'entrainement depuis le log."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with open('evaluation/rl_training_log.json', 'r') as f:
        log = json.load(f)

    if not log:
        logger.warning("Pas de log d'entrainement trouve")
        return

    episodes = [e['episode'] for e in log]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Sharpe Ratio Validation",
            "Return Validation (%)",
            "Epsilon",
        ),
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=[e['val_sharpe'] for e in log],
            name='Val Sharpe',
            line=dict(color='#00ff88', width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=[e['val_return'] for e in log],
            name='Val Return',
            line=dict(color='#4488ff', width=2),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=[e['epsilon'] for e in log],
            name='Epsilon',
            line=dict(color='#ffaa00', width=2),
        ),
        row=3, col=1,
    )

    fig.update_layout(
        template='plotly_dark',
        height=800,
        title="Courbes d'entrainement DQN",
    )

    fig.write_html('evaluation/rl_training_curves.html')
    #try:
    #    fig.write_image(
    #        'evaluation/rl_training_curves.png',
    #        width=1200, height=800,
    #    )
    #except (ImportError, ValueError) as e:
    #    logger.warning(
    #        f"write_image impossible ({e}). "
    #        f"Installez kaleido: pip install kaleido"
    #    )
    logger.success("Courbes d'entrainement sauvegardees")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Entrainement (50 ep x 2000 steps = 100k steps)
    agent = train(
        max_episodes=50,
        early_stopping_patience=10,
        val_freq=5,
        checkpoint_freq=10,
        max_steps_per_episode=2000,
        seed=42,
    )

    # Courbes entrainement
    plot_training_curves()

    # Evaluation finale
    evaluate_final('models/v1/rl_best.pth')


if __name__ == "__main__":
    main()
