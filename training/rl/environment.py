"""
Environnement de trading GBP/USD pour RL.
Compatible gymnasium (gym).
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger

from features.pipeline import FEATURE_COLUMNS


class TradingEnv(gym.Env):
    """
    Environnement de trading GBP/USD M15.

    State  : features techniques + position + pnl_unrealized + drawdown
    Action : 0=SELL, 1=HOLD, 2=BUY
    Reward : PnL ajuste au risque

    Attributes:
        df: DataFrame M15 avec features et close.
        initial_balance: Capital initial.
        transaction_cost: Cout en ratio (0.0002 = 2 pips).
        max_drawdown_limit: Seuil de drawdown pour terminer l'episode.
        reward_scaling: Facteur de scaling du reward.
        use_risk_adjusted_reward: Si True, utilise le reward ajuste au risque.

    Usage:
        env = TradingEnv(df_train)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0002,
        max_drawdown_limit: float = 0.20,
        reward_scaling: float = 100.0,
        use_risk_adjusted_reward: bool = True,
        max_steps_per_episode: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.reward_scaling = reward_scaling
        self.use_risk_adjusted_reward = use_risk_adjusted_reward
        self.max_steps_per_episode = max_steps_per_episode

        # Features utilisees
        self.feature_cols = [
            c for c in FEATURE_COLUMNS if c in df.columns
        ]
        self.n_features = len(self.feature_cols)

        # State = features + [position, pnl_unrealized, drawdown]
        self.n_state = self.n_features + 3

        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state,),
            dtype=np.float32,
        )

        # Etat interne (initialise dans reset)
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0        # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.peak_balance = initial_balance
        self.trade_history: list[float] = []
        self.reward_history: list[float] = []

        # Sous-ensemble par episode
        self._episode_start = 0
        self._episode_end = len(self.df)

        logger.info(
            f"TradingEnv initialise : "
            f"{len(df)} bougies | "
            f"{self.n_features} features | "
            f"state_dim={self.n_state} | "
            f"max_steps={max_steps_per_episode or 'all'}"
        )

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'observation."""
        row = self.df.iloc[self.current_step]

        # Features techniques
        features = row[self.feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Prix courant
        current_price = row['close']

        # PnL non realise (normalise)
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price
        else:
            pnl_pct = 0.0

        # Drawdown courant
        drawdown = (self.balance - self.peak_balance) / self.peak_balance

        # State complet
        state = np.concatenate([
            features,
            [float(self.position),   # -1, 0, 1
             float(pnl_pct),         # PnL non realise
             float(drawdown)],       # Drawdown courant
        ]).astype(np.float32)

        return state

    def _get_info(self) -> Dict[str, Any]:
        """Informations de debug."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'n_trades': len(self.trade_history),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reinitialise l'environnement.

        Si max_steps_per_episode est defini, choisit un sous-ensemble
        aleatoire de bougies consecutives pour cet episode.
        """
        super().reset(seed=seed)

        # Sous-ensemble aleatoire si max_steps defini
        if self.max_steps_per_episode and self.max_steps_per_episode < len(self.df):
            max_start = len(self.df) - self.max_steps_per_episode
            self._episode_start = self.np_random.integers(0, max_start)
            self._episode_end = self._episode_start + self.max_steps_per_episode
        else:
            self._episode_start = 0
            self._episode_end = len(self.df)

        self.current_step = self._episode_start
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.trade_history = []
        self.reward_history = []

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute une action dans l'environnement.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY

        Returns:
            obs, reward, terminated, truncated, info
        """
        current_price = float(self.df.iloc[self.current_step]['close'])
        reward = 0.0
        trade_pnl = 0.0

        # Couts
        cost = current_price * self.transaction_cost

        # === LOGIQUE DE TRADING ===
        if action == 2 and self.position != 1:  # BUY
            if self.position == -1:
                # Ferme short
                trade_pnl = (self.entry_price - current_price) / self.entry_price
                self.balance *= (1 + trade_pnl)
                self.balance -= cost
                self.trade_history.append(trade_pnl)

            self.position = 1
            self.entry_price = current_price
            self.balance -= cost

        elif action == 0 and self.position != -1:  # SELL
            if self.position == 1:
                # Ferme long
                trade_pnl = (current_price - self.entry_price) / self.entry_price
                self.balance *= (1 + trade_pnl)
                self.balance -= cost
                self.trade_history.append(trade_pnl)

            self.position = -1
            self.entry_price = current_price
            self.balance -= cost

        # Mise a jour peak
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # === REWARD ===
        if self.use_risk_adjusted_reward:
            reward = self._risk_adjusted_reward(trade_pnl, current_price)
        else:
            reward = trade_pnl * self.reward_scaling

        self.reward_history.append(reward)

        # Avance d'un step
        self.current_step += 1

        # Terminaison
        terminated = False
        truncated = False

        # Fin des donnees (ou fin du sous-ensemble)
        if self.current_step >= self._episode_end - 1:
            terminated = True
            # Ferme position finale
            if self.position != 0:
                final_price = float(self.df.iloc[self._episode_end - 1]['close'])
                if self.position == 1:
                    final_pnl = (final_price - self.entry_price) / self.entry_price
                else:
                    final_pnl = (self.entry_price - final_price) / self.entry_price
                self.balance *= (1 + final_pnl)

        # Stop si drawdown trop important
        drawdown = (self.balance - self.peak_balance) / self.peak_balance
        if drawdown < -self.max_drawdown_limit:
            terminated = True
            reward -= 10.0  # Penalite forte
            logger.debug(
                f"Stop drawdown : {drawdown:.2%} a step {self.current_step}"
            )

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.n_state, dtype=np.float32)
        )

        return obs, float(reward), terminated, truncated, self._get_info()

    def _risk_adjusted_reward(
        self,
        trade_pnl: float,
        current_price: float,
    ) -> float:
        """
        Reward ajuste au risque.
        Combine PnL, penalite drawdown et penalite sur-trading.
        """
        reward = trade_pnl * self.reward_scaling

        # Penalite drawdown
        drawdown = (self.balance - self.peak_balance) / self.peak_balance
        if drawdown < -0.05:
            reward += drawdown * 2.0  # Penalite proportionnelle

        # Penalite sur-trading (trop de trades = couts eleves)
        if len(self.trade_history) > 0 and self.current_step > 0:
            trade_frequency = len(self.trade_history) / self.current_step
            if trade_frequency > 0.3:  # Plus de 30% des steps = trade
                reward -= 0.1

        return float(reward)

    def get_episode_stats(self) -> Dict[str, float]:
        """Statistiques de fin d'episode."""
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        if self.trade_history:
            win_rate = sum(1 for t in self.trade_history if t > 0) / len(self.trade_history)
            avg_trade = float(np.mean(self.trade_history))
        else:
            win_rate = 0.0
            avg_trade = 0.0

        return {
            'total_return': total_return,
            'final_balance': self.balance,
            'n_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade,
            'total_reward': sum(self.reward_history),
        }
