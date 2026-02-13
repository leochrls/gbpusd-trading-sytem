"""Strategies baseline pour benchmark.

Toutes heritent de BaseStrategy et implementent generate_signals().
Actions possibles : BUY (2), HOLD (1), SELL (0).
"""

from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np
import pandas as pd
from loguru import logger


class Action(IntEnum):
    """Actions possibles pour le systeme de trading."""

    SELL = 0
    HOLD = 1
    BUY = 2


class BaseStrategy(ABC):
    """Classe abstraite pour toutes les strategies.

    Attributes:
        name: Nom de la strategie.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genere les signaux de trading.

        Args:
            df: DataFrame M15 avec features.

        Returns:
            Series d'actions (Action.BUY / SELL / HOLD).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RandomStrategy(BaseStrategy):
    """Strategie aleatoire uniforme (lower bound).

    Attributes:
        seed: Graine pour reproductibilite.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(name="Random")
        self.seed = seed

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Action aleatoire uniforme parmi BUY/SELL/HOLD."""
        rng = np.random.default_rng(self.seed)
        actions = rng.choice(
            [Action.SELL, Action.HOLD, Action.BUY],
            size=len(df),
        )
        logger.debug(
            f"Random : {np.sum(actions == Action.BUY)} BUY | "
            f"{np.sum(actions == Action.SELL)} SELL | "
            f"{np.sum(actions == Action.HOLD)} HOLD"
        )
        return pd.Series(actions, index=df.index, name="signal")


class RuleBasedStrategy(BaseStrategy):
    """Strategie basee sur regles techniques.

    Regles :
        BUY  : ema_20 > ema_50 ET rsi_14 < rsi_oversold
        SELL : ema_20 < ema_50 ET rsi_14 > rsi_overbought
        HOLD : sinon

    Attributes:
        rsi_oversold: Seuil de survente RSI.
        rsi_overbought: Seuil de surachat RSI.
    """

    def __init__(
        self,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
    ) -> None:
        super().__init__(name="RuleBased")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genere signaux bases sur EMA crossover + RSI.

        Args:
            df: DataFrame avec colonnes ema_20, ema_50, rsi_14.

        Returns:
            Series de signaux.

        Raises:
            ValueError: Si colonnes requises manquantes.
        """
        required = ["ema_20", "ema_50", "rsi_14"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        signals = pd.Series(Action.HOLD, index=df.index, name="signal")

        bullish_trend = df["ema_20"] > df["ema_50"]
        bearish_trend = df["ema_20"] < df["ema_50"]
        oversold = df["rsi_14"] < self.rsi_oversold
        overbought = df["rsi_14"] > self.rsi_overbought

        signals[bullish_trend & oversold] = Action.BUY
        signals[bearish_trend & overbought] = Action.SELL

        n_buy = int((signals == Action.BUY).sum())
        n_sell = int((signals == Action.SELL).sum())
        n_hold = int((signals == Action.HOLD).sum())
        logger.debug(f"RuleBased : {n_buy} BUY | {n_sell} SELL | {n_hold} HOLD")
        return signals


class BuyAndHoldStrategy(BaseStrategy):
    """Strategie Buy & Hold (benchmark passif).

    Achete au debut, tient jusqu'a la fin.
    """

    def __init__(self) -> None:
        super().__init__(name="BuyAndHold")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """BUY au debut, HOLD ensuite."""
        signals = pd.Series(Action.HOLD, index=df.index, name="signal")
        if len(df) > 0:
            signals.iloc[0] = Action.BUY
        logger.debug(f"BuyAndHold : 1 BUY + {len(df) - 1} HOLD")
        return signals
