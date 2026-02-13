"""Bloc court terme - Features basees sur prix et volume recents.

Features produites :
    return_1, return_4, ema_20, ema_50, ema_diff, rsi_14,
    rolling_std_20, range_15m, body, upper_wick, lower_wick,
    body_ratio, upper_wick_ratio, lower_wick_ratio
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les rendements sur 1 et 4 periodes.

    Args:
        df: DataFrame avec colonne 'close'.

    Returns:
        DataFrame avec return_1 et return_4 ajoutes.
    """
    df = df.copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_4"] = df["close"].pct_change(4)
    logger.debug("Returns calcules : return_1, return_4")
    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule EMA 20, EMA 50 et leur difference normalisee.

    Args:
        df: DataFrame avec colonne 'close'.

    Returns:
        DataFrame avec ema_20, ema_50, ema_diff ajoutes.
    """
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_diff"] = (df["ema_20"] - df["ema_50"]) / df["close"]
    logger.debug("EMA calcules : ema_20, ema_50, ema_diff")
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calcule le RSI sur `period` periodes.

    Args:
        df: DataFrame avec colonne 'close'.
        period: Periode RSI (defaut 14).

    Returns:
        DataFrame avec rsi_14 ajoute (borne [0, 100]).
    """
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)
    logger.debug("RSI calcule : rsi_14")
    return df


def add_rolling_std(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule la volatilite rolling sur 20 periodes.

    Args:
        df: DataFrame avec colonne 'close'.

    Returns:
        DataFrame avec rolling_std_20 ajoute.
    """
    df = df.copy()
    df["rolling_std_20"] = df["close"].pct_change().rolling(20).std()
    logger.debug("Rolling std calcule : rolling_std_20")
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features de structure de bougie.

    Features :
        range_15m : amplitude high-low
        body : taille absolue du corps
        upper_wick, lower_wick : meches
        body_ratio, upper_wick_ratio, lower_wick_ratio : normalises par range

    Args:
        df: DataFrame avec colonnes OHLC.

    Returns:
        DataFrame avec features bougie ajoutees.
    """
    df = df.copy()
    df["range_15m"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    range_safe = df["range_15m"].replace(0, np.nan)
    df["body_ratio"] = df["body"] / range_safe
    df["upper_wick_ratio"] = df["upper_wick"] / range_safe
    df["lower_wick_ratio"] = df["lower_wick"] / range_safe
    logger.debug("Candle features calculees : range_15m, body, wicks, ratios")
    return df


def compute_short_term_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule toutes les features court terme.

    Args:
        df: DataFrame M15 nettoye.

    Returns:
        DataFrame avec toutes les features court terme.
    """
    logger.info("Calcul des features court terme...")
    df = add_returns(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_rolling_std(df)
    df = add_candle_features(df)

    short_term_cols = [
        "return_1", "return_4",
        "ema_20", "ema_50", "ema_diff",
        "rsi_14",
        "rolling_std_20",
        "range_15m", "body", "upper_wick", "lower_wick",
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    ]
    logger.success(f"{len(short_term_cols)} features court terme calculees")
    return df
