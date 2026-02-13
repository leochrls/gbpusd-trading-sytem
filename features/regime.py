"""Bloc contexte et regime de marche.

Features de tendance long terme, volatilite et force directionnelle.

Features produites :
    ema_200, distance_to_ema200, slope_ema50,
    atr_14, rolling_std_100, volatility_ratio,
    adx_14, macd, macd_signal, macd_histogram
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_ema200(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule EMA 200, distance au prix et slope EMA 50.

    Args:
        df: DataFrame avec colonne 'close'.

    Returns:
        DataFrame avec ema_200, distance_to_ema200, slope_ema50 ajoutes.
    """
    df = df.copy()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["distance_to_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

    if "ema_50" not in df.columns:
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["slope_ema50"] = df["ema_50"].diff(5) / df["ema_50"].shift(5)

    logger.debug("Tendance long terme : ema_200, distance_to_ema200, slope_ema50")
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calcule l'Average True Range (ATR).

    Args:
        df: DataFrame avec 'high', 'low', 'close'.
        period: Periode ATR (defaut 14).

    Returns:
        DataFrame avec atr_14 ajoute.
    """
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(span=period, adjust=False).mean()

    logger.debug("ATR calcule : atr_14")
    return df


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule rolling_std_100 et volatility_ratio.

    volatility_ratio = rolling_std_20 / rolling_std_100
        > 1 : periode agitee
        < 1 : periode calme

    Args:
        df: DataFrame avec colonne 'close'.

    Returns:
        DataFrame avec rolling_std_100 et volatility_ratio ajoutes.
    """
    df = df.copy()
    returns = df["close"].pct_change()
    df["rolling_std_100"] = returns.rolling(100).std()

    if "rolling_std_20" not in df.columns:
        df["rolling_std_20"] = returns.rolling(20).std()

    std_100_safe = df["rolling_std_100"].replace(0, np.nan)
    df["volatility_ratio"] = df["rolling_std_20"] / std_100_safe

    logger.debug("Regime volatilite : rolling_std_100, volatility_ratio")
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calcule l'Average Directional Index (ADX).

    Mesure la FORCE de la tendance (pas la direction).

    Args:
        df: DataFrame avec 'high', 'low', 'close'.
        period: Periode ADX (defaut 14).

    Returns:
        DataFrame avec adx_14 ajoute.
    """
    df = df.copy()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    atr_safe = atr.replace(0, np.nan)

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean()
        / atr_safe
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()
        / atr_safe
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx_14"] = dx.ewm(span=period, adjust=False).mean()

    logger.debug("ADX calcule : adx_14")
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calcule le MACD normalise et sa ligne de signal.

    Args:
        df: DataFrame avec colonne 'close'.
        fast: Periode EMA rapide (defaut 12).
        slow: Periode EMA lente (defaut 26).
        signal: Periode ligne signal (defaut 9).

    Returns:
        DataFrame avec macd, macd_signal, macd_histogram ajoutes.
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = (ema_fast - ema_slow) / df["close"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    logger.debug("MACD calcule : macd, macd_signal, macd_histogram")
    return df


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule toutes les features de regime.

    Args:
        df: DataFrame M15 (peut deja avoir des features court terme).

    Returns:
        DataFrame avec toutes les features de regime.
    """
    logger.info("Calcul des features de regime...")
    df = add_ema200(df)
    df = add_atr(df)
    df = add_volatility_regime(df)
    df = add_adx(df)
    df = add_macd(df)

    regime_cols = [
        "ema_200", "distance_to_ema200", "slope_ema50",
        "atr_14", "rolling_std_100", "volatility_ratio",
        "adx_14", "macd", "macd_signal", "macd_histogram",
    ]
    logger.success(f"{len(regime_cols)} features de regime calculees")
    return df
