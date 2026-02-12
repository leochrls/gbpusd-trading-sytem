"""Phase 3 : Nettoyage des données M15.

Supprime les bougies incomplètes, les prix invalides, et
détecte les gaps anormaux via ATR.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import config


def load_m15_raw() -> pd.DataFrame:
    """Charge les données M15 brutes.

    Returns:
        DataFrame M15 raw.

    Raises:
        FileNotFoundError: Si aucun fichier parquet n'est trouvé.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    parquet_files = list(processed_dir.glob("*_M15_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"Aucun fichier M15 trouvé dans {processed_dir} avec le pattern *_M15_*.parquet"
        )

    dfs = []
    for file_path in parquet_files:
        df_part = pd.read_parquet(file_path)
        # Assurer que timestamp est une colonne
        if "timestamp" not in df_part.columns and df_part.index.name == "timestamp":
            df_part = df_part.reset_index()
        dfs.append(df_part)
        logger.info(f"Chargé : {file_path.name} ({len(df_part):,} lignes)")

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Total M15 raw chargé : {len(df):,} bougies")
    return df


def remove_incomplete(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les bougies avec moins de 15 bougies M1.

    Args:
        df: DataFrame M15 avec colonne 'n_candles' (optionnelle).

    Returns:
        Tuple (DataFrame filtré, nombre de bougies supprimées).
    """
    if "n_candles" not in df.columns:
        logger.warning("Colonne 'n_candles' absente : filtre incomplet ignoré")
        return df, 0

    n_before = len(df)
    df_clean = df[df["n_candles"] == 15].copy()
    n_removed = n_before - len(df_clean)

    if n_removed > 0:
        logger.warning(f"{n_removed:,} bougies incomplètes supprimées")
    else:
        logger.info("Aucune bougie incomplète à supprimer")

    return df_clean.reset_index(drop=True), n_removed


def remove_negative_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes avec prix négatifs ou nuls.

    Args:
        df: DataFrame M15 avec colonnes OHLC.

    Returns:
        Tuple (DataFrame filtré, nombre de lignes supprimées).
    """
    mask = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    )
    n_removed = int((~mask).sum())

    if n_removed > 0:
        logger.warning(f"{n_removed} lignes avec prix négatifs/nuls supprimées")
        df = df[mask].reset_index(drop=True)
    else:
        logger.info("Aucun prix négatif détecté")

    return df, n_removed


def check_ohlc_coherence(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Vérifie et corrige la cohérence OHLC.

    Vérifie :
        - low <= open <= high
        - low <= close <= high
        - low <= high

    Args:
        df: DataFrame M15.

    Returns:
        Tuple (DataFrame filtré, nombre de lignes supprimées).
    """
    mask_valid = (
        (df["low"] <= df["high"])
        & (df["low"] <= df["open"])
        & (df["open"] <= df["high"])
        & (df["low"] <= df["close"])
        & (df["close"] <= df["high"])
    )
    n_removed = int((~mask_valid).sum())

    if n_removed > 0:
        logger.warning(f"{n_removed} lignes OHLC incohérentes supprimées")
        df = df[mask_valid].reset_index(drop=True)
    else:
        logger.info("Cohérence OHLC OK")

    return df, n_removed


def detect_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les gaps anormaux via ATR.

    Un gap est anormal si abs(open[t] - close[t-1]) > 3 * ATR_14.

    Args:
        df: DataFrame M15 trié chronologiquement.

    Returns:
        DataFrame avec colonne 'is_gap' ajoutée.
    """
    atr_period = config.get("features.regime.atr_period", 14)

    # Calcul True Range
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    # ATR
    atr = true_range.rolling(window=atr_period, min_periods=1).mean()

    # Gap = différence entre open actuel et close précédent
    gap = (df["open"] - df["close"].shift(1)).abs()

    # Gap anormal si > 3 * ATR
    df["is_gap"] = gap > (3 * atr)
    # Première ligne ne peut pas être un gap
    df.loc[0, "is_gap"] = False

    n_gaps = int(df["is_gap"].sum())
    if n_gaps > 0:
        logger.warning(f"{n_gaps} gaps anormaux détectés (> 3*ATR)")
    else:
        logger.info("Aucun gap anormal détecté")

    return df


def save_final_stats(
    df: pd.DataFrame,
    removed_incomplete: int,
    removed_invalid: int,
    original_count: int,
) -> None:
    """Sauvegarde les statistiques finales en JSON.

    Args:
        df: DataFrame M15 final.
        removed_incomplete: Nombre de bougies incomplètes supprimées.
        removed_invalid: Nombre total de lignes invalides supprimées.
        original_count: Nombre original de bougies M15.
    """
    stats_path = PROJECT_ROOT / "data" / "processed" / "m15_final_stats.json"

    n_gaps = int(df["is_gap"].sum())
    date_min = str(df["timestamp"].min().date())
    date_max = str(df["timestamp"].max().date())
    coverage_pct = round(len(df) / original_count * 100, 1)

    stats = {
        "total_candles": len(df),
        "original_candles": original_count,
        "removed_incomplete": removed_incomplete,
        "removed_invalid": removed_invalid,
        "gaps_detected": n_gaps,
        "date_range": [date_min, date_max],
        "coverage_pct": coverage_pct,
    }

    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Stats finales sauvegardées : {stats_path}")
    logger.info(f"  Bougies finales : {stats['total_candles']:,}")
    logger.info(f"  Couverture : {stats['coverage_pct']}%")


def save_m15_clean(df: pd.DataFrame) -> None:
    """Sauvegarde le DataFrame M15 nettoyé en Parquet.

    Args:
        df: DataFrame M15 final nettoyé.
    """
    parquet_path = PROJECT_ROOT / config.get("data.processed_m15")

    # Supprimer les colonnes de travail avant sauvegarde finale
    cols_to_drop = ["incomplete", "n_candles"]
    cols_present = [c for c in cols_to_drop if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)

    df.to_parquet(parquet_path, index=False)
    logger.info(
        f"M15 nettoyé sauvegardé : {parquet_path} "
        f"({parquet_path.stat().st_size / 1e6:.1f} MB)"
    )


def main() -> None:
    """Point d'entrée principal : nettoyage M15."""
    logger.info("=" * 50)
    logger.info("PHASE 3 : Nettoyage M15")
    logger.info("=" * 50)

    # Chargement
    df = load_m15_raw()
    original_count = len(df)

    # Suppression incomplètes
    df, removed_incomplete = remove_incomplete(df)

    # Prix négatifs
    df, removed_neg = remove_negative_prices(df)

    # Cohérence OHLC
    df, removed_ohlc = check_ohlc_coherence(df)

    removed_invalid = removed_neg + removed_ohlc

    # Détection gaps
    df = detect_gaps(df)

    # Stats
    save_final_stats(df, removed_incomplete, removed_invalid, original_count)

    # Sauvegarde
    save_m15_clean(df)

    logger.success(f"Phase 3 terminée : {len(df):,} bougies M15 propres")


if __name__ == "__main__":
    main()
