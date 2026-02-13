"""
Preparation des donnees pour le ML.
Creation target, alignement features/target, verification anti-leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Tuple


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cree la target binaire.

    y[t] = 1 si close[t+1] > close[t], sinon 0

    IMPORTANT : Le shift est sur le futur donc on doit aligner
    features[t] avec target[t] = direction de t vers t+1.

    Args:
        df: DataFrame avec colonne 'close'

    Returns:
        DataFrame avec colonne 'target' ajoutee
    """
    df = df.copy()

    # close[t+1] > close[t] -> shift(-1) pour regarder la prochaine bougie
    future_close = df['close'].shift(-1)
    df['target'] = np.where(
        future_close.isna(),
        np.nan,
        (future_close > df['close']).astype(float),
    )

    # Derniere ligne = NaN (pas de t+1 connu) -> on la supprime
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    n_up = df['target'].sum()
    n_down = len(df) - n_up
    balance = n_up / len(df) * 100

    logger.info(
        f"Target creee : {n_up} UP ({balance:.1f}%) | "
        f"{n_down} DOWN ({100-balance:.1f}%)"
    )
    return df


def prepare_ml_data(
    df: pd.DataFrame,
    feature_columns: list,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separe features et target, verifie integrite.

    Args:
        df: DataFrame avec features ET target
        feature_columns: Liste des colonnes features

    Returns:
        Tuple (X, y)
    """
    # Verification colonnes
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes features manquantes : {missing}")

    if 'target' not in df.columns:
        raise ValueError("Colonne 'target' manquante. Lance create_target() d'abord.")

    # Suppression NaN residuels
    df_clean = df[feature_columns + ['target']].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.warning(f"{n_dropped} lignes supprimees (NaN)")

    X = df_clean[feature_columns]
    y = df_clean['target']

    logger.info(f"X shape : {X.shape} | y shape : {y.shape}")
    logger.info(f"Classe 1 (UP) : {y.mean()*100:.1f}%")

    return X, y


def load_ml_splits(
    feature_columns: list,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Charge et prepare les 3 splits pour ML.

    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Chargement des splits pour ML...")

    df_train = pd.read_parquet('data/splits/train_features.parquet')
    df_val = pd.read_parquet('data/splits/val_features.parquet')
    df_test = pd.read_parquet('data/splits/test_features.parquet')

    # Creation targets
    df_train = create_target(df_train)
    df_val = create_target(df_val)
    df_test = create_target(df_test)

    # Separation X/y
    X_train, y_train = prepare_ml_data(df_train, feature_columns)
    X_val, y_val = prepare_ml_data(df_val, feature_columns)
    X_test, y_test = prepare_ml_data(df_test, feature_columns)

    logger.success(
        f"Splits prets -> "
        f"Train: {X_train.shape} | "
        f"Val: {X_val.shape} | "
        f"Test: {X_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
