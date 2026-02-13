"""
Tests unitaires pour le pipeline ML.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from features.pipeline import FEATURE_COLUMNS
from training.ml.models import evaluate_model, get_models
from training.ml.prepare_data import create_target, prepare_ml_data


@pytest.fixture
def sample_df():
    """DataFrame synthetique avec features."""
    n = 300
    np.random.seed(42)
    prices = 1.25 + np.cumsum(np.random.normal(0, 0.0002, n))

    df = pd.DataFrame(
        {col: np.random.normal(0, 1, n) for col in FEATURE_COLUMNS},
        index=pd.date_range('2022-01-01', periods=n, freq='15min'),
    )
    df['close'] = prices
    df['rsi_14'] = 50 + np.random.normal(0, 15, n)
    df['rsi_14'] = df['rsi_14'].clip(0, 100)

    return df


def test_create_target_binary(sample_df):
    """Target doit etre binaire {0, 1}."""
    df = create_target(sample_df)
    assert set(df['target'].unique()).issubset({0, 1})


def test_create_target_no_lookahead(sample_df):
    """Verifie que la target est bien decalee d'une periode."""
    df = create_target(sample_df)

    # target[i] = 1 si close[i+1] > close[i]
    for i in range(min(10, len(df) - 1)):
        expected = int(sample_df['close'].iloc[i + 1] > sample_df['close'].iloc[i])
        assert df['target'].iloc[i] == expected, f"Leakage detecte a l'index {i}"


def test_create_target_drops_last_row(sample_df):
    """La derniere ligne doit etre supprimee (pas de t+1)."""
    df = create_target(sample_df)
    assert len(df) == len(sample_df) - 1


def test_prepare_ml_data_shapes(sample_df):
    """X et y doivent avoir le meme nombre de lignes."""
    df = create_target(sample_df)
    X, y = prepare_ml_data(df, FEATURE_COLUMNS)

    assert len(X) == len(y)
    assert X.shape[1] == len(FEATURE_COLUMNS)


def test_no_nan_in_X(sample_df):
    """X ne doit pas contenir de NaN."""
    df = create_target(sample_df)
    X, y = prepare_ml_data(df, FEATURE_COLUMNS)
    assert X.isna().sum().sum() == 0


def test_models_fit_predict(sample_df):
    """Tous les modeles doivent fitter et predire sans erreur."""
    df = create_target(sample_df)
    X, y = prepare_ml_data(df, FEATURE_COLUMNS)

    # Split simple
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val, f"test_{name}")

        assert 0 <= metrics['accuracy'] <= 1, f"{name}: accuracy hors bornes"
        assert 0 <= metrics['roc_auc'] <= 1, f"{name}: roc_auc hors bornes"
        assert metrics['n_samples'] == len(X_val)


def test_no_data_leakage_temporal(sample_df):
    """Verifie que le split est bien temporel."""
    df = create_target(sample_df)
    X, y = prepare_ml_data(df, FEATURE_COLUMNS)

    split_idx = int(len(X) * 0.8)
    train_dates = X.index[:split_idx]
    val_dates = X.index[split_idx:]

    assert train_dates.max() < val_dates.min(), \
        "LEAKAGE : donnees train et val se chevauchent temporellement !"
