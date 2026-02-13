"""Tests unitaires pour le feature engineering."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.pipeline import WARMUP_PERIODS, FeaturePipeline


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Cree un DataFrame M15 synthetique pour tests."""
    n = 1000
    rng = np.random.default_rng(42)

    prices = 1.25 + np.cumsum(rng.normal(0, 0.0002, n))

    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="15min"),
        "open": prices + rng.normal(0, 0.0001, n),
        "high": prices + np.abs(rng.normal(0, 0.0003, n)),
        "low": prices - np.abs(rng.normal(0, 0.0003, n)),
        "close": prices,
        "volume": rng.integers(100, 1000, n).astype(float),
        "hour": 0,
        "day_of_week": 0,
        "is_session_start": False,
        "is_gap": False,
    })

    # Assure coherence OHLC
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def test_pipeline_fit_transform(sample_df: pd.DataFrame) -> None:
    """Teste fit_transform sur train."""
    pipeline = FeaturePipeline()
    df_feat = pipeline.fit_transform(sample_df)

    assert pipeline.is_fitted, "Pipeline devrait etre fitte"
    assert len(df_feat) == len(sample_df) - WARMUP_PERIODS
    assert len(df_feat) > 0


def test_no_nan_after_warmup(sample_df: pd.DataFrame) -> None:
    """Verifie l'absence de NaN apres warm-up."""
    pipeline = FeaturePipeline()
    df_feat = pipeline.fit_transform(sample_df)

    for col in pipeline.get_feature_names():
        nan_count = df_feat[col].isna().sum()
        assert nan_count == 0, f"NaN detectes dans {col}: {nan_count}"


def test_no_leakage_transform(sample_df: pd.DataFrame) -> None:
    """Verifie que transform ne re-fitte pas le scaler."""
    pipeline = FeaturePipeline()

    train = sample_df.iloc[:600].copy().reset_index(drop=True)
    val = sample_df.iloc[600:].copy().reset_index(drop=True)

    pipeline.fit_transform(train)

    scaler_mean_before = pipeline.scaler.mean_.copy()

    pipeline.transform(val, "val")

    np.testing.assert_array_equal(
        pipeline.scaler.mean_,
        scaler_mean_before,
        err_msg="LEAKAGE : le scaler a ete refitte sur val !",
    )


def test_transform_without_fit_raises() -> None:
    """Verifie qu'on ne peut pas transformer sans fitter."""
    pipeline = FeaturePipeline()

    with pytest.raises(RuntimeError):
        pipeline.transform(pd.DataFrame(), "val")


def test_feature_count(sample_df: pd.DataFrame) -> None:
    """Verifie le nombre de features produit."""
    pipeline = FeaturePipeline()
    df_feat = pipeline.fit_transform(sample_df)

    for col in pipeline.get_feature_names():
        assert col in df_feat.columns, f"Feature manquante : {col}"


def test_rsi_bounds(sample_df: pd.DataFrame) -> None:
    """RSI doit etre entre 0 et 100."""
    from features.short_term import add_rsi

    df_rsi = add_rsi(sample_df.copy())

    assert df_rsi["rsi_14"].min() >= 0, "RSI < 0 !"
    assert df_rsi["rsi_14"].max() <= 100, "RSI > 100 !"


def test_save_and_load(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """Teste sauvegarde et chargement du pipeline."""
    pipeline = FeaturePipeline()
    pipeline.fit_transform(sample_df)

    save_path = str(tmp_path / "test_pipeline.pkl")
    pipeline.save(save_path)

    loaded = FeaturePipeline.load(save_path)
    assert loaded.is_fitted
    assert loaded.get_feature_count() == pipeline.get_feature_count()
    np.testing.assert_array_almost_equal(
        loaded.scaler.mean_, pipeline.scaler.mean_
    )
