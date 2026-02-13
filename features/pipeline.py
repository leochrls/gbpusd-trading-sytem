"""Pipeline de feature engineering complet.

Gere le calcul, le warm-up, la normalisation sans data leakage.

Usage :
    pipeline = FeaturePipeline()
    df_train_feat = pipeline.fit_transform(df_train)
    df_val_feat = pipeline.transform(df_val, "val")
    pipeline.save("models/v1/feature_pipeline.pkl")
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from features.regime import compute_regime_features
from features.short_term import compute_short_term_features

# Colonnes features dans l'ordre attendu par les modeles
FEATURE_COLUMNS: list[str] = [
    # Court terme
    "return_1", "return_4",
    "ema_diff",
    "rsi_14",
    "rolling_std_20",
    "range_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    # Regime
    "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal", "macd_histogram",
]

# Colonnes a NE PAS normaliser (deja en %, ratio, etc.)
COLUMNS_SKIP_NORM: list[str] = [
    "rsi_14",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "volatility_ratio",
]

# Warm-up minimum (EMA 200 a besoin de ~200 bougies)
WARMUP_PERIODS: int = 210


class FeaturePipeline:
    """Pipeline complet de feature engineering.

    Attributes:
        scaler: StandardScaler fitte sur train uniquement.
        is_fitted: True apres fit_transform.
        feature_columns: Liste ordonnee des features.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_fitted: bool = False
        self.feature_columns: list[str] = FEATURE_COLUMNS

    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule toutes les features sans normalisation."""
        df = compute_short_term_features(df)
        df = compute_regime_features(df)
        return df

    def _remove_warmup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les premieres lignes (warm-up EMA 200).

        Args:
            df: DataFrame avec features calculees.

        Returns:
            DataFrame sans les lignes de warm-up.
        """
        n_before = len(df)
        df = df.iloc[WARMUP_PERIODS:].copy()
        df = df.reset_index(drop=True)
        logger.info(f"Warm-up : {n_before - len(df)} bougies supprimees (garde {len(df)})")
        return df

    def _validate_no_nan(self, df: pd.DataFrame, split_name: str = "") -> None:
        """Verifie absence de NaN dans les features."""
        nan_counts = df[self.feature_columns].isna().sum()
        if nan_counts.any():
            problematic = nan_counts[nan_counts > 0]
            logger.warning(f"NaN detectes dans {split_name}:\n{problematic}")
        else:
            logger.info(f"Pas de NaN dans les features ({split_name})")

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Calcule les features et FIT le scaler sur train.

        IMPORTANT : Appeler UNIQUEMENT sur df_train.

        Args:
            df_train: DataFrame train (2022).
            save_path: Chemin pour sauvegarder le pipeline.

        Returns:
            DataFrame train avec features normalisees.
        """
        logger.info("=== FIT_TRANSFORM sur TRAIN ===")

        df = self._compute_all_features(df_train)
        df = self._remove_warmup(df)
        self._validate_no_nan(df, "train")

        cols_to_scale = [
            c for c in self.feature_columns if c not in COLUMNS_SKIP_NORM
        ]

        df_scaled = df.copy()
        df_scaled[cols_to_scale] = self.scaler.fit_transform(
            df[cols_to_scale].fillna(0)
        )

        self.is_fitted = True
        logger.success(
            f"Scaler fitte sur {len(df)} bougies train | "
            f"{len(self.feature_columns)} features"
        )

        if save_path:
            self.save(save_path)

        return df_scaled

    def transform(
        self,
        df: pd.DataFrame,
        split_name: str = "unknown",
    ) -> pd.DataFrame:
        """Calcule les features et applique le scaler (deja fitte).

        IMPORTANT : Ne jamais re-fitter ici (data leakage sinon).

        Args:
            df: DataFrame val ou test.
            split_name: Nom du split pour les logs.

        Returns:
            DataFrame avec features normalisees.

        Raises:
            RuntimeError: Si le pipeline n'est pas fitte.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline non fitte ! Appelle fit_transform(df_train) d'abord."
            )

        logger.info(f"=== TRANSFORM sur {split_name.upper()} ===")

        df = self._compute_all_features(df)
        df = self._remove_warmup(df)
        self._validate_no_nan(df, split_name)

        cols_to_scale = [
            c for c in self.feature_columns if c not in COLUMNS_SKIP_NORM
        ]

        df_scaled = df.copy()
        df_scaled[cols_to_scale] = self.scaler.transform(
            df[cols_to_scale].fillna(0)
        )

        logger.success(f"Transform {split_name} termine : {len(df_scaled)} bougies")
        return df_scaled

    def save(self, path: str) -> None:
        """Sauvegarde le pipeline (scaler inclus).

        Args:
            path: Chemin du fichier pickle.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"Pipeline sauvegarde : {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Charge un pipeline sauvegarde.

        Args:
            path: Chemin du fichier pickle.

        Returns:
            Instance FeaturePipeline chargee.
        """
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline charge : {path}")
        return pipeline

    def get_feature_names(self) -> list[str]:
        """Retourne la liste des features dans l'ordre."""
        return self.feature_columns.copy()

    def get_feature_count(self) -> int:
        """Retourne le nombre de features."""
        return len(self.feature_columns)
