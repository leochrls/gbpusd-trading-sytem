"""Script principal : calcule les features pour train/val/test."""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.pipeline import FeaturePipeline


def main() -> None:
    """Execute le pipeline de feature engineering complet."""
    logger.info("=== DEBUT FEATURE ENGINEERING ===")

    # Chargement splits
    splits_dir = PROJECT_ROOT / "data" / "splits"
    df_train = pd.read_parquet(splits_dir / "train.parquet")
    df_val = pd.read_parquet(splits_dir / "val.parquet")
    df_test = pd.read_parquet(splits_dir / "test.parquet")

    logger.info(
        f"Splits charges : train={len(df_train)}, "
        f"val={len(df_val)}, test={len(df_test)}"
    )

    # Pipeline
    pipeline = FeaturePipeline()

    # FIT + TRANSFORM sur train (scaler fitte ici)
    pipeline_path = str(PROJECT_ROOT / "models" / "v1" / "feature_pipeline.pkl")
    df_train_feat = pipeline.fit_transform(df_train, save_path=pipeline_path)

    # TRANSFORM sur val (scaler du train applique)
    df_val_feat = pipeline.transform(df_val, split_name="val")

    # TRANSFORM sur test (scaler du train applique)
    df_test_feat = pipeline.transform(df_test, split_name="test")

    # Sauvegarde features
    df_train_feat.to_parquet(splits_dir / "train_features.parquet", index=False)
    df_val_feat.to_parquet(splits_dir / "val_features.parquet", index=False)
    df_test_feat.to_parquet(splits_dir / "test_features.parquet", index=False)

    logger.success("Features sauvegardees :")
    logger.success(f"   -> {splits_dir / 'train_features.parquet'}")
    logger.success(f"   -> {splits_dir / 'val_features.parquet'}")
    logger.success(f"   -> {splits_dir / 'test_features.parquet'}")

    # Rapport
    feature_names = pipeline.get_feature_names()
    logger.info(f"\nFeatures ({pipeline.get_feature_count()}) :")
    for i, feat in enumerate(feature_names, 1):
        logger.info(f"   {i:2d}. {feat}")

    # Stats rapides
    logger.info("\nStats features train (apres normalisation) :")
    stats = df_train_feat[feature_names].describe().loc[["mean", "std", "min", "max"]]
    logger.info(f"\n{stats.to_string()}")

    logger.success("=== FEATURE ENGINEERING TERMINE ===")
    return pipeline


if __name__ == "__main__":
    main()
