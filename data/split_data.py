"""Phase 4 : Split temporel strict des données M15.

Split :
    - 2022 : Train
    - 2023 : Validation
    - 2024 : Test final (jamais touché pendant le développement)
"""

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import config


def load_m15_clean() -> pd.DataFrame:
    """Charge les données M15 nettoyées.

    Returns:
        DataFrame M15 propre.

    Raises:
        FileNotFoundError: Si le fichier parquet n'existe pas.
    """
    parquet_path = PROJECT_ROOT / config.get("data.processed_m15")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Fichier M15 nettoyé non trouvé : {parquet_path}. "
            "Lancez d'abord clean_m15.py"
        )

    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"M15 nettoyé chargé : {len(df):,} bougies")
    return df


def split_temporal(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Découpe les données selon les dates du config.yaml.

    Args:
        df: DataFrame M15 complet avec colonne timestamp.

    Returns:
        Dictionnaire {"train": df, "val": df, "test": df}.
    """
    splits: dict[str, pd.DataFrame] = {}

    for name, start_key, end_key in [
        ("train", "split.train_start", "split.train_end"),
        ("val", "split.val_start", "split.val_end"),
        ("test", "split.test_start", "split.test_end"),
    ]:
        start = pd.Timestamp(config.get(start_key))
        end = pd.Timestamp(config.get(end_key)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        splits[name] = df[mask].copy().reset_index(drop=True)

        logger.info(
            f"  {name:5s} : {len(splits[name]):>6,} bougies "
            f"({splits[name]['timestamp'].min().date()} -> "
            f"{splits[name]['timestamp'].max().date()})"
        )

    return splits


def verify_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Vérifie l'intégrité des splits.

    Args:
        splits: Dictionnaire des DataFrames par split.

    Raises:
        ValueError: Si les splits ont un overlap ou un ordre incorrect.
    """
    # Vérifier qu'aucun split n'est vide
    for name, df in splits.items():
        if len(df) == 0:
            raise ValueError(f"Split '{name}' est vide !")

    # Vérifier l'ordre chronologique
    max_train = splits["train"]["timestamp"].max()
    min_val = splits["val"]["timestamp"].min()
    max_val = splits["val"]["timestamp"].max()
    min_test = splits["test"]["timestamp"].min()

    if max_train >= min_val:
        raise ValueError(
            f"Overlap train/val : max_train={max_train}, min_val={min_val}"
        )
    if max_val >= min_test:
        raise ValueError(
            f"Overlap val/test : max_val={max_val}, min_test={min_test}"
        )

    logger.info("Vérification splits OK : pas d'overlap, ordre chronologique respecté")

    # Vérifier couverture
    total = sum(len(df) for df in splits.values())
    logger.info(f"Total splits : {total:,} bougies")


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Sauvegarde chaque split en Parquet.

    Args:
        splits: Dictionnaire des DataFrames par split.
    """
    splits_dir = PROJECT_ROOT / "data" / "splits"

    for name, df in splits.items():
        path = splits_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Sauvegardé : {path} ({path.stat().st_size / 1e6:.1f} MB)")


def save_split_report(splits: dict[str, pd.DataFrame]) -> None:
    """Génère et sauvegarde le rapport de split.

    Args:
        splits: Dictionnaire des DataFrames par split.
    """
    report_path = PROJECT_ROOT / "data" / "splits" / "split_report.json"
    total = sum(len(df) for df in splits.values())

    report: dict = {}
    for name, df in splits.items():
        report[name] = {
            "n_candles": len(df),
            "date_range": [
                str(df["timestamp"].min().date()),
                str(df["timestamp"].max().date()),
            ],
            "pct_total": round(len(df) / total * 100, 1),
        }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Rapport split sauvegardé : {report_path}")


def main() -> None:
    """Point d'entrée principal : split temporel."""
    logger.info("=" * 50)
    logger.info("PHASE 4 : Split temporel")
    logger.info("=" * 50)

    # Chargement
    df = load_m15_clean()

    # Split
    splits = split_temporal(df)

    # Vérifications
    verify_splits(splits)

    # Sauvegarde
    save_splits(splits)

    # Rapport
    save_split_report(splits)

    logger.success("Phase 4 terminée : splits train/val/test sauvegardés")


if __name__ == "__main__":
    main()
