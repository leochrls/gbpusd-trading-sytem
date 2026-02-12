import pandas as pd
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def add_features(file_path: Path):
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Processing {file_path.name}...")
    df = pd.read_parquet(file_path)
    original_cols = df.columns.tolist()
    
    # Add features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        
        # Save back
        df.to_parquet(file_path, index=False)
        logger.success(f"Updated {file_path.name}. Added columns: {list(set(df.columns) - set(original_cols))}")
    else:
        logger.warning(f"Skipping {file_path.name}: 'timestamp' column not found.")

def main():
    splits_dir = PROJECT_ROOT / "data" / "splits"
    for split in ["train.parquet", "val.parquet", "test.parquet"]:
        add_features(splits_dir / split)

if __name__ == "__main__":
    main()
