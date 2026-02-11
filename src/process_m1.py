import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_year(year, raw_path):
    """
    Load raw CSV for a specific year, process it, and return cleaned DataFrame and report string.
    """
    report = []
    report.append(f"Processing Year: {year}")
    
    # Define file path (assuming standard naming convention from data exploration)
    # The user has data/raw/2022/DAT_MT_GBPUSD_M1_2022.csv etc.
    file_path = raw_path / str(year) / f"DAT_MT_GBPUSD_M1_{year}.csv"
    
    if not file_path.exists():
        return None, f"ERROR: File not found: {file_path}\n"

    # Load Data (no header based on previous inspection)
    # Col 0: Date, Col 1: Time, Col 2: Open, Col 3: High, Col 4: Low, Col 5: Close, Col 6: Vol
    try:
        df = pd.read_csv(file_path, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        return None, f"ERROR: Failed to read {file_path}: {e}\n"

    # Merge Date + Time -> Timestamp
    try:
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.drop(columns=['date', 'time'], inplace=True)
    except Exception as e:
        return None, f"ERROR: Timestamp conversion failed for {year}: {e}\n"

    # Chronological Sort
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    
    initial_count = len(df)
    report.append(f"Initial rows: {initial_count}")

    # Remove duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        df = df[~df.index.duplicated(keep='first')]
        report.append(f"Removed duplicates: {duplicates}")
    
    # Sub-Sample checking (Regularity 1 min)
    # We expect a full grid? Forex markets close on weekends. 
    # The requirement "Vérification régularité 1 minute" usually implies checking if we have holes during trading hours.
    # For simplicity and robustness, we will create a full range min-max and report missing ratio.
    # However, filling all gaps (including weekends) with NaNs might be huge. 
    # Let's count gaps > 1 minute.
    
    time_diff = df.index.to_series().diff()
    gaps = time_diff[time_diff > pd.Timedelta(minutes=1)]
    
    # Simple gap report
    report.append(f"Number of gaps > 1 min: {len(gaps)}")
    if len(gaps) > 0:
        report.append(f"Largest gap: {gaps.max()}")

    # Detection Incoherences
    inconsistencies = 0
    
    # 1. High < Low
    bad_hl = df[df['high'] < df['low']]
    if len(bad_hl) > 0:
        report.append(f"WARNING: High < Low detected: {len(bad_hl)} rows")
        inconsistencies += len(bad_hl)
        # Filter them out? usually yes for clean data
        df = df[df['high'] >= df['low']]

    # 2. Price <= 0
    bad_price = df[(df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)]
    if len(bad_price) > 0:
        report.append(f"WARNING: Non-positive prices detected: {len(bad_price)} rows")
        inconsistencies += len(bad_price)
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    # 3. OHLC validity (Open/Close must be within Low-High)
    # Allow small epsilon for floating point issues
    epsilon = 1e-5
    bad_oc = df[
        (df['open'] < df['low'] - epsilon) | (df['open'] > df['high'] + epsilon) |
        (df['close'] < df['low'] - epsilon) | (df['close'] > df['high'] + epsilon)
    ]
    if len(bad_oc) > 0:
        report.append(f"WARNING: Open/Close outside High/Low: {len(bad_oc)} rows")
        inconsistencies += len(bad_oc)
        # Filter strictly
        df = df[
            (df['open'] >= df['low'] - epsilon) & (df['open'] <= df['high'] + epsilon) &
            (df['close'] >= df['low'] - epsilon) & (df['close'] <= df['high'] + epsilon)
        ]

    final_count = len(df)
    report.append(f"Final rows after cleaning: {final_count}")
    report.append("-" * 30 + "\n")
    
    return df, "\n".join(report)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / 'data' / 'raw'
    processed_dir = root_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    years = [2022, 2023, 2024]
    all_reports = []
    
    for year in years:
        df, report = load_and_process_year(year, raw_dir)
        all_reports.append(report)
        
        if df is not None:
            output_file = processed_dir / f"GBPUSD_M1_{year}.parquet"
            df.to_parquet(output_file)
            print(f"Saved {output_file}")
            
            # Also save a small head csv for quick manual check
            # df.head().to_csv(processed_dir / f"GBPUSD_M1_{year}_head.csv")

    # Write full report
    with open(processed_dir / "data_quality_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports))
    print(f"Report saved to {processed_dir / 'data_quality_report.txt'}")

if __name__ == "__main__":
    main()
