import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_m1(year, raw_path):
    """
    Phase 1: Importation M1
    - Fusion date + time -> timestamp
    - Vérification régularité 1 minute
    - Tri chronologique
    - Détection incohérences
    """
    report = []
    report.append(f"=== Processing M1 Year: {year} ===")
    
    file_path = raw_path / str(year) / f"DAT_MT_GBPUSD_M1_{year}.csv"
    if not file_path.exists():
        return None, f"ERROR: File not found: {file_path}\n"

    # -- Importation --
    try:
        # Col 0: Date, Col 1: Time, Col 2: Open, Col 3: High, Col 4: Low, Col 5: Close, Col 6: Vol
        df = pd.read_csv(file_path, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        return None, f"ERROR: Failed to read {file_path}: {e}\n"

    # -- Fusion date + time -> timestamp --
    try:
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df.drop(columns=['date', 'time'], inplace=True)
    except Exception as e:
        return None, f"ERROR: Timestamp conversion failed for {year}: {e}\n"

    # -- Tri chronologique --
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    
    initial_count = len(df)
    report.append(f"Initial M1 rows: {initial_count}")

    # -- Détection incohérences --
    # 1. High < Low
    bad_hl = df[df['high'] < df['low']]
    if len(bad_hl) > 0:
        report.append(f"WARNING: High < Low detected: {len(bad_hl)} rows. Dropping them.")
        df = df[df['high'] >= df['low']]
    
    # 2. Price <= 0
    bad_price = df[(df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)]
    if len(bad_price) > 0:
        report.append(f"WARNING: Non-positive prices detected: {len(bad_price)} rows. Dropping them.")
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    # -- Vérification régularité 1 minute (Gaps) --
    # On regarde les trous > 1 minute
    if len(df) > 1:
        time_diff = df.index.to_series().diff()
        gaps = time_diff[time_diff > pd.Timedelta(minutes=1)]
        report.append(f"Number of gaps > 1 min in M1 data: {len(gaps)}")
        if len(gaps) > 0:
            report.append(f"Largest M1 gap: {gaps.max()}")
    
    return df, "\n".join(report)

def aggregate_m1_to_m15(df_m1):
    """
    Phase 2: Agrégation M1 -> M15
    Variable    Règle
    open_15m    open 1ère minute
    high_15m    max(high) sur 15 minutes
    low_15m     min(low) sur 15 minutes
    close_15m   close dernière minute
    """
    # Resample 15min. 
    # 'closed' needs to be handled carefully. Typically forex candles are [00:00, 00:15).
    # Pandas default 'left' closed is standard for financial time series (label=left output).
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'  # Optional but good to keep
    }
    
    df_m15 = df_m1.resample('15min').apply(ohlc_dict)
    
    # Drop rows that are fully NaN (where no M1 data existed in that 15m window)
    # This is part of the "clean" check but resampling creates rows for gaps, which we might want to drop immediately
    # or keep and flag. The requirement says "Suppression bougies incomplètes" in Phase 3.
    # However, 'apply' on empty slices produces NaNs.
    
    return df_m15

def clean_m15(df_m15):
    """
    Phase 3: Nettoyage M15
    - Suppression bougies incomplètes
    - Contrôle prix négatifs
    - Détection gaps anormaux
    """
    report = []
    report.append("--- Cleaning M15 Data ---")
    initial_count = len(df_m15)
    
    # 1. Suppression bougies incomplètes (NaNs resulting from resampling empty windows)
    # If any OHLC is NaN, it's incomplete.
    incomplete_mask = df_m15[['open', 'high', 'low', 'close']].isna().any(axis=1)
    num_incomplete = incomplete_mask.sum()
    
    if num_incomplete > 0:
        report.append(f"Removing {num_incomplete} incomplete M15 candles (caused by gaps in M1).")
        df_m15 = df_m15.dropna(subset=['open', 'high', 'low', 'close'])
    
    # 2. Contrôle prix négatifs (Should be covered by M1 check, but good to double check)
    bad_price = df_m15[(df_m15['open'] <= 0) | (df_m15['high'] <= 0) | (df_m15['low'] <= 0) | (df_m15['close'] <= 0)]
    if len(bad_price) > 0:
        report.append(f"WARNING: M15 Non-positive prices: {len(bad_price)}. Dropping.")
        df_m15 = df_m15[(df_m15['open'] > 0) & (df_m15['high'] > 0) & (df_m15['low'] > 0) & (df_m15['close'] > 0)]
        
    # 3. Détection gaps anormaux
    # "Anormal" is subjective. Let's flag gaps > 1 hour (4 * 15m) as "Large Gaps".
    if len(df_m15) > 1:
        time_diff = df_m15.index.to_series().diff()
        # Normal gap is 15 min.
        # Check for gaps significantly larger than 15 min.
        # Note: weekends will show as large gaps.
        large_gaps = time_diff[time_diff > pd.Timedelta(minutes=30)] # Missing at least one candle
        if len(large_gaps) > 0:
            report.append(f"M15 Discontinuities (> 30 min): {len(large_gaps)}")
            report.append(f"Largest M15 gap: {large_gaps.max()}")
    
    final_count = len(df_m15)
    report.append(f"Final M15 rows: {final_count}")
    
    return df_m15, "\n".join(report)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / 'data' / 'raw'
    processed_dir = root_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    years = [2022, 2023, 2024]
    
    full_report = []
    
    for year in years:
        # Phase 1
        df_m1, report_m1 = load_and_process_m1(year, raw_dir)
        full_report.append(report_m1)
        
        if df_m1 is not None and not df_m1.empty:
            # Phase 2
            df_m15 = aggregate_m1_to_m15(df_m1)
            
            # Phase 3
            df_m15_clean, report_m15 = clean_m15(df_m15)
            full_report.append(report_m15)
            
            # Save
            output_file = processed_dir / f"GBPUSD_M15_{year}.parquet"
            df_m15_clean.to_parquet(output_file)
            print(f"Saved {output_file} ({len(df_m15_clean)} rows)")
        
        full_report.append("\n" + "="*30 + "\n")

    # Write report
    report_path = processed_dir / "processing_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_report))
    print(f"Full report saved to {report_path}")

if __name__ == "__main__":
    main()
