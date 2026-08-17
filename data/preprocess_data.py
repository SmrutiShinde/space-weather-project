"""
STEP 1: preprocess_data.py
===========================
Reads the real NASA OMNI-style space weather dataset:
    data/nasa_omni_space_weather.csv

Applies:
  - Data cleaning (missing values, outlier removal)
  - Risk label creation (scientific NOAA thresholds)
  - Feature engineering (storm severity, rolling avg, delta features etc.)
  - Saves processed CSV ready for ML training

Run this FIRST before train_models.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATASET_FILE = 'data/nasa_omni_space_weather.csv'


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data():
    print("=" * 55)
    print("  SPACE WEATHER DATA PREPROCESSING")
    print("=" * 55)

    if not os.path.exists(DATASET_FILE):
        print(f"\n❌ Dataset not found: {DATASET_FILE}")
        print("   Please place nasa_omni_space_weather.csv in the data/ folder.")
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    print(f"\n📂 Loading dataset: {DATASET_FILE}")
    df = pd.read_csv(DATASET_FILE, parse_dates=['timestamp'])
    print(f"   Raw shape    : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns      : {df.columns.tolist()}")
    print(f"   Date range   : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df):
    print("\n🧹 Cleaning data...")
    before = len(df)

    # Interpolate small gaps (up to 3 consecutive missing values)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=3)

    # Drop rows that still have NaN after interpolation
    df.dropna(inplace=True)

    # Remove physically impossible values
    df = df[df['solar_wind_speed'].between(200, 2000)]
    df = df[df['kp_index'].between(0, 9)]
    df = df[df['dst_index'].between(-600, 100)]
    df = df[df['proton_density'].between(0.01, 200)]

    df.reset_index(drop=True, inplace=True)
    print(f"   Rows removed : {before - len(df)}")
    print(f"   Clean shape  : {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 3. CREATE RISK LABELS
# ─────────────────────────────────────────────
def create_risk_labels(df):
    print("\n🏷️  Creating risk labels (NOAA scientific thresholds)...")

    def label_risk(row):
        # HIGH: G2+ storm (NOAA scale)
        if row['kp_index'] > 5 or row['dst_index'] < -100:
            return 'High'
        # MEDIUM: G1 storm
        elif 3 <= row['kp_index'] <= 5 or -100 <= row['dst_index'] <= -50:
            return 'Medium'
        # LOW: Quiet conditions
        else:
            return 'Low'

    df['risk_level'] = df.apply(label_risk, axis=1)

    dist = df['risk_level'].value_counts()
    total = len(df)
    print(f"   🟢 Low    : {dist.get('Low',0):>6,}  ({dist.get('Low',0)/total*100:.1f}%)")
    print(f"   🟡 Medium : {dist.get('Medium',0):>6,}  ({dist.get('Medium',0)/total*100:.1f}%)")
    print(f"   🔴 High   : {dist.get('High',0):>6,}  ({dist.get('High',0)/total*100:.1f}%)")
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    print("\n⚙️  Engineering features...")

    df = df.sort_values('timestamp').reset_index(drop=True)

    # ── Physics-based composite features ──────
    # Storm Severity Score (combines 3 key storm drivers)
    df['storm_severity'] = (
        0.4 * df['kp_index'] +
        0.3 * np.abs(df['bz']) +
        0.3 * (df['solar_wind_speed'] / 800)
    )

    # Magnetic field vector magnitude
    df['mag_disturbance'] = np.sqrt(
        df['bx']**2 + df['by']**2 + df['bz']**2
    )

    # Solar wind dynamic pressure (ram pressure)
    df['radiation_pressure'] = df['proton_density'] * df['solar_wind_speed']

    # ── Rolling averages (sustained exposure) ──
    df['kp_rolling_3h']   = df['kp_index'].rolling(window=3, min_periods=1).mean()
    df['bz_rolling_3h']   = df['bz'].rolling(window=3, min_periods=1).mean()
    df['wind_rolling_3h'] = df['solar_wind_speed'].rolling(window=3, min_periods=1).mean()
    df['dst_rolling_6h']  = df['dst_index'].rolling(window=6, min_periods=1).mean()

    # ── Sudden change features (rate of change) ──
    df['delta_bz']     = df['bz'].diff().fillna(0)
    df['delta_wind']   = df['solar_wind_speed'].diff().fillna(0)
    df['delta_proton'] = df['proton_density'].diff().fillna(0)
    df['delta_kp']     = df['kp_index'].diff().fillna(0)

    # ── Binary threshold flags ──────────────────
    df['bz_negative']  = (df['bz'] < 0).astype(int)
    df['extreme_kp']   = (df['kp_index'] > 7).astype(int)
    df['extreme_dst']  = (df['dst_index'] < -150).astype(int)
    df['high_speed']   = (df['solar_wind_speed'] > 600).astype(int)

    print(f"   Total features after engineering: {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 5. SAVE PROCESSED DATA
# ─────────────────────────────────────────────
def save_data(df):
    os.makedirs('data', exist_ok=True)

    # Save raw (with timestamp) for dashboard charts
    try:
        df.to_csv('data/space_weather_raw.csv', index=False)
        print("\n💾 Saved: data/space_weather_raw.csv")
    except PermissionError:
        print("\n❌ ERROR: Permission denied when saving data/space_weather_raw.csv!")
        print("   -> It seems the file is open in another program (e.g., Excel or a running Streamlit app).")
        print("   -> Please close it and try again.")
        return

    # Save processed (without timestamp) for ML training
    ml_cols = [c for c in df.columns if c != 'timestamp']
    try:
        df[ml_cols].to_csv('data/space_weather_processed.csv', index=False)
        print("💾 Saved: data/space_weather_processed.csv")
    except PermissionError:
        print("\n❌ ERROR: Permission denied when saving data/space_weather_processed.csv!")
        print("   -> It seems the file is open in another program (e.g., Excel or a running Streamlit app).")
        print("   -> Please close it and try again.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = create_risk_labels(df)
    df = engineer_features(df)
    save_data(df)

    print("\n" + "=" * 55)
    print(f"  ✅ PREPROCESSING DONE")
    print(f"  Dataset : {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Next    : python models/train_models.py")
    print("=" * 55)
