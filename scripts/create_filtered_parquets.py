from pathlib import Path
import pandas as pd

# =============================================================================
# Config
# =============================================================================

DECODED_DIR = Path("data/decoded")
OUTPUT_PATH = Path("data/processed/highway_fsd_filtered.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Include only these signals
SELECTED_SIGNALS = [
    "GPSLatitude04F",
    "GPSLongitude04F",
    "Odometer3B6",
    "UI_isSunUp",
    "UI_minsToSunrise",
    "UI_minsToSunset",
    "DistanceTrip",
    "DI_gear",
    "Elevation3D8",
    "Altitude",
    "UI_Range",
    "UI_navRouteActive",
    "UI_roadSign",
    "UI_streetCount",
    "UI_mapSpeedLimit",
    "UI_mapSpeedLimitDependency",
    "UI_roadClass",
    "AccelerationX",
    "DI_accelPedalPos",
    "DI_brakePedal",
    "Speed",
    "IBST_sInputRodDriver",
    "AccelerationY",
    "SteeringAngle129",
    "SteeringSpeed129",
    "UI_csaRoadCurvC2",
    "UI_csaRoadCurvC3",
    "DAS_turnIndicatorRequest",
    "DAS_turnIndicatorRequestReason",
    "SCCM_turnIndicatorStalkStatus",
    "SCCM_leftStalkCrc",
    "DI_motorRPM",
    "UI_nextBranchDist",
    "UI_nextBranchRightOffRamp",
    "DAS_TP_vlWidth",
    "t",
]

# Optional: Ignore metadata/static info signals
IGNORED_PREFIXES = [
    "HVP_",
    "HIP_",
    "SOC",
    "BUILD_",
    "CAN_",
    "HW_",
    "TEST_",
    "APP_",
    "SYS_",
]

# =============================================================================
# Utility Functions
# =============================================================================


def filter_irrelevant_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop signals starting with ignored prefixes."""
    drop_cols = [
        col for col in df.columns if any(col.startswith(p) for p in IGNORED_PREFIXES)
    ]
    return df.drop(columns=drop_cols, errors="ignore")


# =============================================================================
# Main Processing
# =============================================================================


def create_filtered_parquet_efficient():
    parquet_files = list(DECODED_DIR.rglob("*.parquet"))
    if not parquet_files:
        print("❌ No .parquet files found.")
        return

    print(f"📁 Found {len(parquet_files)} files. Processing...")

    total_rows = 0
    first_write = True

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)

            # Normalize timestamp
            if "t" in df.columns and "timestamp" in df.columns:
                df.drop(columns=["t"], inplace=True)
            elif "t" in df.columns:
                df.rename(columns={"t": "timestamp"}, inplace=True)

            if "timestamp" not in df.columns:
                print(f"⚠️ Skipping {file.name}: no usable timestamp column")
                continue

            # Filter only selected signals + timestamp
            keep_cols = ["timestamp"] + [
                col
                for col in SELECTED_SIGNALS
                if col in df.columns and col != "timestamp"
            ]
            df = df[keep_cols].copy()

            # Drop static/irrelevant signal types
            df = filter_irrelevant_signals(df)
            df.dropna(axis=1, how="all", inplace=True)  # remove all-NaN cols

            if df.empty:
                print(f"⚠️ {file.relative_to(DECODED_DIR)}: no relevant data")
                continue

            # Append to final parquet file
            df.to_parquet(
                OUTPUT_PATH,
                index=False,
                compression="snappy",
                engine="pyarrow",
                mode="w" if first_write else "append",
            )
            first_write = False

            print(f"✅ Processed: {file.relative_to(DECODED_DIR)} → {len(df)} rows")
            total_rows += len(df)

        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")

    if total_rows > 0:
        print(f"\n🎯 Done! Total rows written: {total_rows:,}")
        print(f"📦 Output saved to: {OUTPUT_PATH.resolve()}")
    else:
        print("⚠️ No data was written. All files were empty or invalid.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    create_filtered_parquet_efficient()
