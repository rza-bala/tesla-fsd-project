from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DECODED_DIR = Path("data/decoded")
OUTPUT_PATH = Path("data/lake/fsd_signals.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "t" in df.columns:
        return df
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "t"})
    return df if "t" in df.columns else None


def extract_signals(df: pd.DataFrame) -> pd.DataFrame:
    return df[[col for col in SELECTED_SIGNALS if col in df.columns]]


def load_and_filter(file: Path) -> pd.DataFrame:
    df = pd.read_parquet(file)
    df = normalize_time_column(df)
    if df is None:
        return None
    df = extract_signals(df)
    df = df.dropna(axis=1, how="all")
    return df if not df.empty else None


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def build_database():
    files = list(DECODED_DIR.rglob("*.parquet"))
    print(f"📁 Found {len(files)} files.")
    total_rows = 0
    all_data = []

    for file in files:
        try:
            df = load_and_filter(file)
            if df is not None:
                all_data.append(df)
                total_rows += len(df)
                print(f"✅ {file.name}: {len(df)} rows")
            else:
                print(f"⚠️ Skipped {file.name}: No usable signals")
        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")

    if not all_data:
        print("⚠️ No data written.")
        return

    final = pd.concat(all_data, ignore_index=True)
    final = final.sort_values("t")  # Optional: time ordering
    final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow", compression="snappy")

    print(f"\n✅ Saved {len(final):,} rows to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    build_database()
