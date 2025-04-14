from pathlib import Path
import pandas as pd
import argparse

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_INPUT = Path("data/lake/fsd_signals.parquet")
DEFAULT_OUTPUT = Path("data/lake/fsd_signals_1hz.parquet")
DEFAULT_RATE = "1S"

# Selected signals for LKA / ACC / LCM / EXITS / MERGES
SELECTED_SIGNALS = [
    # GENERAL
    "UTCday318",
    "UTChour318",
    "UTCminutes318",
    "UTCseconds318",
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
    # PLANNING
    "UI_navRouteActive",
    "UI_roadSign",
    "UI_streetCount",
    "UI_mapSpeedLimit",
    "UI_mapSpeedLimitDependency",
    "UI_roadClass",
    # LONGITUDINAL
    "AccelerationX",
    "DI_accelPedalPos",
    "DI_brakePedal",
    "Speed",
    "IBST_sInputRodDriver",
    # LATERAL
    "AccelerationY",
    "SteeringAngle129",
    "SteeringSpeed129",
    "UI_csaRoadCurvC2",
    "UI_csaRoadCurvC3",
    # LCM
    "DAS_turnIndicatorRequest",
    "DAS_turnIndicatorRequestReason",
    "SCCM_turnIndicatorStalkStatus",
    "SCCM_leftStalkCrc",
    "DI_motorRPM",
    # EXITS / MERGES
    "UI_nextBranchDist",
    "UI_nextBranchRightOffRamp",
    "DAS_TP_vlWidth",
    # TIME
    "t",
]

# =============================================================================
# UTILS
# =============================================================================


def infer_aggregation_rules(df: pd.DataFrame) -> dict:
    agg = {}
    for col in df.columns:
        if col == "t":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "mean"
        else:
            agg[col] = "last"
    return agg


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def downsample_signals(input_path: Path, output_path: Path, rate: str):
    print(f"📥 Reading: {input_path}")
    df = pd.read_parquet(input_path)

    if "t" not in df.columns:
        raise ValueError("Missing required column 't' (timestamp).")

    print("⏱️ Converting 't' to datetime...")
    df["t"] = pd.to_datetime(df["t"], unit="s", origin="unix", errors="coerce")

    print("🔍 Filtering selected signals...")
    filtered = [col for col in SELECTED_SIGNALS if col in df.columns]
    df = df[filtered].set_index("t")

    print("📊 Inferring aggregation rules...")
    agg_rules = infer_aggregation_rules(df)

    print(f"📉 Downsampling to: {rate}")
    df_downsampled = df.resample(rate).agg(agg_rules).dropna(how="all")

    df_downsampled.reset_index(inplace=True)
    df_downsampled = df_downsampled.rename(columns={"t": "time"})
    df_downsampled["time"] = df_downsampled["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"💾 Saving to: {output_path}")
    df_downsampled.to_parquet(output_path, index=False, compression="snappy")

    print(
        f"✅ Done. Rows: {len(df_downsampled):,} | Columns: {len(df_downsampled.columns)}"
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample Tesla FSD signal logs to 1Hz."
    )
    parser.add_argument(
        "--infile", type=str, default=str(DEFAULT_INPUT), help="Input Parquet file path"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--rate", type=str, default=DEFAULT_RATE, help="Resample rate (e.g. 1S, 5S)"
    )
    args = parser.parse_args()

    downsample_signals(Path(args.infile), Path(args.outfile), args.rate)
