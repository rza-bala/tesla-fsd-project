import os
import pandas as pd
import re
from pathlib import Path

# === Configuration ===
DECODED_ROOT = "/home/ad.adasworks.com/roy.zabala/Projects/tesla-fsd-project"
OUTPUT_CSV = os.path.join(DECODED_ROOT, "relevant_signals.csv")

VALID_PREFIXES = ("CAN1_", "CAN9_")

EXCLUDE_KEYWORDS = {
    "dcdc", "frontseat", "matrix", "charge", "counter", "heat", "volt",
    "brick", "vcfront", "oil", "temp", "hvac", "batt", "thermal", "kwh",
    "energy", "debug", "checksum", "multi", "power", "crc", "soc", "cmpd", "epb"
}

MIN_UNIQUE_VALUES = 4  # Filter low-variance signals

# === Helper Functions ===

def has_excluded_keyword(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in EXCLUDE_KEYWORDS)

def parse_can_metadata(path_parts):
    for part in path_parts:
        if part.startswith(VALID_PREFIXES):
            match = re.match(r"(CAN\d+)_(?:ID(\d{5}))?(.+)", part)
            if match:
                channel, can_id, can_name = match.groups()
                can_id = can_id if can_id else ""
                can_name = can_name.strip("_")
                return channel, can_id, can_name
    return None, None, None

def process_parquet_file(file_path):
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"⚠️ Skipped unreadable file {file_path}: {e}")
        return []

    signals = []
    for col in df.columns:
        signal_clean = col.strip()

        if signal_clean.lower() in {"t", "timestamp"}:
            continue
        if has_excluded_keyword(signal_clean):
            continue

        unique_count = df[col].nunique(dropna=True)
        if unique_count < MIN_UNIQUE_VALUES:
            continue

        signals.append((signal_clean, unique_count))

    return signals

# === Main Execution ===

def main():
    decoded_path = Path(DECODED_ROOT)
    signal_records = []

    for file_path in decoded_path.rglob("*.parquet"):
        path_parts = file_path.parts
        CAN_Channel, CAN_ID, CAN_Name = parse_can_metadata(path_parts)

        if not CAN_Channel or not CAN_Name:
            print(f"⚠️ Could not parse CAN metadata from path: {file_path}")
            continue

        if has_excluded_keyword(CAN_Name):
            continue

        signals = process_parquet_file(file_path)
        for signal, unique_count in signals:
            signal_records.append({
                "CAN_Channel": CAN_Channel,
                "CAN_ID": CAN_ID,
                "CAN_Name": CAN_Name,
                "Signal": signal,
                "Unique_Values": unique_count
            })

    if not signal_records:
        print("⚠️ No relevant signals found!")
        return

    df_signals = (
        pd.DataFrame(signal_records)
        .groupby(["CAN_Channel", "CAN_ID", "CAN_Name", "Signal"], as_index=False)
        .agg({"Unique_Values": "sum"})
        .sort_values(by="Unique_Values", ascending=False)
    )

    df_signals.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(df_signals)} signal entries to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
