# This script scans decoded .parquet CAN signal files, extracts signal metadata,
# filters out unwanted signals, and appends both value labels (from VAL_) and units (from SG_)
# defined in DBC files to support deeper interpretation and benchmark analytics.

import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

# =============================================================================
# Config
# =============================================================================

PARQUET_ROOT = Path("data/decoded")  # Path to the decoded .parquet files
OUTPUT_DIR = Path("outputs")  # Output directory for saving results
OUTPUT_CSV = OUTPUT_DIR / "signals_summary.csv"
TOP_N_PREVIEW = 20  # Number of preview rows to print
MIN_COUNT_THRESHOLD = 0  # Filter threshold for minimum non-null values

DBC_DIR = Path("dbc")
DBC_FILES = list(DBC_DIR.glob("can1-*.dbc"))  # List of all relevant DBC files

# Keywords that identify unwanted signals (case-insensitive substring match)
EXCLUDED_KEYWORDS = [
    "ESP",
    "PCS",
    "VCFRONT",
    "VCRIGHT",
    "CP",
    "HVP",
    "VCLEFT",
    "VCSEC",
    "CMPD",
    "BMS",
    "BRICK",
    "BATT",
    "TEMP",
    "EPBL",
    "EPBR",
]

# =============================================================================
# Extract VAL_ label mappings and signal units from DBC files
# =============================================================================


def extract_val_and_unit_mappings(dbc_files):
    val_lookup = defaultdict(dict)
    unit_lookup = {}

    for dbc_file in dbc_files:
        try:
            with open(dbc_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Extract VAL_ entries
                    if line.startswith("VAL_"):
                        match = re.match(r"VAL_ \d+ (\w+)\s+(.*);", line)
                        if match:
                            signal_name, entries = match.groups()
                            labels = re.findall(r'(\d+)\s+"([^\"]+)"', entries)
                            val_lookup[signal_name].update(
                                {int(k): v for k, v in labels}
                            )

                    # Extract SG_ unit definitions
                    elif line.startswith("SG_"):
                        match = re.match(
                            r"SG_ (\w+)\s*:\s*\d+\|\d+@[01][+-] \([^)]*\) \[[^]]*\] \"([^\"]*)\"",
                            line,
                        )
                        if match:
                            signal_name, unit = match.groups()
                            unit_lookup[signal_name] = unit
        except Exception as e:
            print(f"⚠️ Failed to read {dbc_file.name}: {e}")

    return val_lookup, unit_lookup


val_lookup, unit_lookup = extract_val_and_unit_mappings(DBC_FILES)

# =============================================================================
# Aggregate Signal Statistics from .parquet files
# =============================================================================

signal_data = defaultdict(
    lambda: {"non_null_value_count": 0, "unique_values": set(), "dbc_sources": set()}
)

parquet_files = list(PARQUET_ROOT.rglob("*.parquet"))
print(f"🔍 Found {len(parquet_files)} .parquet file(s).\n")

for file in parquet_files:
    try:
        df = pd.read_parquet(file)
        dbc_source = file.relative_to(PARQUET_ROOT).parts[0]

        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in EXCLUDED_KEYWORDS):
                continue
            series = df[col].dropna()
            if not series.empty:
                signal_data[col]["non_null_value_count"] += len(series)
                signal_data[col]["unique_values"].update(series.unique())
                signal_data[col]["dbc_sources"].add(dbc_source)

    except Exception as e:
        print(f"⚠️ Error reading {file.name}: {e}")

# =============================================================================
# Build Final Output Table
# =============================================================================

rows = []

for signal, data in signal_data.items():
    total = data["non_null_value_count"]
    unique_values = data["unique_values"]
    unique_count = len(unique_values)

    labels = val_lookup.get(signal, {})
    unit = unit_lookup.get(signal, "")

    label_summary = (
        ", ".join([f"{k}={v}" for k, v in sorted(labels.items())]) if labels else ""
    )

    if total >= MIN_COUNT_THRESHOLD and unique_count > 1:
        rows.append(
            {
                "signal": signal,
                "non_null_value_count": total,
                "unique_non_null_values_count": unique_count,
                "dbc_source": ", ".join(sorted(data["dbc_sources"])),
                "value_labels": label_summary,
                "unit": unit,
            }
        )

# =============================================================================
# Export Final CSV
# =============================================================================

df_signals = pd.DataFrame(rows).sort_values(by="non_null_value_count", ascending=False)

print(
    f"✅ Found {len(df_signals)} valid signals (excluding keywords) with ≥ {MIN_COUNT_THRESHOLD} non-null values and >1 unique value.\n"
)
print(df_signals.head(TOP_N_PREVIEW))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_signals.to_csv(OUTPUT_CSV, index=False)
print(f"📁 Saved to: {OUTPUT_CSV.resolve()}")
