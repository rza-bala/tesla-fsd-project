# scripts/decode_signals.py

import pandas as pd
import argparse
import ast
from pathlib import Path


def safe_eval(val):
    """Safely parse string representations of dicts."""
    try:
        return ast.literal_eval(val)
    except Exception:
        return None


def load_signal_metadata(path):
    """Load the signal metadata registry and parse value_labels correctly."""
    return pd.read_csv(path, converters={"value_labels": safe_eval})


def decode_signal_column(df, signal_name, label_dict, suffix="_label"):
    """Attach a decoded label column using the provided label dictionary."""
    if signal_name in df.columns and isinstance(label_dict, dict):
        df[signal_name + suffix] = df[signal_name].map(label_dict)
    return df


def decode_all_signals(df, metadata_df, suffix="_label"):
    """
    Decode enum signals only if:
    - The signal exists in the dataframe
    - The metadata value_labels is a dictionary
    - At least some keys in the dictionary match actual values in the dataframe
    """
    for _, row in metadata_df.iterrows():
        signal = row["signal_name"]
        labels = row["value_labels"]

        if isinstance(labels, dict) and signal in df.columns:
            unique_vals = df[signal].dropna().unique()
            if any(val in labels for val in unique_vals):
                df = decode_signal_column(df, signal, labels, suffix)

    return df


def main(input_path, metadata_path, output_path):
    print(f"📥 Reading input: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"📘 Loading metadata: {metadata_path}")
    metadata = load_signal_metadata(metadata_path)

    print("🔄 Decoding enum signals...")
    df = decode_all_signals(df, metadata)

    print(f"💾 Saving output: {output_path}")
    df.to_parquet(output_path, index=False)
    print("✅ Done! Decoded file saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Input Parquet file path")
    parser.add_argument("--metadata", required=True, help="Signal metadata CSV path")
    parser.add_argument("--outfile", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    main(Path(args.infile), Path(args.metadata), Path(args.outfile))
