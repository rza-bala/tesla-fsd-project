#!/usr/bin/env python3
import subprocess
from pathlib import Path

def decode_mf4_to_parquet(mf4_dir, output_dir, tool="./tools/mdf2parquet_decode"):
    dbc_can1 = "dbc/can1-zhim-master.dbc"
    dbc_can9 = "dbc/can9-internal.dbc"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all MF4 files in the given directory
    mf4_files = sorted(Path(mf4_dir).glob("*.MF4"))

    if not mf4_files:
        print(f"❌ No MF4 files found in '{mf4_dir}'. Check the directory.")
        return

    # Iterate over each MF4 file and decode to Parquet
    for mf4 in mf4_files:
        cmd = [
            tool,
            f"--dbc-can1={dbc_can1}",
            f"--dbc-can9={dbc_can9}",
            "--input", str(mf4),
            "--output", str(output_dir)  # ✅ Provide only directory path
        ]
        print(f"📦 Decoding '{mf4.name}' to Parquet...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Successfully decoded '{mf4.name}'!")
        else:
            print(f"⚠️ Error decoding '{mf4.name}':")
            print(result.stderr)

if __name__ == "__main__":
    decode_mf4_to_parquet(
        mf4_dir="raw_data/2025-03-08",
        output_dir="processed_data"
    )
