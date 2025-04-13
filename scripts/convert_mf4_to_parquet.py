"""
decode_mf4_to_parquet.py

This script batch-decodes raw MF4 CAN log files into Parquet format using
a specified set of DBC files. Each MF4 file is decoded multiple times,
once per DBC variant (e.g., onyx, obd, model3), and the results are saved
in dedicated subfolders under `data/decoded/`.

Usage:
    Run this script directly to decode all MF4 files.
"""

import subprocess
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DECODER_EXE = ROOT_DIR / "mdf2parquet_decode.exe"

DBC_CONFIG = {
    "can1-onyx": ("--dbc-can1", ROOT_DIR / "dbc" / "can1-onyx.dbc"),
    "can1-obd": ("--dbc-can1", ROOT_DIR / "dbc" / "can1-obd.dbc"),
    "can1-model3": ("--dbc-can1", ROOT_DIR / "dbc" / "can1-model3.dbc"),
    "can1-opendbc": ("--dbc-can1", ROOT_DIR / "dbc" / "can1-opendbc.dbc"),
    "can1-zim": ("--dbc-can1", ROOT_DIR / "dbc" / "can1-zim.dbc"),
}

DBC_CAN9_PATH = ROOT_DIR / "dbc" / "can9-internal.dbc"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
DECODED_DATA_DIR = ROOT_DIR / "data" / "decoded"


# =============================================================================
# Decode Helpers
# =============================================================================


def decode_mf4_with_dbc(mf4_file: Path, dbc_name: str, dbc_flag: str, dbc_path: Path):
    """
    Decode a single MF4 file using a specific DBC and write to a DBC-specific folder.
    """
    output_dir = DECODED_DATA_DIR / dbc_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(DECODER_EXE),
        f"{dbc_flag}={dbc_path}",
        f"--dbc-can9={DBC_CAN9_PATH}",
        "-O",
        str(output_dir),
        "-i",
        str(mf4_file),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Decoded {mf4_file.name} → {dbc_name}")
    except subprocess.CalledProcessError as err:
        print(
            f"❌ Error decoding {mf4_file.name} with {dbc_name}: exit code {err.returncode}"
        )


def decode_all_mf4_files():
    """
    Loop through all .MF4 files in the raw directory and decode them using all DBCs.
    """
    if not DECODER_EXE.exists():
        print(f"❌ Decoder not found: {DECODER_EXE}")
        return

    mf4_files = sorted(RAW_DATA_DIR.glob("*.MF4"))
    if not mf4_files:
        print(f"⚠️ No MF4 files found in {RAW_DATA_DIR}")
        return

    for mf4_file in mf4_files:
        for dbc_name, (dbc_flag, dbc_path) in DBC_CONFIG.items():
            if dbc_path.exists():
                decode_mf4_with_dbc(mf4_file, dbc_name, dbc_flag, dbc_path)
            else:
                print(f"⚠️ Missing DBC file for {dbc_name}: {dbc_path}")


if __name__ == "__main__":
    decode_all_mf4_files()
