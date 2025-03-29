import subprocess
from pathlib import Path

def decode_mf4_to_parquet(mf4_dir, output_dir, tool="./tools/mdf2parquet_decode"):
    dbc1 = "dbc/can1-obd.dbc"
    dbc9 = "dbc/can9-internal.dbc"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mf4_files = sorted(Path(mf4_dir).glob("*.MF4"))

    for mf4 in mf4_files:
        cmd = [
            tool,
            f"--dbc-can1={dbc1}",
            f"--dbc-can9={dbc9}",
            "--input", str(mf4),
            "--output", str(output_dir)
        ]
        print(f"📦 Decoding {mf4.name} to Parquet...")
        subprocess.run(cmd)

if __name__ == "__main__":
    decode_mf4_to_parquet(
        mf4_dir="raw_data/EEEE0005/00000001",
        output_dir="processed_data"
    )
