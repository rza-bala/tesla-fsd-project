# decode.py - Decode all MF4 logs using a DBC file

import cantools
import pandas as pd
from asammdf import MDF
from pathlib import Path
import argparse

def decode_single_mf4(mf4_path, dbc_path, output_dir):
    db = cantools.database.load_file(dbc_path)
    mdf = MDF(mf4_path)
    decoded_rows = []

    for channel in mdf:
        if not hasattr(channel, "name") or "CAN_DataFrame" not in channel.name:
            continue

        channel_data = mdf.get(channel.name)
        samples = channel_data.samples

        for sample in samples:
            try:
                arbitration_id = sample[0][0]
                data_bytes = sample[0][1]
                timestamp = sample[1]

                if isinstance(data_bytes, list):
                    data = bytes(data_bytes)
                else:
                    data = data_bytes

                message = db.get_message_by_frame_id(arbitration_id)
                decoded = message.decode(data)

                decoded['timestamp'] = timestamp
                decoded['message'] = message.name
                decoded['can_id'] = hex(arbitration_id)

                decoded_rows.append(decoded)
            except Exception:
                continue

    if decoded_rows:
        df = pd.DataFrame(decoded_rows)
        out_file = Path(output_dir) / (Path(mf4_path).stem + ".csv")
        df.to_csv(out_file, index=False)
        print(f"\u2705 Decoded {mf4_path} → {out_file}")
    else:
        print(f"\u26A0\uFE0F No decodable messages found in {mf4_path}")

def decode_all_mf4s(folder, dbc_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for mf4_file in sorted(Path(folder).glob("*.MF4")):
        decode_single_mf4(mf4_file, dbc_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbc", required=True, help="Path to DBC file")
    parser.add_argument("--mf4_dir", required=True, help="Directory with .MF4 files")
    parser.add_argument("--out_dir", default="processed_data", help="Directory for decoded CSVs")
    args = parser.parse_args()

    decode_all_mf4s(args.mf4_dir, args.dbc, args.out_dir)
