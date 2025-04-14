# scripts/plot_gps_map.py

import pandas as pd
import folium
from pathlib import Path
import argparse


def plot_gps_points(
    df, lat_col="GPSLatitude04F", lon_col="GPSLongitude04F", save_path="gps_map.html"
):
    # Get starting center point
    start_lat = df[lat_col].iloc[0]
    start_lon = df[lon_col].iloc[0]

    # Create folium map
    m = folium.Map(
        location=[start_lat, start_lon], zoom_start=13, tiles="OpenStreetMap"
    )

    # Plot each GPS coordinate as a tiny circle
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.5,
        ).add_to(m)

    # Save map to HTML
    m.save(save_path)
    print(f"🗺️ Map saved: {save_path}")


def main(input_path, output_path):
    print(f"📥 Loading: {input_path}")
    df = pd.read_parquet(input_path)

    # Drop null GPS rows just in case
    df = df.dropna(subset=["GPSLatitude04F", "GPSLongitude04F"])

    print("📍 Plotting GPS points...")
    plot_gps_points(df, save_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Decoded 1Hz Parquet file")
    parser.add_argument(
        "--outfile", default="outputs/gps_map.html", help="Output HTML map path"
    )
    args = parser.parse_args()

    main(Path(args.infile), Path(args.outfile))
