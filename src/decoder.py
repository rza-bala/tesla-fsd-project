# utils/decoder.py
import pandas as pd
import ast

def load_signal_metadata(path):
    """Load signal metadata CSV and parse label dictionaries."""
    return pd.read_csv(path, converters={"value_labels": ast.literal_eval})

def decode_signal_column(df, signal_name, label_dict, suffix="_label"):
    """Decode one signal using a label dictionary."""
    if signal_name in df.columns and isinstance(label_dict, dict):
        df[signal_name + suffix] = df[signal_name].map(label_dict)
    return df

def decode_all_signals(df, metadata_df, suffix="_label"):
    """Decode all enum signals from the metadata."""
    for _, row in metadata_df.iterrows():
        signal = row["signal_name"]
        labels = row["value_labels"]
        if isinstance(labels, dict):
            df = decode_signal_column(df, signal, labels, suffix)
    return df
