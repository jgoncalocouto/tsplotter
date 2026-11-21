"""Keep browser previews small without restricting dataset imports or exports."""
import pandas as pd


def read_parquet_checked(source):
    return pd.read_parquet(source)


def preview_frame(frame):
    return frame.iloc[:1000, :50]


def export_allowed(frame):
    return True
