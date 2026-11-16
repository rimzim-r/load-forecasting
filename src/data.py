"""Data loading, quality checks, DST deduplication, and hourly interpolation."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "PJME_hourly.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "pjme_clean.csv"


def clean_load_series(df: pd.DataFrame) -> pd.DataFrame:
    """Sort timestamps, average DST duplicates, interpolate missing hours, and add calendar columns."""
    out = df.sort_values("Datetime").set_index("Datetime").copy()
    out.columns = ["load_mw"]

    full_range = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.groupby(out.index).mean()
    out = out.reindex(full_range)
    out.index.name = "Datetime"
    out["load_mw"] = out["load_mw"].interpolate(method="time")

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["year"] = out.index.year
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def find_iqr_outliers(df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Return extreme load hours beyond Q1 - k*IQR or Q3 + k*IQR."""
    q1, q3 = df["load_mw"].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df["load_mw"] < q1 - multiplier * iqr) | (df["load_mw"] > q3 + multiplier * iqr)]


def main(raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, parse_dates=["Datetime"])
    raw_sorted = raw.sort_values("Datetime").set_index("Datetime")
    full_range = pd.date_range(raw_sorted.index.min(), raw_sorted.index.max(), freq="h")

    print(f"Raw shape: {raw_sorted.shape} | {raw_sorted.index.min()} -> {raw_sorted.index.max()}")
    print(f"Duplicate timestamps: {int(raw_sorted.index.duplicated().sum())}")
    print(f"Missing hours: {len(full_range.difference(raw_sorted.index))}")

    df = clean_load_series(raw)
    outliers = find_iqr_outliers(df)
    print(f"Rows after cleaning: {len(df)} | Missing values: {int(df['load_mw'].isna().sum())}")
    print(f"Extreme outliers (kept as genuine heat-wave peaks): {len(outliers)}")

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path)
    print(f"Saved cleaned series to {clean_path}")
    return df


if __name__ == "__main__":
    main()
