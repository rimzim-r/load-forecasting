"""Data loading, quality checks, DST deduplication, hourly interpolation, and EDA plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "PJME_hourly.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "pjme_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"


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


def plot_eda_figures(df: pd.DataFrame, fig_dir: Path = FIG_DIR) -> None:
    """Generate and save overview and seasonality figures (01 and 02)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    df["load_mw"].plot(ax=ax[0], lw=0.4)
    ax[0].set_title("PJME hourly load, full history")
    ax[0].set_ylabel("MW")

    df.loc["2017-01-01":"2017-01-14", "load_mw"].plot(ax=ax[1])
    ax[1].set_title("Two weeks in detail (daily and weekly pattern)")
    ax[1].set_ylabel("MW")
    plt.tight_layout()
    plt.savefig(fig_dir / "01_load_overview.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4))
    sns.lineplot(data=df, x="hour", y="load_mw", hue="is_weekend", ax=ax2[0])
    ax2[0].set_title("Average load by hour: weekday vs weekend")
    sns.boxplot(data=df, x="dayofweek", y="load_mw", ax=ax2[1])
    ax2[1].set_title("Load by day of week (0 = Monday)")
    sns.boxplot(data=df, x="month", y="load_mw", ax=ax2[2])
    ax2[2].set_title("Load by month")
    plt.tight_layout()
    plt.savefig(fig_dir / "02_seasonality.png", dpi=150)
    plt.close(fig2)


def main(raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH, save_plots: bool = True) -> pd.DataFrame:
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

    if save_plots:
        plot_eda_figures(df)

    return df


if __name__ == "__main__":
    main()
