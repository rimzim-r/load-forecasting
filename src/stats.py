"""Stationarity tests (ADF, KPSS), MSTL decomposition, ACF/PACF, and hypothesis tests."""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss

from src.data import CLEAN_PATH, ROOT
from src.features import HOLIDAY_DATES

warnings.filterwarnings("ignore")
FIG_DIR = ROOT / "reports" / "figures"


def run_stationarity_tests(series: pd.Series, name: str) -> dict[str, float]:
    """Run ADF and KPSS tests on a series."""
    clean = series.dropna()
    adf_stat, adf_p = adfuller(clean, maxlag=48, autolag="AIC")[:2]
    kpss_stat, kpss_p = kpss(clean, regression="c", nlags="auto")[:2]
    print(f"{name:28s} ADF p = {adf_p:.4f} | KPSS p = {kpss_p:.4f}")
    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
    }


def run_mstl_decomposition(series: pd.Series, fig_dir: Path = FIG_DIR) -> pd.Series:
    """Fit MSTL (periods 24 and 168) on 2016 sample and return share of variance explained."""
    sample = series["2016-01-01":"2016-12-31"]
    res = MSTL(sample, periods=[24, 168]).fit()

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = res.plot()
    fig.set_size_inches(13, 9)
    plt.tight_layout()
    plt.savefig(fig_dir / "03_mstl_decomposition.png", dpi=150)
    plt.close(fig)

    var_share = pd.Series({
        "trend": res.trend.var(),
        "daily seasonal": res.seasonal["seasonal_24"].var(),
        "weekly seasonal": res.seasonal["seasonal_168"].var(),
        "residual": res.resid.var(),
    }) / sample.var()
    print("\nVariance share (2016 MSTL):")
    print(var_share.round(4))
    return var_share


def plot_acf_pacf(series: pd.Series, fig_dir: Path = FIG_DIR) -> None:
    """Plot and save 8-day ACF and 48-lag PACF."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    plot_acf(series.dropna(), lags=24 * 8, ax=ax[0])
    ax[0].set_title("ACF (8 days of lags)")
    plot_pacf(series.dropna(), lags=48, method="ywm", ax=ax[1])
    ax[1].set_title("PACF (48 lags)")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_acf_pacf.png", dpi=150)
    plt.close(fig)


def compare_groups(a: pd.Series, b: pd.Series, label: str) -> dict[str, float]:
    """Run two-sided Mann-Whitney U test and compute Cohen's d."""
    _, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    cohen_d = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    print(label)
    print(f"  mean A = {a.mean():,.0f} MW | mean B = {b.mean():,.0f} MW")
    print(f"  Mann-Whitney p = {p_val:.2e} | Cohen's d = {cohen_d:.2f}\n")
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "p_value": float(p_val),
        "cohen_d": cohen_d,
    }


def main(clean_path: Path = CLEAN_PATH) -> None:
    df = pd.read_csv(clean_path, index_col="Datetime", parse_dates=True)
    y = df["load_mw"].asfreq("h")

    run_stationarity_tests(y, "Original series")
    run_stationarity_tests(y.diff(1), "Differenced (lag 1)")
    run_stationarity_tests(y.diff(24), "Seasonal diff (lag 24)")
    run_stationarity_tests(y.diff(168), "Seasonal diff (lag 168)")

    run_mstl_decomposition(y)
    plot_acf_pacf(y)
    print()

    daily = y.resample("D").mean().to_frame("load")
    daily["is_weekend"] = daily.index.dayofweek >= 5
    daily["is_holiday"] = daily.index.isin(HOLIDAY_DATES)

    weekday = daily.loc[~daily["is_weekend"], "load"]
    weekend = daily.loc[daily["is_weekend"], "load"]
    compare_groups(weekday, weekend, "Weekday (A) vs weekend (B)")

    wd = daily[~daily["is_weekend"]]
    compare_groups(
        wd.loc[~wd["is_holiday"], "load"],
        wd.loc[wd["is_holiday"], "load"],
        "Normal weekday (A) vs weekday holiday (B)",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=daily, x="is_weekend", y="load", ax=ax[0])
    ax[0].set_title("Daily mean load: weekday vs weekend")
    sns.boxplot(data=wd, x="is_holiday", y="load", ax=ax[1])
    ax[1].set_title("Weekdays: normal vs holiday")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_group_comparisons.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
