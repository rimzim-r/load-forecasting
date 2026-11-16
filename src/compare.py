"""Model comparison, equal-weight ensemble, Diebold-Mariano HAC tests, and demo exports."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from src.data import ROOT
from src.features import metrics

PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

MODELS = ["naive_24", "naive_168", "lightgbm", "lightgbm_v2", "lstm", "ensemble"]


def dm_test(common: pd.DataFrame, col_a: str, col_b: str) -> tuple[float, float]:
    """Diebold-Mariano style test on daily mean absolute errors (HAC standard errors, maxlags=7)."""
    ea = (common["actual"] - common[col_a]).abs().resample("D").mean()
    eb = (common["actual"] - common[col_b]).abs().resample("D").mean()
    d = (ea - eb).dropna()
    fit = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return float(d.mean()), float(fit.pvalues[0])


def main(processed_dir: Path = PROCESSED_DIR, reports_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    preds = pd.read_csv(processed_dir / "test_predictions.csv", index_col=0, parse_dates=True)
    lstm = pd.read_csv(processed_dir / "lstm_predictions.csv", index_col=0, parse_dates=True).iloc[:, 0]

    common = preds.loc[lstm.index].copy()
    common["lstm"] = lstm
    common["ensemble"] = (common["lightgbm_v2"] + common["lstm"]) / 2

    reports_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame({m: metrics(common["actual"], common[m]) for m in MODELS}).T
    table = table.sort_values("MAE").round(2)
    print(table)
    table.to_csv(reports_dir / "results.csv")

    pairs = [
        ("naive_24", "lightgbm_v2"),
        ("lightgbm", "lightgbm_v2"),
        ("lightgbm_v2", "lstm"),
        ("lightgbm_v2", "ensemble"),
        ("lstm", "ensemble"),
    ]
    rows = []
    for a, b in pairs:
        diff, p = dm_test(common, a, b)
        rows.append({"A": a, "B": b, "mean daily MAE A - B (MW)": round(diff, 1), "p-value": round(p, 4)})
    dm_table = pd.DataFrame(rows)
    print("\nDiebold-Mariano HAC tests:")
    print(dm_table)
    dm_table.to_csv(reports_dir / "dm_tests.csv", index=False)

    sarima_path = processed_dir / "sarima_predictions.csv"
    if sarima_path.exists():
        sarima = pd.read_csv(sarima_path, index_col=0, parse_dates=True).iloc[:, 0]
        sub = common.loc[sarima.index].copy()
        sub["sarima"] = sarima
        cols = ["naive_24", "sarima", "lightgbm_v2", "lstm", "ensemble"]
        sarima_table = pd.DataFrame({m: metrics(sub["actual"], sub[m]) for m in cols}).T.sort_values("MAE").round(2)
        print(f"\nFirst {sub.index.normalize().nunique()} test days:")
        print(sarima_table)
        sarima_table.to_csv(reports_dir / "results_first30days.csv")

    common[["actual", "naive_24", "lightgbm_v2", "lstm", "ensemble"]].round(0).to_csv(
        reports_dir / "demo_predictions.csv"
    )

    sns.set_theme(style="whitegrid")
    err_df = pd.DataFrame({
        m: (common["actual"] - common[m]).abs()
        for m in ["naive_24", "lightgbm_v2", "lstm", "ensemble"]
    })
    fig, ax = plt.subplots(figsize=(12, 4))
    err_df.groupby(err_df.index.hour).mean().plot(ax=ax, lw=1.8)
    ax.set_title("Mean absolute error by hour of day (MW)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE (MW)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_error_by_hour.png", dpi=150)
    plt.close(fig)

    week = common["2018-02-05":"2018-02-11"]
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    ax2.plot(week["actual"], label="Actual", lw=2)
    ax2.plot(week["lightgbm_v2"], label="LightGBM", lw=1.2)
    ax2.plot(week["lstm"], label="LSTM", lw=1.2)
    ax2.plot(week["ensemble"], label="Ensemble", lw=1.6)
    ax2.set_title("One test week: actual vs forecasts (MW)")
    ax2.set_ylabel("MW")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_final_week.png", dpi=150)
    plt.close(fig2)

    return table


if __name__ == "__main__":
    main()
