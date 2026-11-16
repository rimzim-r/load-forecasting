"""Naive baselines, LightGBM (standard & midnight-anchored), and 30-day rolling SARIMA."""
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data import CLEAN_PATH, ROOT
from src.features import (
    TEST_START,
    TRAIN_END,
    VAL_END,
    add_anchored_features,
    make_features,
    metrics,
)

warnings.filterwarnings("ignore")

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

LGB_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


def fit_and_refit_lgb(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMRegressor:
    """Tune tree count with early stopping on 2017 validation, then refit on 2002-2017."""
    X_tr, y_tr = train_df[feature_cols], train_df["load_mw"]
    X_va, y_va = val_df[feature_cols], val_df["load_mw"]

    tuner = lgb.LGBMRegressor(n_estimators=3000, **LGB_PARAMS)
    tuner.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    best_n = tuner.best_iteration_
    print(f"Best number of trees ({len(feature_cols)} features): {best_n}")

    final = lgb.LGBMRegressor(n_estimators=best_n, **LGB_PARAMS)
    final.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    return final


def train_lgb_models(
    series: pd.Series, models_dir: Path = MODELS_DIR
) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, list[str]]:
    """Train LightGBM v1 (>=24h lags) and v2 (midnight-anchored) and save boosters to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)

    feat = make_features(series)
    features_v1 = [c for c in feat.columns if c != "load_mw"]
    tr1, va1 = feat[:TRAIN_END], feat["2017-01-01":VAL_END]
    final_v1 = fit_and_refit_lgb(tr1, va1, features_v1)
    final_v1.booster_.save_model(str(models_dir / "lightgbm.txt"))

    feat2 = add_anchored_features(feat, series)
    features_v2 = [c for c in feat2.columns if c != "load_mw"]
    tr2, va2 = feat2[:TRAIN_END], feat2["2017-01-01":VAL_END]
    final_v2 = fit_and_refit_lgb(tr2, va2, features_v2)
    final_v2.booster_.save_model(str(models_dir / "lightgbm_v2.txt"))

    return final_v1, final_v2, features_v2


def predict_baselines_and_lgb(
    series: pd.Series,
    models_dir: Path = MODELS_DIR,
    start: str = TEST_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Generate naive and LightGBM (v1 & v2) predictions from saved booster files."""
    y_slice = series[start:end] if end else series[start:]
    preds = pd.DataFrame({"actual": y_slice})
    preds["naive_24"] = series.shift(24).loc[y_slice.index]
    preds["naive_168"] = series.shift(168).loc[y_slice.index]

    feat = make_features(series)
    features_v1 = [c for c in feat.columns if c != "load_mw"]
    te1 = feat.loc[feat.index.intersection(y_slice.index)]
    booster_v1 = lgb.Booster(model_file=str(models_dir / "lightgbm.txt"))
    preds["lightgbm"] = pd.Series(booster_v1.predict(te1[features_v1]), index=te1.index)

    feat2 = add_anchored_features(feat, series)
    features_v2 = [c for c in feat2.columns if c != "load_mw"]
    te2 = feat2.loc[feat2.index.intersection(y_slice.index)]
    booster_v2 = lgb.Booster(model_file=str(models_dir / "lightgbm_v2.txt"))
    preds["lightgbm_v2"] = pd.Series(booster_v2.predict(te2[features_v2]), index=te2.index)

    return preds


def forecast_sarima(series: pd.Series, n_days: int = 30) -> pd.Series:
    """Fit SARIMA(1,0,1)(1,1,1,24) on 28 pre-test days and roll forward daily without refitting."""
    fit_window = series["2017-12-04":"2017-12-31 23:00"]
    sarima = SARIMAX(
        fit_window,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    days = pd.date_range(TEST_START, periods=n_days, freq="D")
    sarima_fc = []
    res = sarima
    for day in days:
        sarima_fc.append(res.forecast(24))
        actual_day = series[day : day + pd.Timedelta(hours=23)]
        res = res.append(actual_day, refit=False)

    pred = pd.concat(sarima_fc)
    pred.name = "sarima"
    return pred


def plot_lgb_results(
    preds: pd.DataFrame, model_v2: lgb.LGBMRegressor, features_v2: list[str], fig_dir: Path = FIG_DIR
) -> None:
    """Plot one test week and feature importances for LightGBM."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(13, 8))
    week = preds["2018-02-05":"2018-02-11"]
    ax[0].plot(week["actual"], label="Actual", lw=2)
    ax[0].plot(week["lightgbm_v2"], label="LightGBM (midnight-anchored)", lw=1.5)
    ax[0].plot(week["naive_168"], label="Naive (last week)", lw=1, alpha=0.7)
    ax[0].set_title("One test week: actual vs forecasts")
    ax[0].set_ylabel("MW")
    ax[0].legend()

    imp = pd.Series(model_v2.feature_importances_, index=features_v2).sort_values()
    imp.plot(kind="barh", ax=ax[1], title="LightGBM feature importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "06_lightgbm_results.png", dpi=150)
    plt.close(fig)


def main(clean_path: Path = CLEAN_PATH, run_sarima: bool = True) -> pd.DataFrame:
    df = pd.read_csv(clean_path, index_col="Datetime", parse_dates=True)
    y = df["load_mw"].asfreq("h")

    _, final_v2, features_v2 = train_lgb_models(y, MODELS_DIR)
    preds = predict_baselines_and_lgb(y, MODELS_DIR, start=TEST_START)

    res = {m: metrics(preds["actual"], preds[m]) for m in ["naive_24", "naive_168", "lightgbm", "lightgbm_v2"]}
    print(pd.DataFrame(res).T.round(2))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PROCESSED_DIR / "test_predictions.csv")
    plot_lgb_results(preds, final_v2, features_v2, FIG_DIR)

    if run_sarima:
        sarima_pred = forecast_sarima(y, n_days=30)
        sarima_pred.to_csv(PROCESSED_DIR / "sarima_predictions.csv")

    return preds


if __name__ == "__main__":
    main()
