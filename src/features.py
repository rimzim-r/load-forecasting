"""Shared time-series split constants, evaluation metrics, and feature engineering."""
import holidays
import numpy as np
import pandas as pd

TRAIN_END = "2016-12-31 23:00"
VAL_END = "2017-12-31 23:00"
TEST_START = "2018-01-01"
LOOKBACK = 168
HORIZON = 24

US_HOLIDAYS = holidays.US(years=range(2002, 2019))
HOLIDAY_DATES = pd.to_datetime(list(US_HOLIDAYS.keys()))


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute MAE (MW), RMSE (MW), and MAPE (%)."""
    err = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err / y_true)) * 100),
    }


def make_features(series: pd.Series) -> pd.DataFrame:
    """Create lag (>=24h), rolling, calendar, and cyclical features without data leakage."""
    d = pd.DataFrame({"load_mw": series})

    for lag in [24, 48, 72, 168, 336]:
        d[f"lag_{lag}"] = series.shift(lag)

    past = series.shift(24)
    d["roll_mean_24"] = past.rolling(24).mean()
    d["roll_mean_168"] = past.rolling(168).mean()
    d["roll_std_24"] = past.rolling(24).std()

    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["is_holiday"] = d.index.normalize().isin(HOLIDAY_DATES).astype(int)

    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["doy_sin"] = np.sin(2 * np.pi * d.index.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d.index.dayofyear / 365.25)

    return d.dropna()


def add_anchored_features(feat: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """Add midnight-anchored features (load and rolling stats known at 23:00 the previous day)."""
    def anchored(ser: pd.Series) -> np.ndarray:
        a = ser[ser.index.hour == 23].copy()
        a.index = a.index.normalize() + pd.Timedelta(days=1)
        return a.reindex(feat.index.normalize()).values

    feat2 = feat.copy()
    feat2["anchor_load"] = anchored(series)
    feat2["anchor_lag1"] = anchored(series.shift(1))
    feat2["anchor_lag2"] = anchored(series.shift(2))
    feat2["anchor_mean_24"] = anchored(series.rolling(24).mean())
    feat2["anchor_mean_168"] = anchored(series.rolling(168).mean())
    return feat2.dropna()


def day_features(day: pd.Timestamp) -> np.ndarray:
    """Return 10-element calendar vector (one-hot day-of-week, US holiday flag, sin/cos day-of-year)."""
    dow = np.eye(7)[day.dayofweek]
    hol = float(day.normalize() in HOLIDAY_DATES)
    doy = day.dayofyear
    return np.concatenate([
        dow,
        [hol, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)],
    ])


def build_lstm_windows(
    scaled_series: pd.Series,
    start_day: str,
    end_day: str,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Build daily (history_168h, calendar_10d, target_24h) arrays for the LSTM."""
    xs, xd, y_out, days = [], [], [], []
    for day in pd.date_range(start_day, end_day, freq="D"):
        w_start = day - pd.Timedelta(hours=lookback)
        t_end = day + pd.Timedelta(hours=horizon - 1)
        if w_start < scaled_series.index.min() or t_end > scaled_series.index.max():
            continue
        xs.append(scaled_series[w_start : day - pd.Timedelta(hours=1)].values)
        y_out.append(scaled_series[day:t_end].values)
        xd.append(day_features(day))
        days.append(day)
    return np.array(xs)[..., None], np.array(xd), np.array(y_out), days
