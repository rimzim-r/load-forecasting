import numpy as np
import pandas as pd

from src.data import clean_load_series
from src.features import add_anchored_features, day_features, make_features, metrics


def test_clean_load_series_deduplicates_and_interpolates():
    raw = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2018-01-01 00:00:00",
            "2018-01-01 01:00:00",
            "2018-01-01 01:00:00",  # DST duplicate
            "2018-01-01 03:00:00",  # 02:00 missing
        ]),
        "PJME_MW": [30000.0, 31000.0, 33000.0, 34000.0],
    })
    cleaned = clean_load_series(raw)
    assert len(cleaned) == 4
    assert cleaned.loc["2018-01-01 01:00:00", "load_mw"] == 32000.0
    assert cleaned.loc["2018-01-01 02:00:00", "load_mw"] == 33000.0
    assert cleaned["load_mw"].isna().sum() == 0


def test_make_and_anchored_features_no_leakage():
    idx = pd.date_range("2017-01-01", periods=24 * 20, freq="h")
    s = pd.Series(np.linspace(25000.0, 35000.0, len(idx)), index=idx)
    feat = make_features(s)
    feat2 = add_anchored_features(feat, s)
    assert "lag_24" in feat.columns
    assert "anchor_load" in feat2.columns
    assert not feat2.isna().any().any()


def test_day_features_shape():
    vec = day_features(pd.Timestamp("2018-01-01"))
    assert vec.shape == (10,)
    assert vec[7] == 1.0  # New Year's Day is a US holiday


def test_metrics():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    m = metrics(y_true, y_pred)
    assert round(m["MAE"], 2) == 15.0
    assert round(m["MAPE_%"], 2) == 10.0
