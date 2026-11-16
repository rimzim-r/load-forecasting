"""Unified inference module and CLI for generating 24-hour day-ahead load forecasts from saved models."""
import argparse
from pathlib import Path

import pandas as pd

from src.baselines_ml import MODELS_DIR, PROCESSED_DIR, forecast_sarima, predict_baselines_and_lgb
from src.data import CLEAN_PATH
from src.features import TEST_START, metrics


def run_inference(
    series: pd.Series,
    models_dir: Path = MODELS_DIR,
    start_day: str = TEST_START,
    end_day: str = "2018-08-03",
    include_lstm: bool = True,
) -> pd.DataFrame:
    """Run batch day-ahead inference using saved LightGBM boosters and optional Keras LSTM model."""
    end_ts = str(pd.Timestamp(end_day) + pd.Timedelta(hours=23))
    preds = predict_baselines_and_lgb(series, models_dir=models_dir, start=start_day, end=end_ts)

    lstm_path = models_dir / "lstm.keras"
    if include_lstm and lstm_path.exists():
        from src.lstm_model import predict_lstm

        lstm_pred = predict_lstm(series, model_or_path=lstm_path, start_day=start_day, end_day=end_day)
        preds = preds.loc[lstm_pred.index].copy()
        preds["lstm"] = lstm_pred
        preds["ensemble"] = (preds["lightgbm_v2"] + preds["lstm"]) / 2

    return preds


def forecast_next_24h(
    series: pd.Series,
    target_day: str | pd.Timestamp,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Forecast the 24 hours of `target_day` using data known up to 23:00 the previous day."""
    day_str = str(pd.Timestamp(target_day).date())
    return run_inference(series, models_dir=models_dir, start_day=day_str, end_day=day_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run day-ahead electricity load inference from saved models.")
    parser.add_argument("--start-day", default="2018-01-01", help="First forecast date (YYYY-MM-DD).")
    parser.add_argument("--end-day", default="2018-08-03", help="Last forecast date (YYYY-MM-DD).")
    parser.add_argument("--clean-path", type=Path, default=CLEAN_PATH, help="Path to cleaned hourly CSV.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Directory with saved models.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "test_predictions.csv",
        help="Output CSV path for predictions.",
    )
    parser.add_argument("--run-sarima", action="store_true", help="Also run 30-day rolling SARIMA baseline.")
    return parser.parse_args()


def main() -> pd.DataFrame:
    args = parse_args()
    df = pd.read_csv(args.clean_path, index_col="Datetime", parse_dates=True)
    y = df["load_mw"].asfreq("h")

    preds = run_inference(
        y,
        models_dir=args.models_dir,
        start_day=args.start_day,
        end_day=args.end_day,
    )

    model_cols = [c for c in preds.columns if c != "actual"]
    summary = pd.DataFrame({m: metrics(preds["actual"], preds[m]) for m in model_cols}).T.round(2)
    print(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output)
    print(f"Saved predictions to {args.output}")

    if "lstm" in preds.columns:
        lstm_out = args.output.parent / "lstm_predictions.csv"
        preds["lstm"].to_csv(lstm_out)

    if args.run_sarima:
        sarima_pred = forecast_sarima(y, n_days=30)
        sarima_out = args.output.parent / "sarima_predictions.csv"
        sarima_pred.to_csv(sarima_out)
        print(f"Saved SARIMA predictions to {sarima_out}")

    return preds


if __name__ == "__main__":
    main()
