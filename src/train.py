"""Unified training CLI and entry point for LightGBM and LSTM day-ahead load forecasting models."""
import argparse
from pathlib import Path

import pandas as pd

from src.baselines_ml import MODELS_DIR, train_lgb_models
from src.data import CLEAN_PATH, RAW_PATH
from src.data import main as run_data_pipeline
from src.lstm_model import FIG_DIR, train_lstm


def train_all(
    clean_path: Path = CLEAN_PATH,
    models_dir: Path = MODELS_DIR,
    fig_dir: Path = FIG_DIR,
    model_choice: str = "all",
    epochs: int = 80,
) -> None:
    """Train requested models ('lightgbm', 'lstm', or 'all') and persist artifacts to models_dir."""
    if not clean_path.exists():
        print(f"Cleaned dataset not found at {clean_path}; running data pipeline first...")
        run_data_pipeline(raw_path=RAW_PATH, clean_path=clean_path)

    df = pd.read_csv(clean_path, index_col="Datetime", parse_dates=True)
    y = df["load_mw"].asfreq("h")

    if model_choice in ("all", "lightgbm"):
        print("Training LightGBM v1 and v2 (midnight-anchored) models...")
        train_lgb_models(y, models_dir=models_dir)
        print(f"Saved LightGBM models to {models_dir}")

    if model_choice in ("all", "lstm"):
        print("Training sequence-to-24h LSTM model...")
        train_lstm(y, models_dir=models_dir, fig_dir=fig_dir, epochs=epochs)
        print(f"Saved LSTM model to {models_dir / 'lstm.keras'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train day-ahead electricity load forecasting models.")
    parser.add_argument(
        "--models",
        choices=["all", "lightgbm", "lstm"],
        default="all",
        help="Which model family to train (default: all).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs for LSTM (default: 80).")
    parser.add_argument("--clean-path", type=Path, default=CLEAN_PATH, help="Path to cleaned hourly CSV.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Directory to save trained models.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_all(
        clean_path=args.clean_path,
        models_dir=args.models_dir,
        model_choice=args.models,
        epochs=args.epochs,
    )
