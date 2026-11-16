"""Two-layer sequence-to-24h LSTM model with calendar inputs and early stopping."""
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.data import CLEAN_PATH, ROOT
from src.features import HORIZON, LOOKBACK, TRAIN_END, build_lstm_windows, metrics

warnings.filterwarnings("ignore")

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_model(n_calendar_features: int = 10, lookback: int = LOOKBACK, horizon: int = HORIZON) -> keras.Model:
    """Build functional Keras LSTM taking 168h load history and 10 calendar features."""
    seq_in = keras.Input(shape=(lookback, 1), name="history")
    day_in = keras.Input(shape=(n_calendar_features,), name="calendar")

    x = layers.LSTM(64, return_sequences=True)(seq_in)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Concatenate()([x, day_in])
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(horizon)(x)

    model = keras.Model([seq_in, day_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def main(clean_path: Path = CLEAN_PATH, epochs: int = 80) -> pd.Series:
    set_seeds(42)
    df = pd.read_csv(clean_path, index_col="Datetime", parse_dates=True)
    y = df["load_mw"].asfreq("h")

    scaler = StandardScaler().fit(y[:TRAIN_END].values.reshape(-1, 1))
    y_s = pd.Series(scaler.transform(y.values.reshape(-1, 1)).ravel(), index=y.index)

    Xs_tr, Xd_tr, Y_tr, _ = build_lstm_windows(y_s, "2002-01-01", "2016-12-31")
    Xs_va, Xd_va, Y_va, _ = build_lstm_windows(y_s, "2017-01-01", "2017-12-31")
    Xs_te, Xd_te, _, d_te = build_lstm_windows(y_s, "2018-01-01", "2018-08-03")

    model = build_lstm_model(n_calendar_features=Xd_tr.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        [Xs_tr, Xd_tr], Y_tr,
        validation_data=([Xs_va, Xd_va], Y_va),
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    pred_s = model.predict([Xs_te, Xd_te], verbose=0)
    pred_mw = scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1, HORIZON)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in d_te for h in range(HORIZON)])
    lstm_pred = pd.Series(pred_mw.ravel(), index=idx, name="lstm")

    actual_te = y.loc[lstm_pred.index]
    print("LSTM test metrics:", {k: round(v, 2) for k, v in metrics(actual_te, lstm_pred).items()})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    lstm_pred.to_csv(PROCESSED_DIR / "lstm_predictions.csv")
    model.save(MODELS_DIR / "lstm.keras")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_title("LSTM training curve (MAE, scaled units)")
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_lstm_training.png", dpi=150)
    plt.close(fig)

    return lstm_pred


if __name__ == "__main__":
    main()
