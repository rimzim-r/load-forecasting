import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Load Forecasting", layout="wide")
st.title("Day-Ahead Electricity Load Forecasting (PJM East)")
st.caption("Test period: Jan to early Aug 2018. Forecasts are made at midnight for the next 24 hours.")

@st.cache_data
def load():
    return pd.read_csv("reports/demo_predictions.csv", index_col=0, parse_dates=True)

df = load()
models = [c for c in df.columns if c != "actual"]

st.sidebar.header("Options")
chosen = st.sidebar.multiselect("Models", models, default=["lightgbm_v2", "lstm", "ensemble"])
days = sorted(set(df.index.date))
start = st.sidebar.date_input("Start date", days[30], min_value=days[0], max_value=days[-1])
n_days = st.sidebar.slider("Days to show", 1, 14, 7)

view = df.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=n_days)]
st.subheader("Actual vs forecast (MW)")
st.line_chart(view[["actual"] + chosen])

def score(frame, cols):
    rows = {}
    for m in cols:
        e = frame["actual"] - frame[m]
        rows[m] = {"MAE": e.abs().mean(),
                   "RMSE": np.sqrt((e ** 2).mean()),
                   "MAPE_%": (e.abs() / frame["actual"]).mean() * 100}
    return pd.DataFrame(rows).T.round(2)

c1, c2 = st.columns(2)
c1.subheader("Selected window")
c1.dataframe(score(view, models))
c2.subheader("Full test period")
c2.dataframe(score(df, models))