import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Hybrid Sales Forecasting App", layout="wide")

st.title("📊 Hybrid Sales Forecasting App")
st.write("Prophet-like Trend + Gradient Boosting Hybrid Model")

# Upload CSV
uploaded_file = st.file_uploader("Upload Walmart_Sales.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    ts = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    ts = ts.sort_values("Date").reset_index(drop=True)

    st.subheader("Dataset Preview")
    st.dataframe(ts.head())

    # ===============================
    # Prophet-like model
    # ===============================
    X_time = np.arange(len(ts)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X_time, ts["Weekly_Sales"])
    trend = lr.predict(X_time)

    ts["week"] = ts["Date"].dt.isocalendar().week.astype(int)
    seasonal = ts.groupby("week")["Weekly_Sales"].transform("mean")

    ts["prophet_pred"] = trend + (seasonal - seasonal.mean())

    # ===============================
    # Gradient Boosting model
    # ===============================
    ts["lag_1"] = ts["Weekly_Sales"].shift(1)
    ts["lag_2"] = ts["Weekly_Sales"].shift(2)
    ts["rolling_3"] = ts["Weekly_Sales"].rolling(3).mean()
    ts = ts.dropna().reset_index(drop=True)

    X = ts[["lag_1", "lag_2", "rolling_3"]]
    y = ts["Weekly_Sales"]

    train_size = int(len(ts) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    gbm = GradientBoostingRegressor(random_state=42)
    gbm.fit(X_train, y_train)

    ts.loc[X_test.index, "xgb_pred"] = gbm.predict(X_test)

    # ===============================
    # Hybrid model
    # ===============================
    prophet = ts.loc[X_test.index, "prophet_pred"]
    xgb = ts.loc[X_test.index, "xgb_pred"]
    ts.loc[X_test.index, "hybrid_pred"] = (0.4 * prophet) + (0.6 * xgb)

    # ===============================
    # Metrics
    # ===============================
    y_true = y_test.values
    hybrid = ts.loc[X_test.index, "hybrid_pred"].values

    mae = mean_absolute_error(y_true, hybrid)
    rmse = np.sqrt(mean_squared_error(y_true, hybrid))
    mape = np.mean(np.abs((y_true - hybrid) / y_true)) * 100
    accuracy = 100 - mape

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:,.0f}")
    col2.metric("RMSE", f"{rmse:,.0f}")
    col3.metric("Accuracy (%)", f"{accuracy:.2f}")

    # ===============================
    # Visualization
    # ===============================
    st.subheader("📈 Actual vs Hybrid Forecast")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts["Date"][X_test.index], y_true, label="Actual", linewidth=2)
    ax.plot(ts["Date"][X_test.index], ts["hybrid_pred"][X_test.index],
            label="Hybrid Forecast", linestyle="--")

    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
