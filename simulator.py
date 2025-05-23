import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import datetime
import gc
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable

# Register custom loss
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Utility: Temporal features
def add_temporal_features(df):
    df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DaySin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df

# Utility: Sequence builder
def create_sequences(data, window_size):
    X_seq = []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:i + window_size])
    return np.array(X_seq)

# Load model + scalers
@st.cache_resource
def load_model_and_scalers():
    custom_objects = {"mse": mse, "mean_squared_error": MeanSquaredError()}
    model = load_model("traffic_model.h5", custom_objects=custom_objects)
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_and_scalers()

# Load only selected route data
@st.cache_data
def load_data(selected_route):
    files = {
        "MA-A": "./dataset/MA-A.xlsx",
        "NORTHBOUND": "./dataset/NORTHBOUND.xlsx",
        "SOUTHBOUND": "./dataset/SOUTHBOUND.xlsx"
    }
    sheets = ["2018", "2019", "2020", "2021", "2022", "2023"]
    file = files[selected_route]

    df_list = []
    if os.path.exists(file):
        for sheet in sheets:
            try:
                temp = pd.read_excel(file, sheet_name=sheet)
                if "TRAFFIC STATUS" in temp.columns:
                    temp.drop(columns=["TRAFFIC STATUS"], inplace=True)
                temp['TIME(24 HOUR)'] = temp['TIME(24 HOUR)'].astype(str).str.zfill(5) + ':00'
                temp['Datetime'] = pd.to_datetime(temp['DATE'].astype(str) + ' ' + temp['TIME(24 HOUR)'], dayfirst=True, errors='coerce')
                temp.dropna(subset=["Datetime"], inplace=True)
                temp.sort_values("Datetime", inplace=True)
                temp['Hour'] = temp['Datetime'].dt.hour
                temp['DayOfWeek'] = temp['Datetime'].dt.dayofweek
                temp['Month'] = temp['Datetime'].dt.month
                temp['Route'] = selected_route
                df_list.append(temp)
            except Exception as e:
                print(f"❌ Error loading {file} - {sheet}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# App Title
st.title("🚦 Traffic Volume Prediction")

# UI
routes = ["MA-A", "NORTHBOUND", "SOUTHBOUND"]
selected_route = st.selectbox("🛣 Select Route", routes)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.date.today())
with col2:
    end_date = st.date_input("End Date", value=datetime.date.today() + datetime.timedelta(days=1))

if start_date > end_date:
    st.error("⚠️ End date must be after start date!")
    st.stop()

# Load data only for selected route
df = load_data(selected_route)
if df.empty:
    st.warning("⚠️ No data available for this route.")
    st.stop()

# Future window (limit hours to reduce memory)
window_start_date = start_date - pd.Timedelta(days=1)
date_range = pd.date_range(start=window_start_date, end=end_date, freq="H")[:48]  # max 48 hours
df_future = pd.DataFrame({"Datetime": date_range})
df_future["Date"] = df_future["Datetime"].dt.date
df_future["Time"] = df_future["Datetime"].dt.strftime("%H:%M:%S")
df_future["Day of the Week"] = df_future["Datetime"].dt.day_name()
df_future["Hour"] = df_future["Datetime"].dt.hour
df_future["DayOfWeek"] = df_future["Datetime"].dt.dayofweek
df_future["Month"] = df_future["Datetime"].dt.month
df_future = add_temporal_features(df_future)

# Lags
target_column = "TRAFFIC VOLUME"
lags = [1, 2, 3, 6, 12, 24]
window_size = 24

last_df = df.copy()
for lag in lags:
    col = f"{target_column}_lag_{lag}"
    if target_column in last_df.columns:
        lag_values = last_df[target_column].dropna().tail(lag).tolist()
        if len(lag_values) < lag:
            lag_values = [last_df[target_column].mean()] * lag
        df_future[col] = lag_values[-1]
    else:
        df_future[col] = 0

# One-hot encoding
df_future = pd.get_dummies(df_future, columns=["Day of the Week"], drop_first=True)

# Route Encoding
for route in routes:
    df_future[f"Route_{route}"] = 1 if route == selected_route else 0

# Fill missing columns from training
for col in scaler_X.feature_names_in_:
    if col not in df_future.columns:
        df_future[col] = 0

# Arrange features
features = df_future[scaler_X.feature_names_in_].copy()

# Predict
if st.button("🔮 Predict"):
    try:
        X_scaled = scaler_X.transform(features)
        X_seq = create_sequences(X_scaled, window_size)

        if len(X_seq) == 0:
            st.error("❌ Not enough data to make a prediction.")
            st.stop()

        y_scaled = model.predict(X_seq).flatten()
        y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

        prediction_df = df_future.iloc[window_size:].reset_index(drop=True)
        prediction_df["Day of the Week"] = prediction_df["Datetime"].dt.day_name()
        prediction_df["Predicted Traffic Volume"] = y_pred

        st.subheader("📊 Predicted Traffic Volumes")
        st.dataframe(prediction_df[["Datetime", "Day of the Week", "Predicted Traffic Volume"]])

        # Free memory
        del df, df_future, features, X_scaled, X_seq, prediction_df
        gc.collect()

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

