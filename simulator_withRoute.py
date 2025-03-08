import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved scalers and model
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
model = load_model("traffic_forecasting.h5")

def preprocess_new_data(df, scaler_X, window_size=24, is_manual=False):
    df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DaySin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # One-hot encode categorical columns (including ROUTE)
    categorical_columns = ['DAY OF THE WEEK', 'WEATHER', 'ROAD CONDITION', 'HOLIDAY', 'ROUTE']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Ensure all expected ROUTE columns exist
    expected_route_cols = ['ROUTE_NORTHBOUND', 'ROUTE_SOUTHBOUND']
    for col in expected_route_cols:
        if col not in df.columns:
            df[col] = 0  # If missing, set to 0

    # Ensure all expected features exist
    expected_features = scaler_X.feature_names_in_
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with 0

    df = df[expected_features]  # Reorder columns to match training
    X_scaled = scaler_X.transform(df)

    if is_manual:
        X_scaled = np.tile(X_scaled, (window_size, 1))
        return X_scaled.reshape(1, window_size, -1)

    X_seq = []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i + window_size])
    
    return np.array(X_seq)

# Streamlit UI
st.set_page_config(page_title="Traffic Volume Simulator", layout="centered")
st.title("🚦 Traffic Volume Prediction Simulator")
st.write("Upload an Excel file or manually enter details to predict traffic volume.")

# File upload
uploaded_file = st.file_uploader("📂 Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Validate required columns
    required_columns = {'DATE', 'TIME(24 HOUR)', 'DAY OF THE WEEK', 'WEATHER', 'ROAD CONDITION', 
                        'HOLIDAY', 'ACCIDENTS', 'AVERAGE SPEED', 'ROUTE'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        st.error(f"⚠ Missing columns in Excel file: {', '.join(missing_columns)}")
        st.stop()

    df = df.applymap(lambda x: x.title() if isinstance(x, str) else x)
    df['HOLIDAY'] = df['HOLIDAY'].str.upper()
    df['ROUTE'] = df['ROUTE'].str.upper()

    # Ensure valid ROUTE values
    valid_routes = {"MA-A", "NORTHBOUND", "SOUTHBOUND"}
    if not df['ROUTE'].isin(valid_routes).all():
        st.error("⚠ Invalid ROUTE values detected. Must be one of: MA-A, NORTHBOUND, SOUTHBOUND.")
        st.stop()

    df['Datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME(24 HOUR)'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month

    st.write("### 📋 Uploaded Data Preview:")
    st.dataframe(df.head())

    X_new = preprocess_new_data(df, scaler_X)
    
    if len(X_new) > 0:
        y_pred_scaled_new = model.predict(X_new).flatten()
        y_pred_new = scaler_y.inverse_transform(y_pred_scaled_new.reshape(-1, 1)).flatten()

        df['Predicted Traffic Volume'] = np.nan
        df.iloc[-len(y_pred_new):, df.columns.get_loc('Predicted Traffic Volume')] = y_pred_new

        st.write("### 📊 Predictions:")
        st.dataframe(df[['DATE', 'TIME(24 HOUR)', 'ROUTE', 'Predicted Traffic Volume']].dropna())
    else:
        st.error("⚠ Not enough data points for prediction.")

# Manual Input Form
st.write("### ✍ Manual Data Entry")
with st.form("manual_entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("📆 Select Date")
        time = st.slider("⏰ Hour (0-23)", min_value=0, max_value=23, step=1)
        day_of_week = st.selectbox("📅 Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        route = st.selectbox("🛣 Select Route", ["MA-A", "NORTHBOUND", "SOUTHBOUND"])
    with col2:
        weather = st.selectbox("🌦 Weather", ['Clear', 'Cloudy', 'Rainy', 'Snowy'])
        road_condition = st.selectbox("🛣 Road Condition", ['Dry', 'Wet', 'Snowy', 'Icy'])
        holiday = st.selectbox("🎉 Holiday", ['YES', 'NO'])
    
    accidents = st.number_input("⚠ Accidents", min_value=0, step=1)
    avg_speed = st.number_input("🚗 Average Speed (km/h)", min_value=0, step=1)
    submitted = st.form_submit_button("🔮 Predict Traffic Volume")

    if submitted:
        manual_data = pd.DataFrame({
            'DATE': [date],
            'TIME(24 HOUR)': [f"{time:02d}:00"],
            'DAY OF THE WEEK': [day_of_week],
            'WEATHER': [weather],
            'ROAD CONDITION': [road_condition],
            'HOLIDAY': [holiday],
            'ACCIDENTS': [accidents],
            'AVERAGE SPEED': [avg_speed],
            'ROUTE': [route]  # Add route selection
        })

        manual_data = manual_data.applymap(lambda x: x.title() if isinstance(x, str) else x)
        manual_data['HOLIDAY'] = manual_data['HOLIDAY'].str.upper()
        manual_data['Datetime'] = pd.to_datetime(manual_data['DATE'].astype(str) + ' ' + manual_data['TIME(24 HOUR)'], errors='coerce')
        manual_data['Hour'] = manual_data['Datetime'].dt.hour
        manual_data['DayOfWeek'] = manual_data['Datetime'].dt.dayofweek
        manual_data['Month'] = manual_data['Datetime'].dt.month

        X_manual = preprocess_new_data(manual_data, scaler_X, is_manual=True)

        if X_manual.shape[1] != 24:
            st.error("⚠ Error: The manual input requires 24-hour sequence data for prediction.")
        else:
            y_pred_manual_scaled = model.predict(X_manual).flatten()
            y_pred_manual = scaler_y.inverse_transform(y_pred_manual_scaled.reshape(-1, 1)).flatten()

            st.success(f"🚗 Predicted Traffic Volume: **{y_pred_manual[0]:.2f}**")
