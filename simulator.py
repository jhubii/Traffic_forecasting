import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_scalers():
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return scaler_X, scaler_y

def load_feature_names():
    return joblib.load("feature_names.pkl")

# Function to preprocess new dataset (WITH WINDOWING)
def preprocess_new_data(df, scaler_X, window_size, is_manual=False):
    # Ensure date column exists and convert to datetime
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df['Month'] = df['DATE'].dt.month
        df['DayOfWeek'] = df['DATE'].dt.weekday  # Monday = 0, Sunday = 6

        # Convert DATE to DAY OF THE WEEK only if it's missing
        if 'DAY OF THE WEEK' not in df.columns:
            weekday_map = {
                0: "Monday", 1: "Tuesday", 2: "Wednesday", 
                3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
            }
            df['DAY OF THE WEEK'] = df['DayOfWeek'].map(weekday_map)

    # Ensure time column exists
    if 'TIME(24 HOUR)' in df.columns:
        df['Hour'] = df['TIME(24 HOUR)'].str[:2].astype(int)

    print(df.head())

    df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DaySin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df = pd.get_dummies(df, columns=['DAY OF THE WEEK', 'WEATHER', 'ROAD CONDITION', 'HOLIDAY'], drop_first=True)

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  

    df = df[expected_features]

    X_scaled = scaler_X.transform(df)

    if is_manual:
        return X_scaled.reshape(1, 1, -1)  # (batch_size=1, time_steps=1, features)

    X_seq = []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i + window_size])

    return np.array(X_seq)

# Load model and scalers
model = load_model("traffic_forecasting.h5")
scaler_X, scaler_y = load_scalers()
expected_features = load_feature_names()

# Streamlit app
st.set_page_config(page_title="Traffic Volume Simulator", layout="centered")
st.title("\U0001F6A6 Traffic Volume Prediction Simulator")
st.write("Upload an Excel file or manually enter details to predict traffic volume.")

uploaded_file = st.file_uploader("\U0001F4C2 Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    required_columns = ['DATE', 'TIME(24 HOUR)', 'DAY OF THE WEEK', 'WEATHER', 'ROAD CONDITION', 'HOLIDAY', 'ACCIDENTS', 'AVERAGE SPEED']

    if all(col in df.columns for col in required_columns):
        window_size = 24
        X_input = preprocess_new_data(df, scaler_X, window_size)

        y_pred_scaled = model.predict(X_input).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        df = df.iloc[window_size:]  # Align predictions with the correct rows
        df['Predicted Traffic Volume'] = y_pred

        st.write("### Predictions:")
        st.dataframe(df[['TIME(24 HOUR)', 'DAY OF THE WEEK', 'WEATHER', 'ROAD CONDITION', 'HOLIDAY', 'ACCIDENTS', 'AVERAGE SPEED', 'Predicted Traffic Volume']])
    else:
        st.error("❌ The uploaded file is missing required columns. Ensure it contains: " + ", ".join(required_columns))
else:
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("📆 Select Date")
        hour = st.slider("⏰ Hour (0-23)", min_value=0, max_value=23, step=1)
        weather = st.selectbox("🌦 Weather", ["Clear", "Sunny", "Passing Clouds", "Overcast", "Rainy", "Thunderstorm"])
    with col2:
        road_condition = st.selectbox("🛣 Road Condition", ["Dry", "Wet"])
        holiday = st.selectbox("🎉 Holiday", ["NO", "YES"])
        accidents = st.text_input("⚠ Accidents", value="0")  # Use text_input
        avg_speed = st.text_input("🚗 Average Speed", value="0")  # Use text_input

    if st.button("🔮 Predict Traffic Volume"):
        try:
            # Validate inputs
            if None in [date, hour, weather, road_condition, holiday, accidents, avg_speed]:
                st.error("Please fill in all the fields.")
                st.stop()  # Stop execution if inputs are invalid

            # Convert accidents and avg_speed to numbers
            try:
                accidents = float(accidents) if accidents else 0.0  # Default to 0 if empty
                avg_speed = float(avg_speed) if avg_speed else 0.0  # Default to 0 if empty
            except ValueError:
                st.error("Invalid input for accidents or average speed. Please enter valid numbers.")
                st.stop()  # Stop execution if conversion fails

            # Ensure accidents and avg_speed are non-negative
            if accidents < 0 or avg_speed < 0:
                st.error("Accidents and average speed must be non-negative.")
                st.stop()  # Stop execution if values are negative

            user_input = pd.DataFrame([{
                'DATE': date.strftime("%Y/%m/%d"),
                'TIME(24 HOUR)': f"{hour:02d}:00",
                'WEATHER': weather,
                'ROAD CONDITION': road_condition,
                'HOLIDAY': holiday,
                'ACCIDENTS': accidents,
                'AVERAGE SPEED': avg_speed
            }])

            window_size = 24
            X_input = preprocess_new_data(user_input, scaler_X, window_size, is_manual=True)

            y_pred_scaled = model.predict(X_input).flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            st.success(f"🚗 Predicted Traffic Volume: **{y_pred[0]:.2f}**")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please ensure all inputs are valid and try again.")
