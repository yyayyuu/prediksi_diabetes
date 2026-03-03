# =========================
# stream_diabetes_light.py
# =========================

import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model & kolom
# =========================
xgb = joblib.load("model_xgb_tomek_light.joblib")
model_columns = joblib.load("model_columns_light.joblib")

st.title("Prediksi Diabetes Ringan (XGBoost + TomekLinks)")

# =========================
# Input manual
# =========================
age = st.number_input("Umur", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
blood_glucose = st.number_input("Blood Glucose", min_value=0, max_value=600, value=100)
smoking_history = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever"])

# =========================
# Tombol prediksi
# =========================
if st.button("Prediksi"):
    data_input = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "BMI": [BMI],
        "blood_glucose_level": [blood_glucose],
        "smoking_history": [smoking_history]
    })

    # One-Hot Encoding
    data_input_encoded = pd.get_dummies(data_input)

    # Tambahkan kolom yang hilang
    for c in model_columns:
        if c not in data_input_encoded.columns:
            data_input_encoded[c] = 0

    # Urutkan kolom
    data_input_encoded = data_input_encoded[model_columns]

    # Prediksi
    prediksi = xgb.predict(data_input_encoded)
    st.success(f"Hasil prediksi diabetes: {prediksi[0]}")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader("Upload CSV pasien baru", type="csv")
if uploaded_file:
    data_baru = pd.read_csv(uploaded_file)
    st.write("Data baru:")
    st.dataframe(data_baru.head())

    # One-Hot Encoding
    for col in ["gender", "smoking_history"]:
        if col in data_baru.columns:
            data_baru[col] = data_baru[col].astype(str)
    data_baru_encoded = pd.get_dummies(data_baru)
    
    for c in model_columns:
        if c not in data_baru_encoded.columns:
            data_baru_encoded[c] = 0
    data_baru_encoded = data_baru_encoded[model_columns]

    # Prediksi
    data_baru["prediksi_diabetes"] = xgb.predict(data_baru_encoded)

    # Download hasil prediksi
    csv = data_baru.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download hasil prediksi",
        csv,
        file_name="hasil_prediksi_light.csv",
        mime="text/csv"
    )