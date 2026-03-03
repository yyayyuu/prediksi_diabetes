# =========================
# stream_diabetes.py - Streamlit Deployment XGBoost + TomekLinks
# =========================

import streamlit as st
import pandas as pd
import pickle

# =========================
# 1️⃣ Load model dan kolom (pickle)
# =========================
@st.cache_resource
def load_model():
    model = pickle.load(open("model_xgb_tomek.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
    return model, model_columns

model, model_columns = load_model()

# =========================
# 2️⃣ Streamlit UI
# =========================
st.title("Prediksi Diabetes Langsung (XGBoost + TomekLinks)")
st.write("Isi data pasien baru untuk memprediksi risiko diabetes.")

# Input form
age = st.number_input("Umur", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=250, value=120)
smoking_history = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever"])

# =========================
# 3️⃣ Tombol Prediksi
# =========================
if st.button("Prediksi"):
    # Buat DataFrame input
    data_input = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "BMI": [BMI],
        "blood_pressure": [blood_pressure],
        "smoking_history": [smoking_history]
    })

    # One-Hot Encoding
    data_input_encoded = pd.get_dummies(data_input)

    # Tambahkan kolom yang hilang sesuai training
    for c in model_columns:
        if c not in data_input_encoded.columns:
            data_input_encoded[c] = 0

    # Urutkan kolom sesuai training
    data_input_encoded = data_input_encoded[model_columns]

    # Prediksi
    prediksi = model.predict(data_input_encoded)
    st.success(f"Hasil prediksi diabetes: {'Ya' if prediksi[0]==1 else 'Tidak'}")