import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("models/crop_model.pkl")
le = joblib.load("models/label_encoder.pkl")

features = ['N','P','K','temperature','humidity','ph','rainfall']

st.title("🌱 Crop Recommendation System")
st.write("AI system that suggests the best crop based on soil and climate conditions.")

# User input
st.subheader("🌱 Enter Farm Data")

N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temp = st.number_input("Temperature (°C)", -10, 50, 25)
humidity = st.number_input("Humidity (%)", 0, 100, 60)
ph = st.number_input("pH level", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0, 500, 100)

if st.button("Predict Crop"):
    user_input = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]], columns=features)
    prediction = model.predict(user_input)
    crop = le.inverse_transform(prediction)[0]

    st.success(f"🌾 Recommended Crop: **{crop.upper()}**")

    # Farm Advisor Logic
    st.subheader("🧑‍🌾 Farm Advisor Suggestions")
    if N < 50:
        st.write("⚠ Low Nitrogen → Add Urea Fertilizer")
    if P < 50:
        st.write("⚠ Low Phosphorus → Improve root growth nutrients")
    if K < 50:
        st.write("⚠ Low Potassium → Use Potash fertilizer")
    if ph < 5.5:
        st.write("⚠ Soil too acidic → Add Lime")
    elif ph > 8:
        st.write("⚠ Soil too alkaline → Add Organic Compost")
    if rainfall > 250:
        st.write("⚠ Heavy rainfall → Improve drainage system")
    if humidity < 40:
        st.write("⚠ Low humidity → Use irrigation system")
    if temp > 40:
        st.write("⚠ High temperature → Protect crops from heat stress")
