import streamlit as st
import pickle
import numpy as np

# Load model
with open('heart_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

st.title("❤️ Heart Disease Predictor")
st.write("Fill in your details below to check heart disease risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0)
    slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1=normal, 2=fixed, 3=reversible)", [1, 2, 3])

if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probability*100:.1f}% probability)")
