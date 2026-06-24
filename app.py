import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Machine Learning Prediction App")

st.write("Enter values below to get a prediction.")

# Example inputs
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):

    # Create input array
    input_data = np.array([[feature1, feature2, feature3]])

    # Prediction
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
