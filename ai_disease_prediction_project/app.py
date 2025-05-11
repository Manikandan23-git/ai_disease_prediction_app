import streamlit as st
import numpy as np
import pickle

# Load the trained model
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'diabetes_model.pkl' not found. Please run model_training.py first.")
    st.stop()

# App title
st.title("🧠 AI-Powered Diabetes Prediction App")

# User input form
with st.form("diabetes_form"):
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
    bp = st.number_input('Blood Pressure', min_value=0, max_value=122, step=1)
    skin = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
    insulin = st.number_input('Insulin', min_value=0.0, max_value=1000.0, step=1.0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input('Age', min_value=1, max_value=120, step=1)

    # Submit button
    submitted = st.form_submit_button("Predict Diabetes")

# Prediction
if submitted:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🧪 Prediction: The patient is likely **Diabetic**.")
    else:
        st.success("✅ Prediction: The patient is **Not Diabetic**.")
