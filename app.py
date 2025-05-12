import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# Title and description
st.title("🧠 AI-Powered Diabetes Prediction App")
st.markdown("Predict whether a patient is likely to have diabetes using this simple form. 🔍")

# Create columns to organize input
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('👶 Pregnancies', min_value=0, max_value=20, step=1)
    bp = st.number_input('💓 Blood Pressure', min_value=0, max_value=122, step=1)
    insulin = st.number_input('💉 Insulin Level', min_value=0.0, max_value=1000.0, step=1.0)
    dpf = st.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)

with col2:
    glucose = st.number_input('🍬 Glucose Level', min_value=0, max_value=200, step=1)
    skin = st.number_input('🩹 Skin Thickness', min_value=0, max_value=100, step=1)
    bmi = st.number_input('⚖️ BMI', min_value=0.0, max_value=70.0, step=0.1)
    age = st.number_input('🎂 Age', min_value=1, max_value=120, step=1)

# Button to predict
st.markdown("---")
if st.button('🔍 Predict Diabetes'):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)
    
    if result[0] == 1:
        st.error("🚨 **Prediction: Diabetic** — Please consult a healthcare professional.")
    else:
        st.success("✅ **Prediction: Not Diabetic** — Keep up the healthy lifestyle!")

# Optional: Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a trained machine learning model (Random Forest / Logistic Regression, etc.) on the **Pima Indians Diabetes Dataset**.")
    st.write("Built with 💖 using Streamlit.")
