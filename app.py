# app.py
import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="ML Prediction App", page_icon="🚀", layout="centered")

# Custom CSS for animation & styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    .main {
        background-color: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);} 
        to {opacity: 1; transform: translateY(0);} 
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>🚀 Smart Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter values and get instant predictions</p>", unsafe_allow_html=True)

# Input fields (modify according to your model features)
st.subheader("Input Features")
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

# Prediction button
if st.button("Predict 🔮"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)


# requirements.txt content:
# streamlit
# numpy
# scikit-learn
