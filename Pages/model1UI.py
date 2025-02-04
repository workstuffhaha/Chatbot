import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Model and Preprocessor
with open("Models/model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Models/preprocessor1.pkl", "rb") as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Title
st.title("Heart Disease Prediction")

st.write("Answer the following questions to assess heart disease risk.")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.user_inputs = []

def reset_chat():
    st.session_state.step = 0
    st.session_state.user_inputs = []

questions = [
    ("Enter Age:", "number", 1, 120),
    ("Select Sex:", "select", ["Male", "Female"]),
    ("Select Chest Pain Type:", "select", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]),
    ("Enter Resting Blood Pressure:", "number", 80, 200),
    ("Enter Cholesterol Level:", "number", 100, 400),
    ("Select Resting ECG Result:", "select", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]),
    ("Enter Maximum Heart Rate Achieved:", "number", 50, 250),
    ("Do you have Exercise-Induced Angina?", "select", ["Yes", "No"]),
    ("Enter Oldpeak (ST Depression Induced by Exercise):", "number", 0.0, 6.0),
    ("Select ST Slope:", "select", ["Upsloping", "Flat", "Downsloping"])
]

# Define column order for DataFrame
columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

if st.session_state.step < len(questions):
    question, q_type, *options = questions[st.session_state.step]
    
    st.write(question)
    
    if q_type == "number":
        response = st.number_input("", min_value=options[0], max_value=options[1], key=st.session_state.step)
    else:
        response = st.selectbox("", options[0], key=st.session_state.step)
    
    if st.button("Next"):
        st.session_state.user_inputs.append(response)
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == len(questions):
    st.write("### All inputs received! Click 'Predict' to see the result.")
    if st.button("Predict"):
        # Convert user inputs to pandas DataFrame
        user_data_df = pd.DataFrame([st.session_state.user_inputs], columns=columns)
        
        # Preprocess input
        processed_data = preprocessor.transform(user_data_df)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Display result
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        st.success(result)
    
    if st.button("Restart Test"):
        reset_chat()
        st.rerun()
