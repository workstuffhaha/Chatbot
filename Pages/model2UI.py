import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("Models/student_depression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Models/student_depression_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Student Depression Prediction")
st.write("Provide the following details to assess the risk of depression.")

# Define questions
questions = [
    {"text": "Enter Age:", "type": "number", "key": "age", "min": 15, "max": 60, "step": 1},
    {"text": "Enter CGPA:", "type": "number", "key": "cgpa", "min": 0.0, "max": 10.0, "step": 0.01},
    {"text": "Select Current Dietary Habits:", "type": "select", "key": "dietary_habits", "options": ["Healthy", "Moderate", "Unhealthy", "Others"]},
    {"text": "Enter Work/Study Hours per Day:", "type": "number", "key": "study_hours", "min": 0, "max": 18, "step": 1}
]

diet_mapping = {"Healthy": 1, "Moderate": 2, "Unhealthy": 3, "Others": 4}

# Session state to track progress
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

# Display questions one at a time
q = questions[st.session_state.current_question]
if q["type"] == "number":
    response = st.number_input(q["text"], min_value=q["min"], max_value=q["max"], step=q["step"], key=q["key"])
elif q["type"] == "select":
    response = st.selectbox(q["text"], q["options"], key=q["key"])

if st.button("Next"):
    st.session_state.responses[q["key"]] = response
    if st.session_state.current_question < len(questions) - 1:
        st.session_state.current_question += 1
        st.rerun()

# Prediction step
if st.session_state.current_question == len(questions) - 1 and st.button("Predict"):
    age = st.session_state.responses.get("age", 0)
    cgpa = st.session_state.responses.get("cgpa", 0.0)
    dietary_habits = st.session_state.responses.get("dietary_habits", "Healthy")
    study_hours = st.session_state.responses.get("study_hours", 0)
    
    diet_value = diet_mapping[dietary_habits]
    user_input = np.array([age, cgpa, diet_value, study_hours]).reshape(1, -1)
    transformed_input = scaler.transform(user_input)
    prediction = model.predict(transformed_input)
    
    result = "Yes, you might have depression." if prediction[0] == 1 else "No, you do not have depression."
    st.success(result)

# Restart Button
if st.button("Restart Test"):
    st.session_state.current_question = 0
    st.session_state.responses = {}
    st.rerun()
