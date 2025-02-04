import streamlit as st

# Set page config
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")

# Title
st.title("Chatbot Prediction System")
st.write("Choose a prediction model:")

# Buttons for navigation
col1, col2 = st.columns(2)

with col1:
    if st.button("Heart Disease Prediction ❤️"):
        st.switch_page("Pages\model1UI.py")

with col2:
    if st.button("Student Depression Prediction 🎓"):
        st.switch_page("Pages\model2UI.py")
