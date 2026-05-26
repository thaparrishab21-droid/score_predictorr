import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/model.pkl")

# ==========================================
# PAGE TITLE
# ==========================================

st.title("🎓 Student Score Predictor")

st.write(
    "Predict student exam scores using Machine Learning."
)

# ==========================================
# USER INPUTS
# ==========================================

study_hours = st.slider(
    "Study Hours",
    min_value=0,
    max_value=12,
    value=5
)

sleep_hours = st.slider(
    "Sleep Hours",
    min_value=0,
    max_value=12,
    value=7
)

attendance = st.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75
)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Score"):

    # Create DataFrame
    input_data = pd.DataFrame(
        [[study_hours, sleep_hours, attendance]],
        columns=[
            'StudyHours',
            'SleepHours',
            'Attendance'
        ]
    )

    # Prediction
    prediction = model.predict(input_data)

    # Output
    st.success(
        f"Predicted Score: {prediction[0]:.2f}"
    )