import streamlit as st
import pandas as pd
import joblib

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/model.pkl")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's final grade (G3) using Machine Learning."
)

# ==========================================
# USER INPUTS
# ==========================================

studytime = st.slider(
    "Study Time (1-4)",
    min_value=1,
    max_value=4,
    value=2
)

failures = st.slider(
    "Past Failures",
    min_value=0,
    max_value=4,
    value=0
)

absences = st.slider(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5
)

g1 = st.slider(
    "G1 Grade",
    min_value=0,
    max_value=20,
    value=10
)

g2 = st.slider(
    "G2 Grade",
    min_value=0,
    max_value=20,
    value=10
)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Final Grade"):

    input_data = pd.DataFrame(
        [[studytime, failures, absences, g1, g2]],
        columns=[
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2"
        ]
    )

    prediction = model.predict(input_data)

    st.success(
        f"🎯 Predicted Final Grade (G3): {prediction[0]:.2f}"
    )

    if prediction[0] >= 16:
        st.balloons()
        st.success("Excellent Performance! 🌟")

    elif prediction[0] >= 10:
        st.info("Good Performance 👍")

    else:
        st.warning("Needs Improvement 📚")
