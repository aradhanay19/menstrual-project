import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Menstrual Health AI", layout="wide")

st.title("🩸 AI-Based Menstrual Health Monitoring System")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📅 Prediction", "📊 Dashboard"])

# ---------------- ANALYSIS TAB ----------------
with tab1:
    st.header("Health Analysis")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 15, 45, key="age")
        cycle = st.slider("Cycle Length", 20, 40, key="cycle")
        pain = st.slider("Pain Level", 1, 10, key="pain")

    with col2:
        flow = st.selectbox("Flow (1=Low, 2=Medium, 3=Heavy)", [1, 2, 3], key="flow")
        fatigue = st.slider("Fatigue Level", 1, 10, key="fatigue")
        mood = st.slider("Mood Swings", 1, 10, key="mood")

    if st.button("Analyze Health"):

        irregular = 1 if cycle > 32 else 0

        data = np.array([[age, cycle, pain, flow, fatigue, mood, irregular]])
        pred = model.predict(data)

        st.subheader("🧠 AI Result")

        # Risk score
        risk_score = flow + fatigue + pain
        st.progress(risk_score / 30)

        if risk_score > 18:
            st.error("🔴 High Risk")
            st.write("Reason: Heavy flow + high fatigue + pain")
            st.write("👉 Recommendation: Eat iron-rich foods, consult doctor")

        elif risk_score > 12:
            st.warning("⚠️ Medium Risk")
            st.write("Reason: Moderate symptoms")
            st.write("👉 Recommendation: Maintain diet and rest")

        else:
            st.success("🟢 Low Risk")
            st.write("👉 Healthy condition, maintain lifestyle")

# ---------------- PREDICTION TAB ----------------
with tab2:
    st.header("📅 Period Prediction")

    last_period = st.date_input("Last Period Date", key="date")
    cycle_days = st.slider("Cycle Length", 20, 40, key="cycle_pred")

    if st.button("Predict Next Period"):

        next_period = last_period + timedelta(days=cycle_days)
        ovulation = last_period + timedelta(days=cycle_days // 2)

        st.success(f"📅 Next Period Date: {next_period}")
        st.info(f"🌼 Ovulation Window: Around {ovulation}")

# ---------------- DASHBOARD TAB ----------------
with tab3:
    st.header("📊 Health Dashboard")

    df = pd.read_csv("data.csv")

    st.subheader("Cycle Length Trend")
    st.line_chart(df["cycle_length"])

    st.subheader("Pain Level Distribution")
    st.bar_chart(df["pain_level"])

    st.subheader("Fatigue vs Flow")
    st.scatter_chart(df[["fatigue", "flow"]])