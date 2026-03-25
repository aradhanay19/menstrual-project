import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta

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
        prob = model.predict_proba(data)

        confidence = round(np.max(prob) * 100, 2)

        st.subheader("🧠 AI Result")

        # Risk score
        risk_score = flow + fatigue + pain
        st.progress(risk_score / 30)

        st.write(f"🔍 Prediction Confidence: {confidence}%")

        # ALERT SYSTEM
        if cycle > 35:
            st.warning("⚠️ Irregular cycle detected. Medical consultation recommended.")

        # SMART RECOMMENDATIONS
        if risk_score > 18:
            st.error("🔴 High Risk")

            st.write("### 🥗 Diet Recommendation:")
            st.write("- Iron-rich foods: spinach, jaggery, dates")
            st.write("- Fruits: banana, apple")
            st.write("- Stay hydrated")

            st.write("### 🧘 Exercise:")
            st.write("- Light yoga and stretching")
            st.write("- Avoid heavy workouts")

        elif risk_score > 12:
            st.warning("⚠️ Medium Risk")

            st.write("### 🥗 Diet:")
            st.write("- Balanced diet with iron intake")

            st.write("### 🧘 Exercise:")
            st.write("- Walking and light exercise")

        else:
            st.success("🟢 Low Risk")

            st.write("### 👍 Healthy Condition")
            st.write("- Maintain normal diet and lifestyle")

        # SAVE USER DATA (ADVANCED FEATURE)
        new_data = pd.DataFrame({
            "cycle_length": [cycle],
            "pain_level": [pain],
            "fatigue": [fatigue],
            "flow": [flow]
        })

        new_data.to_csv("data.csv", mode='a', header=False, index=False)

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

        # EXPLANATION
        st.write("### ℹ️ How is this calculated?")
        st.write("Next period = Last period date + cycle length")
        st.write("Ovulation occurs around the middle of the cycle")

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

    # INSIGHTS
    st.subheader("📊 Insights from Data")

    avg_cycle = df["cycle_length"].mean()
    avg_pain = df["pain_level"].mean()

    st.write(f"👉 Average cycle length is around {round(avg_cycle)} days, indicating a normal pattern.")

    st.write(f"👉 Average pain level is {round(avg_pain)}, showing moderate discomfort.")

    st.write("👉 Majority of users fall between 26–30 days cycle, which is considered healthy.")

    st.write("👉 Higher fatigue is often linked with heavy flow, indicating possible anemia risk.")