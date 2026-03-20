import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

# -------------------------------
# 🎨 App Title & Intro
# -------------------------------
st.markdown(
    "<h1 style='color:#C0392B; text-align:center;'>❤️ Heart Attack Risk Predictor</h1>", 
    unsafe_allow_html=True
)
st.markdown("<h4 style='text-align:center; color:gray;'>Know your risk, take charge of your heart</h4>", unsafe_allow_html=True)
st.divider()

# -------------------------------
# 📂 Load & Prepare Data
# -------------------------------
data = pd.read_csv("cardiovascular_risk_dataset.csv")

le = LabelEncoder()
data["smoking_status"] = le.fit_transform(data["smoking_status"])
data["family_history_heart_disease"] = le.fit_transform(data["family_history_heart_disease"])

x = data.drop(columns=["risk_category","heart_disease_risk_score"], axis=1)
y = data["risk_category"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
lr = LogisticRegression()
lr.fit(x_train, y_train)

X = data.drop(columns=["risk_category","heart_disease_risk_score"], axis=1)
Y = data["heart_disease_risk_score"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
l = LinearRegression()
l.fit(X_train, Y_train)

# -------------------------------
# 🧍 Demographics
# -------------------------------
st.markdown("### <span style='color:#2E86C1'>Demographics</span>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
with col2:
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=120, value=80)

st.divider()

# -------------------------------
# 🩺 Medical Measurements
# -------------------------------
st.markdown("### <span style='color:#2E86C1'>Medical Measurements</span>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
    resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=150, value=70)
with col4:
    family_history = st.number_input("Family History of Heart Disease (1=Yes,0=No)", min_value=0, max_value=1, value=0)
    diet_quality = st.slider("Diet Quality Score (1=Poor,10=Excellent)", 1, 10, 6)

st.divider()

# -------------------------------
# 🏃 Lifestyle Factors
# -------------------------------
with st.expander("Lifestyle Factors"):
    smoking_status = st.number_input("Smoking Status (2=Non-smoker,1=Former,0=Current)", min_value=0, max_value=2, value=0)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000)
    stress_level = st.slider("Stress Level (1=Low,10=High)", 1, 10, 5)
    physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=40.0, value=3.0)
    sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
    alcohol_units = st.number_input("Alcohol Units per Week", min_value=0.0, max_value=50.0, value=5.0)

st.divider()

# -------------------------------
# 📊 Prediction
# -------------------------------
user_input = np.array([[
    age, bmi, systolic_bp, diastolic_bp, cholesterol, resting_hr,
    smoking_status, daily_steps, stress_level, physical_activity,
    sleep_hours, family_history, diet_quality, alcohol_units
]])

predict = lr.predict(user_input)
predict1 = l.predict(user_input)

if st.button("🔍 Predict Risk",use_container_width=True):
    st.subheader("Risk of Heart Attack")

    if predict == "Low":
        st.success(f"✅ Low Risk\nPredicted Score: {predict1[0]:.2f}%")
        st.progress(int(predict1[0]))
    elif predict == "Medium":
        st.warning(f"⚠️ Medium Risk\nPredicted Score: {predict1[0]:.2f}%")
        st.progress(int(predict1[0]))
    else:
        st.error(f"❌ High Risk\nPredicted Score: {predict1[0]:.2f}%")
        st.progress(int(predict1[0]))

st.divider()

# -------------------------------
# ⚠️ Disclaimer
# -------------------------------
st.info("""
⚠️ **Disclaimer**  
This app is for **educational purposes only**.  
It provides an estimate of heart attack risk but is **not a medical diagnosis**.  
Always consult a doctor for medical advice.
""")
