import numpy as np
import pandas as pd
import streamlit as st

st.title("Heart Attack Risk Predictor")
st.markdown("Know your risk, take charge of your heart ❤️")

data = pd.read_csv("cardiovascular_risk_dataset.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["smoking_status"]=le.fit_transform(data["smoking_status"])
data["family_history_heart_disease"]=le.fit_transform(data["family_history_heart_disease"])

x = data.drop(columns = ["risk_category","heart_disease_risk_score"] , axis = 1)
y = data["risk_category"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42,test_size = 0.2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

X = data.drop(columns = ["risk_category","heart_disease_risk_score"] , axis = 1)
Y = data["heart_disease_risk_score"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 42,test_size = 0.2)

from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(X_train,Y_train)

# Demographics
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)

# Blood Pressure
systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=120, value=80)

# Medical Measurements
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=150, value=70)

# Lifestyle Factors
smoking_status = st.number_input("Smoking Status (2 = Non-smoker ,1 = Former smoker , 0 = Current smoker )",min_value=0, max_value=2, value=0)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000)
stress_level = st.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 5)
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=40.0, value=3.0)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)

# Family & Diet
family_history = st.number_input("Family History of Heart Disease (1 = Yes, 0 = No)",min_value=0, max_value=1, value=0)
diet_quality = st.slider("Diet Quality Score (1 = Poor, 10 = Excellent)", 1, 10, 6)
alcohol_units = st.number_input("Alcohol Units per Week", min_value=0.0, max_value=50.0, value=5.0)

user_input = np.array([[
    age,
    bmi,
    systolic_bp,
    diastolic_bp,
    cholesterol,
    resting_hr,
    smoking_status,
    daily_steps,
    stress_level,
    physical_activity,
    sleep_hours,
    family_history,
    diet_quality,
    alcohol_units
]])

predict = lr.predict(user_input)
predict1 = l.predict(user_input)

if st.button("Predict Risk"):
    st.subheader("Risk of Heart Attack")

    if predict == "Low":
        st.success(predict[0])
        st.success(f"Predicted Risk: {predict1[0]:.2f}%")

    elif predict == "Medium":
        st.warning(predict[0])
        st.warning(f"Predicted Risk: {predict1[0]:.2f}%")

    else:
        st.error(predict[0])
        st.error(f"Predicted Risk: {predict1[0]:.2f}%")
st.write("---")
st.markdown("""
⚠️ **Disclaimer**

This app is designed for **educational purposes only**.  
It provides an estimate of heart attack risk based on the information you enter, but it is **not a medical diagnosis**.  
Your results should be seen as a guide to better understand how lifestyle and health factors may influence heart health.  

💡 *Think of this tool as a way to learn more about your numbers and habits — but always rely on your doctor for medical decisions.*
""")

        




