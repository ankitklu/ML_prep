import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Decision_Tree_heart.pkl')
scaler = joblib.load('Scaler_heart.pkl')
columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction')
st.write('Enter patient details to estimate the risk of heart disease.')

age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex', options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: {
    0: 'Typical angina', 1: 'Atypical angina', 2: 'Non-anginal pain', 3: 'Asymptomatic'
}[x])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=220, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2], format_func=lambda x: {
    0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'
}[x])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: {
    0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'
}[x])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3], format_func=lambda x: {
    0: 'Unknown', 1: 'Normal', 2: 'Fixed defect', 3: 'Reversible defect'
}[x])

if st.button('Predict'):
    input_row = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    input_df = pd.DataFrame([input_row])[columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f'High risk of heart disease (confidence: {proba:.1%})')
    else:
        st.success(f'Low risk of heart disease (confidence: {1 - proba:.1%})')
