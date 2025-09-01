"""
Streamlit app for Credit Risk Prediction using trained model and pipeline
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and pipeline components
def load_components():
    model = joblib.load("credit_risk_model_gradient_boosting.joblib")
    scaler = joblib.load("credit_risk_scaler.joblib")
    label_encoders = joblib.load("credit_risk_label_encoders.joblib")
    target_encoder = joblib.load("credit_risk_target_encoder.joblib")
    feature_columns = joblib.load("credit_risk_features.joblib")
    return model, scaler, label_encoders, target_encoder, feature_columns

def preprocess_input(data, label_encoders, scaler, feature_columns):
    df = pd.DataFrame([data])
    # Feature engineering (as in training)
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['Young', 'Young Adult', 'Adult', 'Senior'])
    df['Credit_Amount_Category'] = pd.cut(df['Credit amount'], bins=[0, 2000, 5000, 10000, 20000], labels=['Low', 'Medium', 'High', 'Very High'])
    df['Duration_Category'] = pd.cut(df['Duration'], bins=[0, 12, 24, 48, 100], labels=['Short', 'Medium', 'Long', 'Very Long'])
    df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
    df['Checking account'] = df['Checking account'].fillna('no_inf')
    for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_Category', 'Credit_Amount_Category', 'Duration_Category']:
        if col in label_encoders:
            df[col + '_encoded'] = label_encoders[col].transform(df[col])
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    return X_scaled

def main():
    st.title("Credit Risk Prediction App")
    st.write("Enter customer details to predict credit risk.")

    st.sidebar.header("Load Example/Test Case")
    if st.sidebar.button("Fill Example Case"):
        st.session_state['age'] = 35
        st.session_state['sex'] = 'male'
        st.session_state['job'] = 3
        st.session_state['housing'] = 'own'
        st.session_state['saving_accounts'] = 'rich'
        st.session_state['checking_account'] = 'rich'
        st.session_state['credit_amount'] = 3000
        st.session_state['duration'] = 12
        st.session_state['purpose'] = 'car'

    # Use session state for input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.get('age', 30), key='age')
    sex = st.selectbox("Sex", ["male", "female"], index=["male", "female"].index(st.session_state.get('sex', 'male')), key='sex')
    job = st.selectbox("Job (0=unemployed, 3=skilled)", [0, 1, 2, 3], index=[0, 1, 2, 3].index(st.session_state.get('job', 0)), key='job')
    housing = st.selectbox("Housing", ["own", "rent", "free"], index=["own", "rent", "free"].index(st.session_state.get('housing', 'own')), key='housing')
    saving_accounts = st.selectbox("Saving accounts", ["no_inf", "little", "moderate", "quite rich", "rich"], index=["no_inf", "little", "moderate", "quite rich", "rich"].index(st.session_state.get('saving_accounts', 'no_inf')), key='saving_accounts')
    checking_account = st.selectbox("Checking account", ["no_inf", "little", "moderate", "rich"], index=["no_inf", "little", "moderate", "rich"].index(st.session_state.get('checking_account', 'no_inf')), key='checking_account')
    credit_amount = st.number_input("Credit amount", min_value=100, max_value=20000, value=st.session_state.get('credit_amount', 2000), key='credit_amount')
    duration = st.number_input("Duration (months)", min_value=4, max_value=72, value=st.session_state.get('duration', 12), key='duration')
    purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "education", "business", "domestic appliance", "repairs", "vacation/others"], index=["car", "furniture/equipment", "radio/TV", "education", "business", "domestic appliance", "repairs", "vacation/others"].index(st.session_state.get('purpose', 'car')), key='purpose')

    if st.button("Predict Risk"):
        model, scaler, label_encoders, target_encoder, feature_columns = load_components()
        input_data = {
            'Age': age,
            'Sex': sex,
            'Job': job,
            'Housing': housing,
            'Saving accounts': saving_accounts,
            'Checking account': checking_account,
            'Credit amount': credit_amount,
            'Duration': duration,
            'Purpose': purpose
        }
        X_scaled = preprocess_input(input_data, label_encoders, scaler, feature_columns)
        prediction = model.predict(X_scaled)
        probability = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        risk = target_encoder.inverse_transform(prediction)[0]
        st.subheader(f"Predicted Risk: {risk.upper()}")
        if probability is not None:
            st.write("Prediction Probabilities:")
            for i, risk_class in enumerate(target_encoder.classes_):
                st.write(f"{risk_class}: {probability[0][i]:.2%}")

if __name__ == "__main__":
    main()
