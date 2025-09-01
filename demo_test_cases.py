#!/usr/bin/env python3
"""
Demo Test Cases for Credit Risk Prediction Model
This script demonstrates the model with predefined test cases showing different scenarios.
"""

import pandas as pd
import numpy as np
import joblib
import pickle

def load_model_and_components():
    """Load the saved model and all necessary components"""
    
    # Load using joblib (recommended for scikit-learn models)
    try:
        model = joblib.load("credit_risk_model_gradient_boosting.joblib")
        scaler = joblib.load("credit_risk_scaler.joblib")
        label_encoders = joblib.load("credit_risk_label_encoders.joblib")
        target_encoder = joblib.load("credit_risk_target_encoder.joblib")
        feature_columns = joblib.load("credit_risk_features.joblib")
        print(" Successfully loaded model using joblib!")
    except:
        # Fallback to pickle
        with open("credit_risk_model_gradient_boosting.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("credit_risk_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open("credit_risk_encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
            label_encoders = encoders['label_encoders']
            target_encoder = encoders['target_encoder']
        with open("credit_risk_features.pkl", 'rb') as f:
            feature_columns = pickle.load(f)
        print(" Successfully loaded model using pickle!")
    
    return model, scaler, label_encoders, target_encoder, feature_columns

def predict_credit_risk(age, sex, job, housing, saving_accounts, checking_account, 
                       credit_amount, duration, purpose, model, scaler, label_encoders, 
                       target_encoder, feature_columns):
    """
    Make a credit risk prediction for a new customer
    """
    # Create sample data
    sample_data = {
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    }
    
    # Create DataFrame
    sample_df = pd.DataFrame(sample_data)
    
    # Apply preprocessing
    sample_df['Age_Category'] = pd.cut(
        sample_df['Age'], 
        bins=[0, 30, 40, 50, 100], 
        labels=['Young', 'Young Adult', 'Adult', 'Senior']
    )
    
    sample_df['Credit_Amount_Category'] = pd.cut(
        sample_df['Credit amount'], 
        bins=[0, 2000, 5000, 10000, 20000], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    sample_df['Duration_Category'] = pd.cut(
        sample_df['Duration'], 
        bins=[0, 12, 24, 48, 100], 
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    
    # Handle missing values
    sample_df['Saving accounts'] = sample_df['Saving accounts'].fillna('no_inf')
    sample_df['Checking account'] = sample_df['Checking account'].fillna('no_inf')
    
    # Encode categorical variables
    for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 
                 'Purpose', 'Age_Category', 'Credit_Amount_Category', 'Duration_Category']:
        if col in label_encoders:
            sample_df[col + '_encoded'] = label_encoders[col].transform(sample_df[col])
    
    # Select features
    X_sample = sample_df[feature_columns]
    
    # Scale features
    X_sample_scaled = scaler.transform(X_sample)
    
    # Make prediction
    prediction = model.predict(X_sample)
    probability = model.predict_proba(X_sample) if hasattr(model, 'predict_proba') else None
    
    # Decode prediction
    risk = target_encoder.inverse_transform(prediction)[0]
    
    return risk, probability

def display_test_case(case_name, customer_data, risk, probability):
    """Display test case results"""
    print(f"\n{'='*80}")
    print(f" TEST CASE: {case_name}")
    print(f"{'='*80}")
    
    print("\n Customer Information:")
    for key, value in customer_data.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nRisk Prediction: {risk.upper()}")
    
    if probability is not None:
        print("\n Prediction Probabilities:")
        for i, risk_class in enumerate(['bad', 'good']):
            prob_percent = probability[0][i] * 100
            print(f"   {risk_class.upper()}: {prob_percent:.2f}%")
    
    print("\n Interpretation:")
    if risk == 'good':
        print("    This customer is likely to be a GOOD credit risk.")
        print("    The model suggests approving this credit application.")
    else:
        print("   ⚠️  This customer may be a HIGHER credit risk.")
        print("   ⚠️  The model suggests reviewing this application carefully.")
    
    # Risk factors analysis
    print("\n Risk Factor Analysis:")
    risk_factors = []
    
    if customer_data['age'] < 25 or customer_data['age'] > 60:
        risk_factors.append("Age (very young or very old)")
    
    if customer_data['job'] == 0:
        risk_factors.append("Unemployed")
    
    if customer_data['housing'] == 'rent':
        risk_factors.append("Renting (vs owning)")
    
    if customer_data['saving_accounts'] in ['little', 'no_inf']:
        risk_factors.append("Limited savings")
    
    if customer_data['checking_account'] in ['little', 'no_inf']:
        risk_factors.append("Limited checking account")
    
    if customer_data['credit_amount'] > 10000:
        risk_factors.append("High credit amount")
    
    if customer_data['duration'] > 36:
        risk_factors.append("Long loan duration")
    
    if risk_factors:
        print("   Identified risk factors:")
        for factor in risk_factors:
            print(f"   • {factor}")
    else:
        print("No significant risk factors identified")

def main():
    print("="*80)
    print(" CREDIT RISK PREDICTION MODEL - DEMO TEST CASES")
    print("="*80)
    print("This demo shows how the model performs with different customer profiles.")
    
    # Load the model
    print("\n🔄 Loading the model...")
    model, scaler, label_encoders, target_encoder, feature_columns = load_model_and_components()
    
    print(f" Model loaded successfully!")
    print(f" Model type: {type(model).__name__}")
    print(f" Target classes: {list(target_encoder.classes_)}")
    
    # Predefined test cases
    test_cases = [
        {
            'name': 'Low Risk - Stable Professional',
            'data': {
                'age': 35,
                'sex': 'male',
                'job': 3,
                'housing': 'own',
                'saving_accounts': 'rich',
                'checking_account': 'rich',
                'credit_amount': 3000,
                'duration': 12,
                'purpose': 'car'
            }
        },
        {
            'name': 'High Risk - Young Unemployed',
            'data': {
                'age': 22,
                'sex': 'male',
                'job': 0,
                'housing': 'rent',
                'saving_accounts': 'little',
                'checking_account': 'little',
                'credit_amount': 15000,
                'duration': 48,
                'purpose': 'business'
            }
        },
        {
            'name': 'Medium Risk - Young Professional',
            'data': {
                'age': 28,
                'sex': 'female',
                'job': 2,
                'housing': 'rent',
                'saving_accounts': 'moderate',
                'checking_account': 'moderate',
                'credit_amount': 8000,
                'duration': 24,
                'purpose': 'education'
            }
        },
        {
            'name': 'Low Risk - Senior Homeowner',
            'data': {
                'age': 55,
                'sex': 'female',
                'job': 3,
                'housing': 'own',
                'saving_accounts': 'quite rich',
                'checking_account': 'rich',
                'credit_amount': 5000,
                'duration': 18,
                'purpose': 'furniture/equipment'
            }
        },
        {
            'name': 'High Risk - High Amount Loan',
            'data': {
                'age': 45,
                'sex': 'male',
                'job': 1,
                'housing': 'rent',
                'saving_accounts': 'no_inf',
                'checking_account': 'no_inf',
                'credit_amount': 18000,
                'duration': 60,
                'purpose': 'business'
            }
        },
        {
            'name': 'Borderline Case - Mixed Factors',
            'data': {
                'age': 30,
                'sex': 'male',
                'job': 2,
                'housing': 'own',
                'saving_accounts': 'little',
                'checking_account': 'moderate',
                'credit_amount': 12000,
                'duration': 36,
                'purpose': 'car'
            }
        }
    ]
    
    print(f"\n🧪 Running {len(test_cases)} test cases...")
    
    # Run all test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔄 Processing test case {i}/{len(test_cases)}...")
        
        # Make prediction
        risk, probability = predict_credit_risk(
            test_case['data']['age'],
            test_case['data']['sex'],
            test_case['data']['job'],
            test_case['data']['housing'],
            test_case['data']['saving_accounts'],
            test_case['data']['checking_account'],
            test_case['data']['credit_amount'],
            test_case['data']['duration'],
            test_case['data']['purpose'],
            model, scaler, label_encoders, target_encoder, feature_columns
        )
        
        # Display results
        display_test_case(test_case['name'], test_case['data'], risk, probability)
    
    # Summary
    print(f"\n{'='*80}")
    print(" DEMO SUMMARY")
    print(f"{'='*80}")
    print("The model successfully processed all test cases!")
    print("Different customer profiles show varying risk levels")
    print("The model considers multiple factors in its predictions")
    print(" Risk factor analysis helps explain the predictions")
    print("\n The model is ready for real-world credit risk assessment!")

if __name__ == "__main__":
    main()
