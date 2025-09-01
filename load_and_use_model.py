#!/usr/bin/env python3
"""
Load and Use the Saved Credit Risk Prediction Model
This script demonstrates how to load the saved model and make predictions.
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
        print("Successfully loaded model using joblib!")
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
        print("Successfully loaded model using pickle!")
    
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

def main():
    print("=" * 60)
    print("CREDIT RISK PREDICTION MODEL - LOAD AND USE")
    print("=" * 60)
    
    # Load the model and components
    model, scaler, label_encoders, target_encoder, feature_columns = load_model_and_components()
    
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns}")
    print(f"Target classes: {target_encoder.classes_}")
    
    # Test cases
    test_cases = [
        {
            'name': 'Low Risk Customer',
            'age': 35,
            'sex': 'male',
            'job': 2,
            'housing': 'own',
            'saving_accounts': 'moderate',
            'checking_account': 'moderate',
            'credit_amount': 3000,
            'duration': 18,
            'purpose': 'car'
        },
        {
            'name': 'High Risk Customer',
            'age': 22,
            'sex': 'male',
            'job': 0,
            'housing': 'rent',
            'saving_accounts': 'little',
            'checking_account': 'little',
            'credit_amount': 15000,
            'duration': 48,
            'purpose': 'business'
        },
        {
            'name': 'Medium Risk Customer',
            'age': 45,
            'sex': 'female',
            'job': 1,
            'housing': 'own',
            'saving_accounts': 'rich',
            'checking_account': 'rich',
            'credit_amount': 8000,
            'duration': 24,
            'purpose': 'education'
        }
    ]
    
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        
        # Print customer details
        for key, value in test_case.items():
            if key != 'name':
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Make prediction
        risk, probability = predict_credit_risk(
            test_case['age'],
            test_case['sex'],
            test_case['job'],
            test_case['housing'],
            test_case['saving_accounts'],
            test_case['checking_account'],
            test_case['credit_amount'],
            test_case['duration'],
            test_case['purpose'],
            model, scaler, label_encoders, target_encoder, feature_columns
        )
        
        print(f"\n  Predicted Risk: {risk.upper()}")
        if probability is not None:
            print("  Prediction Probabilities:")
            for j, risk_class in enumerate(target_encoder.classes_):
                print(f"    {risk_class}: {probability[0][j]:.4f}")
        
        # Add interpretation
        if risk == 'good':
            print("  Interpretation: This customer is likely to be a good credit risk.")
        else:
            print("  Interpretation: This customer may be a higher credit risk.")
    
    print("\n" + "=" * 60)
    print("MODEL USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. Load the model using the load_model_and_components() function")
    print("2. Use predict_credit_risk() function to make predictions")
    print("3. The model expects the following input features:")
    print("   - Age: integer (19-75)")
    print("   - Sex: 'male' or 'female'")
    print("   - Job: integer (0-3, where 0=unemployed)")
    print("   - Housing: 'own', 'rent', or 'free'")
    print("   - Saving accounts: 'little', 'moderate', 'rich', 'quite rich', or 'no_inf'")
    print("   - Checking account: 'little', 'moderate', 'rich', or 'no_inf'")
    print("   - Credit amount: integer (250-18424)")
    print("   - Duration: integer (4-72 months)")
    print("   - Purpose: 'car', 'business', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'domestic appliances', or 'vacation/others'")
    print("\n4. The model returns:")
    print("   - Risk prediction: 'good' or 'bad'")
    print("   - Probability scores for each risk class")

if __name__ == "__main__":
    main()
