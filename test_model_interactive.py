#!/usr/bin/env python3
"""
Interactive Credit Risk Prediction Model Tester
This script allows you to test the model with your own input data.
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
        print("✅ Successfully loaded model using joblib!")
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
        print("✅ Successfully loaded model using pickle!")
    
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

def get_user_input():
    """Get user input for prediction"""
    print("\n" + "="*60)
    print("ENTER CUSTOMER INFORMATION")
    print("="*60)
    
    # Age
    while True:
        try:
            age = int(input("Age (19-75): "))
            if 19 <= age <= 75:
                break
            else:
                print("❌ Age must be between 19 and 75")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Sex
    while True:
        sex = input("Sex (male/female): ").lower()
        if sex in ['male', 'female']:
            break
        else:
            print("❌ Please enter 'male' or 'female'")
    
    # Job
    print("\nJob levels:")
    print("0 = unemployed")
    print("1 = unskilled and non-resident")
    print("2 = unskilled and resident")
    print("3 = skilled employee/official")
    while True:
        try:
            job = int(input("Job (0-3): "))
            if 0 <= job <= 3:
                break
            else:
                print("❌ Job must be between 0 and 3")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Housing
    print("\nHousing options:")
    print("own = own")
    print("rent = rent")
    print("free = free")
    while True:
        housing = input("Housing (own/rent/free): ").lower()
        if housing in ['own', 'rent', 'free']:
            break
        else:
            print("❌ Please enter 'own', 'rent', or 'free'")
    
    # Saving accounts
    print("\nSaving accounts options:")
    print("little = little")
    print("moderate = moderate")
    print("rich = rich")
    print("quite rich = quite rich")
    print("no_inf = no information")
    while True:
        saving_accounts = input("Saving accounts: ").lower()
        if saving_accounts in ['little', 'moderate', 'rich', 'quite rich', 'no_inf']:
            break
        else:
            print("❌ Please enter a valid option")
    
    # Checking account
    print("\nChecking account options:")
    print("little = little")
    print("moderate = moderate")
    print("rich = rich")
    print("no_inf = no information")
    while True:
        checking_account = input("Checking account: ").lower()
        if checking_account in ['little', 'moderate', 'rich', 'no_inf']:
            break
        else:
            print("❌ Please enter a valid option")
    
    # Credit amount
    while True:
        try:
            credit_amount = int(input("Credit amount (250-18424): "))
            if 250 <= credit_amount <= 18424:
                break
            else:
                print("❌ Credit amount must be between 250 and 18424")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Duration
    while True:
        try:
            duration = int(input("Duration in months (4-72): "))
            if 4 <= duration <= 72:
                break
            else:
                print("❌ Duration must be between 4 and 72 months")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Purpose
    print("\nPurpose options:")
    print("car = car")
    print("business = business")
    print("education = education")
    print("furniture/equipment = furniture/equipment")
    print("radio/TV = radio/TV")
    print("repairs = repairs")
    print("domestic appliances = domestic appliances")
    print("vacation/others = vacation/others")
    while True:
        purpose = input("Purpose: ").lower()
        if purpose in ['car', 'business', 'education', 'furniture/equipment', 'radio/tv', 'repairs', 'domestic appliances', 'vacation/others']:
            break
        else:
            print("❌ Please enter a valid purpose")
    
    return {
        'age': age,
        'sex': sex,
        'job': job,
        'housing': housing,
        'saving_accounts': saving_accounts,
        'checking_account': checking_account,
        'credit_amount': credit_amount,
        'duration': duration,
        'purpose': purpose
    }

def display_prediction(customer_data, risk, probability):
    """Display the prediction results"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print("\n📋 Customer Information:")
    for key, value in customer_data.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎯 Risk Prediction: {risk.upper()}")
    
    if probability is not None:
        print("\n📊 Prediction Probabilities:")
        for i, risk_class in enumerate(['bad', 'good']):
            prob_percent = probability[0][i] * 100
            print(f"   {risk_class.upper()}: {prob_percent:.2f}%")
    
    print("\n💡 Interpretation:")
    if risk == 'good':
        print("   ✅ This customer is likely to be a GOOD credit risk.")
        print("   ✅ The model suggests approving this credit application.")
    else:
        print("   ⚠️  This customer may be a HIGHER credit risk.")
        print("   ⚠️  The model suggests reviewing this application carefully.")
    
    # Risk factors analysis
    print("\n🔍 Risk Factor Analysis:")
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
        print("   ✅ No significant risk factors identified")

def main():
    print("="*60)
    print("🎯 CREDIT RISK PREDICTION MODEL - INTERACTIVE TESTER")
    print("="*60)
    print("This tool allows you to test the credit risk prediction model")
    print("with your own customer data.")
    
    # Load the model
    print("\n🔄 Loading the model...")
    model, scaler, label_encoders, target_encoder, feature_columns = load_model_and_components()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model).__name__}")
    print(f"🎯 Target classes: {list(target_encoder.classes_)}")
    
    while True:
        try:
            # Get user input
            customer_data = get_user_input()
            
            # Make prediction
            print("\n🔄 Making prediction...")
            risk, probability = predict_credit_risk(
                customer_data['age'],
                customer_data['sex'],
                customer_data['job'],
                customer_data['housing'],
                customer_data['saving_accounts'],
                customer_data['checking_account'],
                customer_data['credit_amount'],
                customer_data['duration'],
                customer_data['purpose'],
                model, scaler, label_encoders, target_encoder, feature_columns
            )
            
            # Display results
            display_prediction(customer_data, risk, probability)
            
            # Ask if user wants to test another customer
            print("\n" + "="*60)
            another = input("Would you like to test another customer? (y/n): ").lower()
            if another not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again.")
    
    print("\n🎉 Thank you for testing the Credit Risk Prediction Model!")

if __name__ == "__main__":
    main()

