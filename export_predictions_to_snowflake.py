#!/usr/bin/env python3
"""
Export Credit Risk Predictions for Snowflake Integration
This script generates predictions for the entire dataset and exports them to CSV/Parquet files
for loading into Snowflake.
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import os

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

def create_risk_target(df):
    """Create risk target based on business logic"""
    conditions = [
        # High risk conditions
        (df['Credit amount'] > 10000) & (df['Duration'] > 36),
        (df['Age'] < 25) & (df['Job'] == 0),
        (df['Saving accounts'] == 'little') & (df['Checking account'] == 'little'),
        (df['Credit amount'] > 15000) & (df['Duration'] > 48),
        (df['Job'] == 0) & (df['Housing'] == 'rent'),
        
        # Medium risk conditions
        (df['Credit amount'] > 8000) & (df['Duration'] > 24),
        (df['Age'] < 30) & (df['Job'] <= 1),
        (df['Saving accounts'] == 'little') | (df['Checking account'] == 'little')
    ]
    
    choices = ['bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad']
    df['Risk'] = np.select(conditions, choices, default='good')
    return df

def preprocess_data(df):
    """Apply the same preprocessing as during training"""
    
    # Create risk target if not exists
    if 'Risk' not in df.columns:
        df = create_risk_target(df)
    
    # Create feature categories
    df['Age_Category'] = pd.cut(
        df['Age'], 
        bins=[0, 30, 40, 50, 100], 
        labels=['Young', 'Young Adult', 'Adult', 'Senior']
    )
    
    df['Credit_Amount_Category'] = pd.cut(
        df['Credit amount'], 
        bins=[0, 2000, 5000, 10000, 20000], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    df['Duration_Category'] = pd.cut(
        df['Duration'], 
        bins=[0, 12, 24, 48, 100], 
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    
    # Handle missing values
    df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
    df['Checking account'] = df['Checking account'].fillna('no_inf')
    
    return df

def generate_predictions_for_dataset():
    """Generate predictions for the entire dataset and export to files"""
    
    print("=" * 60)
    print("EXPORTING PREDICTIONS FOR SNOWFLAKE INTEGRATION")
    print("=" * 60)
    
    # Load model and components
    model, scaler, label_encoders, target_encoder, feature_columns = load_model_and_components()
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("german_credit_data.csv")
    print(f"Dataset loaded: {len(df)} records")
    
    # Preprocess the data
    print("Preprocessing data...")
    df_processed = preprocess_data(df.copy())
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 
                 'Purpose', 'Age_Category', 'Credit_Amount_Category', 'Duration_Category']:
        if col in label_encoders:
            df_processed[col + '_encoded'] = label_encoders[col].transform(df_processed[col])
    
    # Select features for prediction
    X = df_processed[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Decode predictions
    risk_predictions = target_encoder.inverse_transform(predictions)
    
    # Create results DataFrame
    print("Creating results DataFrame...")
    results_df = pd.DataFrame({
        'customer_id': range(1, len(df) + 1),
        'predicted_label': predictions,  # 0 = good, 1 = bad
        'predicted_risk': risk_predictions,
        'probability_good': probabilities[:, 0],
        'probability_bad': probabilities[:, 1],
        'timestamp': datetime.now()
    })
    
    # Add original features for optional storage
    feature_df = df.copy()
    feature_df['customer_id'] = range(1, len(df) + 1)
    feature_df['timestamp'] = datetime.now()
    
    # Export to CSV
    print("Exporting to CSV files...")
    results_df.to_csv('credit_risk_scores.csv', index=False)
    feature_df.to_csv('customer_features.csv', index=False)
    
    # Export to Parquet (more efficient for large datasets)
    print("Exporting to Parquet files...")
    results_df.to_parquet('credit_risk_scores.parquet', index=False)
    feature_df.to_parquet('customer_features.parquet', index=False)
    
    # Create summary statistics
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total customers: {len(results_df)}")
    print(f"Good risk predictions: {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"Bad risk predictions: {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
    print(f"Average probability (good): {results_df['probability_good'].mean():.3f}")
    print(f"Average probability (bad): {results_df['probability_bad'].mean():.3f}")
    
    # Risk distribution by age
    print("\nRisk Distribution by Age Category:")
    age_risk = pd.DataFrame({
        'Age': df['Age'],
        'Predicted_Risk': risk_predictions
    })
    age_risk['Age_Category'] = pd.cut(age_risk['Age'], bins=[0, 30, 40, 50, 100], labels=['Young', 'Young Adult', 'Adult', 'Senior'])
    age_risk_summary = age_risk.groupby('Age_Category')['Predicted_Risk'].value_counts().unstack(fill_value=0)
    print(age_risk_summary)
    
    # Risk distribution by credit amount
    print("\nRisk Distribution by Credit Amount Category:")
    credit_risk = pd.DataFrame({
        'Credit_Amount': df['Credit amount'],
        'Predicted_Risk': risk_predictions
    })
    credit_risk['Amount_Category'] = pd.cut(credit_risk['Credit_Amount'], bins=[0, 2000, 5000, 10000, 20000], labels=['Low', 'Medium', 'High', 'Very High'])
    credit_risk_summary = credit_risk.groupby('Amount_Category')['Predicted_Risk'].value_counts().unstack(fill_value=0)
    print(credit_risk_summary)
    
    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)
    print("1. credit_risk_scores.csv - Main predictions file for Snowflake")
    print("2. credit_risk_scores.parquet - Parquet version (more efficient)")
    print("3. customer_features.csv - Original features for optional storage")
    print("4. customer_features.parquet - Parquet version of features")
    
    return results_df, feature_df

def create_snowflake_sql_scripts():
    """Create SQL scripts for Snowflake table creation and data loading"""
    
    print("\n" + "=" * 60)
    print("CREATING SNOWFLAKE SQL SCRIPTS")
    print("=" * 60)
    
    # SQL for creating the main risk scores table
    risk_scores_sql = """
-- Create the main credit risk scores table
CREATE OR REPLACE TABLE CREDIT_RISK_SCORES (
    customer_id INT,
    predicted_label INT,   -- 1 = default/bad, 0 = repaid/good
    predicted_risk STRING, -- 'good' or 'bad'
    probability_good FLOAT,
    probability_bad FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the customer features table (optional, for richer dashboards)
CREATE OR REPLACE TABLE CUSTOMER_FEATURES (
    customer_id INT,
    age INT,
    sex STRING,
    job INT,
    housing STRING,
    saving_accounts STRING,
    checking_account STRING,
    credit_amount INT,
    duration INT,
    purpose STRING,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create a view that combines both tables
CREATE OR REPLACE VIEW CREDIT_RISK_DASHBOARD AS
SELECT 
    cf.*,
    crs.predicted_label,
    crs.predicted_risk,
    crs.probability_good,
    crs.probability_bad,
    crs.timestamp as prediction_timestamp
FROM CUSTOMER_FEATURES cf
JOIN CREDIT_RISK_SCORES crs ON cf.customer_id = crs.customer_id;
"""
    
    # SQL for loading data from CSV files
    load_data_sql = """
-- Load credit risk scores from CSV
COPY INTO CREDIT_RISK_SCORES (customer_id, predicted_label, predicted_risk, probability_good, probability_bad, timestamp)
FROM @your_stage/credit_risk_scores.csv
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"');

-- Load customer features from CSV
COPY INTO CUSTOMER_FEATURES (customer_id, age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose, timestamp)
FROM @your_stage/customer_features.csv
FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"');

-- Alternative: Load from Parquet files (more efficient)
COPY INTO CREDIT_RISK_SCORES (customer_id, predicted_label, predicted_risk, probability_good, probability_bad, timestamp)
FROM @your_stage/credit_risk_scores.parquet
FILE_FORMAT = (TYPE = 'PARQUET');

COPY INTO CUSTOMER_FEATURES (customer_id, age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose, timestamp)
FROM @your_stage/customer_features.parquet
FILE_FORMAT = (TYPE = 'PARQUET');
"""
    
    # SQL for useful queries
    useful_queries_sql = """
-- Query 1: Risk distribution
SELECT 
    predicted_risk,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM CREDIT_RISK_SCORES
GROUP BY predicted_risk
ORDER BY customer_count DESC;

-- Query 2: Risk by age group
SELECT 
    CASE 
        WHEN age < 30 THEN 'Young (<30)'
        WHEN age < 40 THEN 'Young Adult (30-39)'
        WHEN age < 50 THEN 'Adult (40-49)'
        ELSE 'Senior (50+)'
    END as age_group,
    predicted_risk,
    COUNT(*) as customer_count
FROM CREDIT_RISK_DASHBOARD
GROUP BY age_group, predicted_risk
ORDER BY age_group, predicted_risk;

-- Query 3: Risk by credit amount range
SELECT 
    CASE 
        WHEN credit_amount < 2000 THEN 'Low (<2K)'
        WHEN credit_amount < 5000 THEN 'Medium (2K-5K)'
        WHEN credit_amount < 10000 THEN 'High (5K-10K)'
        ELSE 'Very High (10K+)'
    END as credit_range,
    predicted_risk,
    COUNT(*) as customer_count
FROM CREDIT_RISK_DASHBOARD
GROUP BY credit_range, predicted_risk
ORDER BY credit_range, predicted_risk;

-- Query 4: Average probability by risk level
SELECT 
    predicted_risk,
    ROUND(AVG(probability_good), 3) as avg_probability_good,
    ROUND(AVG(probability_bad), 3) as avg_probability_bad
FROM CREDIT_RISK_SCORES
GROUP BY predicted_risk;
"""
    
    # Write SQL files
    with open('snowflake_create_tables.sql', 'w') as f:
        f.write(risk_scores_sql)
    
    with open('snowflake_load_data.sql', 'w') as f:
        f.write(load_data_sql)
    
    with open('snowflake_useful_queries.sql', 'w') as f:
        f.write(useful_queries_sql)
    
    print("SQL files created:")
    print("1. snowflake_create_tables.sql - Table creation scripts")
    print("2. snowflake_load_data.sql - Data loading scripts")
    print("3. snowflake_useful_queries.sql - Useful queries for analysis")

def main():
    """Main function to run the export process"""
    
    # Generate predictions and export files
    results_df, feature_df = generate_predictions_for_dataset()
    
    # Create Snowflake SQL scripts
    create_snowflake_sql_scripts()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR SNOWFLAKE INTEGRATION")
    print("=" * 60)
    print("1. Upload the CSV/Parquet files to your Snowflake stage")
    print("2. Run the SQL scripts in snowflake_create_tables.sql")
    print("3. Run the data loading scripts in snowflake_load_data.sql")
    print("4. Use the queries in snowflake_useful_queries.sql for analysis")
    print("5. Connect Power BI to Snowflake using the native connector")
    print("6. Build dashboards using the CREDIT_RISK_DASHBOARD view")

if __name__ == "__main__":
    main()
