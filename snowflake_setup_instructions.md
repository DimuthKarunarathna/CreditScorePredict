
# Snowflake Setup Instructions

## Prerequisites
1. Snowflake account with appropriate privileges
2. Python environment with Snowpark installed
3. Model files generated from training

## Step 1: Install Snowpark
```bash
pip install snowflake-snowpark-python
```

## Step 2: Configure Snowflake
1. Create a new database for credit risk analysis
2. Create a warehouse for processing
3. Create a schema for the project
4. Create a stage for model files

```sql
-- Create database
CREATE DATABASE CREDIT_RISK_DB;

-- Create warehouse
CREATE WAREHOUSE CREDIT_RISK_WH 
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE;

-- Create schema
CREATE SCHEMA CREDIT_RISK_DB.CREDIT_RISK_SCHEMA;

-- Create stage for model files
CREATE STAGE CREDIT_RISK_DB.CREDIT_RISK_SCHEMA.MODEL_STAGE;
```

## Step 3: Upload Model Files
Upload the following files to the MODEL_STAGE:
- credit_risk_model_gradient_boosting.joblib
- credit_risk_scaler.joblib
- credit_risk_label_encoders.joblib
- credit_risk_target_encoder.joblib
- credit_risk_features.joblib

## Step 4: Run Deployment Script
1. Update connection parameters in snowpark_deployment.py
2. Run the deployment script:
```bash
python snowpark_deployment.py
```

## Step 5: Test the Deployment
```sql
-- Test prediction UDF
SELECT predict_credit_risk(35, 'male', 2, 'own', 'moderate', 'moderate', 5000, 24, 'car');

-- Run batch prediction
CALL BATCH_CREDIT_RISK_PREDICTION();

-- Check results
SELECT * FROM CREDIT_PREDICTIONS LIMIT 10;
```

## Step 6: Monitor and Maintain
1. Set up monitoring for UDF performance
2. Schedule regular model retraining
3. Update model files in the stage
4. Monitor prediction accuracy

## Troubleshooting
- Check warehouse status: `SHOW WAREHOUSES;`
- Check UDF status: `SHOW USER FUNCTIONS;`
- Check stage contents: `LIST @MODEL_STAGE;`
- Check logs: `SELECT * FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) ORDER BY START_TIME DESC LIMIT 10;`
