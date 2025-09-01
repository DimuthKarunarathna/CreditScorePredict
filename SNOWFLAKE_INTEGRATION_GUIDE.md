# Credit Risk Model - Snowflake & Power BI Integration Guide

## Overview
This guide provides step-by-step instructions for deploying your trained credit risk prediction model to Snowflake and creating Power BI dashboards for real-time credit risk analysis.

## 📁 Project Structure
```
CreditScorePredict/
├── credit_risk_prediction_fixed_final.py    # Main training script
├── load_and_use_model.py                    # Model loading and testing
├── export_predictions_to_snowflake.py       # Export predictions to CSV/Parquet
├── deploy_model_to_snowflake.py             # Create Snowflake deployment files
├── snowpark_deployment.py                   # Snowflake deployment script
├── snowflake_create_tables.sql              # SQL for table creation
├── snowflake_load_data.sql                  # SQL for data loading
├── snowflake_useful_queries.sql             # Useful analysis queries
├── powerbi_connection_guide.md              # Power BI setup guide
├── snowflake_setup_instructions.md          # Snowflake setup instructions
├── credit_risk_scores.csv                   # Exported predictions (CSV)
├── credit_risk_scores.parquet               # Exported predictions (Parquet)
├── customer_features.csv                    # Customer features (CSV)
├── customer_features.parquet                # Customer features (Parquet)
└── german_credit_data.csv                   # Original dataset
```

## 🚀 Quick Start

### Step 1: Export Predictions
The predictions have already been exported. You can find them in:
- `credit_risk_scores.csv` - Main predictions file
- `credit_risk_scores.parquet` - Parquet version (more efficient)
- `customer_features.csv` - Original features for analysis

### Step 2: Set Up Snowflake
1. Follow the instructions in `snowflake_setup_instructions.md`
2. Create your Snowflake account and configure the environment
3. Run the SQL scripts in `snowflake_create_tables.sql`

### Step 3: Load Data into Snowflake
1. Upload the CSV/Parquet files to your Snowflake stage
2. Run the data loading scripts in `snowflake_load_data.sql`

### Step 4: Deploy Model (Optional)
1. Update connection parameters in `snowpark_deployment.py`
2. Run the deployment script to create UDFs for real-time inference

### Step 5: Connect Power BI
1. Follow the guide in `powerbi_connection_guide.md`
2. Connect to Snowflake and import the views
3. Create your dashboards

## 📊 Data Schema

### CREDIT_RISK_SCORES Table
```sql
CREATE TABLE CREDIT_RISK_SCORES (
    customer_id INT,
    predicted_label INT,        -- 1 = default/bad, 0 = repaid/good
    predicted_risk STRING,      -- 'good' or 'bad'
    probability_good FLOAT,     -- Probability of good risk
    probability_bad FLOAT,      -- Probability of bad risk
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### CUSTOMER_FEATURES Table
```sql
CREATE TABLE CUSTOMER_FEATURES (
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
```

## 📈 Power BI Dashboard Examples

### 1. Risk Distribution Dashboard
- **Total Applications**: Card showing total number of applications
- **Risk Distribution**: Pie chart showing good vs bad risk percentages
- **Risk by Age Group**: Bar chart showing risk distribution by age
- **Risk by Credit Amount**: Bar chart showing risk by credit amount range

### 2. Portfolio Trends Dashboard
- **Applications Over Time**: Line chart showing application volume
- **Risk Ratio Trends**: Line chart showing risk ratio changes
- **Credit Amount Distribution**: Area chart showing amount distribution

### 3. Segment Analysis Dashboard
- **Risk by Job & Housing**: Matrix showing risk by job type and housing
- **Age vs Credit Amount**: Scatter plot colored by risk
- **Risk by Purpose**: Bar chart showing risk by loan purpose
- **Overall Risk Score**: Gauge showing portfolio risk score

### 4. Real-time Application Dashboard
- **Recent Applications**: Table showing latest applications with predictions
- **Key Metrics**: Cards showing approval rate, average probability
- **Filters**: Date range, risk level, credit amount range

## 🔧 Useful SQL Queries

### Risk Distribution
```sql
SELECT 
    predicted_risk,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM CREDIT_RISK_SCORES
GROUP BY predicted_risk
ORDER BY customer_count DESC;
```

### Risk by Age Group
```sql
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
```

### Risk by Credit Amount
```sql
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
```

## 📋 DAX Measures for Power BI

### Approval Rate
```dax
Approval Rate = 
DIVIDE(
    CALCULATE(COUNT(CREDIT_RISK_DASHBOARD[customer_id]), 
              CREDIT_RISK_DASHBOARD[predicted_risk] = "good"),
    COUNT(CREDIT_RISK_DASHBOARD[customer_id]),
    0
)
```

### Average Credit Amount by Risk
```dax
Avg Credit Amount by Risk = 
AVERAGE(CREDIT_RISK_DASHBOARD[credit_amount])
```

### Risk Score
```dax
Risk Score = 
AVERAGE(CREDIT_RISK_DASHBOARD[probability_bad])
```

## 🔄 Real-time Model Deployment

### Using Snowpark UDFs
The `snowpark_deployment.py` script creates User-Defined Functions (UDFs) that allow you to:
- Make real-time predictions on new applications
- Process batch predictions efficiently
- Integrate predictions directly into your data pipeline

### Testing the UDF
```sql
-- Test single prediction
SELECT predict_credit_risk(35, 'male', 2, 'own', 'moderate', 'moderate', 5000, 24, 'car');

-- Run batch prediction
CALL BATCH_CREDIT_RISK_PREDICTION();

-- Check results
SELECT * FROM CREDIT_PREDICTIONS LIMIT 10;
```

## 📊 Current Model Performance

Based on the exported predictions:
- **Total Customers**: 1,000
- **Good Risk Predictions**: 783 (78.3%)
- **Bad Risk Predictions**: 217 (21.7%)
- **Average Probability (Good)**: 0.735
- **Average Probability (Bad)**: 0.265

### Risk Distribution by Age
- **Young (<30)**: 411 customers (350 bad, 61 good)
- **Young Adult (30-39)**: 315 customers (231 bad, 84 good)
- **Adult (40-49)**: 161 customers (122 bad, 39 good)
- **Senior (50+)**: 113 customers (80 bad, 33 good)

### Risk Distribution by Credit Amount
- **Low (<2K)**: 432 customers (364 bad, 68 good)
- **Medium (2K-5K)**: 380 customers (303 bad, 77 good)
- **High (5K-10K)**: 148 customers (98 bad, 50 good)
- **Very High (10K+)**: 40 customers (18 bad, 22 good)

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Issues**
   - Verify Snowflake account credentials
   - Check warehouse status: `SHOW WAREHOUSES;`
   - Ensure proper permissions

2. **Data Loading Issues**
   - Check file format compatibility
   - Verify stage contents: `LIST @your_stage;`
   - Check for data type mismatches

3. **UDF Issues**
   - Check UDF status: `SHOW USER FUNCTIONS;`
   - Verify package dependencies
   - Check logs: `SELECT * FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY()) ORDER BY START_TIME DESC LIMIT 10;`

4. **Power BI Issues**
   - Verify Snowflake connector installation
   - Check data refresh settings
   - Ensure proper data source permissions

## 📞 Support

For issues with:
- **Model Training**: Check `credit_risk_prediction_fixed_final.py`
- **Snowflake Setup**: Refer to `snowflake_setup_instructions.md`
- **Power BI**: Refer to `powerbi_connection_guide.md`
- **Data Export**: Check `export_predictions_to_snowflake.py`

## 🔄 Next Steps

1. **Monitor Performance**: Set up monitoring for prediction accuracy
2. **Retrain Model**: Schedule regular model retraining with new data
3. **Scale Up**: Consider larger warehouses for bigger datasets
4. **Security**: Implement row-level security in Snowflake
5. **Automation**: Set up automated data pipelines for continuous predictions

## 📚 Additional Resources

- [Snowflake Documentation](https://docs.snowflake.com/)
- [Snowpark Python Guide](https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html)
- [Power BI Snowflake Connector](https://docs.microsoft.com/en-us/power-bi/connect-data/service-connect-snowflake)
- [Credit Risk Analysis Best Practices](https://www.risk.net/risk-management/credit-risk)

---

**Note**: This integration provides a complete end-to-end solution for credit risk prediction, from model training to real-time deployment and visualization. The system is designed to be scalable, maintainable, and easily extensible for additional features.
