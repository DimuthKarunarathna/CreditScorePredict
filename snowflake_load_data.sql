
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
