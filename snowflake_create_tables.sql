
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
