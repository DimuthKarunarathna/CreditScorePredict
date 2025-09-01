
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
