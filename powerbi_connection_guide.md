
# Power BI Connection Guide for Snowflake

## Step 1: Install Snowflake Connector
1. Open Power BI Desktop
2. Go to File > Get Data
3. Search for "Snowflake" and install the connector

## Step 2: Connect to Snowflake
1. Click "Get Data" > "Database" > "Snowflake"
2. Enter your Snowflake connection details:
   - Server: your_account.snowflakecomputing.com
   - Database: your_database
   - Warehouse: your_warehouse
   - Username: your_username
   - Password: your_password

## Step 3: Import Data
Select the following views/tables:
- CREDIT_RISK_DASHBOARD (for comprehensive analysis)
- REALTIME_CREDIT_RISK (for real-time applications)
- CREDIT_RISK_ANALYTICS (for risk analytics)

## Step 4: Create Visualizations

### 1. Risk Distribution Dashboard
- **Card Visual**: Total applications, Good risk count, Bad risk count
- **Pie Chart**: Risk distribution (Good vs Bad)
- **Bar Chart**: Risk by age group
- **Bar Chart**: Risk by credit amount range

### 2. Portfolio Trends Dashboard
- **Line Chart**: Applications over time
- **Line Chart**: Risk ratio trends
- **Area Chart**: Credit amount distribution over time

### 3. Segment Analysis Dashboard
- **Matrix**: Risk by job type and housing
- **Scatter Plot**: Age vs Credit Amount (colored by risk)
- **Bar Chart**: Risk by purpose
- **Gauge**: Overall portfolio risk score

### 4. Real-time Application Dashboard
- **Table**: Recent applications with predictions
- **Cards**: Key metrics (approval rate, average probability)
- **Filters**: Date range, risk level, credit amount range

## Step 5: Set Up Refresh Schedule
1. Go to File > Options and settings > Data source settings
2. Configure automatic refresh (e.g., every hour)
3. Publish to Power BI Service for sharing

## Sample DAX Measures

```
// Approval Rate
Approval Rate = 
DIVIDE(
    CALCULATE(COUNT(CREDIT_RISK_DASHBOARD[customer_id]), 
              CREDIT_RISK_DASHBOARD[predicted_risk] = "good"),
    COUNT(CREDIT_RISK_DASHBOARD[customer_id]),
    0
)

// Average Credit Amount by Risk
Avg Credit Amount by Risk = 
AVERAGE(CREDIT_RISK_DASHBOARD[credit_amount])

// Risk Score (weighted average)
Risk Score = 
AVERAGE(CREDIT_RISK_DASHBOARD[probability_bad])
```

## Best Practices
1. Use incremental refresh for large datasets
2. Create calculated columns for better performance
3. Use bookmarks for different dashboard views
4. Set up row-level security if needed
5. Create drill-through reports for detailed analysis
