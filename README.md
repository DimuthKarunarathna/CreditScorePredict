# Credit Risk Prediction Model

This project demonstrates a complete machine learning pipeline for predicting credit risk using the German Credit Dataset. The original notebook had several issues that have been fixed, and a trained model has been saved for deployment.

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
python3 credit_risk_prediction_fixed_final.py
```

### Test the Model

**Interactive Testing:**
```bash
python3 test_model_interactive.py
```

**Demo with Predefined Cases:**
```bash
python3 demo_test_cases.py
```

**Load and Use the Saved Model:**
```bash
python3 load_and_use_model.py
```

## 📁 Clean Project Structure

```
CreditScorePredict/
├── german_credit_data.csv                    # Original dataset
├── credit_risk_prediction_fixed_final.py     # Complete training pipeline
├── test_model_interactive.py                 # Interactive model tester
├── demo_test_cases.py                        # Demo with predefined test cases
├── load_and_use_model.py                     # Model loading and prediction script
├── requirements.txt                          # Python dependencies
├── README.md                                 # This documentation
├── credit_risk_model_gradient_boosting.joblib # Trained model (joblib format)
├── credit_risk_scaler.joblib                 # Feature scaler
├── credit_risk_label_encoders.joblib         # Label encoders
├── credit_risk_target_encoder.joblib         # Target encoder
└── credit_risk_features.joblib               # Feature column names
```

## 🔧 Issues Fixed

### Original Notebook Problems:
1. **Missing Risk Column**: The original dataset didn't have a "Risk" target column
2. **Incorrect Data Path**: The notebook was looking for a non-existent file
3. **Deprecated Plotly Methods**: Used outdated `py.iplot()` instead of modern Plotly
4. **Missing Dependencies**: XGBoost had compatibility issues on macOS
5. **No Model Saving**: The original notebook didn't save the trained model

### Solutions Implemented:
1. **Created Risk Target**: Built a risk scoring system based on multiple features
2. **Fixed Data Loading**: Updated to use the correct dataset file
3. **Modern Visualization**: Used matplotlib and seaborn for reliable plotting
4. **Dependency Management**: Created requirements.txt and handled XGBoost issues
5. **Model Persistence**: Saved models in joblib format for optimal performance

## 📊 Dataset Information

The German Credit Dataset contains 1000 samples with the following features:

- **Age**: Customer age (19-75)
- **Sex**: Gender ('male', 'female')
- **Job**: Employment status (0=unemployed, 1-3=employed levels)
- **Housing**: Housing situation ('own', 'rent', 'free')
- **Saving accounts**: Savings account status
- **Checking account**: Checking account status
- **Credit amount**: Loan amount (250-18424)
- **Duration**: Loan duration in months (4-72)
- **Purpose**: Loan purpose

## 🎯 Model Performance

The best performing model is **Gradient Boosting** with the following metrics:

- **Accuracy**: 97.50%
- **F1 Score**: 97.52%
- **Precision**: 97.68%
- **Recall**: 97.50%

### Feature Importance (Top 5):
1. **Duration** (43.2%)
2. **Credit amount** (24.6%)
3. **Age** (8.1%)
4. **Checking account** (8.0%)
5. **Housing** (7.4%)

## 🚀 Usage Examples

### Interactive Testing

Run the interactive tester to input your own customer data:

```bash
python3 test_model_interactive.py
```

This will prompt you to enter customer information and show real-time predictions.

### Demo Test Cases

See how the model performs with different scenarios:

```bash
python3 demo_test_cases.py
```

This runs 6 predefined test cases showing:
- Low Risk - Stable Professional
- High Risk - Young Unemployed
- Medium Risk - Young Professional
- Low Risk - Senior Homeowner
- High Risk - High Amount Loan
- Borderline Case - Mixed Factors

### Programmatic Usage

```python
from load_and_use_model import load_model_and_components, predict_credit_risk

# Load the model
model, scaler, label_encoders, target_encoder, feature_columns = load_model_and_components()

# Make a prediction
risk, probability = predict_credit_risk(
    age=35,
    sex='male',
    job=2,
    housing='own',
    saving_accounts='moderate',
    checking_account='moderate',
    credit_amount=5000,
    duration=24,
    purpose='car',
    model=model,
    scaler=scaler,
    label_encoders=label_encoders,
    target_encoder=target_encoder,
    feature_columns=feature_columns
)

print(f"Predicted Risk: {risk}")
print(f"Probabilities: {probability}")
```

### Input Requirements

| Feature | Type | Values |
|---------|------|--------|
| Age | int | 19-75 |
| Sex | str | 'male', 'female' |
| Job | int | 0-3 (0=unemployed) |
| Housing | str | 'own', 'rent', 'free' |
| Saving accounts | str | 'little', 'moderate', 'rich', 'quite rich', 'no_inf' |
| Checking account | str | 'little', 'moderate', 'rich', 'no_inf' |
| Credit amount | int | 250-18424 |
| Duration | int | 4-72 months |
| Purpose | str | 'car', 'business', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'domestic appliances', 'vacation/others' |

## 🔄 Model Retraining

To retrain the model with new data:

1. Replace `german_credit_data.csv` with your new dataset
2. Ensure the new dataset has the same column structure
3. Run the training script:
   ```bash
   python3 credit_risk_prediction_fixed_final.py
   ```

## 🛠️ Deployment

### API Integration

The saved model can be easily integrated into web APIs:

```python
# Flask example
from flask import Flask, request, jsonify
from load_and_use_model import load_model_and_components, predict_credit_risk

app = Flask(__name__)
model, scaler, label_encoders, target_encoder, feature_columns = load_model_and_components()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    risk, probability = predict_credit_risk(
        data['age'], data['sex'], data['job'], data['housing'],
        data['saving_accounts'], data['checking_account'],
        data['credit_amount'], data['duration'], data['purpose'],
        model, scaler, label_encoders, target_encoder, feature_columns
    )
    return jsonify({
        'risk': risk,
        'probability': probability.tolist() if probability is not None else None
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📋 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues, please open an issue in the repository.

---

**Note**: This model is for educational purposes. For production use in financial services, additional validation, testing, and compliance measures should be implemented.
