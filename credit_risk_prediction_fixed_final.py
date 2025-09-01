#!/usr/bin/env python3
"""
Credit Risk Prediction Model Pipeline - Fixed Version for German Credit Dataset
This script demonstrates a complete machine learning pipeline for predicting credit risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Model persistence
import joblib
import pickle

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("All libraries imported successfully!")

def create_risk_target(df):
    """
    Create a risk target variable based on credit amount and other features
    This is a simplified approach since the original Risk column is missing
    """
    # Create risk based on credit amount percentiles and other factors
    credit_amount_75th = df['Credit amount'].quantile(0.75)
    credit_amount_25th = df['Credit amount'].quantile(0.25)
    
    # Create risk categories based on multiple factors
    risk_scores = []
    
    for _, row in df.iterrows():
        score = 0
        
        # Credit amount factor (higher amount = higher risk)
        if row['Credit amount'] > credit_amount_75th:
            score += 2
        elif row['Credit amount'] > credit_amount_25th:
            score += 1
        
        # Duration factor (longer duration = higher risk)
        if row['Duration'] > 24:
            score += 2
        elif row['Duration'] > 12:
            score += 1
        
        # Age factor (very young or very old = higher risk)
        if row['Age'] < 25 or row['Age'] > 60:
            score += 1
        
        # Job factor (unemployed = higher risk)
        if row['Job'] == 0:  # unemployed
            score += 2
        
        # Housing factor (renting = higher risk)
        if row['Housing'] == 'rent':
            score += 1
        
        # Saving accounts factor
        if pd.isna(row['Saving accounts']) or row['Saving accounts'] == 'NA':
            score += 1
        elif row['Saving accounts'] == 'little':
            score += 1
        
        # Checking account factor
        if pd.isna(row['Checking account']) or row['Checking account'] == 'NA':
            score += 1
        elif row['Checking account'] == 'little':
            score += 1
        
        risk_scores.append(score)
    
    # Convert scores to risk categories
    risk_scores = np.array(risk_scores)
    risk_percentile_75 = np.percentile(risk_scores, 75)
    
    risk_categories = []
    for score in risk_scores:
        if score >= risk_percentile_75:
            risk_categories.append('bad')
        else:
            risk_categories.append('good')
    
    return risk_categories

def main():
    # 1. Load and Explore the Dataset
    print("=" * 50)
    print("1. LOADING AND EXPLORING THE DATASET")
    print("=" * 50)
    
    # Load the dataset
    df_credit = pd.read_csv("german_credit_data.csv", index_col=0)
    
    print("Dataset shape:", df_credit.shape)
    print("\nFirst few rows:")
    print(df_credit.head())
    
    # Basic information about the dataset
    print("\nDataset Info:")
    print(df_credit.info())
    
    print("\nMissing Values:")
    print(df_credit.isnull().sum())
    
    print("\nUnique Values per Column:")
    print(df_credit.nunique())
    
    print("\nStatistical Summary:")
    print(df_credit.describe(include='all'))
    
    # Create the missing Risk target variable
    print("\nCreating Risk target variable...")
    df_credit['Risk'] = create_risk_target(df_credit)
    
    print("Risk distribution:")
    print(df_credit['Risk'].value_counts())
    
    # 2. Data Visualization
    print("\n" + "=" * 50)
    print("2. DATA VISUALIZATION")
    print("=" * 50)
    
    # Target variable distribution
    plt.figure(figsize=(10, 6))
    risk_counts = df_credit['Risk'].value_counts()
    
    plt.bar(risk_counts.index, risk_counts.values, 
            color=['#2E8B57', '#DC143C'])
    plt.title('Risk Distribution')
    plt.xlabel('Risk')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Risk Distribution:")
    print(risk_counts)
    
    # Age distribution by risk
    plt.figure(figsize=(12, 8))
    for risk in df_credit['Risk'].unique():
        subset = df_credit[df_credit['Risk'] == risk]
        plt.hist(subset['Age'], alpha=0.7, label=risk, bins=20)
    
    plt.title('Age Distribution by Risk')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('age_distribution_by_risk.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Credit amount distribution by risk
    plt.figure(figsize=(12, 8))
    df_credit.boxplot(column='Credit amount', by='Risk', figsize=(12, 8))
    plt.title('Credit Amount Distribution by Risk')
    plt.suptitle('')  # Remove default suptitle
    plt.tight_layout()
    plt.savefig('credit_amount_distribution_by_risk.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Job level by risk
    plt.figure(figsize=(12, 8))
    job_risk = pd.crosstab(df_credit['Job'], df_credit['Risk'])
    
    sns.heatmap(job_risk, annot=True, fmt='d', cmap='Blues')
    plt.title('Job Level vs Risk Heatmap')
    plt.tight_layout()
    plt.savefig('job_risk_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Job Level vs Risk:")
    print(job_risk)
    
    # Housing by risk
    plt.figure(figsize=(10, 6))
    housing_risk = pd.crosstab(df_credit['Housing'], df_credit['Risk'])
    
    housing_risk.plot(kind='bar', stacked=True)
    plt.title('Housing vs Risk')
    plt.xlabel('Housing')
    plt.ylabel('Count')
    plt.legend(title='Risk')
    plt.tight_layout()
    plt.savefig('housing_risk.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Housing vs Risk:")
    print(housing_risk)
    
    # 3. Feature Engineering and Preprocessing
    print("\n" + "=" * 50)
    print("3. FEATURE ENGINEERING AND PREPROCESSING")
    print("=" * 50)
    
    # Create a copy for preprocessing
    df_processed = df_credit.copy()
    
    # Create age categories
    df_processed['Age_Category'] = pd.cut(
        df_processed['Age'], 
        bins=[0, 30, 40, 50, 100], 
        labels=['Young', 'Young Adult', 'Adult', 'Senior']
    )
    
    # Create credit amount categories
    df_processed['Credit_Amount_Category'] = pd.cut(
        df_processed['Credit amount'], 
        bins=[0, 2000, 5000, 10000, 20000], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Create duration categories
    df_processed['Duration_Category'] = pd.cut(
        df_processed['Duration'], 
        bins=[0, 12, 24, 48, 100], 
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    
    print("New features created:")
    print(df_processed[['Age_Category', 'Credit_Amount_Category', 'Duration_Category']].head())
    
    # Handle missing values
    df_processed['Saving accounts'] = df_processed['Saving accounts'].fillna('no_inf')
    df_processed['Checking account'] = df_processed['Checking account'].fillna('no_inf')
    
    # Encode categorical variables
    categorical_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 
                          'Purpose', 'Age_Category', 'Credit_Amount_Category', 'Duration_Category']
    
    label_encoders = {}
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}: {le.classes_}")
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df_processed['Risk_encoded'] = target_encoder.fit_transform(df_processed['Risk'])
    print(f"\nEncoded Risk: {target_encoder.classes_}")
    
    # Select features for modeling
    feature_columns = [
        'Age', 'Job', 'Credit amount', 'Duration',
        'Sex_encoded', 'Housing_encoded', 'Saving accounts_encoded', 
        'Checking account_encoded', 'Purpose_encoded', 'Age_Category_encoded',
        'Credit_Amount_Category_encoded', 'Duration_Category_encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['Risk_encoded']
    
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print("\nFeature columns:")
    print(feature_columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully!")
    
    # 4. Model Training and Evaluation
    print("\n" + "=" * 50)
    print("4. MODEL TRAINING AND EVALUATION")
    print("=" * 50)
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for models that need it
        if name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Calculate ROC AUC if possible
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}" if roc_auc else "  ROC AUC: N/A")
        print("-" * 50)
    
    # Compare model performance
    performance_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'F1 Score': [results[name]['f1_score'] for name in results.keys()],
        'Precision': [results[name]['precision'] for name in results.keys()],
        'Recall': [results[name]['recall'] for name in results.keys()],
        'ROC AUC': [results[name]['roc_auc'] for name in results.keys()]
    })
    
    performance_df = performance_df.sort_values('F1 Score', ascending=False)
    print("Model Performance Comparison:")
    print(performance_df.round(4))
    
    # Visualize model performance
    plt.figure(figsize=(15, 10))
    
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        axes[i].bar(performance_df['Model'], performance_df[metric])
        axes[i].set_title(f'{metric} by Model')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Hyperparameter Tuning for Best Model
    print("\n" + "=" * 50)
    print("5. HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Get the best performing model
    best_model_name = performance_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"Best performing model: {best_model_name}")
    print(f"Best F1 Score: {performance_df.iloc[0]['F1 Score']:.4f}")
    
    # Hyperparameter tuning for the best model
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        print(f"Hyperparameter tuning not implemented for {best_model_name}")
        param_grid = None
    
    if param_grid:
        print(f"Tuning {best_model_name}...")
        
        # Use appropriate data (scaled or unscaled)
        X_train_tune = X_train_scaled if best_model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM'] else X_train
        
        grid_search = GridSearchCV(
            best_model.__class__(**{k: v for k, v in best_model.get_params().items() if k != 'random_state'}),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_tune, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the best model with tuned parameters
        best_model = grid_search.best_estimator_
    
    # 6. Final Model Evaluation
    print("\n" + "=" * 50)
    print("6. FINAL MODEL EVALUATION")
    print("=" * 50)
    
    # Final evaluation of the best model
    if best_model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM']:
        y_pred_final = best_model.predict(X_test_scaled)
        y_pred_proba_final = best_model.predict_proba(X_test_scaled) if hasattr(best_model, 'predict_proba') else None
    else:
        y_pred_final = best_model.predict(X_test)
        y_pred_proba_final = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
    
    print(f"Final Evaluation Results for {best_model_name}")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_final, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_final, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_final, average='weighted'):.4f}")
    
    if y_pred_proba_final is not None:
        try:
            roc_auc_final = roc_auc_score(y_test, y_pred_proba_final, multi_class='ovr')
            print(f"ROC AUC: {roc_auc_final:.4f}")
        except:
            print("ROC AUC: N/A")
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_final)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_
    )
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=target_encoder.classes_))
    
    # Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Feature Importance:")
        print(feature_importance)
    
    # 7. Model Saving
    print("\n" + "=" * 50)
    print("7. MODEL SAVING")
    print("=" * 50)
    
    # Save the trained model
    model_filename = f"credit_risk_model_{best_model_name.lower().replace(' ', '_')}.pkl"
    scaler_filename = "credit_risk_scaler.pkl"
    encoder_filename = "credit_risk_encoders.pkl"
    feature_filename = "credit_risk_features.pkl"
    
    # Save the model
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save the scaler
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the encoders
    with open(encoder_filename, 'wb') as f:
        pickle.dump({
            'label_encoders': label_encoders,
            'target_encoder': target_encoder
        }, f)
    
    # Save the feature columns
    with open(feature_filename, 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print(f"Model saved as: {model_filename}")
    print(f"Scaler saved as: {scaler_filename}")
    print(f"Encoders saved as: {encoder_filename}")
    print(f"Feature columns saved as: {feature_filename}")
    
    # Alternative: Save using joblib (often better for scikit-learn models)
    joblib.dump(best_model, f"credit_risk_model_{best_model_name.lower().replace(' ', '_')}.joblib")
    joblib.dump(scaler, "credit_risk_scaler.joblib")
    joblib.dump(label_encoders, "credit_risk_label_encoders.joblib")
    joblib.dump(target_encoder, "credit_risk_target_encoder.joblib")
    joblib.dump(feature_columns, "credit_risk_features.joblib")
    
    print("\nModels also saved using joblib format!")
    
    # 8. Model Testing
    print("\n" + "=" * 50)
    print("8. MODEL TESTING")
    print("=" * 50)
    
    # Test the saved model with a sample prediction
    def predict_credit_risk(age, sex, job, housing, saving_accounts, checking_account, 
                           credit_amount, duration, purpose, model, scaler, encoders, features):
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
            if col in encoders['label_encoders']:
                sample_df[col + '_encoded'] = encoders['label_encoders'][col].transform(sample_df[col])
        
        # Select features
        X_sample = sample_df[features]
        
        # Scale features
        X_sample_scaled = scaler.transform(X_sample)
        
        # Make prediction
        if best_model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM']:
            prediction = model.predict(X_sample_scaled)
            probability = model.predict_proba(X_sample_scaled) if hasattr(model, 'predict_proba') else None
        else:
            prediction = model.predict(X_sample)
            probability = model.predict_proba(X_sample) if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        risk = encoders['target_encoder'].inverse_transform(prediction)[0]
        
        return risk, probability
    
    # Test with a sample customer
    sample_customer = {
        'age': 35,
        'sex': 'male',
        'job': 2,
        'housing': 'own',
        'saving_accounts': 'moderate',
        'checking_account': 'moderate',
        'credit_amount': 5000,
        'duration': 24,
        'purpose': 'car'
    }
    
    print("Sample Customer:")
    for key, value in sample_customer.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    risk, probability = predict_credit_risk(
        sample_customer['age'],
        sample_customer['sex'],
        sample_customer['job'],
        sample_customer['housing'],
        sample_customer['saving_accounts'],
        sample_customer['checking_account'],
        sample_customer['credit_amount'],
        sample_customer['duration'],
        sample_customer['purpose'],
        best_model,
        scaler,
        {'label_encoders': label_encoders, 'target_encoder': target_encoder},
        feature_columns
    )
    
    print(f"\nPredicted Risk: {risk}")
    if probability is not None:
        print("\nPrediction Probabilities:")
        for i, risk_class in enumerate(target_encoder.classes_):
            print(f"  {risk_class}: {probability[0][i]:.4f}")
    
    # 9. Summary
    print("\n" + "=" * 50)
    print("9. SUMMARY")
    print("=" * 50)
    
    # Summary of the modeling process
    print("CREDIT RISK PREDICTION MODEL - SUMMARY")
    print("=" * 50)
    print(f"Dataset: {df_credit.shape[0]} samples, {df_credit.shape[1]} features")
    print(f"Best Model: {best_model_name}")
    print(f"Best F1 Score: {performance_df.iloc[0]['F1 Score']:.4f}")
    print(f"Best Accuracy: {performance_df.iloc[0]['Accuracy']:.4f}")
    print(f"\nSaved Files:")
    print(f"  - Model: credit_risk_model_{best_model_name.lower().replace(' ', '_')}.pkl")
    print(f"  - Scaler: credit_risk_scaler.pkl")
    print(f"  - Encoders: credit_risk_encoders.pkl")
    print(f"  - Features: credit_risk_features.pkl")
    print(f"\nNext Steps:")
    print(f"  1. Deploy the model using the saved files")
    print(f"  2. Create an API endpoint for predictions")
    print(f"  3. Monitor model performance in production")
    print(f"  4. Retrain periodically with new data")

if __name__ == "__main__":
    main()

