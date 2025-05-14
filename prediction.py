import pandas as pd
import numpy as np
import joblib

def create_features(df):
    """Create engineered features from raw data."""
    df_new = df.copy()
    
    # Utilization to income ratio
    df_new['util_to_income'] = df_new['revolvingutilizationofunsecuredlines'] / df_new['monthlyincome'].replace(0, 0.001)
    
    # Age groups (binned)
    df_new['age_group'] = pd.cut(df_new['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=[0, 1, 2, 3, 4, 5])
    
    # Delinquency features
    df_new['total_delinquencies'] = (df_new['numberoftime3059dayspastduenotworse'] + 
                                    df_new['numberoftime6089dayspastduenotworse'] + 
                                    df_new['numberoftimes90dayslate'])
    
    # Has delinquency flag
    df_new['has_delinquency'] = (df_new['total_delinquencies'] > 0).astype(int)
    
    # Credit lines to income ratio
    df_new['credit_lines_to_income'] = df_new['numberofopencreditlinesandloans'] / df_new['monthlyincome'].replace(0, 0.001)
    
    # Real estate loans to total credit lines ratio
    df_new['realestate_to_credit_ratio'] = df_new['numberrealestateloansorlines'] / df_new['numberofopencreditlinesandloans'].replace(0, 0.001)
    
    # High utilization flag
    df_new['high_utilization'] = (df_new['revolvingutilizationofunsecuredlines'] > 0.5).astype(int)
    
    # Income per dependent 
    df_new['income_per_dependent'] = df_new['monthlyincome'] / (df_new['numberofdependents'] + 1)
    
    # Compute logarithmic transforms for skewed features
    for feature in ['revolvingutilizationofunsecuredlines', 'debtratio', 'monthlyincome', 'util_to_income']:
        # Add a small constant to avoid log(0)
        df_new[f'log_{feature}'] = np.log1p(df_new[feature].replace(0, 0.001))
        
    return df_new

def predict_credit_risk(applicant_data):
    """
    Make credit risk predictions for new applicant data.
    
    Parameters:
    applicant_data (dict or DataFrame): Data for one or more loan applicants
    
    Returns:
    DataFrame: Predicted probabilities and risk categories
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(applicant_data, dict):
        applicant_data = pd.DataFrame([applicant_data])
    
    # Load models and preprocessing info
    try:
        balanced_rf = joblib.load('models/balanced_rf_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        preprocessing_info = joblib.load('models/preprocessing_info.pkl')
    except FileNotFoundError:
        raise Exception("Model files not found. Make sure to train and save the models first.")
    
    # Preprocess the data (handle missing values)
    for col in ['monthlyincome', 'numberofdependents']:
        if col in applicant_data.columns and applicant_data[col].isnull().any():
            median_value = preprocessing_info[f'median_{col}']
            applicant_data[col] = applicant_data[col].fillna(median_value)
    
    # Feature engineering
    applicant_data = create_features(applicant_data)
    
    # Get required features
    required_features = preprocessing_info['selected_features']
    
    # Handle missing features
    for feature in required_features:
        if feature not in applicant_data.columns:
            applicant_data[feature] = 0  # Default value
    
    # Select only the features used by the model
    applicant_data = applicant_data[required_features]
    
    # Make predictions with each model
    pred_proba_brf = balanced_rf.predict_proba(applicant_data)[:, 1]
    pred_proba_xgb = xgb_model.predict_proba(applicant_data)[:, 1]
    pred_proba_rf = rf_model.predict_proba(applicant_data)[:, 1]
    
    # Create ensemble prediction
    pred_proba_ensemble = (0.4 * pred_proba_brf +
                           0.4 * pred_proba_xgb +
                           0.2 * pred_proba_rf)
    
    # Determine risk category
    risk_categories = []
    for prob in pred_proba_ensemble:
        if prob < 0.05:
            risk_categories.append("Very Low Risk")
        elif prob < 0.15:
            risk_categories.append("Low Risk")
        elif prob < 0.30:
            risk_categories.append("Moderate Risk")
        elif prob < 0.50:
            risk_categories.append("High Risk")
        else:
            risk_categories.append("Very High Risk")
    
    # Create result DataFrame
    results = pd.DataFrame({
        'default_probability': pred_proba_ensemble,
        'risk_category': risk_categories
    })
    
    return results