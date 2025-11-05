import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# MODEL LOADING
# -------------------------------

# Build path dynamically (safe even if run from different directories)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'model_data.joblib')

# Load model and components safely
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    cols_to_scale = model_data['cols_to_scale']
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}. Please check the path and try again.")
except Exception as e:
    raise RuntimeError(f"⚠️ Error while loading model: {e}")


# -------------------------------
# INPUT PREPARATION FUNCTION
# -------------------------------

def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):
    """Prepare the input data in the same format as the model was trained."""

    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,

        # One-hot encoding for categorical features
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummy placeholders for missing numeric columns (ensures scaler runs smoothly)
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # Ensure all required scaling columns exist
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0

    # Apply MinMaxScaler transformation
    try:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    except Exception as e:
        raise RuntimeError(f"Error during scaling: {e}")

    # Ensure DataFrame includes all model features (add missing ones as 0)
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    df = df[features]

    return df


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    """Predicts default probability, credit score, and rating."""

    # Prepare input data
    input_df = prepare_input(
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency, delinquency_ratio,
        credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


# -------------------------------
# CREDIT SCORE CALCULATION
# -------------------------------

def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """Converts logistic regression output to probability, score, and rating."""

    try:
        # For Logistic Regression-like models
        x = np.dot(input_df.values, model.coef_.T) + model.intercept_
        default_probability = 1 / (1 + np.exp(-x))
    except AttributeError:
        # For other models (e.g. RandomForest, XGBoost)
        default_probability = model.predict_proba(input_df)[:, 1]

    non_default_probability = 1 - default_probability

    # Convert to 300–900 range
    credit_score = base_score + non_default_probability.flatten() * scale_length
    credit_score = int(credit_score[0])

    # Determine Rating
    rating = get_rating(credit_score)

    return float(default_probability.flatten()[0]), credit_score, rating


# -------------------------------
# RATING LOGIC
# -------------------------------

def get_rating(score):
    """Assign rating category based on credit score."""
    if 300 <= score < 500:
        return 'Poor'
    elif 500 <= score < 650:
        return 'Average'
    elif 650 <= score < 750:
        return 'Good'
    elif 750 <= score <= 900:
        return 'Excellent'
    else:
        return 'Undefined'
