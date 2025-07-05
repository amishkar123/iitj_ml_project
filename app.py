import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# --- Load the trained model and feature columns ---
try:
    model = joblib.load('gradient_boosting_model.joblib')
    model_features = joblib.load('model_features.joblib')
    print("Model and features loaded successfully.")
except FileNotFoundError:
    print("Error: Model or feature columns file not found.")
    print("Please ensure 'gradient_boosting_model.joblib' and 'model_features.joblib' are in the same directory.")
    print("Run the model training script first to generate these files.")
    model = None # Set model to None to prevent errors if files are missing
    model_features = []

# Define the original categorical features for consistent one-hot encoding
# This list must match the original categorical_features used during training
ORIGINAL_CATEGORICAL_FEATURES = ['Employment_Status', 'Loan_Purpose']

# Define ALL possible categories for each categorical feature
# This is crucial for consistent one-hot encoding (especially with drop_first=True)
# Ensure these lists match the unique values present in your training data
ALL_POSSIBLE_CATEGORIES = {
    'Employment_Status': ['Employed', 'Retired', 'Self-Employed', 'Unemployed'], # Alphabetical order for consistency with get_dummies default
    'Loan_Purpose': ['Car', 'Debt Consolidation', 'Education', 'Home', 'Other'] # Alphabetical order
}


@app.route('/')
def home():
    """Renders the home page with the credit application form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the web form."""
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded. Please check server logs.")

    # Get data from the form
    form_data = request.form.to_dict()

    # Convert form data to a pandas DataFrame row
    # Ensure numerical types are correctly cast
    try:
        input_data = {
            'Age': int(form_data['age']),
            'Income': int(form_data['income']),
            'Credit_Score': int(form_data['credit_score']),
            'Loan_Amount_Requested': int(form_data['loan_amount_requested']),
            'Employment_Status': form_data['employment_status'],
            'Years_at_Current_Job': int(form_data['years_at_job']),
            'Existing_Debt': int(form_data['existing_debt']),
            'Num_Credit_Accounts': int(form_data['num_credit_accounts']),
            'Delinquencies_in_2_Years': int(form_data['delinquencies']),
            'Loan_Purpose': form_data['loan_purpose']
        }
    except ValueError as e:
        return render_template('index.html', prediction_text=f"Invalid input: {e}. Please ensure all numerical fields are correct.")

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # --- IMPORTANT: Re-apply Feature Engineering consistently ---
    # These calculations MUST match any feature engineering done in train_and_save_model.py
    input_df['Debt_to_Income_Ratio'] = input_df['Existing_Debt'] / (input_df['Income'] + 1e-6)
    input_df['Loan_to_Income_Ratio'] = input_df['Loan_Amount_Requested'] / (input_df['Income'] + 1e-6)


    # --- Robust One-Hot Encoding for new input ---
    # Create a dummy DataFrame with all possible one-hot encoded columns
    # This ensures consistency even if a specific category isn't in the current input
    dummy_df = pd.DataFrame(columns=model_features)
    dummy_df.loc[0] = 0 # Initialize all values to 0

    # Populate numerical and engineered features directly
    for col in input_df.columns:
        if col in model_features: # Check if the original column (numerical/engineered) is a direct feature
            dummy_df.loc[0, col] = input_df.loc[0, col]
        elif col in ORIGINAL_CATEGORICAL_FEATURES:
            # Handle one-hot encoding for the categorical features
            category_value = input_df.loc[0, col]
            # Construct the dummy column name as it would be after get_dummies(drop_first=True)
            # This assumes get_dummies sorts columns alphabetically and drops the first one.
            # You might need to adjust ALL_POSSIBLE_CATEGORIES order if your training data's get_dummies
            # resulted in a different dropped column.
            if category_value != ALL_POSSIBLE_CATEGORIES[col][0]: # If it's not the 'dropped' category
                dummy_col_name = f"{col}_{category_value}"
                if dummy_col_name in model_features: # Ensure this dummy column exists in trained features
                    dummy_df.loc[0, dummy_col_name] = 1

    # Ensure the final input DataFrame for prediction has all columns in the correct order
    # and fills any missing ones (e.g., one-hot encoded columns not present in this specific input) with 0.
    input_processed = dummy_df.reindex(columns=model_features, fill_value=0)

    # --- DEBUGGING STEP: Print the processed input to compare with expected features ---
    print("\n--- Processed Input for Prediction (DataFrame) ---")
    print(input_processed)
    print(f"Columns match model_features in count: {len(input_processed.columns) == len(model_features)}")
    print(f"Columns match model_features in order and name: {list(input_processed.columns) == model_features}")
    print(f"Raw prediction probability for acceptance: {model.predict_proba(input_processed)[0][1]:.8f}")


    # Make prediction
    prediction_proba = model.predict_proba(input_processed)[0][1] # Probability of acceptance (class 1)
    prediction_status = "Accepted" if prediction_proba >= 0.5 else "Rejected"

    return render_template('index.html',
                           prediction_text=f"Credit Application Status: {prediction_status}",
                           probability=f"Probability of Acceptance: {prediction_proba:.2f}")

if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode for development (auto-reloads on code changes)
