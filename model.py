import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib # Import joblib for saving/loading models

# --- 1. Data Generation ---
np.random.seed(42) # for reproducibility

num_applicants = 1000

data = {
    'Age': np.random.randint(20, 70, num_applicants),
    'Income': np.random.randint(30000, 150000, num_applicants),
    'Credit_Score': np.random.randint(300, 850, num_applicants),
    'Loan_Amount_Requested': np.random.randint(5000, 50000, num_applicants),
    'Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], num_applicants, p=[0.6, 0.2, 0.1, 0.1]),
    'Years_at_Current_Job': np.random.randint(0, 20, num_applicants),
    'Existing_Debt': np.random.randint(0, 30000, num_applicants),
    'Num_Credit_Accounts': np.random.randint(1, 10, num_applicants),
    'Delinquencies_in_2_Years': np.random.randint(0, 5, num_applicants),
    'Loan_Purpose': np.random.choice(['Home', 'Car', 'Debt Consolidation', 'Education', 'Other'], num_applicants, p=[0.25, 0.2, 0.3, 0.15, 0.1])
}

df = pd.DataFrame(data)

# Create a synthetic 'Acceptance_Status' target variable
# *** ADJUSTED LOGIC FOR STRONGER BIAS TOWARDS ACCEPTANCE ***
df['Acceptance_Probability'] = (
    0.5 * (df['Credit_Score'] / 850) +        # Increased weight for good credit score
    0.4 * (df['Income'] / 150000) -          # Increased weight for income
    0.1 * (df['Loan_Amount_Requested'] / 50000) - # Significantly reduced negative weight
    0.02 * (df['Existing_Debt'] / 30000) +   # Further reduced negative weight for debt
    0.15 * (df['Years_at_Current_Job'] / 20) + # Slightly increased positive weight
    0.1 # Added a constant positive bias to shift probabilities upwards
)
# Add some randomness, centered around 0
df['Acceptance_Probability'] = df['Acceptance_Probability'] + np.random.rand(num_applicants) * 0.1 - 0.05 # Reduced randomness

# Convert probability to binary acceptance status (e.g., threshold at 0.5)
df['Acceptance_Status'] = (df['Acceptance_Probability'] > 0.5).astype(int)

# Drop the intermediate probability column
df = df.drop(columns=['Acceptance_Probability'])

print("--- Synthetic Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
df.info()
print("\n--- Acceptance Status Distribution ---")
print(df['Acceptance_Status'].value_counts(normalize=True))

# --- 2. Data Preprocessing & Feature Engineering (Combined for consistency) ---

# Separate features (X) and target (y)
X = df.drop('Acceptance_Status', axis=1)
y = df['Acceptance_Status']

# --- Feature Engineering: Apply BEFORE one-hot encoding ---
X['Debt_to_Income_Ratio'] = X['Existing_Debt'] / (X['Income'] + 1e-6)
X['Loan_to_Income_Ratio'] = X['Loan_Amount_Requested'] / (X['Income'] + 1e-6)

# Identify categorical features for one-hot encoding
# Ensure this list is consistent with ORIGINAL_CATEGORICAL_FEATURES in app.py
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Apply One-Hot Encoding to categorical features
# drop_first=True is important for consistent behavior with Flask app
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print("\n--- Features after Feature Engineering and One-Hot Encoding Head ---")
print(X.head())
print("\n--- Features Info after Encoding ---")
X.info()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 3. Model Selection and Training (Gradient Boosting) ---

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
print("\n--- Training the Gradient Boosting Model ---")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Model Evaluation ---

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (acceptance)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\n--- Confusion Matrix ---")
print(conf_matrix)
print("Interpretation: [[TN, FP], [FN, TP]]")

# Feature Importance (for interpretability)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Top 10 Feature Importances ---")
print(feature_importances.head(10))

# --- Save the trained model and feature columns ---
model_filename = 'gradient_boosting_model.joblib'
feature_columns_filename = 'model_features.joblib'

joblib.dump(model, model_filename)
joblib.dump(X.columns.tolist(), feature_columns_filename) # Save as a list for easier loading

print(f"\nModel saved to {model_filename}")
print(f"Feature columns saved to {feature_columns_filename}")
