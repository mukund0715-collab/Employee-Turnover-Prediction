import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Load dataset
# -----------------------------
file_path = input("Enter File Name: ")
df = pd.read_csv(file_path)

# Keep original Name column for output later
employee_names = df['Name']

# Drop Name from features
df.drop(columns=['Name'], inplace=True)

# Encode categorical features
le_salary = LabelEncoder()
df['salary'] = le_salary.fit_transform(df['salary'])

le_sales = LabelEncoder()
df['sales'] = le_sales.fit_transform(df['sales'])

# -----------------------------
# Calculate efficiency_score
# -----------------------------
num_cols = ['last_evaluation', 'average_montly_hours', 'number_project', 'time_spend_company', 'Work_accident']
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

df['efficiency_score'] = (
    0.4 * df_scaled['last_evaluation'] +
    0.25 * df_scaled['average_montly_hours'] +
    0.15 * df_scaled['number_project'] +
    0.15 * df_scaled['time_spend_company'] -
    0.05 * df_scaled['Work_accident']
)

# Drop rows where target (left) is NaN
df.dropna(subset=['left'], inplace=True)

# Create efficiency label (High = 1, Low = 0)
thresh = df['efficiency_score'].median()
df['efficiency_label'] = (df['efficiency_score'] > thresh).astype(int)

# -----------------------------
# Features and targets
# -----------------------------
features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
            'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']

X = df[features]
y_turnover = df['left']
y_efficiency = df['efficiency_label']

# -----------------------------
# Train/Test split
# -----------------------------
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_turnover, test_size=0.2, random_state=42)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)

# Imputation for LR and RF
imputer = SimpleImputer(strategy='median')
X_train_t_imp = imputer.fit_transform(X_train_t)
X_test_t_imp = imputer.transform(X_test_t)
X_train_e_imp = imputer.fit_transform(X_train_e)
X_test_e_imp = imputer.transform(X_test_e)

# Scale for LR
scaler_X = StandardScaler()
X_train_t_scaled = scaler_X.fit_transform(X_train_t_imp)
X_test_t_scaled = scaler_X.transform(X_test_t_imp)
X_train_e_scaled = scaler_X.fit_transform(X_train_e_imp)
X_test_e_scaled = scaler_X.transform(X_test_e_imp)

# -----------------------------
# Models
# -----------------------------
log_reg_turnover = LogisticRegression(max_iter=1000)
rf_turnover = RandomForestClassifier(n_estimators=100, random_state=42)
hgb_turnover = HistGradientBoostingClassifier(random_state=42)

log_reg_efficiency = LogisticRegression(max_iter=1000)
rf_efficiency = RandomForestClassifier(n_estimators=100, random_state=42)
hgb_efficiency = HistGradientBoostingClassifier(random_state=42)

# Train models
log_reg_turnover.fit(X_train_t_scaled, y_train_t)
rf_turnover.fit(X_train_t_imp, y_train_t)
hgb_turnover.fit(X_train_t, y_train_t)  # Handles NaN

log_reg_efficiency.fit(X_train_e_scaled, y_train_e)
rf_efficiency.fit(X_train_e_imp, y_train_e)
hgb_efficiency.fit(X_train_e, y_train_e)  # Handles NaN

# Predictions
y_pred_lr_turnover = log_reg_turnover.predict(X_test_t_scaled)
y_pred_rf_turnover = rf_turnover.predict(X_test_t_imp)
y_pred_hgb_turnover = hgb_turnover.predict(X_test_t)

y_pred_lr_efficiency = log_reg_efficiency.predict(X_test_e_scaled)
y_pred_rf_efficiency = rf_efficiency.predict(X_test_e_imp)
y_pred_hgb_efficiency = hgb_efficiency.predict(X_test_e)

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_model(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1 Score': round(f1_score(y_true, y_pred), 4)
    }

# Evaluate all models
results = []
results.append(evaluate_model(y_test_t, y_pred_lr_turnover, 'LogReg - Turnover'))
results.append(evaluate_model(y_test_t, y_pred_rf_turnover, 'RandomForest - Turnover'))
results.append(evaluate_model(y_test_t, y_pred_hgb_turnover, 'HGB - Turnover'))

results.append(evaluate_model(y_test_e, y_pred_lr_efficiency, 'LogReg - Efficiency'))
results.append(evaluate_model(y_test_e, y_pred_rf_efficiency, 'RandomForest - Efficiency'))
results.append(evaluate_model(y_test_e, y_pred_hgb_efficiency, 'HGB - Efficiency'))

results_df = pd.DataFrame(results)
print(results_df)

# -----------------------------
# Generate full dataset predictions using HGB (best for NaN)
# -----------------------------
df['Predicted_Turnover'] = hgb_turnover.predict(X)

# Re-attach Name to original features
df_full_output = df.copy()
df_full_output['Name'] = employee_names
df_full_output['efficiency_score'] = df['efficiency_score']

# -----------------------------
# Add Turnover Amount
# -----------------------------
BASE_COST = 50000   # fixed cost per turnover
FACTOR = 10000      # scaling factor with efficiency_score

df_full_output['Turnover_Amount'] = BASE_COST + (FACTOR * df_full_output['efficiency_score'])

# Reorder columns to show Name first
cols = ['Name', 'efficiency_score', 'Predicted_Turnover', 'Turnover_Amount'] + \
       [col for col in df_full_output.columns if col not in ['Name', 'efficiency_score', 'Predicted_Turnover', 'Turnover_Amount']]
df_full_output = df_full_output[cols]

# -----------------------------
# Save files
# -----------------------------
base_name = os.path.splitext(os.path.basename(file_path))[0]

# 1. Sorted by efficiency (employees predicted to leave only)
sorted_efficiency_file = f"{base_name}_sorted_efficiency.csv"
employees_to_leave = df_full_output[df_full_output['Predicted_Turnover'] == 1]
employees_to_leave = employees_to_leave.sort_values(by='efficiency_score', ascending=False)
employees_to_leave.to_csv(sorted_efficiency_file, index=False)

# 2. Sorted by Predicted Turnover (all employees, leavers first)
sorted_turnover_file = f"{base_name}_sorted_turnover.csv"
df_sorted_turnover = df_full_output.sort_values(by=['Predicted_Turnover', 'efficiency_score'], ascending=[False, False])
df_sorted_turnover.to_csv(sorted_turnover_file, index=False)

print("Files created:")
print(f"1. Sorted Efficiency File: {sorted_efficiency_file}")
print(f"2. Sorted Predicted Turnover File: {sorted_turnover_file}")

