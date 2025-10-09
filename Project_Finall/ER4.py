import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
file_path = input("Enter File Name: ")
df = pd.read_csv(file_path)

# Preserve employee names for output
employee_names = df['Name']
df.drop(columns=['Name'], inplace=True)

# -------------------------------------------------
# 2. Encode categorical columns
# -------------------------------------------------
le_salary = LabelEncoder()
df['salary'] = le_salary.fit_transform(df['salary'])

le_sales = LabelEncoder()
df['sales'] = le_sales.fit_transform(df['sales'])

# -------------------------------------------------
# 3. Efficiency score (raw z-score version)
# -------------------------------------------------
num_cols = ['last_evaluation', 'average_montly_hours',
            'number_project', 'time_spend_company', 'Work_accident']

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

df['efficiency_score_raw'] = (
    0.4 * df_scaled['last_evaluation'] +
    0.25 * df_scaled['average_montly_hours'] +
    0.15 * df_scaled['number_project'] +
    0.15 * df_scaled['time_spend_company'] -
    0.05 * df_scaled['Work_accident']
)

# -------------------------------------------------
# 4. Normalize efficiency to 0–100 for readability
# -------------------------------------------------
minmax = MinMaxScaler(feature_range=(0, 100))
df['efficiency_score'] = minmax.fit_transform(
    df[['efficiency_score_raw']]
)

# Drop rows where target is NaN
df.dropna(subset=['left'], inplace=True)

# High/Low efficiency label
thresh = df['efficiency_score'].median()
df['efficiency_label'] = (df['efficiency_score'] > thresh).astype(int)

# -------------------------------------------------
# 5. Features and targets
# -------------------------------------------------
features = ['satisfaction_level', 'last_evaluation', 'number_project',
            'average_montly_hours', 'time_spend_company',
            'Work_accident', 'promotion_last_5years',
            'sales', 'salary']

X = df[features]
y_turnover = df['left']
y_efficiency = df['efficiency_label']

# -------------------------------------------------
# 6. Train/Test split
# -------------------------------------------------
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y_turnover, test_size=0.2, random_state=42)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X, y_efficiency, test_size=0.2, random_state=42)

# Imputer
imputer = SimpleImputer(strategy='median')
X_train_t_imp = imputer.fit_transform(X_train_t)
X_test_t_imp = imputer.transform(X_test_t)
X_train_e_imp = imputer.fit_transform(X_train_e)
X_test_e_imp = imputer.transform(X_test_e)

# Scale for Logistic Regression
scaler_X = StandardScaler()
X_train_t_scaled = scaler_X.fit_transform(X_train_t_imp)
X_test_t_scaled = scaler_X.transform(X_test_t_imp)
X_train_e_scaled = scaler_X.fit_transform(X_train_e_imp)
X_test_e_scaled = scaler_X.transform(X_test_e_imp)

# -------------------------------------------------
# 7. Models
# -------------------------------------------------
log_reg_turnover = LogisticRegression(max_iter=1000)
rf_turnover = RandomForestClassifier(n_estimators=100, random_state=42)
hgb_turnover = HistGradientBoostingClassifier(random_state=42)

log_reg_efficiency = LogisticRegression(max_iter=1000)
rf_efficiency = RandomForestClassifier(n_estimators=100, random_state=42)
hgb_efficiency = HistGradientBoostingClassifier(random_state=42)

# Train
log_reg_turnover.fit(X_train_t_scaled, y_train_t)
rf_turnover.fit(X_train_t_imp, y_train_t)
hgb_turnover.fit(X_train_t, y_train_t)

log_reg_efficiency.fit(X_train_e_scaled, y_train_e)
rf_efficiency.fit(X_train_e_imp, y_train_e)
hgb_efficiency.fit(X_train_e, y_train_e)

# -------------------------------------------------
# 8. Evaluate models
# -------------------------------------------------
def evaluate_model(y_true, y_pred, name):
    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall': round(recall_score(y_true, y_pred), 4),
        'F1 Score': round(f1_score(y_true, y_pred), 4)
    }

results = [
    evaluate_model(y_test_t, log_reg_turnover.predict(X_test_t_scaled), 'LogReg - Turnover'),
    evaluate_model(y_test_t, rf_turnover.predict(X_test_t_imp),        'RandomForest - Turnover'),
    evaluate_model(y_test_t, hgb_turnover.predict(X_test_t),           'HGB - Turnover'),
    evaluate_model(y_test_e, log_reg_efficiency.predict(X_test_e_scaled), 'LogReg - Efficiency'),
    evaluate_model(y_test_e, rf_efficiency.predict(X_test_e_imp),         'RandomForest - Efficiency'),
    evaluate_model(y_test_e, hgb_efficiency.predict(X_test_e),            'HGB - Efficiency')
]
print(pd.DataFrame(results))

# -------------------------------------------------
# 9. Predictions for full dataset
# -------------------------------------------------
df['Predicted_Turnover'] = hgb_turnover.predict(X)

# Attach names back
df_full_output = df.copy()
df_full_output['Name'] = employee_names

# -------------------------------------------------
# 10. Turnover amount scaled 20k – 65k
# -------------------------------------------------
# Using normalized efficiency (0–100)
MIN_COST, MAX_COST = 20_000, 65_000
df_full_output['Turnover_Amount'] = (
    MIN_COST + (df_full_output['efficiency_score'] / 100) * (MAX_COST - MIN_COST)
)

# Arrange columns
cols = ['Name', 'efficiency_score', 'Predicted_Turnover',
        'Turnover_Amount'] + \
       [c for c in df_full_output.columns
        if c not in ['Name', 'efficiency_score', 'Predicted_Turnover', 'Turnover_Amount']]
df_full_output = df_full_output[cols]

# -------------------------------------------------
# 11. Save output files
# -------------------------------------------------
base_name = os.path.splitext(os.path.basename(file_path))[0]

# a) Only predicted leavers, sorted by efficiency
sorted_eff_file = f"{base_name}_sorted_efficiency.csv"
leavers = df_full_output[df_full_output['Predicted_Turnover'] == 1]
leavers.sort_values(by='efficiency_score', ascending=False).to_csv(sorted_eff_file, index=False)

# b) All employees, sorted by predicted turnover then efficiency
sorted_turnover_file = f"{base_name}_sorted_turnover.csv"
df_full_output.sort_values(by=['Predicted_Turnover', 'efficiency_score'],
                           ascending=[False, False]).to_csv(sorted_turnover_file, index=False)

print("\nFiles created:")
print(f"1. Employees predicted to leave (sorted by efficiency): {sorted_eff_file}")
print(f"2. All employees sorted by turnover prediction:         {sorted_turnover_file}")
