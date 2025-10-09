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

# Preserve employee ID for output
employee_ID = df['Emp_ID']
df.drop(columns=['Emp_ID'], inplace=True)

beginning_employees = len(df)

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

weights = {
    'w0_intercept': -4.0,  # This is the bias or intercept term
    'w1_satisfaction': 2.5,
    'w2_hours_worked': -0.01,
    'w3_projects': 0.3
}

df['z'] = (
    weights['w0_intercept'] +
    (weights['w1_satisfaction'] * df['satisfaction_level']) +
    (weights['w2_hours_worked'] * df['average_montly_hours']) +
    (weights['w3_projects'] * df['number_project'])
)

# --- 4. Apply the Sigmoid Function to Calculate Probability ---
# This is the core of the formula: P(turnover) = 1 / (1 + e^-z)
df['efficiency_score_raw'] = 1 / (1 + np.exp(-df['z']))

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
def evaluate_model(y_true, y_pred, ID):
    return {
        'Model': ID,
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
employees_left = int(df['Predicted_Turnover'].sum())
ending_employees = beginning_employees - employees_left


# Attach ID back
df_full_output = df.copy()
df_full_output['Emp_ID'] = employee_ID

 

# To find the average number of employees: (Beginning + Ending) / 2
average_employees = (beginning_employees + ending_employees) / 2

# Formula: (Number of Employees Who Left / Average Number of Employees)
# We calculate both the decimal and percentage for later use.
turnover_rate_decimal = employees_left / average_employees
turnover_rate_percent = turnover_rate_decimal * 100


# --- 2. Determine the Cost Per Departure ---
# This section breaks down the costs associated with a single employee leaving.
# These are estimates and should be adjusted for your specific company.
cost_per_departure = {
    'Recruitment Costs': 5000,      # Advertising, agency fees, background checks
    'Onboarding Costs': 2500,       # Administrative costs, IT setup, orientation
    'Training Costs': 7500,         # Materials, trainers' salaries, equipment
    'Lost Productivity': 25000,     # Ramp-up time for the replacement
    'Knowledge Loss': 8000,         # Value of experience the employee took
    'Indirect Costs': 2000          # Time spent by managers/colleagues
}

# Sum all the individual costs to get the average cost for one departure.
average_cost_per_departure = sum(cost_per_departure.values())


# --- 3. Calculate the Total Turnover Cost ---
# Formula: Total Employees x Turnover Rate (decimal) x Average Cost Per Departure
# Note: We use the average number of employees for this calculation.
total_annual_turnover_cost = (
    average_employees *
    turnover_rate_decimal *
    average_cost_per_departure
)

# Arrange columns
cols = ['Emp_ID', 'efficiency_score', 
        'Predicted_Turnover'] + \
       [c for c in df_full_output.columns
        if c not in ['Emp_ID', 'efficiency_score', 'Predicted_Turnover']]
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
print("The Total TurnOver cost of the year: ", int(total_annual_turnover_cost))
print(f"1. Employees predicted to leave (sorted by efficiency): {sorted_eff_file}")
print(f"2. All employees sorted by turnover prediction:         {sorted_turnover_file}")
