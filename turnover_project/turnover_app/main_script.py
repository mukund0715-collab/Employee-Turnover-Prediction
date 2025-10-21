import pandas as pd
import numpy as np
import os
import pickle
import tempfile # Added for saving reports to a temporary location
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# NOTE: In your Django project, these modules must exist as separate files
# and be importable. The main function below assumes they are available.
from .efficiency_module import calculate_efficiency
from .turnover_model import load_or_train_turnover_model, PREPROCESSOR_FILENAMES, MODEL_FILENAME
from .cost_module import calculate_turnover_cost_and_priority

# --- CONFIGURATION / GLOBAL VARIABLES ---
FEATURES = ['satisfaction_level', 'last_evaluation', 'number_project',
            'average_montly_hours', 'time_spend_company',
            'Work_accident', 'promotion_last_5years',
            'sales', 'salary']

def run_turnover_analysis(file_path, base_name):
    """
    Runs the full ML pipeline (Load, Preprocess, Warm-Start Train, Predict, Cost)
    on the provided CSV file and saves the reports.

    Args:
        file_path (str): The path to the uploaded CSV file.
        base_name (str): The file name base (e.g., 'EmployeeData_Jan') for report naming.

    Returns:
        tuple: (path_to_leavers_report_csv, path_to_full_list_csv)
    """

    # -------------------------------------------------
    # 1. Load dataset & Setup (Using arguments instead of input())
    # -------------------------------------------------
    df = pd.read_csv(file_path)

    # Use Emp_ID as the identifier
    employee_ID = df['Emp_ID'].reset_index(drop=True)
    df.drop(columns=['Emp_ID'], inplace=True)

    # -------------------------------------------------
    # 2. Encode categorical columns
    # -------------------------------------------------
    le_salary = LabelEncoder()
    df['salary'] = le_salary.fit_transform(df['salary'])
    le_sales = LabelEncoder()
    df['sales'] = le_sales.fit_transform(df['sales'])

    # -------------------------------------------------
    # 3. Calculate Efficiency (Using Module)
    # -------------------------------------------------
    df, df_efficiency_scaled = calculate_efficiency(df)
    df.dropna(subset=['left'], inplace=True)

    X = df[FEATURES]
    y_turnover = df['left']

    # -------------------------------------------------
    # 4. Train/Test split
    # -------------------------------------------------
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X, y_turnover, test_size=0.2, random_state=42)

    # -------------------------------------------------
    # 5. Train/Update Turnover Model (Using Module)
    # -------------------------------------------------
    (log_reg_turnover, imputer, scaler_X, X_test_t_scaled) = \
        load_or_train_turnover_model(X_train_t, X_test_t, y_train_t, FEATURES)

    # -------------------------------------------------
    # 6. Evaluate Model (LogReg only)
    # -------------------------------------------------
    def evaluate_model(y_true, y_pred, name):
        # Evaluation function for immediate comparison (optional to keep in Django)
        return {
            'Model': name,
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall': round(recall_score(y_true, y_pred), 4),
            'F1 Score': round(f1_score(y_true, y_pred), 4)
        }

    y_pred_t = log_reg_turnover.predict(X_test_t_scaled)
    # print("\n--- Model Performance Comparison (on Current Test Data) ---")
    # print(pd.DataFrame([evaluate_model(y_test_t, y_pred_t, 'LogReg - Turnover (Updated/Trained)')]))

    # -------------------------------------------------
    # 7. Predictions & Cost Calculation (Using Module)
    # -------------------------------------------------
    X_imp_full = imputer.transform(X)
    X_scaled_full = scaler_X.transform(X_imp_full)

    # Calculate predictions and cost (passing employee_ID)
    df_full_output = calculate_turnover_cost_and_priority(
        df, employee_ID, log_reg_turnover, X_scaled_full, df_efficiency_scaled
    )

    # -------------------------------------------------
    # 8. Save Trained Model and Reports
    # -------------------------------------------------
    # a) Save Updated Model Files (Fixed Names)
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(log_reg_turnover, file)
    with open(PREPROCESSOR_FILENAMES['SCALER'], 'wb') as file:
        pickle.dump(scaler_X, file)
    with open(PREPROCESSOR_FILENAMES['IMPUTER'], 'wb') as file:
        pickle.dump(imputer, file)

    # b) Save Output Reports (Dynamic Names, saving to TEMP_DIR for Django access)
    temp_dir = tempfile.gettempdir()
    
    REPORT_LEAVERS_FILE = os.path.join(temp_dir, f"{base_name}_priority_leavers_report.csv")
    leavers = df_full_output[df_full_output['Predicted_Turnover'] == 1]
    leavers.sort_values(by='Retention_Priority_Score', ascending=False).to_csv(REPORT_LEAVERS_FILE, index=False)

    REPORT_FULL_LIST_FILE = os.path.join(temp_dir, f"{base_name}_priority_full_list_report.csv")
    df_full_output.sort_values(by=['Predicted_Turnover', 'Retention_Priority_Score'],
                               ascending=[False, False]).to_csv(REPORT_FULL_LIST_FILE, index=False)

    # Return the paths to the generated reports for the Django view to handle zipping/download
    return REPORT_LEAVERS_FILE, REPORT_FULL_LIST_FILE

# NOTE: The input() function has been removed. 
# The function will run when called by the Django view.
