import pandas as pd
import numpy as np

def calculate_turnover_cost_and_priority(df, employee_ID, log_reg_model, X_scaled_full, df_efficiency_scaled):
    """
    Calculates predicted turnover probability, estimated cost, and retention priority score.

    Args:
        df (pd.DataFrame): The main DataFrame containing features and labels.
        employee_names (pd.Series): The series of employee names.
        log_reg_model (LogisticRegression): The trained turnover model.
        X_scaled_full (np.array): All features scaled (Z-scores).
        df_efficiency_scaled (pd.DataFrame): Z-scores of efficiency features.

    Returns:
        pd.DataFrame: The final output DataFrame with all calculated metrics.
    """
    
    # 1. Predictions
    # Calculate Probability_Turnover using the trained model
    df['Probability_Turnover'] = log_reg_model.predict_proba(X_scaled_full)[:, 1]

    THRESHOLD = 0.5
    df['Predicted_Turnover'] = (df['Probability_Turnover'] >= THRESHOLD).astype(int)

    df_full_output = df.copy()
    df_full_output['Emp_ID'] = employee_ID.reset_index(drop=True)
    
    # 2. Financial Integration
    SALARY_MAP = {0: 45000, 1: 65000, 2: 100000}
    BASE_COST_MULTIPLIER = 0.5 # 50% base cost

    # Weighting for Difficulty-to-Replace Multiplier (Heuristic based on complexity)
    W_TENURE = 0.20
    W_PROJECTS = 0.15
    W_EFFICIENCY = 0.10
    
    df_full_output['Annual_Salary_Temp'] = df_full_output['salary'].map(SALARY_MAP)

    # Create scaled features DataFrame from the numpy array for easy access
    features = ['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company',
                'Work_accident', 'promotion_last_5years',
                'sales', 'salary']
    X_scaled_full_df = pd.DataFrame(X_scaled_full, columns=features)
    X_scaled_full_df.index = df_full_output.index

    # Calculate Complexity Multiplier
    # NOTE: df_efficiency_scaled['efficiency_score_raw'] contains the Z-scores used for efficiency
    df_full_output['Complexity_Multiplier'] = (
        BASE_COST_MULTIPLIER +
        W_TENURE * X_scaled_full_df['time_spend_company'] +
        W_PROJECTS * X_scaled_full_df['number_project'] +
        W_EFFICIENCY * df['efficiency_score_raw']
    )
    df_full_output['Complexity_Multiplier'] = df_full_output['Complexity_Multiplier'].clip(lower=0.5)

    # Calculate Estimated Turnover Cost
    df_full_output['Estimated_Turnover_Cost'] = (
        df_full_output['Annual_Salary_Temp'] * df_full_output['Complexity_Multiplier']
    )

    # Calculate Retention Priority Score
    df_full_output['Retention_Priority_Score'] = (
        df_full_output['Probability_Turnover'] * df_full_output['Estimated_Turnover_Cost']
    )

    # Remove temporary columns
    df_full_output.drop(columns=['Annual_Salary_Temp', 'Complexity_Multiplier', 
                                 'efficiency_score_raw', 'efficiency_label', 'left', 'salary', 'sales'] +
                                 list(df_efficiency_scaled.columns), 
                        errors='ignore', inplace=True)

    # Arrange final columns
    final_cols = ['Emp_ID', 'efficiency_score', 'Probability_Turnover', 'Predicted_Turnover',
                  'Estimated_Turnover_Cost', 'Retention_Priority_Score'] + \
                 [c for c in df_full_output.columns
                  if c not in ['Emp_ID', 'efficiency_score', 'Probability_Turnover', 'Predicted_Turnover',
                               'Estimated_Turnover_Cost', 'Retention_Priority_Score']]
    df_full_output = df_full_output[final_cols]
    
    return df_full_output