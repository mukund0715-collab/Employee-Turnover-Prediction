import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_efficiency(df):
    """
    Calculates the raw Z-score based efficiency score and the 0-100 normalized score.

    Args:
        df (pd.DataFrame): The main DataFrame.

    Returns:
        tuple: (df, df_scaled_efficiency) - The updated DataFrame and a DataFrame 
               containing only the Z-scores used for efficiency (needed later for cost).
    """
    num_cols = ['last_evaluation', 'average_montly_hours',
                'number_project', 'time_spend_company', 'Work_accident']

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

    # Efficiency score calculation using weighted sum of Z-scores
    df['efficiency_score_raw'] = (
        0.4 * df_scaled['last_evaluation'] +
        0.25 * df_scaled['average_montly_hours'] +
        0.15 * df_scaled['number_project'] +
        0.15 * df_scaled['time_spend_company'] -
        0.05 * df_scaled['Work_accident']
    )
    
    # Store the raw Z-scores used for efficiency and align index
    df_efficiency_scaled = df_scaled[['last_evaluation', 'average_montly_hours',
                                     'number_project', 'time_spend_company', 'Work_accident']].copy()
    df_efficiency_scaled.index = df.index

    # Normalize efficiency to 0–100 for readability
    minmax = MinMaxScaler(feature_range=(0, 100))
    df['efficiency_score'] = minmax.fit_transform(
        df[['efficiency_score_raw']]
    )

    # High/Low efficiency label (for potential classification task, not used in final report)
    thresh = df['efficiency_score'].median()
    df['efficiency_label'] = (df['efficiency_score'] > thresh).astype(int)
    
    return df, df_efficiency_scaled