import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

def generate_visualizations(input_file, full_report_file, leavers_report_file, output_dir):
    """
    Generates all visualization plots, calculates key metrics, and returns a structured dictionary of results.
    
    Args:
        input_file (str): Path to the original employee data CSV.
        full_report_file (str): Path to the full priority list report CSV.
        leavers_report_file (str): Path to the priority leavers report CSV.
        output_dir (str): Directory where image files should be saved.
        
    Returns:
        dict: Structured metrics and image file paths for Django consumption.
    """
    
    # --- Data Loading ---
    try:
        df_input = pd.read_csv(input_file)
        df_full = pd.read_csv(full_report_file)
        df_leavers = pd.read_csv(leavers_report_file)
    except FileNotFoundError as e:
        return {"error": f"Error loading required files during visualization: {e}"}

    # Setup directories and styles
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150 # Higher resolution for charts
    
    # Dictionary to store file paths for output
    visualization_paths = {}
    
    # --- 1. CALCULATE SUMMARY METRICS (Requested) ---
    total_employees = df_full.shape[0]
    predicted_leavers = df_full['Predicted_Turnover'].sum()
    percentage_turnover_risk = (predicted_leavers / total_employees) * 100

    # Define "High Priority Intervention" as the top 25% of predicted leavers by score
    if not df_leavers.empty:
        # Calculate the 75th percentile score among only the predicted leavers
        priority_threshold = df_leavers['Retention_Priority_Score'].quantile(0.75)
        
        # Count how many employees in the FULL list meet this high priority threshold
        high_priority_employees = df_full[
            (df_full['Predicted_Turnover'] == 1) & 
            (df_full['Retention_Priority_Score'] >= priority_threshold)
        ].shape[0]
        
        percentage_to_retain_focus = (high_priority_employees / total_employees) * 100
    else:
        high_priority_employees = 0
        percentage_to_retain_focus = 0.0

    # --- VISUALIZATION 1: Distribution of Retention Priority Score ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df_full['Retention_Priority_Score'], kde=True, bins=30, color='teal')
    plt.title('1. Distribution of Retention Priority Score', fontsize=14)
    path = os.path.join(output_dir, '1_priority_score_distribution.png')
    plt.savefig(path)
    plt.close()
    visualization_paths['priority_distribution'] = path

    # --- VISUALIZATION 2: Relationship between Cost and Probability ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Estimated_Turnover_Cost', 
        y='Probability_Turnover', 
        hue='Predicted_Turnover', 
        data=df_full, 
        palette='viridis', 
        alpha=0.6,
        size='Retention_Priority_Score',
        sizes=(20, 200)
    )
    plt.title('2. Turnover Risk vs. Estimated Cost', fontsize=14)
    plt_max_cost = df_full['Estimated_Turnover_Cost'].max() * 1.05
    plt.xlim(0, plt_max_cost)
    plt.legend(title='Predicted Turnover', loc='lower right')
    path = os.path.join(output_dir, '2_cost_vs_probability.png')
    plt.savefig(path)
    plt.close()
    visualization_paths['cost_vs_probability'] = path

    # --- VISUALIZATION 3: Turnover Rate by Salary Level ---
    turnover_rate = df_input.groupby('salary')['left'].mean().reset_index(name='Turnover_Rate')
    salary_order = ['low', 'medium', 'high']
    turnover_rate['salary'] = pd.Categorical(turnover_rate['salary'], categories=salary_order, ordered=True)
    turnover_rate.sort_values('salary', ascending=True, inplace=True)
    
    plt.figure(figsize=(8, 5))
    # Using specific colors to ensure robustness against palette errors
    sns.barplot(x='salary', y='Turnover_Rate', data=turnover_rate, palette=['#FF9999', '#99FF99', '#9999FF'], order=salary_order)
    plt.title('3. Employee Turnover Rate by Salary Level', fontsize=14)
    plt.ylim(0, turnover_rate['Turnover_Rate'].max() * 1.1)
    path = os.path.join(output_dir, '3_turnover_by_salary.png')
    plt.savefig(path)
    plt.close()
    visualization_paths['turnover_by_salary'] = path

    # --- VISUALIZATION 4: Top 10 Employees by Retention Priority Score ---
    top_10 = df_leavers.sort_values('Retention_Priority_Score', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Retention_Priority_Score', 
        y='Emp_ID', 
        data=top_10, 
        palette='Reds_d',
        order=top_10['Emp_ID'] 
    )
    plt.title('4. Top 10 Employees for Retention Intervention', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, '4_top_10_priority.png')
    plt.savefig(path)
    plt.close()
    visualization_paths['top_10_priority'] = path

    # --- VISUALIZATION 5: Efficiency vs Predicted Turnover ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Predicted_Turnover', y='efficiency_score', data=df_full, palette='Set3')
    plt.title('5. Employee Efficiency Score vs Predicted Turnover', fontsize=14)
    plt.xticks([0, 1], ['0 - Predicted Stay', '1 - Predicted Leave'])
    path = os.path.join(output_dir, '5_efficiency_vs_predicted_turnover.png')
    plt.savefig(path)
    plt.close()
    visualization_paths['efficiency_vs_turnover'] = path

    # --- VISUALIZATION 6: Project Count vs Average Monthly Hours ---
    df_left = df_input[df_input['left'] == 1] 
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='number_project', 
        y='average_montly_hours', 
        data=df_input, 
        hue='left',
        palette=['blue', 'red'],
        alpha=0.6,
        style='left',
        s=50
    )
    plt.title('6. Project Count vs. Average Monthly Hours (Burnout Indicator)', fontsize=14)
    plt.legend(title='Actually Left', labels=['Stayed (0)', 'Left (1)'])
    path = os.path.join(output_dir, '6_projects_vs_hours.png')
    plt.savefig(path)
    plt.close()
    visualization_paths['projects_vs_hours'] = path
    
    # --- FINAL STRUCTURED OUTPUT ---
    final_output = {
        "metrics": {
            "total_employees": total_employees,
            "predicted_leavers_count": predicted_leavers,
            "turnover_risk_percent": round(percentage_turnover_risk, 2),
            "high_priority_employees_count": high_priority_employees,
            "high_priority_retain_percent": round(percentage_to_retain_focus, 2)
        },
        "visualization_paths": visualization_paths
    }
    
    return final_output

# NOTE: The calling code in Django's main_script.py should now import this 
# function and use its dictionary return value directly.
# This prevents potential issues with capturing stdout JSON.