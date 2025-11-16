import pandas as pd
import matplotlib
# We still set the backend to Agg, as Seaborn might otherwise try to use a GUI backend
matplotlib.use('Agg')
from matplotlib.figure import Figure  # <-- Import Figure directly
import seaborn as sns
import os

def visualize_data(input_file, full_report_file, leavers_report_file):
    """
    Generates and saves visualizations for the employee turnover data.
    ... (rest of your docstring) ...
    """
    try:
        # Load all required dataframes
        df_input = pd.read_csv(input_file)
        df_full = pd.read_csv(full_report_file)
        df_leavers = pd.read_csv(leavers_report_file) 
        df_input = df_input.drop(columns=['left'])
        
    except FileNotFoundError as e:
        return {"success": False, "error": f"Error loading required files during visualization: {e}"}

    try:
        sns.set_style("whitegrid")
        # plt.rcParams is part of pyplot, so we set it on the matplotlib module
        matplotlib.rcParams['figure.dpi'] = 150
        
        dataset_name = os.path.splitext(os.path.basename(input_file))[0]
        output_folder = os.path.join('visualizations', dataset_name)
        os.makedirs(output_folder, exist_ok=True)

        image_paths = []
        
        df = df_full

        # 6. Total Turnover Cost KPI Card
        if 'Predicted_Turnover' in df.columns and 'Estimated_Turnover_Cost' in df.columns:
            print("Generating total turnover cost KPI (from full_report_file)...")
            if df['Predicted_Turnover'].dtype == 'object':
                leavers_df = df[df['Predicted_Turnover'] == 'Yes']
            else:
                leavers_df = df[df['Predicted_Turnover'] == 1]
                
            total_cost = leavers_df['Estimated_Turnover_Cost'].sum()
            cost_text = f"${total_cost:,.2f}"

            # --- New OO Plot ---
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            
            ax.text(0.5, 0.6, 'Total Predicted Turnover Cost',
                        ha='center', va='center', fontsize=20, color='gray')
            ax.text(0.5, 0.35, cost_text,
                        ha='center', va='center', fontsize=40, color='black', weight='bold')
            ax.axis('off')
            
            img_path = os.path.join(output_folder, 'total_turnover_cost_kpi.png')
            fig.savefig(img_path, bbox_inches='tight', pad_inches=0.1) # Use fig.savefig
            abs_img_path = os.path.abspath(img_path)
            image_paths.append(abs_img_path)
            # No plt.close() needed
        else:
            print("Warning: 'Predicted_Turnover' or 'Turnover_Cost' column not found in full_report_file. Skipping KPI card.")
        
        if 'Predicted_Turnover' in df.columns:
            print("Generating turnover percentage KPI (from full_report_file)...")
            turnover_counts = df['Predicted_Turnover'].value_counts()
            
            if not turnover_counts.empty:
                if df['Predicted_Turnover'].dtype == 'object':
                    leavers_count = turnover_counts.get('Yes', 0)
                    stayers_count = turnover_counts.get('No', 0)
                else:
                    leavers_count = turnover_counts.get(1, 0)
                    stayers_count = turnover_counts.get(0, 0)

                total_count = leavers_count + stayers_count
                
                if total_count > 0:
                    percent_leavers = (leavers_count / total_count) * 100
                    percent_stayers = (stayers_count / total_count) * 100
                else:
                    percent_leavers = 0.0
                    percent_stayers = 0.0

                # --- New OO Plot ---
                fig = Figure(figsize=(8, 4))
                ax = fig.add_subplot(111)
                
                ax.text(0.5, 0.7, 'Predicted Turnover Percentage',
                            ha='center', va='center', fontsize=20, color='gray')
                ax.text(0.25, 0.4, 'To Leave',
                            ha='center', va='center', fontsize=18, color='black')
                ax.text(0.25, 0.2, f"{percent_leavers:.1f}%",
                            ha='center', va='center', fontsize=30, color='#d9534f', weight='bold') # Red color
                ax.text(0.75, 0.4, 'To Stay',
                            ha='center', va='center', fontsize=18, color='black')
                ax.text(0.75, 0.2, f"{percent_stayers:.1f}%",
                            ha='center', va='center', fontsize=30, color='#5cb85c', weight='bold') # Green color
                ax.axis('off')
                
                img_path = os.path.join(output_folder, 'turnover_percentage_kpi.png')
                fig.savefig(img_path, bbox_inches='tight', pad_inches=0.1) # Use fig.savefig
                abs_img_path = os.path.abspath(img_path)
                image_paths.append(abs_img_path)
                # No plt.close() needed
            else:
                    print("Warning: 'Predicted_Turnover' column is empty in full_report_file. Skipping percentage KPI.")
        else:
            print("Warning: 'Predicted_Turnover' column not found in full_report_file. Skipping percentage KPI.")
            
        if 'Predicted_Turnover' in df.columns and 'efficiency_score' in df.columns:
            print("Generating efficiency vs. turnover boxplot (from full_report_file)...")
            
            # --- New OO Plot ---
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111) 
            sns.boxplot(x='Predicted_Turnover', y='efficiency_score', data=df, palette='Set3', ax=ax) # Pass ax=ax
            ax.set_title('Efficiency Score vs Predicted Turnover') # Use ax.set_title
            img_path = os.path.join(output_folder, 'efficiency_vs_turnover.png')
            fig.tight_layout() # Use fig.tight_layout
            fig.savefig(img_path) # Use fig.savefig
            abs_img_path = os.path.abspath(img_path) 
            image_paths.append(abs_img_path)
            # No plt.close() needed
        else:
            print("Warning: 'Predicted_Turnover' or 'efficiency_score' column not found in full_report_file. Skipping boxplot.")
            
        df = df_input 
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # 1. Distribution plots for numeric columns
        print("Generating distribution plots (from input_file)...")
        for col in numeric_cols:
            # --- New OO Plot ---
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            sns.histplot(df[col], bins=30, kde=True, ax=ax) # Pass ax=ax
            ax.set_title(f'Distribution of {col}') # Use ax.set_title
            
            img_path = os.path.join(output_folder, f'distribution_{col}.png')
            fig.tight_layout() # Use fig.tight_layout
            fig.savefig(img_path) # Use fig.savefig
            abs_img_path = os.path.abspath(img_path) 
            image_paths.append(abs_img_path)
            # No plt.close() needed

        # 2. Correlation heatmap
        if not numeric_cols.empty:
            print("Generating correlation heatmap (from input_file)...")
            
            # --- New OO Plot ---
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax) # Pass ax=ax
            ax.set_title('Correlation Heatmap') # Use ax.set_title
            
            img_path = os.path.join(output_folder, 'correlation_heatmap.png')
            fig.tight_layout() # Use fig.tight_layout
            fig.savefig(img_path) # Use fig.savefig
            abs_img_path = os.path.abspath(img_path) 
            image_paths.append(abs_img_path)
            # No plt.close() needed

        print(f"Visualization plots saved in: {output_folder}")
        
        return image_paths

    except Exception as e:
        return {"success": False, "error": f"An error occurred during plot generation: {e}"}
