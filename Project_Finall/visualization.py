import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Input file
input_file = input("Enter Employee efficiency CSV file : ")

# Extract folder name from file name
dataset_name = os.path.splitext(os.path.basename(input_file))[0]
output_folder = os.path.join('visualizations', dataset_name)
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(input_file)

# Set style
sns.set(style="whitegrid")

# Numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# 1. Distribution plots for numeric columns
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(output_folder, f'distribution_{col}.png'))
    plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
plt.close()

# 3. Countplot for Predicted Turnover
plt.figure(figsize=(6, 4))
sns.countplot(x='Predicted_Turnover', data=df, palette='Set2')
plt.title('Predicted Turnover Distribution')
plt.savefig(os.path.join(output_folder, 'predicted_turnover_count.png'))
plt.close()

# 4. Efficiency vs Turnover
plt.figure(figsize=(8, 5))
sns.boxplot(x='Predicted_Turnover', y='efficiency_score', data=df, palette='Set3')
plt.title('Efficiency Score vs Predicted Turnover')
plt.savefig(os.path.join(output_folder, 'efficiency_vs_turnover.png'))
plt.close()

print(f"Visualization plots saved in: {output_folder}")
