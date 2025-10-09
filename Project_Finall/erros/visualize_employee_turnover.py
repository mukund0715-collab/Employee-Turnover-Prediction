# visualize_employee_turnover.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ========== CONFIGURATION ==========
INPUT_CSV = 'HR_comma_sep.csv'  # original input file name
PREDICTED_CSV = 'employee_predictions_full.csv'

# Create folder for analysis
folder_name = os.path.splitext(INPUT_CSV)[0] + '_analysis'
plots_folder = os.path.join(folder_name, 'plots')
os.makedirs(plots_folder, exist_ok=True)

# Copy predicted CSV for reference
if os.path.exists(PREDICTED_CSV):
    os.makedirs(folder_name, exist_ok=True)
    pd.read_csv(PREDICTED_CSV).to_csv(os.path.join(folder_name, PREDICTED_CSV), index=False)

# ========== LOAD DATA ==========
df = pd.read_csv(PREDICTED_CSV)

# Basic columns
features = ['last_evaluation', 'average_montly_hours', 'number_project', 'time_spend_company', 'Work_accident']

# Encode categorical if needed
if 'salary' in df.columns:
    le = LabelEncoder()
    df['salary'] = le.fit_transform(df['salary'].astype(str))
    features.append('salary')
if 'Department' in df.columns:
    le = LabelEncoder()
    df['Department'] = le.fit_transform(df['Department'].astype(str))
    features.append('Department')

# ========== FEATURE IMPORTANCE (RF & HGB) ==========
X = df[features]
y = df['Predicted_Turnover']

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=rf_importances, y=rf_importances.index)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'feature_importance_rf.png'))
plt.close()

# HistGradientBoosting
hgb = HistGradientBoostingClassifier(random_state=42)
hgb.fit(X, y)
hgb_importances = pd.Series(hgb.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=hgb_importances, y=hgb_importances.index)
plt.title('Feature Importance - HGB')
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'feature_importance_hgb.png'))
plt.close()

# ========== CORRELATION HEATMAP ==========
plt.figure(figsize=(10,8))
sns.heatmap(df[features + ['efficiency_score']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'correlation_heatmap.png'))
plt.close()

# ========== EFFICIENCY DISTRIBUTION ==========
plt.figure(figsize=(8,5))
sns.histplot(df['efficiency_score'], bins=20, kde=True)
plt.title('Efficiency Score Distribution')
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'efficiency_distribution.png'))
plt.close()

# ========== EFFICIENCY VS TURNOVER ==========
plt.figure(figsize=(8,5))
sns.boxplot(x='Predicted_Turnover', y='efficiency_score', data=df)
plt.title('Efficiency vs Predicted Turnover')
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'efficiency_vs_turnover.png'))
plt.close()

# ========== CONFUSION MATRIX & ROC ==========
# Split again for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'HGB': HistGradientBoostingClassifier(random_state=42)
}

plt.figure(figsize=(8,6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})')

    # Confusion Matrix for HGB
    if name == 'HGB':
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Leave'])
        disp.plot(cmap='Blues')
        plt.savefig(os.path.join(plots_folder, 'confusion_matrix.png'))
        plt.close()

# Final ROC
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'roc_curve.png'))
plt.close()

print(f'All plots saved in: {plots_folder}')
