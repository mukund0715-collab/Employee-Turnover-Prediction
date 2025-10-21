import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# --- FIXED FILENAMES for Persistence ---
MODEL_BASE_NAME = "turnover_analysis"
MODEL_FILENAME = f"{MODEL_BASE_NAME}_log_reg_turnover_model.pkl"
PREPROCESSOR_FILENAMES = {
    'SCALER': f"{MODEL_BASE_NAME}_scaler_X.pkl",
    'IMPUTER': f"{MODEL_BASE_NAME}_imputer.pkl"
}

def load_or_train_turnover_model(X_train, X_test, y_train, features):
    """
    Loads saved preprocessors if they exist, or fits new ones.
    Loads the previous LogReg model for warm start, or initializes a new one.
    Trains/updates the LogReg model and saves all objects.

    Returns:
        tuple: (log_reg_turnover, imputer, scaler_X, X_test_scaled)
    """

    # 1. Handle Preprocessing (Imputer and Scaler)
    if os.path.exists(PREPROCESSOR_FILENAMES['IMPUTER']) and os.path.exists(PREPROCESSOR_FILENAMES['SCALER']):
        print("Loading saved Imputer and Scaler for consistent transformation...")
        with open(PREPROCESSOR_FILENAMES['IMPUTER'], 'rb') as f:
            imputer = pickle.load(f)
        with open(PREPROCESSOR_FILENAMES['SCALER'], 'rb') as f:
            scaler_X = pickle.load(f)

        X_train_imp = imputer.transform(X_train)
        X_test_imp = imputer.transform(X_test)
        X_train_scaled = scaler_X.transform(X_train_imp)
        X_test_scaled = scaler_X.transform(X_test_imp)

    else:
        print("Fitting new Imputer and Scaler...")
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_imp)
        X_test_scaled = scaler_X.transform(X_test_imp)
    
    # 2. Handle Model Loading and Initialization
    if os.path.exists(MODEL_FILENAME):
        with open(MODEL_FILENAME, 'rb') as f:
            log_reg_turnover = pickle.load(f)
        log_reg_turnover.set_params(warm_start=True, max_iter=1000)
    else:
        log_reg_turnover = LogisticRegression(max_iter=1000, warm_start=True)

    # 3. Train/Update the Model
    log_reg_turnover.fit(X_train_scaled, y_train)

    return log_reg_turnover, imputer, scaler_X, X_test_scaled