import pandas as pd
import os
import joblib
import logging
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# **Step 2: Create Output Folder**
xgb_folder = "XGB_Folder"
os.makedirs(xgb_folder, exist_ok=True)

# **Step 3: Load and Prepare Data**
data_folder = "Prepare_datavisual"
data_file = "Expanded_Fingerprints_Data_Cleaned.csv"
data_path = os.path.join(data_folder, data_file)
data = pd.read_csv(data_path)

# Define Input (Fingerprint bits) and Output (Target)
X = data[[col for col in data.columns if col.startswith("FP_")]]
y = data["Aij"]  # Modify if training for other targets
target_name = y.name  # Store the name of the target variable

# **Step 4: Apply Variance Thresholding**
selector = VarianceThreshold(threshold=0.03)  # Remove low-variance fingerprint bits
X_reduced = selector.fit_transform(X)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
X_reduced = pd.DataFrame(X_reduced, columns=selected_features)

# Save selected features for reference
selected_features_txt = os.path.join(xgb_folder, f"selected_features_{target_name}.txt")
selected_features_excel = os.path.join(xgb_folder, f"selected_features_{target_name}.xlsx")

# Save to TXT
with open(selected_features_txt, "w") as f:
    f.write("\n".join(selected_features))

# Save to Excel
pd.DataFrame(selected_features, columns=["Selected Features"]).to_excel(selected_features_excel, index=False)

logger.info(f"Selected features saved to {selected_features_txt} and {selected_features_excel}")

# **Step 5: Standardize the Selected Features**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Save the scaler
scaler_path = os.path.join(xgb_folder, f"scaler_{target_name}.pkl")
joblib.dump(scaler, scaler_path)
logger.info(f"Scaler saved to {scaler_path}")

# **Step 6: Cross-validation with XGBoost**
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# **Step 7: Define XGBoost Hyperparameter Space for Bayesian Optimization**
param_grid = {
    'n_estimators': Integer(100, 1500),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'max_depth': Integer(4, 50),
    'subsample': Real(0.7, 1.0),
    'colsample_bytree': Real(0.1, 1.0),
    'gamma': Real(0, 5),
    'reg_alpha': Real(0, 0.8),
    'reg_lambda': Real(0.1, 10, prior='log-uniform')
}

# **Step 8: Bayesian Optimization**
bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=param_grid,
    scoring='r2',
    n_iter=200,
    cv=kf,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

logger.info(f"Starting Bayesian Optimization for XGBoost on target '{target_name}'...")
bayes_search.fit(X_scaled, y)

# Get the best model
best_xgb = bayes_search.best_estimator_

# **Step 9: Save Best Model**
model_path = os.path.join(xgb_folder, f"best_xgboost_model_{target_name}.pkl")
joblib.dump(best_xgb, model_path)
logger.info(f"Best XGBoost model for target '{target_name}' saved to {model_path}")

# **Step 10: Save Hyperparameter Log**
log_file = os.path.join(xgb_folder, f"tuning_log_{target_name}.txt")
with open(log_file, "w") as f:
    f.write(f"Target: {target_name}\n")
    f.write(f"Best Parameters:\n{bayes_search.best_params_}\n")
logger.info(f"Hyperparameter log for target '{target_name}' saved to {log_file}")

# **Step 11: Cross-validation Results**
cv_results = cross_val_score(best_xgb, X_scaled, y, cv=kf, scoring='r2', n_jobs=-1)

# Log the cross-validation scores
logger.info(f"Cross-validation R² scores for target '{target_name}': {cv_results}")
logger.info(f"Mean R² score: {np.mean(cv_results):.4f}, Standard Deviation: {np.std(cv_results):.4f}")