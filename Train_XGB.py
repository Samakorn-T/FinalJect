import pandas as pd
import numpy as np
import os
import joblib
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# **Step 2: Create Output Folder**
xgb_folder = "XGB_Folder"
os.makedirs(xgb_folder, exist_ok=True)

# **Step 3: Load and Prepare Data**
data_folder = "Prepare_datavisual"
data_file = "Filtered_Fingerprints_Data_Updated.xlsx"
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

# Define Input (Fingerprint bits) and Output (Bij Target)
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input features
y = data["Bij"]  # Train only for "Bij" target

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler inside XGB_Folder
scaler_path = os.path.join(xgb_folder, "scaler.pkl")
joblib.dump(scaler, scaler_path)

# Set up K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# **Step 4: Define XGBoost Hyperparameter Space**
param_grid = {
    'n_estimators': Integer(100, 1000),
    'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 20),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 5),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0.1, 10, prior='log-uniform')
}

# **Step 5: Bayesian Optimization with XGBoost**
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=param_grid,
    scoring='r2',
    n_iter=150,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

logger.info("Starting Bayesian Optimization for XGBoost...")
bayes_search.fit(X_scaled, y)

# Get the best model
best_xgb = bayes_search.best_estimator_

# **Step 6: Model Evaluation**
y_pred = best_xgb.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

logger.info(f"XGBoost Best Parameters: {bayes_search.best_params_}")
logger.info(f"XGBoost Performance - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

# **Step 7: Save Best Model**
model_path = os.path.join(xgb_folder, "best_xgboost_model.pkl")
joblib.dump(best_xgb, model_path)
logger.info(f"Best XGBoost model saved to {model_path}")

# **Step 8: Save Hyperparameter Log**
log_file = os.path.join(xgb_folder, "tuning_log.txt")
with open(log_file, "w") as f:
    f.write(f"Best Parameters:\n{bayes_search.best_params_}\n")
    f.write("Performance Metrics:\n")
    f.write(f"R²: {r2:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}\n")
logger.info(f"Hyperparameter log saved to {log_file}")

# **Step 9: Plot Actual vs Predicted**
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.7, label="Predicted vs Actual", color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Fit")
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("XGBoost Prediction for Bij", fontsize=14)
plt.legend(loc="upper left", fontsize=10)

# Add metrics as a text box
plt.text(0.95, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}",
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment="top",
         horizontalalignment="right",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plot_path = os.path.join(xgb_folder, "XGBoost_Bij_Prediction.png")
plt.savefig(plot_path)
plt.show()

logger.info(f"Saved prediction plot to {plot_path}")