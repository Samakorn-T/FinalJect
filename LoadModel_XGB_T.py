import pandas as pd
import os
import joblib
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# **Step 2: Create Output Folder**
load_folder = "Load_Model"
os.makedirs(load_folder, exist_ok=True)

# **Step 3: Load Data**
data_folder = "Prepare_datavisual"
data_file = "Expanded_Fingerprints_Data_Cleaned.csv"
data_path = os.path.join(data_folder, data_file)
data = pd.read_csv(data_path)

# Extract "Name i" and "Name j" for reference
name_i = data["Name i"]
name_j = data["Name j"]

# Define Input (Fingerprint bits)
X = data[[col for col in data.columns if col.startswith("FP_")]]
y = data["Aij"]  # Target
target_name = y.name  # Store the target variable name

# **Step 4: Load Preprocessing Tools and Model**
xgb_folder = "XGB_Folder"
selected_features_path = os.path.join(xgb_folder, f"selected_features_{target_name}.txt")

# Handle missing feature selection file
if os.path.exists(selected_features_path):
    with open(selected_features_path, "r") as f:
        selected_features = f.read().splitlines()
else:
    logger.warning(f"Feature selection file {selected_features_path} not found! Using all features.")
    selected_features = list(X.columns)

# Filter the dataset to match selected features
X_selected = X[selected_features]

# Load the scaler and model
scaler_path = os.path.join(xgb_folder, f"scaler_{target_name}.pkl")
model_path = os.path.join(xgb_folder, f"best_xgboost_model_{target_name}.pkl")

scaler = joblib.load(scaler_path)
best_xgb = joblib.load(model_path)
logger.info("Loaded saved model, scaler, and selected features.")

# **Step 5: Apply Standardization**
X_scaled = scaler.transform(X_selected)

# **Step 6: Make Predictions**
y_pred = best_xgb.predict(X_scaled)

# **Step 7: Compute Metrics**
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

logger.info(f"Model Performance on Full Data - R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

# **Step 8: Save Predictions to Excel**
predictions_df = pd.DataFrame({
    "Name i": name_i,
    "Name j": name_j,
    "Actual Bij": y,
    "Predicted Bij": y_pred
})

excel_path = os.path.join(load_folder, f"XGBoost_Predictions_{target_name}.xlsx")
predictions_df.to_excel(excel_path, index=False)
logger.info(f"Predictions saved to {excel_path}")

# **Step 9: Plot Actual vs Predicted**
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.7, label="Predicted vs Actual", color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Fit")
plt.xlabel("Actual Bij", fontsize=12)
plt.ylabel("Predicted Bij", fontsize=12)
plt.title(f"XGBoost Prediction on Full Dataset ({target_name})", fontsize=14)
plt.legend(loc="upper left", fontsize=10)

# Add metrics as a text box
plt.text(0.95, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}",
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment="top",
         horizontalalignment="right",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

# **Save Plot as PNG**
plot_path = os.path.join(load_folder, f"XGBoost_FullData_Prediction_{target_name}.png")
plt.savefig(plot_path)
plt.show()
logger.info(f"Saved full dataset prediction plot to {plot_path}")