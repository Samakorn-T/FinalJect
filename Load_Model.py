import os
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# **Step 2: Define Paths**
main_output_folder = "Main_Output_Folder"
model_folder = os.path.join(main_output_folder, "Models")
metrics_folder = os.path.join(main_output_folder, "Metrics")
plot_main_folder = os.path.join(main_output_folder, "Plots")
plot_eval_folder = os.path.join(plot_main_folder, "Model_Evaluation")  # New subfolder for evaluation plots

# Create directories if they don't exist
os.makedirs(metrics_folder, exist_ok=True)
os.makedirs(plot_eval_folder, exist_ok=True)

# **Step 3: Load Dataset**
data_folder = 'Prepare_datavisual'
data_file = 'Filtered_Fingerprints_Data_Updated.xlsx'
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

# Extract Inputs and Targets
X = data[[col for col in data.columns if col.startswith("FP_")]]
y = data[["Aij", "Bij", "Alpha"]]

# Load StandardScaler used in training
scaler_path = os.path.join(main_output_folder, "scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    logger.info("Loaded existing StandardScaler.")
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    logger.info("StandardScaler not found. Re-trained and saved.")

# **Step 4: Load Saved Models**
saved_models = {}
for model_file in os.listdir(model_folder):
    if model_file.endswith(".pkl"):
        model_name = model_file.replace(".pkl", "")
        model_path = os.path.join(model_folder, model_file)
        saved_models[model_name] = joblib.load(model_path)
        logger.info(f"Loaded model: {model_name}")

# **Step 5: Evaluate Models**
metrics_log = []
epsilon = 1e-10  # Small constant to avoid division by zero

for model_name, model in saved_models.items():
    target = model_name.split("_")[-1]  # Extract target variable name
    y_true = y[target].values
    y_pred = model.predict(X_scaled)

    # Compute Metrics (Safe Division)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100  # MAPE
    mspe = np.mean(((y_true - y_pred) / (y_true + epsilon)) ** 2) * 100  # MSPE

    # Log Results
    metrics_log.append({
        "Model": model_name.split("_")[0],  # Extract model name
        "Target": target,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "MAPE (%)": mape,
        "MSPE (%)": mspe
    })

    logger.info(f"{model_name} - R²: {r2:.4f}, MAPE: {mape:.4f}%, MSPE: {mspe:.4f}%")

    # **Step 6: Plot Actual vs. Predicted Scatter Plot**
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label='Predicted vs Actual', color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{model_name} - Actual vs Predicted for {target}", fontsize=14)
    plt.legend(loc='upper left', fontsize=10)

    # Add Metrics as a Text Box
    plt.text(0.95, 0.05, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}\nMAPE: {mape:.4f}%\nMSPE: {mspe:.4f}%",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # Save the plot
    plot_path = os.path.join(plot_eval_folder, f"{model_name}_{target}_Actual_vs_Predicted.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    logger.info(f"Saved actual vs predicted plot: {plot_path}")

# **Step 7: Save Updated Metrics Log**
metrics_df = pd.DataFrame(metrics_log)
metrics_log_file = os.path.join(metrics_folder, "Updated_Metrics_Log.csv")
metrics_df.to_csv(metrics_log_file, index=False)
logger.info(f"Saved updated metrics log to {metrics_log_file}")

# **Step 8: Bar Plots for MAPE, MSPE, and R²**
sns.set_style("whitegrid")

for metric in ["MAPE (%)", "MSPE (%)", "R2"]:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Target", y=metric, hue="Model", data=metrics_df, palette="viridis")
    plt.title(f"{metric} for Different Models", fontsize=14)
    plt.xlabel("Target Variable", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the plot
    plot_path = os.path.join(plot_eval_folder, f"{metric.replace(' ', '_')}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    logger.info(f"Saved plot: {plot_path}")