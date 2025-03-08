import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# **Step 2: Load and Prepare Data**
data_folder = 'Prepare_datavisual'
data_file = 'Filtered_Fingerprints_Data_Updated.xlsx'
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

# Define Input (Fingerprint bits) and Output (Targets)
X = data[[col for col in data.columns if col.startswith("FP_")]]  # Input
y = data[["Aij", "Bij", "Alpha"]]  # Output

# Standardize the input data (important for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# **Step 3: Define Hyperparameter Spaces**
param_grids = {
    "OLS": {},  # OLS has no hyperparameters
    "DecisionTree": {
        'max_depth': Integer(3, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'criterion': Categorical(['squared_error', 'absolute_error'])
    },
    "SVR": {
        'C': Real(0.1, 100, prior='log-uniform'),
        'epsilon': Real(0.01, 0.2, prior='log-uniform'),
        'kernel': Categorical(['rbf'])
    },
    "XGBoost": {
        'n_estimators': Integer(50, 500),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 20),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'gamma': Real(0, 1),
        'reg_alpha': Real(0, 1),
        'reg_lambda': Real(0.1, 10, prior='log-uniform')
    }
}

models_to_tune = {
    "OLS": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1)
}

# **Step 4: Set Up Output Folders**
main_output_folder = "Main_Output_Folder"
os.makedirs(main_output_folder, exist_ok=True)

# Subfolders for models, metrics, and plots
model_folder = os.path.join(main_output_folder, "Models")
metrics_folder = os.path.join(main_output_folder, "Metrics")
plot_folder = os.path.join(main_output_folder, "Plots")

os.makedirs(model_folder, exist_ok=True)
os.makedirs(metrics_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# **Step 5: Train Models with Bayesian Optimization**
metrics_log = []  # List to store metrics for all models and targets
tuned_models = {}

for model_name, model in tqdm(models_to_tune.items(), desc="Tuning Models with Bayesian Optimization"):
    logger.info(f"Starting Bayesian optimization for {model_name}...")
    
    for target in y.columns:
        logger.info(f"Tuning {model_name} for target {target}...")
        
        if model_name == "OLS":
            # OLS has no hyperparameter tuning
            best_model = model.fit(X_scaled, y[target])
        else:
            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=param_grids[model_name],
                scoring='r2',
                n_iter=50,  # Number of iterations for optimization
                cv=kfold,
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
            
            # Perform Bayesian optimization
            bayes_search.fit(X_scaled, y[target])
            best_model = bayes_search.best_estimator_
        
        # Save the best model
        tuned_models[f"{model_name}_{target}"] = best_model
        
        # Evaluate on training data
        y_pred = best_model.predict(X_scaled)
        mae = mean_absolute_error(y[target], y_pred)
        mse = mean_squared_error(y[target], y_pred)
        r2 = r2_score(y[target], y_pred)
        
        metrics_log.append({
            "Model": model_name,
            "Target": target,
            "Best Parameters": bayes_search.best_params_ if model_name != "OLS" else "N/A",
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        })
        
        logger.info(f"Target: {target}, R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

# **Step 6: Save Metrics Log**
metrics_df = pd.DataFrame(metrics_log)
metrics_log_file = os.path.join(metrics_folder, "Metrics_Log.csv")
metrics_df.to_csv(metrics_log_file, index=False)
logger.info(f"Saved metrics log to {metrics_log_file}")

# Save Tuned Models
for model_key, model in tuned_models.items():
    model_path = os.path.join(model_folder, f"{model_key}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved model {model_key} to {model_path}")

# **Step 7: Visualize Results**
for metric in metrics_df.itertuples():
    model_name = metric.Model
    target = metric.Target
    mae = metric.MAE
    mse = metric.MSE
    r2 = metric.R2
    best_model = tuned_models[f"{model_name}_{target}"]
    y_pred = best_model.predict(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(y[target], y_pred, alpha=0.7, label='Predicted vs Actual', color='blue')
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{model_name} Prediction for {target}", fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    
    # Add metrics as text box
    plt.text(0.95, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    
    plot_path = os.path.join(plot_folder, f"{model_name}_{target}_Prediction.png")
    plt.savefig(plot_path)
    plt.show()
    logger.info(f"Saved plot to {plot_path}")