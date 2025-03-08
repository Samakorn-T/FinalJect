#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:17:24 2025
@author: padprow
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer for M1/M2 Macs
from keras_tuner import Hyperband
import tensorflow.keras.backend as K
from keras_tuner import Objective
import shutil

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
y = data[["Bij", "Alpha"]]  # Output

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# **Step 3: Define Custom Metrics**
def r2_metric(y_true, y_pred):
    """Calculate the R^2 score."""
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + K.epsilon()))
    return r2

def rmse_metric(y_true, y_pred):
    """Calculate the RMSE (Root Mean Squared Error)."""
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

# **Step 4: Define Model Structure**
def build_model(hp):
    model = Sequential()

    # Input layer
    model.add(Dense(
        hp.Int('units_input', min_value=128, max_value=2048, step=128),
        activation=hp.Choice('activation_input', ['relu', 'tanh', 'selu', 'swish']),
        input_shape=(X_scaled.shape[1],)
    ))
    model.add(BatchNormalization())

    # Dynamically add hidden layers
    for i in range(hp.Int('num_hidden_layers', min_value=5, max_value=30)):
        model.add(Dense(
            hp.Int(f'units_hidden_{i}', min_value=128, max_value=2048, step=128),
            activation=hp.Choice(f'activation_hidden_{i}', ['relu', 'tanh', 'selu', 'swish']),
            kernel_regularizer='l2'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f'dropout_hidden_{i}', min_value=0.0, max_value=0.6, step=0.1)))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log')),
        loss='huber_loss',  # Huber loss
        metrics=['mae', rmse_metric, r2_metric]
    )

    return model

# **Step 5: Hyperparameter Tuning**
metrics_log = []
tuned_models = {}

main_folder = "MLP_Folder"
os.makedirs(main_folder, exist_ok=True)

# Subfolders within MLP_Folder
model_folder = os.path.join(main_folder, "MLP_Models")
plot_folder = os.path.join(main_folder, "MLP_Plots")
tuning_results_folder = os.path.join(main_folder, "tuning_results")

# Create the subfolders
os.makedirs(model_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

for target in y.columns:
    logger.info(f"Starting tuning for target {target}...")
    
    # Set up Keras Tuner
    tuner = Hyperband(
        build_model,
        objective=Objective("val_r2_metric", direction="max"),  # Use R² as the tuning objective
        max_epochs=150,
        factor=3,
        directory=tuning_results_folder,
        project_name=f"MLP_{target}"
    )
    
    # Cross-Validation with K-Fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y[target])):
        logger.info(f"Processing Fold {fold+1} for target {target}...")
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[target].iloc[train_idx], y[target].iloc[val_idx]
        
        tuner.search(X_train, y_train, epochs=150, validation_data=(X_val, y_val))
    
    # Get Best Hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train Best Model on Full Dataset
    callbacks = []  # Disable all model checkpoint callbacks
    best_model.fit(X_scaled, y[target], epochs=150, validation_split=0.2, verbose=0, callbacks=callbacks)
    tuned_models[target] = best_model

    # ลบ tuning_results_folder หลังจากเสร็จงานสำหรับ target
    if os.path.exists(tuning_results_folder):
        shutil.rmtree(tuning_results_folder)
        logger.info(f"Deleted tuning results folder: {tuning_results_folder}")

    # Evaluate Model
    y_pred = best_model.predict(X_scaled).flatten()
    mae = mean_absolute_error(y[target], y_pred)
    mse = mean_squared_error(y[target], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y[target], y_pred)
    
    metrics_log.append({
        "Target": target,
        "Best Hyperparameters": best_hps.values,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })
    
    logger.info(f"Target: {target}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# **Step 6: Save Metrics and Models**
metrics_df = pd.DataFrame(metrics_log)
metrics_log_file = os.path.join(model_folder, "Metrics_Log.csv")
metrics_df.to_csv(metrics_log_file, index=False)
logger.info(f"Saved metrics log to {metrics_log_file}")

# Save Tuned Models
for target, model in tuned_models.items():
    model_path = os.path.join(model_folder, f"MLP_{target}.h5")
    model.save(model_path)
    logger.info(f"Saved model for {target} to {model_path}")

# **Step 7: Summarize Best Hyperparameters and Architecture**
summary_folder = os.path.join(main_folder, "Model_Summaries")
os.makedirs(summary_folder, exist_ok=True)

summary_file = os.path.join(summary_folder, "Best_Hyperparameters_and_Architectures.txt")

with open(summary_file, "w") as summary:
    for target, best_model in tuned_models.items():
        summary.write(f"Target: {target}\n")
        summary.write("Best Model Architecture:\n")
        for i, layer in enumerate(best_model.layers):
            summary.write(f"  Layer {i+1}: {layer.name} ({layer.__class__.__name__})\n")
            if hasattr(layer, "units"):
                summary.write(f"    Units: {layer.units}\n")
            if hasattr(layer, "activation"):
                summary.write(f"    Activation: {layer.activation.__name__}\n")
            if hasattr(layer, "rate"):
                summary.write(f"    Dropout Rate: {layer.rate}\n")
        summary.write("\n" + "-" * 50 + "\n\n")

logger.info(f"Saved best hyperparameters and architecture summaries to {summary_file}")

# **Step 8: Visualize Results**
for metric in metrics_df.itertuples():
    target = metric.Target
    mae = metric.MAE
    mse = metric.MSE
    rmse = metric.RMSE
    r2 = metric.R2
    best_model = tuned_models[target]
    y_pred = best_model.predict(X_scaled).flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(y[target], y_pred, alpha=0.7, label='Predicted vs Actual', color='blue')
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], 'r--', label="Perfect Fit")
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"MLP Prediction for {target}", fontsize=14)
    plt.legend(loc='upper left', fontsize=10)

    # Add metrics as a text box at the top-right corner
    text_str = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
    plt.text(
        0.95, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    # Save and show the plot
    plot_path = os.path.join(plot_folder, f"MLP_{target}_Prediction.png")
    plt.savefig(plot_path)
    plt.show()
    logger.info(f"Saved plot for {target} to {plot_path}")
    