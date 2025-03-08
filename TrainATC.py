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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer for M1/M2 Macs
from keras_tuner import Hyperband, Objective
import shutil

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# **Step 2: Load and Prepare Data**
data_folder = 'Prepare_datavisual'
data_file = 'Filtered_Fingerprints_Data_Updated.xlsx'
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

X = data[[col for col in data.columns if col.startswith("FP_")]]
y = data[["Bij", "Alpha"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# **Step 3: Define Autoencoder Model for Feature Extraction**
def build_autocoder(hp):
    input_dim = X_scaled.shape[1]
    encoding_dim = hp.Int('encoding_dim', min_value=64, max_value=2048, step=64)
    
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation=hp.Choice('activation', ['relu', 'tanh', 'selu', 'swish']))(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1))(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autocoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autocoder.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-5, 1e-2, sampling='log')), loss='mse')
    return autocoder, encoder

# **Step 4: Hyperparameter Tuning for Autoencoder**
autocoder_tuner = Hyperband(
    lambda hp: build_autocoder(hp)[0],
    objective=Objective("val_loss", direction="min"),
    max_epochs=100,
    factor=3,
    directory="autocoder_tuning",
    project_name="Autocoder"
)

autocoder_tuner.search(X_scaled, X_scaled, epochs=150, validation_split=0.2, verbose=1)

best_hp = autocoder_tuner.get_best_hyperparameters(1)[0]
best_autocoder, best_encoder = build_autocoder(best_hp)

# **Step 5: Train Best Autoencoder and Extract Features**
best_autocoder.fit(X_scaled, X_scaled, epochs=150, validation_split=0.2, verbose=1)
X_encoded = best_encoder.predict(X_scaled)

# **Step 6: Define ATC Model for Regression**
def build_regressor(hp):
    model = Sequential()
    model.add(Dense(
        hp.Int('units', min_value=64, max_value=2048, step=64),
        activation=hp.Choice('activation', ['relu', 'tanh', 'selu', 'swish']),
        input_shape=(X_encoded.shape[1],)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-5, 1e-2, sampling='log')), loss='huber_loss', metrics=['mae'])
    return model

# **Step 7: Hyperparameter Tuning and Training for Regression**
metrics_log = []
tuned_models = {}
main_folder = "ATC_Folder"
os.makedirs(main_folder, exist_ok=True)
model_folder = os.path.join(main_folder, "ATC_Models")
plot_folder = os.path.join(main_folder, "ATC_Plots")
os.makedirs(model_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

for target in y.columns:
    logger.info(f"Starting tuning for target {target}...")
    regressor_tuner = Hyperband(
        build_regressor,
        objective=Objective("val_mae", direction="min"),
        max_epochs=150,
        factor=4,
        directory="regressor_tuning",
        project_name=f"Regressor_{target}"
    )
    regressor_tuner.search(X_encoded, y[target], epochs=150, validation_split=0.2, verbose=1)
    best_hp_regressor = regressor_tuner.get_best_hyperparameters(1)[0]
    best_regressor = build_regressor(best_hp_regressor)
    best_regressor.fit(X_encoded, y[target], epochs=150, validation_split=0.2, verbose=1)
    tuned_models[target] = best_regressor

    # **Step 8: Evaluate Model Performance**
    y_pred = best_regressor.predict(X_encoded).flatten()
    mae = mean_absolute_error(y[target], y_pred)
    mse = mean_squared_error(y[target], y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y[target], y_pred)
    
    metrics_log.append({"Target": target, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
    
    # **Step 9: Plot Predictions**
    plt.figure(figsize=(8, 6))
    plt.scatter(y[target], y_pred, alpha=0.7, color='blue')
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"ATC Prediction for {target}")
    plt.text(0.95, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}", transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plot_path = os.path.join(plot_folder, f"ATC_{target}_Prediction.png")
    plt.savefig(plot_path)
    plt.show()
    logger.info(f"Saved plot for {target} to {plot_path}")
