import pandas as pd
import os
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.preprocessing import StandardScaler

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Define Paths
model_folder = "XGB_Folder"
output_folder = "Predictions"
os.makedirs(output_folder, exist_ok=True)

def smiles_to_fingerprint(smiles, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.tolist()
    return None

def load_model_and_scaler(target_name):
    model_path = os.path.join(model_folder, f"best_xgboost_model_{target_name}.pkl")
    scaler_path = os.path.join(model_folder, f"scaler_{target_name}.pkl")
    feature_path = os.path.join(model_folder, f"selected_features_{target_name}.txt")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(f"Missing model or scaler for {target_name}.")
        return None, None, None

    selected_features = None
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            selected_features = f.read().splitlines()
    else:
        logger.warning(f"Feature selection file {feature_path} not found!")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, selected_features

def predict_values(smiles_i, smiles_j):
    fp_i, fp_j = smiles_to_fingerprint(smiles_i), smiles_to_fingerprint(smiles_j)
    if fp_i is None or fp_j is None:
        logger.error("Invalid SMILES input. Could not generate fingerprint.")
        return None
    
    feature_names = [f"FP_i_{i}" for i in range(2048)] + [f"FP_j_{i}" for i in range(2048)]
    X_input = pd.DataFrame([fp_i + fp_j], columns=feature_names)

    def get_model_predictions(X_input):
        results = {}
        for target in ["Aij", "Bij", "Alpha"]:
            model, scaler, selected_features = load_model_and_scaler(target)
            if not model or not scaler or not selected_features:
                continue
            X_selected = X_input[selected_features]
            X_scaled = scaler.transform(X_selected)
            results[target] = model.predict(X_scaled)[0]
        return results

    original_results = get_model_predictions(X_input)
    swapped_results = get_model_predictions(pd.DataFrame([fp_j + fp_i], columns=feature_names))
    
    final_results = {
        "Aij": original_results.get("Aij"), "Bij": original_results.get("Bij"), "Alpha": original_results.get("Alpha"),
        "Aji": swapped_results.get("Aij"), "Bji": swapped_results.get("Bij")
    }
    logger.info(f"Predictions: {final_results}")
    return final_results

def antoine_eq(A, B, C, T):
    return 10 ** (A - B / (T + C))

def gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T):
    R = 8.314  # J/(mol·K)
    tau12 = (Aij + Bij / (T + 273.15)) / R
    tau21 = (Aji + Bji / (T + 273.15)) / R
    G12, G21 = np.exp(-alpha * tau12), np.exp(-alpha * tau21)
    gamma1 = np.exp(x2**2 * (tau21 * G21 / (x1 + x2 * G21))**2)
    gamma2 = np.exp(x1**2 * (tau12 * G12 / (x2 + x1 * G12))**2)
    return gamma1, gamma2

def find_boiling_point(A, B, C):
    return (B / (A - np.log10(760))) - C

def plot_vle_manual(smiles_i, A_i, B_i, C_i, smiles_j, A_j, B_j, C_j):
    prediction = predict_values(smiles_i, smiles_j)
    if not prediction:
        return
    
    Aij, Aji, Bij, Bji, alpha = prediction["Aij"], prediction["Aji"], prediction["Bij"], prediction["Bji"], prediction["Alpha"]
    T_boil_i, T_boil_j = find_boiling_point(A_i, B_i, C_i), find_boiling_point(A_j, B_j, C_j)
    Tmin, Tmax = min(T_boil_i, T_boil_j), max(T_boil_i, T_boil_j)
    
    x1_values = np.linspace(0, 1, 20)
    y1_values, T_values = [], []

    for x1 in x1_values:
        x2 = 1 - x1
        T_guess = Tmin + (Tmax - Tmin) * x1  # Linearly spaced T-values
        gamma1, gamma2 = gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T_guess)
        Psat1, Psat2 = antoine_eq(A_i, B_i, C_i, T_guess), antoine_eq(A_j, B_j, C_j, T_guess)
        y1 = (x1 * gamma1 * Psat1) / (x1 * gamma1 * Psat1 + x2 * gamma2 * Psat2)
        y1_values.append(y1)
        T_values.append(T_guess)

    plt.figure(figsize=(8, 6))
    plt.plot(x1_values, T_values, label="Bubble Point", marker="o")
    plt.plot(y1_values, T_values, label="Dew Point", marker="s")
    plt.xlabel(f"Mole Fraction ({smiles_i})")
    plt.ylabel("Temperature (°C)")
    plt.title("T-xy Diagram")
    plt.legend()
    plt.grid(True, which='both', linestyle='-', color='gray', alpha=0.7)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', color='black', alpha=0.4)
    plt.show()

# Run Prediction
plot_vle_manual("CCO", 8.20417, 1642.89, 230.300,"O", 8.07131, 1730.63, 233.426)
