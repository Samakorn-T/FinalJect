import pandas as pd
import os
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# **Step 1: Configure Logging**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# **Step 2: Define Paths**
model_folder = "XGB_Folder"
output_folder = "Predictions"
os.makedirs(output_folder, exist_ok=True)

# **Step 3: Convert SMILES to Fingerprint**
def smiles_to_fingerprint(smiles, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=nBits)
        return list(fp)  # Convert fingerprint to a list
    return None

# **Step 4: Load Models and Preprocessing Tools**
def load_model_and_scaler(target_name):
    model_path = os.path.join(model_folder, f"best_xgboost_model_{target_name}.pkl")
    scaler_path = os.path.join(model_folder, f"scaler_{target_name}.pkl")
    feature_path = os.path.join(model_folder, f"selected_features_{target_name}.txt")

    # Load selected features
    if os.path.exists(feature_path):
        with open(feature_path, "r") as f:
            selected_features = f.read().splitlines()
    else:
        logger.warning(f"Feature selection file {feature_path} not found!")
        selected_features = None  

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, selected_features

# **Step 5: Function to Predict NRTL Parameters**
def predict_values(smiles_i, smiles_j):
    fp_i = smiles_to_fingerprint(smiles_i)
    fp_j = smiles_to_fingerprint(smiles_j)

    if fp_i is None or fp_j is None:
        logger.error("Invalid SMILES input. Could not generate fingerprint.")
        return None

    feature_names = [f"FP_i_{i}" for i in range(2048)] + [f"FP_j_{i}" for i in range(2048)]
    X_input = pd.DataFrame([fp_i + fp_j], columns=feature_names)

    def get_model_predictions(X_input):
        results = {}
        for target in ["Aij", "Bij", "Alpha"]:
            model, scaler, selected_features = load_model_and_scaler(target)

            if selected_features is None:
                logger.error(f"Missing selected features for {target}. Skipping prediction.")
                continue
            
            X_selected = X_input[selected_features]
            X_scaled = scaler.transform(X_selected)

            results[target] = model.predict(X_scaled)[0]

        return results

    original_results = get_model_predictions(X_input)

    X_swapped = pd.DataFrame([fp_j + fp_i], columns=feature_names)
    swapped_results = get_model_predictions(X_swapped)

    final_results = {
        "Aij": original_results.get("Aij", None),
        "Bij": original_results.get("Bij", None),
        "Alpha": original_results.get("Alpha", None),
        "Aji": swapped_results.get("Aij", None),
        "Bji": swapped_results.get("Bij", None)
    }
    if final_results:
        logger.info(f"Predictions: {final_results}")
    return final_results

# **Step 6: VLE Calculation and T-xy Diagram (Manual Input for Antoine Coefficients)**
def antoine_eq(A, B, C, T):
    """Calculates Psat using Antoine equation"""
    return 10 ** (A - B / (T + C))  # mmHg

def gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T):
    """NRTL model for activity coefficients"""
    R = 8.314  # J/(mol·K)
    tau12 = (Aij + Bij / (T + 273.15)) / R
    tau21 = (Aji + Bji / (T + 273.15)) / R
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)
    
    gamma1 = np.exp(x2**2 * (tau21 * G21 / (x1 + x2 * G21))**2)
    gamma2 = np.exp(x1**2 * (tau12 * G12 / (x2 + x1 * G12))**2)
    
    return gamma1, gamma2

# **Step 7: Plot VLE with Manual Input for Antoine Coefficients**
def plot_vle_manual(smiles_i, A_i, B_i, C_i, smiles_j, A_j, B_j, C_j):
    prediction = predict_values(smiles_i, smiles_j)
    if not prediction:
        logger.error("Failed to get NRTL parameters.")
        return
    
    Aij, Aji, Bij, Bji, alpha = prediction["Aij"], prediction["Aji"], prediction["Bij"], prediction["Bji"], prediction["Alpha"]

    # Get Antoine coefficients from manual input
    antoine_i = {"A": A_i, "B": B_i, "C": C_i}
    antoine_j = {"A": A_j, "B": B_j, "C": C_j}

    if not antoine_i or not antoine_j:
        logger.error("Failed to get Antoine coefficients.")
        return

    T_values = np.linspace(70, 100, 20)
    x1_values = np.linspace(0, 1, 20)
    y1_values = []

    for x1 in x1_values:
        x2 = 1 - x1
        T = 80  
        Psat1 = antoine_eq(**antoine_i, T=T)
        Psat2 = antoine_eq(**antoine_j, T=T)
        gamma1, gamma2 = gamma_nrtl(x1, x2, Aij, Aji, Bij, Bji, alpha, T)
        y1_values.append((x1 * gamma1 * Psat1) / (x1 * gamma1 * Psat1 + x2 * gamma2 * Psat2))

    # Plot with major and minor grids
    plt.plot(x1_values, T_values, label="Bubble Point")
    plt.plot(y1_values, T_values, label="Dew Point")
    plt.xlabel(f"Mole Fraction ({smiles_i})")
    plt.ylabel("Temperature (°C)")
    plt.title("T-xy Diagram")
    plt.legend()

    # Enable minor ticks for subgrid
    plt.minorticks_on()

    # Add both major and minor grids
    plt.grid(True, which='both', linestyle='-', color='gray', alpha=1)

    # Add minor grid lines
    plt.grid(True, which='minor', linestyle=':', color='black', alpha=0.4)

    # Show the plot
    plt.show()

# **Run Prediction**
# Example manual input:
smiles_i = "O"  # SMILES I
A_i, B_i, C_i = 8.07131, 1730.63, 233.426  # Antoine coefficients for I
smiles_j = "CCO"  # SMILES J
A_j, B_j, C_j = 8.20417, 1642.89, 230.300  # Antoine coefficients for J

plot_vle_manual(smiles_i, A_i, B_i, C_i, smiles_j, A_j, B_j, C_j)