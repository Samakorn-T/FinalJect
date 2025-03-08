import pandas as pd
import os
import joblib
import logging
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
        selected_features = None  # Keep it None instead of assuming 2048 bits

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, selected_features

# **Step 5: Function to Predict Values**
def predict_values(smiles_i, smiles_j):
    # Convert SMILES to fingerprints
    fp_i = smiles_to_fingerprint(smiles_i)
    fp_j = smiles_to_fingerprint(smiles_j)

    if fp_i is None or fp_j is None:
        logger.error("Invalid SMILES input. Could not generate fingerprint.")
        return None

    # **Fix Feature Naming**
    feature_names = [f"FP_i_{i}" for i in range(2048)] + [f"FP_j_{i}" for i in range(2048)]
    X_input = pd.DataFrame([fp_i + fp_j], columns=feature_names)

    def get_model_predictions(X_input):
        results = {}
        for target in ["Aij", "Bij", "Alpha"]:
            model, scaler, selected_features = load_model_and_scaler(target)

            if selected_features is None:
                logger.error(f"Missing selected features for {target}. Skipping prediction.")
                continue
            
            # **Ensure selected features exist in input**
            missing_features = [f for f in selected_features if f not in X_input.columns]
            if missing_features:
                logger.warning(f"Missing features for {target}: {missing_features}")

            # Select only relevant features
            X_selected = X_input[selected_features]
            X_scaled = scaler.transform(X_selected)

            # Predict
            results[target] = model.predict(X_scaled)[0]

        return results

    # **Predict Aij, Bij, Alpha**
    original_results = get_model_predictions(X_input)

    # **Predict Aji, Bji using swapped fingerprints**
    X_swapped = pd.DataFrame([fp_j + fp_i], columns=feature_names)
    swapped_results = get_model_predictions(X_swapped)

    # **Final Output**
    final_results = {
        "Aij": original_results.get("Aij", None),
        "Bij": original_results.get("Bij", None),
        "Alpha": original_results.get("Alpha", None),
        "Aji": swapped_results.get("Aij", None),  # Aji is calculated by swapping inputs
        "Bji": swapped_results.get("Bij", None)   # Bji is calculated by swapping inputs
    }

    return final_results

# **Step 6: Input Usage**
smiles_i = "O"  # Input SMILE i
smiles_j = "CCO"    # Input SMILE j

prediction = predict_values(smiles_i, smiles_j)

if prediction:
    logger.info(f"Predictions: {prediction}")