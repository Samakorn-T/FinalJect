#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:31:28 2025

@author: padprow
"""

import os
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# %% Create Temp
print("Create Temp")
# Read data from Excel Zone
print("Phase 1: Reading")

input_folder = 'Import'
input_file = 'Interaction.xlsx'

input_path = os.path.join(input_folder, input_file)

print("Current working directory:", os.getcwd())
df = pd.read_excel(input_path, usecols=['CAS i', 'CAS j'])

# Combine CAS i and CAS j into a single column and remove duplicates
print("Phase 2: Combining columns and removing duplicates")

combined_cas = pd.concat([df['CAS i'], df['CAS j']])
combined_cas_unique = combined_cas.drop_duplicates().reset_index(drop=True)
combined_cas_unique = pd.DataFrame(combined_cas_unique, columns=['CAS'])

# Convert CAS to Name & SMILES Zone
print("Phase 3: Converting CAS to Name & SMILES")

def get_name_smiles_from_cas(cas_id):
    try:
        compounds = pcp.get_compounds(cas_id, 'name')
        if compounds:
            compound = compounds[0]
            name = compound.iupac_name if compound.iupac_name else "N/A"
            smiles = compound.isomeric_smiles if compound.isomeric_smiles else "N/A"
            return name, smiles
        else:
            print(f"SMILES not found for CAS: {cas_id}")
            return "N/A", "N/A"
    except Exception as e:
        print(f"Error fetching data for CAS {cas_id}: {e}")
        return "N/A", "N/A"

total_cas = len(combined_cas_unique)
for idx, cas in enumerate(combined_cas_unique['CAS'], start=0):
    name, smiles = get_name_smiles_from_cas(cas)
    combined_cas_unique.at[idx, 'Name'] = name
    combined_cas_unique.at[idx, 'SMILES'] = smiles
    if smiles == "N/A":
        combined_cas_unique.at[idx, 'Manual_SMILES'] = ""  # Add empty column for manual input
    print(f"Processed {idx+1}/{total_cas} substances")

# Add a column for Manual_SMILES
combined_cas_unique['Manual_SMILES'] = combined_cas_unique['SMILES'].apply(
    lambda x: "" if x == "N/A" else x
)

# Save the processed file for manual input
temp_folder = 'Temp'
os.makedirs(temp_folder, exist_ok=True)
Processed_file = 'Processed_Interaction_With_Manual_SMILES.xlsx'
Processed_path = os.path.join(temp_folder, Processed_file)
combined_cas_unique.to_excel(Processed_path, index=False)
print(f"Data saved for manual input to: {Processed_path}")

# Read the file after manual input
manual_smiles_path = os.path.join(temp_folder, 'Processed_Interaction_With_Manual_SMILES.xlsx')
combined_cas_unique = pd.read_excel(manual_smiles_path)

# Replace missing SMILES with Manual_SMILES if provided
combined_cas_unique['SMILES'] = combined_cas_unique.apply(
    lambda row: row['Manual_SMILES'] if row['SMILES'] == "N/A" and row['Manual_SMILES'] else row['SMILES'],
    axis=1
)

# Convert SMILES to Fingerprint Zone
print("Phase 4: Converting SMILES to Fingerprint")

def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fingerprint = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
            return list(fingerprint)
        else:
            return None
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None

combined_cas_unique['Fingerprint'] = combined_cas_unique['SMILES'].map(smiles_to_fingerprint)

# Temp Collect
print("Phase 5: Collecting Temp")

Processed_file = 'Processed_Interaction.xlsx'
Processed_path = os.path.join(temp_folder, Processed_file)
combined_cas_unique.to_excel(Processed_path, index=False)

# %% Merge Data
print("Merge Data")
# Phase 1: Read Original and Processed Data
print("Phase 1: Reading Data")

input_file = 'Interaction.xlsx'
input_path = os.path.join(input_folder, input_file)

# Read the original file that contains CAS i and CAS j
df_original = pd.read_excel(input_path, usecols=['CAS i', 'CAS j', 'Aij', 'Aji', 'Bij', 'Bji', 'Alpha'])

# Read the processed file that contains CAS, Name, SMILES, and Fingerprint
temp_file = 'Processed_Interaction.xlsx'
temp_path = os.path.join(temp_folder, temp_file)
df_processed = pd.read_excel(temp_path)

# Phase 2: Merge Processed Data with Original Data
print("Phase 2: Merging Data")

# Merge data for CAS i
df_merged_i = pd.merge(df_original, df_processed, how='left', left_on='CAS i', right_on='CAS', suffixes=('', '_i'))
df_merged_i = df_merged_i.drop(columns=['CAS'])  # Remove the redundant CAS column from processed data

# Rename columns for CAS i
df_merged_i = df_merged_i.rename(columns={
    'Name': 'Name i',
    'SMILES': 'SMILES i',
    'Fingerprint': 'Fingerprint i'
})

# Merge data for CAS j
df_merged = pd.merge(df_merged_i, df_processed, how='left', left_on='CAS j', right_on='CAS', suffixes=('', '_j'))
df_merged = df_merged.drop(columns=['CAS'])  # Remove the redundant CAS column from processed data

# Rename columns for CAS j
df_merged = df_merged.rename(columns={
    'Name': 'Name j',
    'SMILES': 'SMILES j',
    'Fingerprint': 'Fingerprint j'
})

# Reorganize columns in the desired order: CAS i, CAS j, Name i, Name j, SMILES i, SMILES j, Fingerprint i, Fingerprint j
df_merged = df_merged[['CAS i', 'CAS j', 'Name i', 'Name j', 'SMILES i', 'SMILES j', 'Fingerprint i', 'Fingerprint j', 'Aij', 'Aji', 'Bij', 'Bji', 'Alpha']]

# Phase 3: Save the Final Merged Data
print("Phase 3: Saving Merged Data")

Merge_file = 'Merged_Interaction.xlsx'
Merge_path = os.path.join(temp_folder, Merge_file)
df_merged.to_excel(Merge_path, index=False)

# Display final row count
print(f"Total rows in merged data: {len(df_merged)}")