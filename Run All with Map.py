#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:00:14 2024

@author: padprow
"""

import os
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

# ปิดการแจ้งเตือน
RDLogger.DisableLog('rdApp.warning')
#%%Create Temp
print("Create Temp")
# Read data from Excel Zone
print("Phase 1:Reading")

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
            return "N/A", "N/A"
    except Exception as e:
        print(f"Error fetching data for CAS {cas_id}: {e}")
        return "N/A", "N/A"

total_cas = len(combined_cas_unique)
for idx, cas in enumerate(combined_cas_unique['CAS'], start=0):
    name, smiles = get_name_smiles_from_cas(cas)
    combined_cas_unique.at[idx, 'Name'] = name
    combined_cas_unique.at[idx, 'SMILES'] = smiles
    print(f"Processed {idx+1}/{total_cas} substances")

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

temp_folder = 'Temp'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
Processed_file = 'Processed_Interaction.xlsx'
Processed_path = os.path.join(temp_folder, Processed_file)
combined_cas_unique.to_excel(Processed_path, index=False)

#%% Merge Data
print("Merge Data")
# Phase 1: Read Original and Processed Data
print("Phase 1: Reading Data")

input_folder = 'Import'
input_file = 'Interaction.xlsx'
input_path = os.path.join(input_folder, input_file)


# Read the original file that contains CAS i and CAS j
df_original = pd.read_excel(input_path, usecols=['CAS i', 'CAS j', 'Aij', 'Aji', 'Bij', 'Bji', 'Alpha'])

# Read the processed file that contains CAS, Name, SMILES, and Fingerprint
temp_folder = 'Temp'
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

temp_folder = 'Temp'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
Merge_file = 'Merged_Interaction.xlsx'
Merge_path = os.path.join(temp_folder, Merge_file)
df_merged.to_excel(Merge_path, index=False)
#%%Swap Data
print("Swap Data")
# Input and output paths
input_path = 'Temp/Merged_Interaction.xlsx'
output_folder = 'Temp'
os.makedirs(output_folder, exist_ok=True)
output_file = 'Expanded_Interaction_Data.xlsx'
output_path = os.path.join(output_folder, output_file)

# Read the Excel file
print("Reading Excel Data...")
df = pd.read_excel(input_path)

# Ensure required columns are present
required_columns = [
    'CAS i', 'CAS j', 'Name i', 'Name j', 
    'SMILES i', 'SMILES j', 'Fingerprint i', 'Fingerprint j',
    'Aij', 'Aji', 'Bij', 'Bji', 'Alpha'
]

if not all(col in df.columns for col in required_columns):
    raise ValueError("Some required columns are missing from the input file.")

# Create the swapped DataFrame
print("Creating swapped data...")
df_swapped = df.copy()

# Swap the relevant columns
df_swapped[['CAS i', 'CAS j']] = df[['CAS j', 'CAS i']].values
df_swapped[['Name i', 'Name j']] = df[['Name j', 'Name i']].values
df_swapped[['SMILES i', 'SMILES j']] = df[['SMILES j', 'SMILES i']].values
df_swapped[['Fingerprint i', 'Fingerprint j']] = df[['Fingerprint j', 'Fingerprint i']].values
df_swapped[['Aij', 'Aji']] = df[['Aji', 'Aij']].values
df_swapped[['Bij', 'Bji']] = df[['Bji', 'Bij']].values

# Combine the original and swapped DataFrames
print("Combining original and swapped data...")
df_combined = pd.concat([df, df_swapped], ignore_index=True)

# Save the combined data to a new Excel file
print(f"Saving combined data to {output_file}...")
df_combined.to_excel(output_path, index=False)
print(f"Data saved successfully to: {output_path}")

# Display final row count
print(f"Total rows after combining: {len(df_combined)}")

#%%Expand FP
print("Expand Fingerprint")
# Phase 1: Reading Excel data
input_folder = 'Temp'
input_file = 'Expanded_Interaction_Data.xlsx'
input_path = os.path.join(input_folder, input_file)

print("Reading Excel data...")
df = pd.read_excel(input_path)

# Function to expand fingerprint columns
def expand_fingerprint_column(df, col_name, prefix):
    """Expand a fingerprint list into multiple columns."""
    # Convert string representation of lists to actual lists
    df[col_name] = df[col_name].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # Create a DataFrame from the expanded lists
    fingerprints_df = pd.DataFrame(df[col_name].to_list(), 
                                   columns=[f'{prefix}_{i}' for i in range(len(df[col_name][0]))])
    
    # Drop the original fingerprint column and concatenate the new columns
    df = df.drop(columns=[col_name])
    return pd.concat([df, fingerprints_df], axis=1)

# Expand 'Fingerprint i' and 'Fingerprint j' into individual columns
print("Expanding fingerprints...")
df = expand_fingerprint_column(df, 'Fingerprint i', 'FP_i')
df = expand_fingerprint_column(df, 'Fingerprint j', 'FP_j')

# Display the first few rows to verify
print(df.head())

# Save the expanded DataFrame to a new Excel file
output_folder = 'Prepare_datavisual'
os.makedirs(output_folder, exist_ok=True)
output_file = 'Expanded_Fingerprints_Data.xlsx'
output_path = os.path.join(output_folder, output_file)
df.to_excel(output_path, index=False)

print(f"Expanded data saved to: {output_path}")

#%%=================================================================
#ส่วนทำ Mapping
import os
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from rdkit.Chem import Draw

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.warning')

#%% Create Temp
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
            return "N/A", "N/A"
    except Exception as e:
        print(f"Error fetching data for CAS {cas_id}: {e}")
        return "N/A", "N/A"

total_cas = len(combined_cas_unique)
for idx, cas in enumerate(combined_cas_unique['CAS'], start=0):
    name, smiles = get_name_smiles_from_cas(cas)
    combined_cas_unique.at[idx, 'Name'] = name
    combined_cas_unique.at[idx, 'SMILES'] = smiles
    print(f"Processed {idx+1}/{total_cas} substances")

# Convert SMILES to Fingerprint and Map Bit-to-Substructure Zone
print("Phase 4: Converting SMILES to Fingerprint and Mapping Bits")

def smiles_to_fingerprint_with_mapping(smiles, cas, image_output_folder):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            bit_info = {}
            # Generate Morgan fingerprint with bitInfo to trace substructures
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bit_info)

            # Identify which bits are on
            active_bits = list(fp.GetOnBits())
            bit_mapping = []

            for bit in active_bits:
                substructure_info = bit_info.get(bit, [])
                substructure_details = []
                for atom_idx, radius in substructure_info:
                    # Capture details about the substructure centered around `atom_idx`
                    atom = mol.GetAtomWithIdx(atom_idx)
                    substructure_details.append({
                        "bit": bit,
                        "atom": atom.GetSymbol(),
                        "atom_index": atom_idx,
                        "radius": radius
                    })
                    # Generate substructure image
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    atoms_to_highlight = set([atom_idx])
                    bonds_to_highlight = []

                    for bidx in env:
                        bond = mol.GetBondWithIdx(bidx)
                        bonds_to_highlight.append(bidx)
                        atoms_to_highlight.add(bond.GetBeginAtomIdx())
                        atoms_to_highlight.add(bond.GetEndAtomIdx())

                    # Save image of the substructure and name it "bit_X.png"
                    img = Draw.MolToImage(mol, highlightAtoms=list(atoms_to_highlight), highlightBonds=bonds_to_highlight)
                    img_file = os.path.join(image_output_folder, f'bit_{bit}.png')  # Example: "bit_1.png"
                    img.save(img_file)
                    substructure_details[-1]["image_path"] = img_file  # Store the image path

                bit_mapping.append(substructure_details)
            return list(fp), bit_mapping
        else:
            return None, None
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None, None

# Create columns to store bit-mapping info for CAS entries
image_output_folder = 'Substructure_Images'
if not os.path.exists(image_output_folder):
    os.makedirs(image_output_folder)

combined_cas_unique['Fingerprint'] = combined_cas_unique['SMILES'].map(lambda x: smiles_to_fingerprint_with_mapping(x, combined_cas_unique['CAS'], image_output_folder)[0])
combined_cas_unique['Bit_Mapping'] = combined_cas_unique['SMILES'].map(lambda x: smiles_to_fingerprint_with_mapping(x, combined_cas_unique['CAS'], image_output_folder)[1])

# Save temporary file with fingerprint and mapping data
temp_folder = 'Temp'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
Processed_file = 'Processed_Interaction_with_Bit_Mapping.xlsx'
Processed_path = os.path.join(temp_folder, Processed_file)
combined_cas_unique.to_excel(Processed_path, index=False)

#%% Create Bit Mapping Excel with Images
print("Phase 5: Creating Bit Mapping Excel with Images")

# Prepare a list to store detailed bit-to-substructure mapping for each CAS entry
bit_mapping_data = []

for idx, row in combined_cas_unique.iterrows():
    cas = row['CAS']
    name = row['Name']
    smiles = row['SMILES']
    bit_mapping = row['Bit_Mapping']

    if bit_mapping:
        for substructure_list in bit_mapping:
            for substructure in substructure_list:
                bit_mapping_data.append({
                    'CAS': cas,
                    'Name': name,
                    'SMILES': smiles,
                    'Bit': substructure['bit'],
                    'Atom': substructure['atom'],
                    'Atom Index': substructure['atom_index'],
                    'Radius': substructure['radius'],
                    'Substructure Image': substructure['image_path']
                })

# Convert the bit mapping data to a DataFrame
bit_mapping_df = pd.DataFrame(bit_mapping_data)

# Save the bit-to-substructure mapping to an Excel file
output_folder = 'Mapping_Output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
Bit_Mapping_file = 'Bit_Substructure_Mapping_with_Images.xlsx'
Bit_Mapping_path = os.path.join(output_folder, Bit_Mapping_file)
bit_mapping_df.to_excel(Bit_Mapping_path, index=False)

print(f"Bit-to-substructure mapping with images saved successfully to: {Bit_Mapping_path}")