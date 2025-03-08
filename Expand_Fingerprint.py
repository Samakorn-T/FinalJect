#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:43:33 2024

@author: padprow
"""

import os
import pandas as pd

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