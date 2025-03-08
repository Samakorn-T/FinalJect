#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:18:23 2025

@author: padprow
"""

import pandas as pd
import os

# Load the data
data_folder = "Prepare_datavisual"
data_file = "Expanded_Fingerprints_Data.xlsx"
data_path = os.path.join(data_folder, data_file)

# Ensure the file exists before proceeding
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

# Read the Excel file correctly
data = pd.read_excel(data_path)

# Define numerical columns for outlier detection
num_cols = ["Bij", "Bji"]

# Dictionary to store outlier information
outlier_info = {}

# Detect and count outliers using IQR
for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    outlier_info[col] = {"Lower Bound": lower_bound, "Upper Bound": upper_bound, "Outliers": outliers}

# Print outlier count information
for col, info in outlier_info.items():
    print(f"{col}: {info['Outliers']} outliers detected.")

# Remove outliers from dataset
cleaned_data = data.copy()
for col in num_cols:
    Q1 = cleaned_data[col].quantile(0.25)
    Q3 = cleaned_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Apply filtering to remove outliers
    cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]

# Ensure the output folder exists
os.makedirs(data_folder, exist_ok=True)

# Save cleaned data to a new csv file
cleaned_file_csv = "Expanded_Fingerprints_Data_Cleaned.csv"
cleaned_file_path_csv = os.path.join(data_folder, cleaned_file_csv)
cleaned_data.to_csv(cleaned_file_path_csv, index=False)

print(f"Cleaned data saved to {cleaned_file_path_csv}")