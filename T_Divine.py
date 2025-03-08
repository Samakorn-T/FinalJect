#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:56:33 2025

@author: padprow
"""


import pandas as pd
import os

# Load the Excel file
data_folder = 'Prepare_datavisual'
data_file = 'Filtered_Fingerprints_Data.xlsx'
data_path = os.path.join(data_folder, data_file)
data = pd.read_excel(data_path)

# Perform the operation: Divide Bij column by 303.15
data['Bij'] = data['Bij']
data['Bji'] = data['Bji']

# Save the updated DataFrame to a new Excel file
output_file = "Filtered_Fingerprints_Data_Updated.csv"
output_path = os.path.join(data_folder, output_file)
data.to_csv(output_path, index=False)

print(f"Updated file saved as {output_path}")