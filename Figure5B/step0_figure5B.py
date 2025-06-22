# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:25:59 2025

@author: mikin
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Define folder path
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "r")

# Define mappings
label_mapping = {0: 'control', 1: 'withdrawal', 2: 'relapse'}

# Define filenames
filenames_heroin = [
    "features_ac.csv", "features_dm.csv", "features_lat.csv",
    "features_vm.csv", "features_pc.csv"
]

filenames_sucrose = [
    "features_ac_sucrose.csv", "features_dm_sucrose.csv", "features_lat_sucrose.csv",
    "features_vm_sucrose.csv", "features_pc_sucrose.csv"
]

# Combine all filenames
all_filenames = filenames_heroin + filenames_sucrose

# Container for final results
all_means = []

# Loop through each file
for filename in all_filenames:
    path = os.path.join(data_dir, filename)
    print(f"Processing: {filename}")
    
    # Determine treatment
    treatment = 'sucrose' if 'sucrose' in filename else 'heroin'
    
    # Get subregion from filename
    subregion = filename.split('_')[1].replace('.csv', '')
  
    # Read file (cvs)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    
    # Identify label column
    label_col = [col for col in df.columns if 'label' in col][0]
    df['condition'] = df[label_col].map(label_mapping)
    print("df['condition']", df['condition'])
    # Extract feature columns
    exclude_cols = {'id', label_col, 'condition'}
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Normalize features using MinMaxScaler if max > 1
    scaler = MinMaxScaler()
    for col in feature_cols:
        col_data = df[col].values.reshape(-1, 1)
        if np.max(col_data) > 1:
            df[col] = scaler.fit_transform(col_data).flatten()

    # Compute mean per condition
    medians = df.groupby('condition')[feature_cols].median().reset_index()
    
    print("subregion", subregion)
    # Melt into long format
    melted = medians.melt(id_vars='condition', var_name='feature', value_name='value')
    melted['subregion'] = subregion
    melted['treatment'] = treatment
    melted.dropna(subset=['value'], inplace=True)
    
    # Append to result list
    all_means.append(melted)

# Concatenate all results
df_all = pd.concat(all_means, ignore_index=True)

# Pivot to MultiIndex dataframe
df_pivot = df_all.pivot_table(index=['feature', 'condition'],
                               columns=['treatment', 'subregion'],
                               values='value')

# Rename index levels
df_pivot.index.rename(['feature', 'class'], inplace=True)

# Clean column labels
df_pivot.columns = pd.MultiIndex.from_tuples([
    (treat, subregion.replace('.csv', '')) for treat, subregion in df_pivot.columns
])

# Reset index
df_pivot_reset = df_pivot.reset_index()

# Extract index and class
feature_col = df_pivot_reset['feature']
class_col = df_pivot_reset['class']
df_data = df_pivot_reset.drop(columns=['feature', 'class'])

# Organize data columns
subregion_order = ['ac', 'dm', 'lat', 'vm', 'pc']
data_columns = [(t, s) for t in ['heroin', 'sucrose'] for s in subregion_order]
df_data = df_data[data_columns]

# Build final table
df_final = pd.concat([feature_col, class_col, class_col, df_data], axis=1)

level0 = ["", "heroin", "sucrose"] + [t for t, s in data_columns]
level1 = ["", "class", "class"] + [s for t, s in data_columns]
header_rows = pd.DataFrame([level0, level1])
 # Ensure alignment
header_rows.columns = range(len(level0)) 

# Flatten columns to numeric labels for writing
df_final.columns = range(df_final.shape[1])

# Save to Excel with manual headers
with pd.ExcelWriter("heatmap_median_values.xlsx", engine="xlsxwriter") as writer:
    header_rows.to_excel(writer, index=False, header=False, startrow=0)
    df_final.to_excel(writer, index=False, header=False, startrow=2)





