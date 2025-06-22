# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:29:29 2024

@author: mikin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('heatmap_median_values.xlsx', header=[0, 1])

shape_feats_list = ['area','area filled', 'equivalent diameter area', 'feret diameter max',
  'eccentricity', 'perimeter', 'perimeter to surface ratio', 'sphericity',
  'spherical dispropotion', 'solidity', 'major axis', 'minor axis', 'elogantion',
  'fractal dimension','lacunarity']

row_mapping = {0: 'area',
               1: 'area filled', 
               2: 'equivalent diameter area', 
               3: 'feret diameter max',
               4: 'eccentricity', 
               5: 'perimeter', 
               6: 'perimeter to surface ratio', 
               7: 'sphericity',
               8: 'spherical dispropotion', 
               9: 'solidity', 
               10: 'major axis', 
               11: 'minor axis', 
               12: 'elogantion',
               13: 'fractal dimension',
               14: 'lacunarity'}


#print(df)
feature_names = df.iloc[:, 0].values.tolist()

# Create a list to store unique elements in order
unique_feature_names = []

# Loop through the original list
for name in feature_names:
    # Add to the unique list only if not already present
    if name not in unique_feature_names:
        unique_feature_names.append(name)

# Display the updated list without duplicates and maintaining order
# print(unique_feature_names)

# Select the entire 'opioids' row
df_opioids = df.loc[:, pd.IndexSlice['heroin', :]]
df_sucrose = df.loc[:, pd.IndexSlice['sucrose', :]]

# Filter the DataFrame based on the 'opioids' level for 'control'
control_df_opioids = df_opioids[df_opioids[('heroin', 'class')] == 'control'].droplevel(0, axis=1)

# Filter the DataFrame based on the 'opioids' level for 'withdrawal'
withdrawal_df_opioids = df_opioids[df_opioids[('heroin', 'class')] == 'withdrawal'].droplevel(0, axis=1)

# Filter the DataFrame based on the 'opioids' level for 'relapse'
relapse_df_opioids = df_opioids[df_opioids[('heroin', 'class')] == 'relapse'].droplevel(0, axis=1)



# Filter the DataFrame based on the 'sucrose' level for 'control'
control_df_sucrose = df_sucrose[df_sucrose[('sucrose', 'class')] == 'control'].droplevel(0, axis=1)

# Filter the DataFrame based on the 'sucrose' level for 'withdrawal'
withdrawal_df_sucrose = df_sucrose[df_sucrose[('sucrose', 'class')] == 'withdrawal'].droplevel(0, axis=1)

# Filter the DataFrame based on the 'sucrose' level for 'relapse'
relapse_df_sucrose = df_sucrose[df_sucrose[('sucrose', 'class')] == 'relapse'].droplevel(0, axis=1)

control_df_opioids = control_df_opioids.drop(columns='class')
control_df_opioids_median = control_df_opioids.values

withdrawal_df_opioids = withdrawal_df_opioids.drop(columns='class')
withdrawal_df_opioids_median = withdrawal_df_opioids.values

relapse_df_opioids = relapse_df_opioids.drop(columns='class')
relapse_df_opioids_median = relapse_df_opioids.values

control_df_sucrose = control_df_sucrose.drop(columns='class')
control_df_sucrose_median = control_df_sucrose.values

withdrawal_df_sucrose = withdrawal_df_sucrose.drop(columns='class')
withdrawal_df_sucrose_median = withdrawal_df_sucrose.values

relapse_df_sucrose = relapse_df_sucrose.drop(columns='class')
relapse_df_sucrose_median = relapse_df_sucrose.values


median_withdrawal_df_sucrose =  withdrawal_df_sucrose_median - control_df_sucrose_median
median_relapse_df_sucrose =  relapse_df_sucrose_median - control_df_sucrose_median

median_with_and_relapse_df_sucrose = withdrawal_df_sucrose_median - relapse_df_sucrose_median


median_withdrawal_df_opioids =  withdrawal_df_opioids_median - control_df_opioids_median
median_relapse_df_opioids =  relapse_df_opioids_median - control_df_opioids_median

median_with_and_relapse_df_opioids = withdrawal_df_opioids_median - relapse_df_opioids_median


median_withdrawal_df_sucrose = pd.DataFrame(median_withdrawal_df_sucrose)
median_withdrawal_df_sucrose.rename(index=row_mapping, inplace=True)

median_relapse_df_sucrose = pd.DataFrame(median_relapse_df_sucrose)
median_relapse_df_sucrose.rename(index=row_mapping, inplace=True)

median_with_rel_df_sucrose = pd.DataFrame(median_with_and_relapse_df_sucrose) #
median_with_rel_df_sucrose.rename(index=row_mapping, inplace=True)

median_withdrawal_df_opioids = pd.DataFrame(median_withdrawal_df_opioids)
median_withdrawal_df_opioids.rename(index=row_mapping, inplace=True)

median_relapse_df_opioids = pd.DataFrame(median_relapse_df_opioids)
median_relapse_df_opioids.rename(index=row_mapping, inplace=True)

median_with_rel_df_opioids = pd.DataFrame(median_with_and_relapse_df_opioids)
median_with_rel_df_opioids.rename(index=row_mapping, inplace=True)
# # Calculate the difference between control and withdrawal/relapse
# heatmap_data = control_df.subtract(withdrawal_df).append(control_df.subtract(relapse_df))

# # heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".3f", linewidths=.5)
# plt.title('Heatmap: Control - Withdrawal/Relapse')
# plt.show()


combined_df_sucrose = pd.concat([median_withdrawal_df_sucrose, median_relapse_df_sucrose, median_with_rel_df_sucrose],
                                keys=['sucrose withdrawal vs control', 'sucrose relapse vs control', 'sucrose withdrawal vs relapse'])
combined_df_opioids = pd.concat([median_withdrawal_df_opioids, median_relapse_df_opioids, median_with_rel_df_opioids],
                                keys=['heroin withdrawal vs control', 'heroin relapse vs control','heroin withdrawal vs relapse'])
column_mapping = {0: 'AC', 1: 'DM', 2: 'LAT', 3: 'VM', 4: 'PC'}


#combined_df_sucrose.index.set_levels([row_mapping[idx] for idx in combined_df_sucrose.index.levels[0]], level=0, inplace=True)
#combined_df_opioids.index.set_levels([row_mapping[idx] for idx in combined_df_opioids.index.levels[0]], level=0, inplace=True)

combined_df_sucrose.rename(columns=column_mapping, level=0, inplace=True)
combined_df_opioids.rename(columns=column_mapping, level=0, inplace=True)

combined_df = pd.concat([combined_df_opioids, combined_df_sucrose]) #, keys=['sucrose', 'opioids']


feature_names = combined_df.index.get_level_values(1).unique().tolist()
level_names = combined_df.index.get_level_values(0).unique().tolist()

# Reshape the DataFrame for seaborn heatmap
heatmap_data = combined_df.unstack()
# print("heatmap_data", heatmap_data)
heatmap_data.iloc[[0,1,2]] = heatmap_data.iloc[[1,0,2]].values 
heatmap_data.iloc[[3,4,5]] = heatmap_data.iloc[[4,3,5]].values

# Switching names of rows
new_row_names = heatmap_data.index.tolist()
new_row_names[0], new_row_names[1] = new_row_names[1], new_row_names[0]
new_row_names[3], new_row_names[4] = new_row_names[4], new_row_names[3]

# Assigning new row names
heatmap_data.index = new_row_names


# Extracting class names
class_names = heatmap_data.columns.get_level_values(0).unique().tolist()


fig, axes = plt.subplots(nrows=len(class_names), ncols=1, figsize=(10, 15), sharex=True, gridspec_kw={'height_ratios': [1] * len(class_names)})

cmap = "coolwarm"
# Plot each class in a subplot
for i, class_name in enumerate(class_names):
    sns.heatmap(heatmap_data[class_name], cmap=cmap, annot=False,  ax=axes[i], vmin=-0.20, vmax=0.25)# , cbar_kws={'label': 'Colorbar Label'}
    axes[i].set_title(class_name, fontsize=16, fontname='Arial')
    axes[i].set_yticklabels(axes[i].get_yticklabels(), size='large')  # Increase label size
    #axes[i].set_xlabel('Features', size='large')

# Adjust layout
plt.tight_layout()

# plt.savefig('heatmap.png', bbox_inches='tight')

plt.show()



