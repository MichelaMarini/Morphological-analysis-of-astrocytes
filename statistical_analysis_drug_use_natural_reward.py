# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:21:21 2024

@author: mikin
"""

import os
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import MinMaxScaler


'''
The following code perform ANOVA and TukeyHSD test and for each experiment and each NAc location.
It saves the results (F-values, p-values from ANOVA and adj p-values from TukeyHSD test) in .cvs format 
P_Values: sheet name for adj p-values The adjusted p-values are denoted by asterisks, where:
  - *** indicates a highly significant result (p < 0.001)
  - ** indicates a very significant result (p < 0.01)
  - * indicates a significant result (p < 0.05)
  
P_Num_Values: sheet name for numerical adj p-values 

ANOVA: sheet name for F-values and p-values

'''

# Get the current working directory
current_dir = os.getcwd()


def statistical_analysis(path):
    
    filename = os.path.splitext(os.path.basename(path))[0]
    data = pd.read_csv(path)
    
    columns_to_normalize = data.columns[2:]
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    
    for column in columns_to_normalize:
        
        column_data = data[column].values.reshape(-1, 1)  # Reshape for scaling
        if np.max(column_data) > 1 :
           
            normalized_column = scaler.fit_transform(column_data)
            data[column] = normalized_column.flatten() 
    

    
    feature = ['Area_bbox','Area_filled', 'equivalent_diameter_area', 'feret_diameter_max',
    'Eccentricity', 'Perimeter', 'Perimeter_to_Surface_Ratio', 'Sphericity',
    'Spherical Dispropotion', 'Solidity', 'AXIS', 'axis', 'Elogantion',
    'Fractal Dimension','Lacunarity']
    
    
    df =  data.drop(['ID'], axis=1) 
    label_names = {
        0: 'control',
        1: 'withdrawal',
        2: 'relapse'
    }
    
    
    def ANOVA(df): 
       
        # Initialize dictionaries to store p-values and f-values for each feature
        p_values = {}
        f_values = {}
        # Create empty dictionaries to store comparison results
        comparison_results = {}
        
        # Iterate over each feature
        for feature in df.columns:
            
            if feature != 'LABEL':
                # Create an empty list to store the feature values for each label
                feature_values_tot = []
                
                # Iterate over each label
                for label in range(3):
                    # Filter the DataFrame based on the label
                    label_data = df[df['LABEL'] == label]
                   
                    feature_values_tot.append(label_data[feature])
                    
                # One-way ANOVA with the feature values for all labels
                f_value, p_value = stats.f_oneway(*feature_values_tot)
                
                # Pairwise Tukey's HSD comparisons
                tukey_results = pairwise_tukeyhsd(df[feature], df['LABEL'])
               
                # Store the p-value, f_value and comparison results in the dictionaries
                p_values[feature], f_values[feature] = p_value, f_value
                formatted_p_values = {feature: f"{p_value:.3e}" for feature, p_value in p_values.items()}
                formatted_f_values = {feature: f"{f_value:.3f}" for feature, f_value in f_values.items()}
                
                comparison_results[feature] = tukey_results
                
    
        
        return formatted_p_values, formatted_f_values, comparison_results
        
        
     
    def TukeyHSD(comparison_results, feature_of_interest, label_names): 
        
        tukey_result = comparison_results[feature_of_interest]
        comparison_table = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
        
        comparison_table['group1'] = comparison_table['group1'].map(label_names)
        comparison_table['group2'] = comparison_table['group2'].map(label_names)
        filtered_table = comparison_table[['group1', 'group2', 'meandiff', 'p-adj', 'reject']]
        
        pops = filtered_table[['group1', 'group2']]
       
        p_values_tukey = filtered_table['p-adj']
        hyp_test = filtered_table['reject']
        hyp_test = hyp_test.to_numpy()
        p_values_tukey = p_values_tukey.to_numpy()
       
        pops = pops.copy()
        pops['classes'] = pops.apply(lambda row: f"{row['group1']}-{row['group2']}", axis=1)
        classes = pops['classes']
        classes = classes.tolist()
       
        return p_values_tukey
    
    
    
    classes = ['c-w', 'c-r', 'w-r']
    
    feats_list = ['area','area filled', 'equivalent diameter area', 'feret diameter max',
      'eccentricity', 'perimeter', 'perimeter to surface ratio', 'sphericity',
      'spherical dispropotion', 'solidity', 'major axis', 'minor axis', 'elogantion',
      'fractal dimension','lacunarity']
    
    matrix_p_values = np.empty((16,4), dtype=object)   
    matrix_p_values[0, 1:] = classes
    matrix_p_values[1:, 0] = feats_list
    
    
    matrix_p_num_values = np.empty((16,4), dtype=object)   
    matrix_p_num_values[0, 1:] = classes
    matrix_p_num_values[1:, 0] = feats_list
    
    formatted_p_values, formatted_f_values, comparison_results = ANOVA(df)
    
    
    features = list(formatted_p_values.keys())
    combined_data = {
        'Feature': features,
        'f_value': [formatted_f_values[feature] for feature in features],
        'p_value': [formatted_p_values[feature] for feature in features]
    }
    
    ANOVA_Results= pd.DataFrame(combined_data)
    
    # Set the 'Feature' column as the index
    ANOVA_Results.set_index('Feature', inplace=True)
    
    # Display the DataFrame
    print("ANOVA Results", ANOVA_Results)
    
    
    for j in range(len(feature)):
        
        
        p_values_tukey = TukeyHSD(comparison_results, feature[j], label_names)
        #p_values, classes, f_values, p_values_tukey = get_p_values(feature[j])
        p_values_star = ['***' if value < 0.001 else
                '**' if 0.001 <= value < 0.01 else
                '*' if 0.01 <= value < 0.05 else
                #'.' if 0.05 <= value < 0.1 else
                ' ' for value in p_values_tukey]
    
        matrix_p_values[j+1,1:] = p_values_star
        matrix_p_num_values[j+1,1:] = p_values_tukey
        
        
    
    df_p_values = pd.DataFrame(matrix_p_values[1:], columns=matrix_p_values[0])
    df_p_num_values = pd.DataFrame(matrix_p_num_values[1:], columns=matrix_p_num_values[0])
    
    output_filename = f"statistical_results_{filename}.xlsx"
    # Save DataFrames to the same Excel file on different sheets
    with pd.ExcelWriter(output_filename) as writer:
        df_p_values.to_excel(writer, sheet_name='P_Values', index=False) 
        df_p_num_values.to_excel(writer, sheet_name='P_Num_Values', index=False)
        ANOVA_Results.to_excel(writer, sheet_name='ANOVA', index=True)
    
    '''
    If you need the results in Latex 
    '''
    # from tabulate import tabulate
    # # Generate LaTeX code using tabulate
    # latex_p_values = tabulate(df_p_values, headers='keys', tablefmt='latex')
    # latex_p_num_values = tabulate(df_p_num_values, headers='keys', tablefmt='latex')
    
    # # Print LaTeX code 
    # print("LaTeX code for P Values Table:")
    # print(latex_p_values)
    # print("\nLaTeX code for P Num Values Table:")
    # print(latex_p_num_values)


filenames = ["features_ac.csv", "features_dm.csv", "features_lat.csv", "features_vm.csv", "features_pc.csv"]
# Loop through each filename
for filename in filenames: #for opioids experiment 
    # Construct the full path
    path = os.path.join(current_dir, "r", filename)
    print(f"Processing: {filename}")
    # Call the function with the constructed path
    statistical_analysis(path)
    
'''
sucrose experiment
'''
filenames_sucrose = ["features_ac_sucrose.csv", "features_dm_sucrose.csv", "features_lat_sucrose.csv", "features_vm_sucrose.csv", "features_pc_sucrose.csv"]
# Loop through each filename
for filename in filenames_sucrose: #for sucrose experiment 
    # Construct the full path
    path = os.path.join(current_dir, "r", filename)
    print(f"Processing: {filename}")
    # Call the function with the constructed path
    statistical_analysis(path)
    



