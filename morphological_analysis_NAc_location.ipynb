# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:57:14 2024

@author: mikin
"""

import cv2 as cv
import os
import csv
import sys


import numpy as np

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import scipy


from scipy.stats import norm, gaussian_kde
from scipy.special import kl_div
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.plotting import parallel_coordinates
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance


sns.set(context='paper', style='white', font_scale=1.5, rc={"lines.linewidth": 2.5, 'font.family': 'Arial'}) #sans-serif
sns.set_palette('muted')


path = os.path.join(os.getcwd(),"r","features_control_brain_region.csv") # features_control_sucrose
data = pd.read_csv(path)


columns_to_normalize = data.columns[2:]

# Initialize MinMaxScaler
scaler = MinMaxScaler()


for column in columns_to_normalize:
    
    column_data = data[column].values.reshape(-1, 1)  # Reshape for scaling
    if np.max(column_data) > 1 :
       
        normalized_column = scaler.fit_transform(column_data)
        data[column] = normalized_column.flatten() 
        

'''
in case you want to save the normalized features to a cvs file
'''
# output_path = os.path.join(os.getcwd(), "r", "standardized_features_control_brain_region.csv")

# # Save the DataFrame to a CSV file
# data.to_csv(output_path, index=False)      

# Extract data vectors for each class
df_ac = data[(data['LABEL']==0)] 
df_dm = data[(data['LABEL']==1)] 
df_lat = data[(data['LABEL']==2)] 
df_m = data[(data['LABEL']==3)] 
df_pc = data[(data['LABEL']==4)] 


lengths = [len(df_ac), len(df_dm), len(df_lat), len(df_m), len(df_pc)]


max_length, min_length = max(lengths), min(lengths)


# Extract the feature vectors for each class
features_ac = df_ac.iloc[:, 2:].values
features_dm = df_dm.iloc[:, 2:].values
features_lat = df_lat.iloc[:, 2:].values
features_m = df_m.iloc[:, 2:].values
features_pc = df_pc.iloc[:, 2:].values

conc_feature_for_corr = np.concatenate((features_ac, features_dm, features_lat, features_m, features_pc), axis=0)
corr_matrix = np.zeros((conc_feature_for_corr.shape[1], conc_feature_for_corr.shape[1]))


for i in range(conc_feature_for_corr.shape[1]):
    for j in range(conc_feature_for_corr.shape[1]):
        
        corr_index, _ = scipy.stats.spearmanr(conc_feature_for_corr[:,i], conc_feature_for_corr[:,j])
        corr_matrix[i][j] = corr_index


corr_df = pd.DataFrame(corr_matrix)

# # Save the correlation matrix to a CSV file 
# corr_df.to_csv(output_path, index=False, header=False)

# print(f"Correlation matrix saved to {output_path}")
        

shape_feats_list = ['Area bbox','Area', 'Equivalent diameter area', 'Feret diameter max',
  'Eccentricity', 'Perimeter', 'Perimeter to Surface Ratio', 'Sphericity',
  'Spherical Dispropotion', 'Solidity', 'Major axis', 'Minor axis', 'Elogantion',
  'Fractal Dimension','Lacunarity']

plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=shape_feats_list, yticklabels=shape_feats_list)
plt.title("Spearman Correlation Matrix")
plt.show()


population_features = {
    'ac': features_ac,
    'lat': features_lat,
    'vm': features_m,
    'pc': features_pc,
    'dm': features_dm,
}

# for pop, data in population_features.items():
#     fig, axs = plt.subplots(3, 5, figsize=(20, 10), sharex=True, sharey=True)
#     fig.suptitle(pop, fontsize=16)
    
#     # Loop through the features and plot normalized histograms
#     for i in range(len(shape_feats_list)):
#         ax = axs[i // 5, i % 5]
        
#         # Extract normalized y-values for the current feature
#         y_values = data[:, i].flatten()
       
#         # Plot the histogram and get histogram values (heights) and bin edges (x values)
#         heights, bin_edges, _ = ax.hist(y_values, color='lightgreen', edgecolor='black', bins=int(np.sqrt(min_length)), density=True)
        
#         ax.set_title(shape_feats_list[i])
#         # Print x values and corresponding histogram heights
#         # print(f"Feature: {shape_feats_list[i]}")
#         # print("y_values", len(y_values))
#         # print("heights:", heights)
#         # print("Histogram Heights normalized:", normalized_heights)
#         # # Calculate the integral of the normalized histogram
#         # integral = np.sum(normalized_heights * bin_widths)
    
#         # print(f"Feature: {shape_feats_list[i]}")
#         # print("Integral of the normalized histogram:", integral)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust title position
#     plt.show()




# Create a subplot grid of size 3 by 5
fig, axs = plt.subplots(3, 5, figsize=(20, 10))
fig.suptitle('Distribution of Features for Different Classes', fontsize=16)

features_dict = {}
weight_dict= {}

# Loop through the features and plot distributions for each class
for i, feature_name in enumerate(shape_feats_list):
    ax = axs[i // 5, i % 5]
    
    # Create a list of colors for each class
    class_colors = ['blue', 'red', 'green', 'yellow', 'purple']
    
    feature_normalized = {}
    weigths_normalized = {}
    
    # Plot histograms for each class
    for j, (class_name, data) in enumerate(population_features.items()):
        # Extract normalized y-values for the current feature
        y_values = data[:, i].flatten()
        num_points = int(np.sqrt(min_length))
         
        heights, bin_edges, _ = ax.hist(y_values, color=class_colors[j], alpha=0.35, bins=num_points, label=class_name, 
                                        density=True)
        
        
        #ax.plot(bin_edges[:-1], heights, color=class_colors[j])      
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2      
        # Store data in nested dictionaries
        feature_normalized[class_name] = midpoints 
        weigths_normalized[class_name] = heights
    
    features_dict['features_' + feature_name] = feature_normalized
    weight_dict['features_' + feature_name] = weigths_normalized
    
    ax.set_title(feature_name)
    ax.legend()
   

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #title position
# plt.savefig('Distribution of features for different classes.png', dpi=300, bbox_inches='tight')
plt.show()



shape_feats_list = ['area','area filled', 'equivalent diameter area', 'feret diameter max',
  'eccentricity', 'perimeter', 'perimeter to surface ratio', 'sphericity',
  'spherical dispropotion', 'solidity', 'major axis', 'minor axis', 'elogantion',
  'fractal dimension','lacunarity']


population_features = {
    'AC': features_ac,
    'LAT': features_lat,
    'VM': features_m,
    'PC': features_pc,
    'DM': features_dm,
}




'''
plot distribution for each feature
'''

for i, feature_name in enumerate(shape_feats_list):
    # Create a new figure for each feature
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Distribution of {shape_feats_list[i]}', fontsize=16, fontfamily='Arial')

    # Create a list of colors for each class
    class_colors = ['blue', 'red', 'green', 'yellow', 'purple']
    
    # Lists to store the means for each class
    class_means = []
    y_max_tot = []
    # Plot lines for each class and fill the area under the curve with transparency
    for j, (class_name, data) in enumerate(population_features.items()):
        
        y_values = data[:, i].flatten()
        midpoints = np.linspace(min(y_values), max(y_values), 100)  # Create midpoints for the line plot
        kernel = scipy.stats.gaussian_kde(y_values)
        pdf_values = kernel(midpoints)
        
        # Plot the line
        line, = ax.plot(midpoints, pdf_values, color='black')
        x_line_data, y_line_data = line.get_data()
        ax.fill_between(midpoints, pdf_values, alpha=0.35, color=class_colors[j], label=class_name)
        
        # Calculate and store the mean
        mean_value = round(np.median(y_values),2)
        y_max = 0
        x_value_mean = 0
        for k in range(len(x_line_data)-1):
            if round(np.mean(x_line_data[k:k+1]),2) -0.02 <= mean_value <= round(np.mean(x_line_data[k:k+1]),2)+0.02:  
                
                y_max = y_line_data[k]
                x_value_mean = x_line_data[k]
         
        y_max_tot.append(y_max)
        class_means.append(x_value_mean)
    
    # Plot dashed lines after all other elements
    for j, (class_name, data) in enumerate(population_features.items()):
        
        ax.plot([class_means[j], class_means[j]], [0, y_max_tot[j]], color=class_colors[j], linestyle='--')
        
        ax.legend()
        # print("shape", shape_feats_list[i])
        # print("y_max_tot", y_max_tot[j], class_colors[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # title position
    #plt.savefig(f'Distribution of {shape_feats_list[i]}', dpi=300, bbox_inches='tight')
    plt.show()

'''
uncomment the code below if you want to plot the distributions for the most 2 separeted classes
by feature
'''
# for i, feature_name in enumerate(shape_feats_list):
#     # Create a new figure for each feature
#     fig, ax = plt.subplots(figsize=(8, 6))
#     fig.suptitle(f'{shape_feats_list[i]}', fontsize=24, fontfamily='Arial', y=0.88)

#     # Create a list of colors for each class
#     class_colors = ['#FF6BA9',  # Pink
#                  '#FFDB58',  # Yellow
#                  '#87CEEB',  # Blue
#                  '#808080',  # Gray
#                  '#9ACD32']  # Green
#     #['#FFD6F1', '#FFF2CC', '#DEEBF7', '#E2F0D9', '#C3C3C3'] 
#     #['blue', 'red', 'green', 'yellow', 'purple']

#     # Lists to store the means for each class
#     class_means = []
#     y_max_tot = []
#     # Extract normalized y-values for the current feature for all classes
#     y_values_all = [data[:, i].flatten() for class_name, data in population_features.items()]
#     y_values_all_means = [np.median(y_values) for y_values in y_values_all] #np.median(y_values)

#     # Find the indices of the classes with minimum and maximum mean values
#     min_mean_idx = np.argmin(y_values_all_means)
#     max_mean_idx = np.argmax(y_values_all_means)
    
    
#     # Remove the max and min values
#     y_values_without_min_max = np.delete(y_values_all_means, [min_mean_idx, max_mean_idx])
    
#     # Find index of the element before the max value
#     value_before_max = np.max(y_values_without_min_max)
    
#     # Find the index of value_before_max in y_values_all_means
#     index_before_max = np.where(y_values_all_means == value_before_max)[0][0]
    
#     # Find index of the element after the min value
#     value_after_min = np.min(y_values_without_min_max)
    
#     # Find the index of value_after_min in y_values_all_means
#     index_after_min = np.where(y_values_all_means == value_after_min)[0][0]
    

#     # Plot lines for classes with minimum and maximum mean values
#     for j, (class_name, data) in enumerate(population_features.items()):
#         print("class_name", class_name)
#         print("y_values_all_means", y_values_all_means)
#         print("shape_feats_list[i]", shape_feats_list[i])
#         if j == min_mean_idx or j == max_mean_idx: #or j == index_after_min
#             # Extract normalized y-values for the current feature
#             y_values = data[:, i].flatten()
#             midpoints = np.linspace(min(y_values), max(y_values), 100)  # Create midpoints for the line plot
#             kernel = scipy.stats.gaussian_kde(y_values)
#             pdf_values = kernel(midpoints)

#             # Plot the line
#             line, = ax.plot(midpoints, pdf_values, color='black')
#             ax.fill_between(midpoints, pdf_values, alpha=0.35, color=class_colors[j], label=class_name)

#             # Store max y-value for dashed line
#             y_max_tot.append(max(pdf_values))
#             class_means.append(midpoints[np.argmax(pdf_values)])
           

#     # Plot dashed lines for min and max mean classes
#     ax.plot([class_means[0], class_means[0]], [0, y_max_tot[0]], color='black', linestyle='--', alpha=0.75)
#     ax.plot([class_means[1], class_means[1]], [0, y_max_tot[1]], color='black', linestyle='--', alpha=0.75)
#     #ax.plot([class_means[2], class_means[2]], [0, y_max_tot[2]], color='black', linestyle='--', alpha=0.75)

#     legend = ax.legend(prop={'size': 16, 'family': 'Arial'})
#     for text in legend.get_texts():
#         text.set_fontsize(16)
#         text.set_fontfamily('Arial')

#     ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#     ax.tick_params(axis='x', labelsize=16)
#     ax.tick_params(axis='y', labelsize=16)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # title position
#     # plt.savefig(f'distribution of {shape_feats_list[i]}', dpi=300, bbox_inches='tight')
#     plt.show()



populations = ['AC', 'LAT', 'VM', 'PC', 'DM']

features_ac_hist = np.zeros((num_points, len(shape_feats_list)))
features_dm_hist = np.zeros((num_points, len(shape_feats_list)))
features_lat_hist = np.zeros((num_points, len(shape_feats_list)))
features_m_hist = np.zeros((num_points, len(shape_feats_list)))
features_pc_hist = np.zeros((num_points, len(shape_feats_list)))

for i, (feature_name, feature_data) in enumerate(features_dict.items()):
    
    features_ac_hist[:,i] = feature_data.get('ac') 
    features_dm_hist[:,i] = feature_data.get('dm') 
    features_lat_hist[:,i] = feature_data.get('lat') 
    features_m_hist[:,i] = feature_data.get('vm') 
    features_pc_hist[:,i] = feature_data.get('pc') 
    

weights_ac_hist = np.zeros((num_points, len(shape_feats_list)))
weights_dm_hist = np.zeros((num_points, len(shape_feats_list)))
weights_lat_hist = np.zeros((num_points, len(shape_feats_list)))
weights_m_hist = np.zeros((num_points, len(shape_feats_list)))
weights_pc_hist = np.zeros((num_points, len(shape_feats_list)))


for i, (feature_name, feature_data) in enumerate(weight_dict.items()):
    
    weights_ac_hist[:,i] = feature_data.get('ac') 
    weights_dm_hist[:,i] = feature_data.get('dm') 
    weights_lat_hist[:,i] = feature_data.get('lat') 
    weights_m_hist[:,i] = feature_data.get('vm') 
    weights_pc_hist[:,i] = feature_data.get('pc') 
        
    


features_list = [features_ac_hist, features_lat_hist, features_m_hist, features_pc_hist, features_dm_hist]
weights_list = [weights_ac_hist, weights_lat_hist, weights_m_hist, weights_pc_hist, weights_dm_hist]

'''
Plot of the EMD distances for single features
'''

tot_distance_matrix = np.zeros((5, 5))
distance_matrix_all_features = np.zeros((len(shape_feats_list),5,5))
# Loop through the features and calculate distance matrices

for i in range(len(shape_feats_list)):
    
    distance_matrix = np.zeros((5, 5))
    
    for j in range(5):
        for k in range(5):
                
                distance_matrix[j, k] = wasserstein_distance(features_list[j][:, i], features_list[k][:, i],
                                                                  weights_list[j][:,i], weights_list[k][:,i]
                                                                  )
                    
                distance_matrix[k, j] = wasserstein_distance(features_list[k][:, i], features_list[j][:, i],
                                                                weights_list[k][:,i], weights_list[j][:,i]
                                                            )
                
                   
                distance_matrix_all_features[i,j,k] = distance_matrix[j, k]
                distance_matrix_all_features[i,k,j] = distance_matrix[k, j]
                
                 
    # Add values inside each cell
    for x in range(5):
        for y in range(5):
            if y < x:  # Only display numbers in the lower triangular part
                ax.text(y, x, f'{distance_matrix[x, y]:.2f}', va='center', ha='center', color='black')
            else:
                ax.text(y, x, '', va='center', ha='center')  # Empty string for upper triangular part


    
    # Plot the distance matrix
    ax = axs[i // 5, i % 5]
    im = ax.imshow(distance_matrix, cmap='Blues')  
    ax.set_title(shape_feats_list[i], fontsize=20, fontfamily='Arial', pad=15)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(populations)
    ax.set_yticklabels(populations)
    
    
# Remove the unused subplots 
num_subplots = len(shape_feats_list)
for i in range(num_subplots, axs.size):
    fig.delaxes(axs.flatten()[i])
# Add the color legend below the plots
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), location='right', pad=0.015, aspect=30)
# Set the color limit for the colorbar
mappable = cbar.mappable   
mappable.set_clim(vmin=0, vmax=0.3)

max_value = np.max(distance_matrix_all_features[:])
distance_matrix_upper = np.zeros((len(shape_feats_list),5,5))
# Create subplots for each feature using distance_matrix_all_features
num_features = len(shape_feats_list)
fig, axs = plt.subplots(num_features // 5, 5, figsize=(25, 10), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.3})
fig.suptitle("Wasserstein Distance Matrices", fontsize=16)

# Loop through distance_matrix_all_features to create subplots
for i in range(num_features):
    
    for x in range(5):
            for y in range(5):
                if y < x:  # Only display colors in the upper triangular part
                    distance_matrix_upper[i, x, y] = 0 
                else:
                    distance_matrix_upper[i, x, y] = distance_matrix_all_features[i, x, y]
                
    ax = axs[i // 5, i % 5]
      
    im = ax.imshow(distance_matrix_upper[i], cmap='Blues', vmax=max_value)  
    ax.set_title(shape_feats_list[i], fontsize=16, fontfamily='Arial', pad=12)
    
    # Set labels only for external sides
    if i // 5 == num_features // 5 - 1:  # Bottom row
        ax.set_xticks(range(5))
        ax.set_xticklabels(populations)
        
    else:
        ax.set_xticks([])
 
    if i % 5 == 0:  # Leftmost column
        ax.set_yticks(range(5))
        ax.set_yticklabels(populations)

    
    for x in range(5):
            for y in range(5):
                if y < x:  # Only display numbers in the lower triangular part
                    ax.text(y, x, f'{distance_matrix_all_features[i, x, y]:.2f}', va='center', ha='center', color='black', fontsize=10)
                else:
                    ax.text(y, x, '', va='center', ha='center')  # Empty string for upper triangular part
    
    
    

# Add the common colorbar below the plots
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), location='right', pad=0.015, aspect=30)
# Set the color limit for the colorbar
mappable = cbar.mappable
mappable.set_clim(vmin=0, vmax=max_value)
plt.savefig('EMD Distance Matrices.png', dpi=300, bbox_inches='tight')
plt.show()
tick_positions = np.arange(len(populations))
tick_labels = ['AC', 'LAT', 'VM', 'PC', 'DM']  


'''
Aggregated EMD 
'''
I15 = np.eye(15)
upper_matrix_tot =  np.zeros((5,5))
for i in range(len(shape_feats_list)):
        
        single_feat_distance_matrix = distance_matrix_all_features[i,:,:]
    
        for k in range (0,5):
            for l in range (0,5):
                
                dist_single_feature = single_feat_distance_matrix[k,l]
                # print("dist_single_feature", dist_single_feature)
                
                for j in range(i,len(shape_feats_list)):
                    
                    dist_single_feat_vs_rest = distance_matrix_all_features[j,k,l]
    
                    value = I15[i,j]-((1/15)*np.abs(corr_matrix[i,j]-I15[i,j]))
                    tot_distance_matrix[k,l] += dist_single_feature.T*value*dist_single_feat_vs_rest
                    

tot_distance_matrix = np.sqrt(tot_distance_matrix)

for x in range(5):
        for y in range(5):
            if y < x:  # Only display numbers in the lower triangular part
                plt.text(y, x, f'{tot_distance_matrix[x, y]:.2f}', va='center', ha='center', color='black')
            else:
                plt.text(y, x, '', va='center', ha='center')  # Empty string for upper triangular part

for x in range(5):
        for y in range(5):
            if y < x:  # Only display colors in the upper triangular part
                upper_matrix_tot[x, y] = 0 
            else:
                upper_matrix_tot[x, y] = tot_distance_matrix[x, y]
                
plt.imshow(upper_matrix_tot, cmap='Blues') 
plt.title('Aggregated distance matrix', fontfamily='Arial', pad=12)
plt.xticks(tick_positions, tick_labels)
plt.yticks(tick_positions, tick_labels)
plt.colorbar(location='right') 
plt.savefig('Aggragated EMD Matrix.png', dpi=300, bbox_inches='tight')
plt.show()
