# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:10:55 2024

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
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity



sns.set(context='paper', style='white', font_scale=1.5, rc={"lines.linewidth": 2.5, 'font.family': 'Arial'})
sns.set_palette('muted')


# Get the current working directory
current_dir = os.getcwd()


def morphological_analysis(path):
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
            
 
    df_control = data[(data['LABEL']==0)] 
    df_withdrawal = data[(data['LABEL']==1)] 
    df_relapse = data[(data['LABEL']==2)] 
      
    # Calculate the length of each DataFrame
    lengths = [len(df_control), len(df_withdrawal), len(df_relapse)]
      
    min_length = min(lengths)

    # Extract the feature vectors for each class
    features_control = df_control.iloc[:, 2:].values
    features_withdrawal = df_withdrawal.iloc[:, 2:].values
    features_relapse = df_relapse.iloc[:, 2:].values
    
    
    conc_feature_for_corr = np.concatenate((features_control, features_withdrawal, features_relapse), axis=0)   
    corr_matrix = np.zeros((conc_feature_for_corr.shape[1], conc_feature_for_corr.shape[1]))
    
    for i in range(conc_feature_for_corr.shape[1]):
        for j in range(conc_feature_for_corr.shape[1]):
          
            corr_index, _ = scipy.stats.spearmanr(conc_feature_for_corr[:,i], conc_feature_for_corr[:,j])
            corr_matrix[i][j] = corr_index
    
    shape_feats_list = ['area','area filled', 'equivalent diameter area', 'feret diameter max',
      'eccentricity', 'perimeter', 'perimeter to surface ratio', 'sphericity',
      'spherical dispropotion', 'solidity', 'major axis', 'minor axis', 'elogantion',
      'fractal dimension','lacunarity']
    
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=shape_feats_list, yticklabels=shape_feats_list)
    # plt.title("Spearman Correlation Matrix")
    # plt.show()
    
    
    population_features = {
        'control': features_control,
        'withdrawal': features_withdrawal,
        'relapse': features_relapse,
    }
    
    for pop, data in population_features.items():
        fig, axs = plt.subplots(3, 5, figsize=(20, 10), sharex=True, sharey=True)
        fig.suptitle(pop, fontsize=16)
        
        # Loop through the features and plot normalized histograms
        for i in range(len(shape_feats_list)):
            ax = axs[i // 5, i % 5]
            
            # Extract normalized y-values for the current feature
            y_values = data[:, i].flatten()
           
            # Plot the histogram and get histogram values (heights) and bin edges (x values)
            heights, bin_edges, _ = ax.hist(y_values, color='lightgreen', edgecolor='black', bins=int(np.sqrt(min_length)), density=True)
            
            ax.set_title(shape_feats_list[i])
           
    
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust title position
        # plt.show()
        plt.close(fig)
    
    
    # Create a subplot grid of size 3 by 5
    fig, axs = plt.subplots(3, 5, figsize=(20, 10))
    fig.suptitle('Distribution of Features for Different Classes', fontsize=16)
    
    
    features_dict = {}
    weight_dict= {}
    
    
    # Loop through the features and plot distributions for each class
    for i, feature_name in enumerate(shape_feats_list):
        ax = axs[i // 5, i % 5]
        
        # Create a list of colors for each class
        class_colors = ['green', 'blue', 'red']
        
        feature_normalized = {}
        weigths_normalized = {}
        
        # Plot histograms for each class
        for j, (class_name, data) in enumerate(population_features.items()):
            # Extract normalized y-values for the current feature
            y_values = data[:, i].flatten()
            num_points = int(np.sqrt(min_length))
            
            # Plot the histogram using normalized heights
            heights, bin_edges, _ = ax.hist(y_values, color=class_colors[j], alpha=0.35, bins=num_points, label=class_name, 
                                            density=True, histtype='bar')
            
            
            midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            

            # Store data in nested dictionaries
            feature_normalized[class_name] = midpoints 
            weigths_normalized[class_name] = heights
        
        features_dict['features_' + feature_name] = feature_normalized
        weight_dict['features_' + feature_name] = weigths_normalized
        
        ax.set_title(feature_name)
        ax.legend()
       
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #title position
    # uncomment if you want to save the histrograms
    # plt.savefig(f'Distribution of features for {filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # comment if you want to visualize the histrograms 
    # plt.show()  #uncomment if you want to visualize the histrograms 
    
    
    for i, feature_name in enumerate(shape_feats_list):
        # Create a new figure for each feature
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'{shape_feats_list[i]}', fontsize=16, fontfamily='Arial')
    
        # Create a list of colors for each class
        class_colors = ['blue', 'red', 'green']
        
        # Lists to store the means for each class
        class_means = []
        y_max_tot = []
        # Plot lines for each class and fill the area under the curve with transparency
        for j, (class_name, data) in enumerate(population_features.items()):
            # Extract normalized y-values for the current feature
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
            
            ax.plot([class_means[j], class_means[j]], [0, y_max_tot[j]], color=class_colors[j], linestyle='--') #, alpha=0.75
            
            legend = ax.legend(prop={'size': 16, 'family': 'Arial'})
            for text in legend.get_texts():
                text.set_fontsize(16)
                text.set_fontfamily('Arial')
            
                       
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # uncomment if you want to save the distributions 
        # plt.savefig(f'{shape_feats_list[i]}_distribution_{filename}', dpi=300, bbox_inches='tight')
        plt.close(fig) # comment if you want to visualize the distributions 
        # plt.show()  #uncomment if you want to visualize the distributions 

    

    populations = ['control', 'withdrawal', 'relapse',]
    features_ac_hist = np.zeros((num_points, len(shape_feats_list)))
    features_dm_hist = np.zeros((num_points, len(shape_feats_list)))
    features_lat_hist = np.zeros((num_points, len(shape_feats_list)))
    
    
    for i, (feature_name, feature_data) in enumerate(features_dict.items()):
        
        features_ac_hist[:,i] = feature_data.get('control') 
        features_dm_hist[:,i] = feature_data.get('withdrawal') 
        features_lat_hist[:,i] = feature_data.get('relapse') 
          
    
    weights_ac_hist = np.zeros((num_points, len(shape_feats_list)))
    weights_dm_hist = np.zeros((num_points, len(shape_feats_list)))
    weights_lat_hist = np.zeros((num_points, len(shape_feats_list)))
    
    for i, (feature_name, feature_data) in enumerate(weight_dict.items()):
        
        weights_ac_hist[:,i] = feature_data.get('control') 
        weights_dm_hist[:,i] = feature_data.get('withdrawal') 
        weights_lat_hist[:,i] = feature_data.get('relapse') 
        

    
    features_list = [features_ac_hist, features_dm_hist, features_lat_hist]
    weights_list = [weights_ac_hist,weights_dm_hist, weights_lat_hist]
    
    # Creating the subplot of size 3 by 5
    fig, axs = plt.subplots(3, 5, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle(f"Wasserstein Distance Matrices of {filename}", fontsize=16)
    
    
    tot_distance_matrix = np.zeros((3, 3))
    distance_matrix_all_features = np.zeros((len(shape_feats_list),3,3))
    # Loop through the features and calculate distance matrices
    for i in range(len(shape_feats_list)):
        
        distance_matrix = np.zeros((3, 3))
        # corr_coeff = corr_matrix[]
        for j in range(0, 3):
            for k in range(0, 3):
                
                distance_matrix[j, k] = wasserstein_distance(features_list[j][:, i], features_list[k][:, i],
                                                              weights_list[j][:,i], weights_list[k][:,i]
                                                              )
                
                distance_matrix[k, j] = wasserstein_distance(features_list[k][:, i], features_list[j][:, i],
                                                            weights_list[k][:,i], weights_list[j][:,i]
                                                            )
                
                
                
                distance_matrix_all_features[i,j,k] = distance_matrix[j, k]
                distance_matrix_all_features[i,k,j] = distance_matrix[k, j]

    
        # Plot the distance matrix
        ax = axs[i // 5, i % 5]
        im = ax.imshow(distance_matrix, cmap='Blues')
        ax.set_title(shape_feats_list[i])
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(populations, fontsize=8)
        ax.set_yticklabels(populations, fontsize=8)
    
    # Remove the unused subplots 
    num_subplots = len(shape_feats_list)
    for i in range(num_subplots, axs.size):
        fig.delaxes(axs.flatten()[i])
    
    # Add the color legend below the plots
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), location='right', pad=0.015, aspect=30)
    mappable = cbar.mappable
    # mappable.set_clim(vmin=0, vmax=0.06)  # Set the color limit for the colorbar
    # plt.savefig(f'EMD matrices for {filename}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    tick_positions = np.arange(len(populations))
    tick_labels = ['control', 'withdrawal', 'relapse']  
   


    I15 = np.eye(15)
    upper_matrix_tot =  np.zeros((3,3))
    for i in range(len(shape_feats_list)):
            
            single_feat_distance_matrix = distance_matrix_all_features[i,:,:]
        
            for k in range (0,3):
                for l in range (0,3):
                    
                    dist_single_feature = single_feat_distance_matrix[k,l]
                    # print("dist_single_feature", dist_single_feature)
                    
                    for j in range(i,len(shape_feats_list)):
                        
                        dist_single_feat_vs_rest = distance_matrix_all_features[j,k,l]
                
                        value = I15[i,j]-((1/15)*np.abs(corr_matrix[i,j]-I15[i,j]))
                        tot_distance_matrix[k,l] += dist_single_feature.T*value*dist_single_feat_vs_rest
                        
                        
    tot_distance_matrix = np.sqrt(tot_distance_matrix)
    
    for x in range(3):
            for y in range(3):
                if y < x:  # Only display numbers in the lower triangular part :)
                    plt.text(y, x, f'{tot_distance_matrix[x, y]:.2f}', va='center', ha='center', color='black')
                else:
                    plt.text(y, x, '', va='center', ha='center')  # Empty string for upper triangular part
    
    for x in range(3):
            for y in range(3):
                if y < x:  # Only display colors in the upper triangular part
                    upper_matrix_tot[x, y] = 0 
                else:
                    upper_matrix_tot[x, y] = tot_distance_matrix[x, y]
                    
    im = plt.imshow(upper_matrix_tot, cmap='Blues') 
    plt.title(f'Wasserstein Distance Matrices of {filename}', fontfamily='Arial', pad=12)
    plt.yticks([])
    plt.xticks(tick_positions, tick_labels)
    #plt.yticks(tick_positions, tick_labels)
    cbar = plt.colorbar(im, location='right', pad=0.015, aspect=30)
    mappable = cbar.mappable
    # mappable.set_clim(vmin=0, vmax=0.4)
    # Set the colorbar alpha (transparency) to 0 to hide it
    cbar.set_alpha(0.0)
    
    # plt.savefig(f'EMD Matrix {filename}.png', dpi=300, bbox_inches='tight')
    plt.show()


filenames = ["features_ac.csv", "features_dm.csv", "features_lat.csv", "features_vm.csv", "features_pc.csv"]
# Loop through each filename
for filename in filenames: #for opioids experiment 
    # Construct the full path
    path = os.path.join(current_dir, "r", filename)
    print(f"Processing: {filename}")
    # Call the function with the constructed path
    morphological_analysis(path)
    
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
    morphological_analysis(path)
