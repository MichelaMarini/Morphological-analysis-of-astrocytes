# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:39:42 2024

@author: mikin
"""

import sys 
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

import os
import csv
import seaborn as sns
import random

from scipy.stats import norm


from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score


import matplotlib as mpl

# Set font to Arial
mpl.rcParams['font.family'] = 'Arial'

sns.set(context='paper', style='white', font_scale=1.5, rc={"lines.linewidth": 2.5, 'font.family': 'Arial'})
sns.set_palette('muted')


def create_synthetic_data(data_fewest_samples, max_length):
    
    ''' application of the SMOTE algortihm
    Wang, S., Dai, Y., Shen, J. & Xuan, J. Research on expansion and classification of imbalanced data based on 
    SMOTE algorithm. Sci Rep 11, 24039 (2021). https://doi.org:10.1038/s41598-021-03430-5
    '''
    
    min_length = len(data_fewest_samples)
    central_point = data_fewest_samples.iloc[:, 2:].mean()
       
    # Estimate the normal distribution and calculate standard deviation
    inbalanced_ratio = int(max_length) #*0.9
    tot_synt_data = inbalanced_ratio - min_length
    
    # Create an empty matrix
    num_columns = len(data_fewest_samples.columns[2:])
    
    # Estimate the normal distribution and calculate standard deviation
    std_deviations = {}
    
    for column in data_fewest_samples.columns[2:]:
        column_values = data_fewest_samples[column]
        mean = column_values.mean()
        std = column_values.std()
        # Estimate normal distribution using mean and standard deviation
        distribution = norm(loc=mean, scale=std)
        std_deviations[column] = distribution.std()
    
    matrix = np.zeros((tot_synt_data, num_columns))
    matrix_list = []
    for i in range(matrix.shape[0]):
        for column, std_dev in std_deviations.items():

            random_std_dev = np.random.uniform(0, std_dev/3)
            function = np.random.normal(loc=1, scale=random_std_dev, size=num_columns)
            
        matrix_list.append(function)
            
    matrix = np.array(matrix_list)
    
    
    data_features_minority =  data_fewest_samples.iloc[:, 2:]
    data_features_minority = data_features_minority.to_numpy()
    data_central_point = central_point.to_numpy()
    repeated_data_features_minority = np.tile(data_features_minority, 
                              (tot_synt_data // data_features_minority.shape[0] + 1, 1))[:tot_synt_data]
    
    # Perform the operation
    synthetic_data = repeated_data_features_minority + matrix * (data_central_point - repeated_data_features_minority)
    
    features_fewest_samples = data_fewest_samples.drop(['ID', 'LABEL'], axis=1)
    features_fewest_samples = features_fewest_samples.to_numpy()
   
    resampled_data = np.concatenate((features_fewest_samples, synthetic_data), axis=0)
    
    return(resampled_data)


            
path = os.path.join(os.getcwd(),"r","features_control_brain_region.csv") 
save_name = os.path.join(os.getcwd(),"r","classification_control_brain_region.csv")


data = pd.read_csv(path)

columns_to_normalize = data.columns[2:]
subset_data = data[columns_to_normalize]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the subset of columns
normalized_values = scaler.fit_transform(subset_data)

# Assign the normalized values back to the original DataFrame
data.loc[:, columns_to_normalize] = normalized_values


'''only shape features'''
df_name_features = ['area', 'area filled', 'equivalent diameter area', 'feret diameter max', 
                'eccentricity', 'perimeter', 'perimeter to surface ratio', 'sphericity', 
                'spherical dispropotion', 'solidity', 'major axis', 'minor axis', 'elogantion',
                'fractal dimension', 'lacunarity']  


df_name_features = np.array(df_name_features)

df_ac = data[(data['LABEL']==0)] 
df_dm = data[(data['LABEL']==1)]
df_lat = data[(data['LABEL']==2)] 
df_m = data[(data['LABEL']==3)] 
df_pc = data[(data['LABEL']==4)] 

# Calculate the length of each DataFrame
lengths = [len(df_ac), len(df_dm), len(df_lat), len(df_m), len(df_pc)]

max_length = max(lengths)

df_feat_ac = create_synthetic_data(df_ac, max_length)
df_feat_lat = create_synthetic_data(df_lat, max_length)
df_feat_m = create_synthetic_data(df_m, max_length)
df_feat_pc = create_synthetic_data(df_pc, max_length)

df_features_dm = df_dm.drop(['ID', 'LABEL'], axis=1)  
df_feat_dm = df_features_dm.to_numpy()


df_labels_ac = df_ac['LABEL']
df_labels_dm = df_dm['LABEL'] 
df_labels_lat = df_lat['LABEL']
df_labels_m = df_m['LABEL']
df_labels_pc = df_pc['LABEL']


df_labels_ac = df_labels_ac.to_numpy()
df_lab_ac = np.tile(df_labels_ac, 
                    (df_feat_ac.shape[0] // df_labels_ac.shape[0] + 1))[:df_feat_ac.shape[0]]

df_lab_dm = df_labels_dm.to_numpy()

df_lab_lat = df_labels_lat.to_numpy()
df_lab_lat = np.tile(df_labels_lat, 
                    (df_feat_lat.shape[0] // df_labels_lat.shape[0] + 1))[:df_feat_lat.shape[0]]

df_lab_m = df_labels_m.to_numpy()
df_lab_m = np.tile(df_labels_m, 
                    (df_feat_m.shape[0] // df_labels_m.shape[0] + 1))[:df_feat_m.shape[0]]

df_lab_pc = df_labels_pc.to_numpy()
df_lab_pc = np.tile(df_labels_pc, 
                    (df_feat_pc.shape[0] // df_labels_pc.shape[0] + 1))[:df_feat_pc.shape[0]]

df_feat  =  np.concatenate((df_feat_ac,df_feat_dm,df_feat_lat,
                            df_feat_m,df_feat_pc), axis=0)

df_lab = np.concatenate((df_lab_ac,df_lab_dm,df_lab_lat,
                         df_lab_m,df_lab_pc), axis=0)


tol=1e-4
datasets = [[df_feat, df_lab]]
rf_classifier = RandomForestClassifier(n_estimators = 100, 
                        max_depth=20, 
                        criterion='gini',
                        random_state=0)
classifiers= [
    #LogisticRegression(C=C, penalty='l1', tol=tol, class_weight='balanced', solver='saga', max_iter=10000),
    #LogisticRegression(C=C, penalty='l2', tol=tol, class_weight='balanced', solver='lbfgs', max_iter=10000),
    #SVC(kernel='linear',class_weight='balanced'),
    #DecisionTreeClassifier(),
    #SVC(kernel='linear'),
    
    # BalancedBaggingClassifier(estimator=RandomForestClassifier(),
    #                             sampling_strategy='all',
    #                             max_features=1.0,
    #                             replacement=False,
    #                             random_state=8),
    OneVsRestClassifier(rf_classifier)]
    #                         # bootstrap = True,
    #                         #class_weight="balanced")]

    # ExtraTreesClassifier(n_estimators = 100,
    #                      max_depth=10,
    #                      criterion="entropy",
    #                      class_weight="balanced")]
    
# names = ["BalancedBagging","RandomForest"]
names = ["RandomForest"]

brain_results = [["Classifier","Accuracy on Test Set", "Std"]]
results=[brain_results]

num_iter = 10


for i in range(num_iter):
    
    performance_scores = []
    
    for ds_cnt, ds in enumerate(datasets):
            X, y = ds
            X = StandardScaler().fit_transform(X)
        
            # Split the data into training and testing sets
            random_state_1 = random.randint(0,12)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state_1, stratify=y)
            # Initialize the classifier
            clf = classifiers[ds_cnt]
        
            # Aggregate the feature importances from each estimator
            importances_list = []

            for name, clf in zip(names, classifiers):
                print("classifier", name)
                # Fit the random forest classifier to the bootstrap training set
                clf.fit(X_train, y_train)

                # Aggregate the feature importances from each binary classifier
                feature_importances = np.zeros(X.shape[1])
                importances_list = []
                for estimator in clf.estimators_:
                    feature_importances = estimator.feature_importances_
                    importances_list.append(feature_importances)
                
                # Convert the list of feature importances into a numpy array
                importances_array = np.array(importances_list)
                
                # Calculate the standard deviation of feature importances across estimators
                std = np.std(importances_array, axis=0)
                
                # Sort the features by their importance (reversed order)
                sorted_indices = np.argsort(np.mean(importances_array, axis=0))[::-1]
                
                # Reverse the order for plotting
                sorted_indices = sorted_indices[::-1]
                
                # Create an array with the names of the top features
                top_feature_names = np.array([df_name_features[index] for index in sorted_indices])
                top_feature_importances = np.mean(importances_array[:, sorted_indices], axis=0) * 100
                top_feature_importances = np.round(top_feature_importances, 1)
                
                std_important_features = std[sorted_indices]
                forest_importances = pd.Series(top_feature_importances, index=top_feature_names)
                
                '''
                Plot the feature importances usign MDA
                '''
                # # Plot the feature importances using bar plot
                # fig, ax = plt.subplots(figsize=(12, 8))
                # forest_importances.plot.barh(ax=ax, xerr=std_important_features)
                # ax.set_title("Feature importances using MDA", fontsize=22)
                # ax.set_xlabel("MDA (%)", fontsize=16)
                # ax.tick_params(axis='y', labelsize=16)
                # ax.tick_params(axis='x', labelsize=16)
                # sns.despine()
                # fig.tight_layout()
                # # plt.savefig('Feature importances using MDA.png', dpi=300, bbox_inches='tight')
                # plt.show()
               
                
                
                # Iterate over each class
                target_names = ['ac', 'dm', 'lat', 'm', 'pc']
                num_classes = len(target_names)
                
                for class_label in range(num_classes):
                    # Get the feature importances for the current class
                    class_feature_importances = clf.estimators_[class_label].feature_importances_
                    
                    # Sort the feature importances in descending order
                    sorted_indices_k = np.argsort(class_feature_importances)[::-1]
                    
                    # Get the top-k most significant features for the current class
                    top_k_features = sorted_indices[:11]
                    top_feature_names_k = np.array([df_name_features[index] for index in sorted_indices_k]) #sorted_indices
                    top_feature_importances_k = np.array([class_feature_importances[index] for index in sorted_indices_k])
                    top_feature_importances_k = top_feature_importances_k*100
                    top_feature_importances_k = np.round(top_feature_importances_k,1)
                    
                    # # Print the class label and its corresponding top-k features
                    # print("Class:", class_label)
                    # for i in range(len(top_feature_names_k[:11])):
                    #     print(f"Feature '{top_feature_names_k[i]}': Importance {top_feature_importances[i]} %")
                    
                    '''
                    Plot the feature importances for each class using MDI 
                    '''
                    # # Plot the feature importances for the current class
                    # fig, ax = plt.subplots()
                    # ax.bar(top_feature_names_k, top_feature_importances_k)
                    # ax.set_title(f"Feature Importances using MDI for Class {target_names[class_label]}")
                    # ax.set_ylabel("MDI (%)")
                    # sns.despine()
                    # plt.xticks(rotation=90)
                    # fig.tight_layout()
                    # plt.show()
                    
                # testing 
                y_pred = clf.predict_proba(X_test)
                pred_train = clf.predict(X_train)
                pred_test = clf.predict(X_test)
                
            
                # Evaluate the performance metrics for the test dataset
                roc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
                # print("roc", roc)
                score = clf.score(X_test, y_test)
                conf = confusion_matrix(y_test, pred_test)
                
                report = classification_report(y_test, pred_test)
                print("classification report for iteration", i+1)
                print(report)

                # Print the evaluation metrics
                # print(f"Iteration: {i+1}")
                # print("ROC AUC:", roc)
                # print("Confusion Matrix:")
                # for row in conf:
                #     print('\t'.join([str(cell) for cell in row]))
               
                
                # print("Classification report for training:")
                # print(classification_report(y_train, pred_train, target_names=target_names))
                # print("Classification report for test:")
                # print(classification_report(y_test, pred_test, target_names=target_names))
                accuracy = accuracy_score(y_test, pred_test)
    
                # Append the performance score to the list
                performance_scores.append(accuracy)
            
                average_performance = sum(performance_scores) / len(performance_scores)
                std_dev_performance = np.std(performance_scores)
                results[ds_cnt].append([str(i), name, average_performance, std_dev_performance])
        
        # print("Average Performance Score:", average_performance)     
        # print("performance_scores", performance_scores)        

            
results[0].append("")
brain_final_results=[]
final_results=[brain_final_results]


for i in range(len(classifiers)):
    j=i+1
    entries=[j]
    ACC_i=[]
    
    for k in range(num_iter-1):
        j=j+len(classifiers)
        entries.append(j)
    
    for l in entries:
        
        ACC_i.append(results[0][l][2])
        # print(ACC_i)
    final_results[0].append([names[i], np.mean(ACC_i), np.std(ACC_i)])
    
    print(brain_results[0])
    print(final_results[0])
              

