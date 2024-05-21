# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:01:19 2024

@author: mikin
"""

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os
import csv



from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score


# Get the current working directory
current_dir = os.getcwd()


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
    
    matrix = np.empty((tot_synt_data, num_columns))
    
    for i in range(matrix.shape[0]):
        for column, std_dev in std_deviations.items():

            random_std_dev = np.random.uniform(0, std_dev/3)
            function = np.random.normal(loc=1, scale=random_std_dev, size=num_columns)
            matrix[i, :] = function[:matrix.shape[1]]
            
    # print("matrix", matrix.shape)
    
    data_features_minority =  data_fewest_samples.iloc[:, 2:]
    data_features_minority = data_features_minority.to_numpy()
    data_central_point = central_point.to_numpy()
    repeated_data_features_minority = np.tile(data_features_minority, 
                              (tot_synt_data // data_features_minority.shape[0] + 1, 1))[:tot_synt_data]
    
    # Perform the operation
    synthetic_data = repeated_data_features_minority + matrix * (data_central_point - repeated_data_features_minority)
    
    features_fewest_samples = data_fewest_samples.drop(['ID', 'LABEL'], axis=1)
    features_fewest_samples = features_fewest_samples.to_numpy()
    # print(synthetic_data.shape)
    resampled_data = np.concatenate((features_fewest_samples, synthetic_data), axis=0)
    
    return(resampled_data)



def experiment_classification(path):
    
    data = pd.read_csv(path)
    
    columns_to_normalize = data.columns[2:]
    subset_data = data[columns_to_normalize]
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize the subset of columns
    normalized_values = scaler.fit_transform(subset_data)
    
    # Assign the normalized values back to the original DataFrame
    data.loc[:, columns_to_normalize] = normalized_values
    
    # Unique labels in the 'LABEL' column
    unique_labels = data['LABEL'].unique()
    
    df_name_features = ['area', 'area filled', 'equivalent diameter area', 'feret diameter max', 
                    'eccentricity', 'perimeter', 'perimeter to surface ratio', 'sphericity', 
                    'spherical dispropotion', 'solidity', 'major axis', 'minor axis', 'elogantion',
                    'fractal dimension', 'lacunarity']   
        
    
    df_name_features = np.array(df_name_features)
    
    
    # Dictionary to store synthetic data for each label
    synthetic_data = {}
    
    # Calculate max length
    max_length = data.groupby('LABEL').size().max()
    
    # Loop through each unique label
    for label in unique_labels:
       
        df_subset = data[data['LABEL'] == label]
        synthetic_data[label] = create_synthetic_data(df_subset, max_length)
    
    
    class_features = []
    class_labels = []
    
    
    for label, data_array in synthetic_data.items():
        
        # Extract features and labels
        features = data_array[:, :-1]  # Assuming the last column is the label
        labels = np.full((features.shape[0],), label)  # Create an array of labels
        
        class_features.extend(features.tolist())
        class_labels.extend(labels.tolist())
        
    
    datasets = [[class_features, class_labels]]
    
    
    
    n_trees = 100
    
    rf_classifier = RandomForestClassifier(n_estimators = n_trees, 
                           max_depth=10, 
                           criterion='gini',
                           random_state=8)
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
        #                             random_state=42),
        OneVsRestClassifier(rf_classifier)]
                               # bootstrap = True,
                               #class_weight="balanced")]
        # ExtraTreesClassifier(n_estimators = 100,
        #                      max_depth=10,
        #                      criterion="entropy",
        #                      class_weight="balanced")]
    #names = ["Balancd L1 Regression", " DecisionTree", "Linear SVM", "RandomForest"]
    # names = ["BalancedBagging","RandomForest"]
    names = ["RandomForest"]
    
    brain_results = [["Classifier","Accuracy on Test Set", "Std"]]
    results=[brain_results]
    
    
    num_iter = 10
    
        
    for i in range(num_iter):
        # iterate over datasets
        performance_scores = []
        f1_scores = []
        for ds_cnt, ds in enumerate(datasets):
             
            # preprocess dataset, split into training and test part
            X, y = ds
            X = StandardScaler().fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
                       
            for name, clf in zip(names, classifiers):
               
                # Fit the random forest classifier to the bootstrap training set
                clf.fit(X_train, y_train)
    
    
                # Aggregate the feature importances from each binary classifier
                feature_importances = np.zeros(X.shape[1])
                importances_list = []
                for estimator in clf.estimators_:
                    feature_importances = estimator.feature_importances_
                    importances_list.append(feature_importances)
    
                importances_array = np.array(importances_list)
                std = np.std(importances_array, axis=0)
                          
                # Sort the features by their importance
                sorted_indices = np.argsort(feature_importances)[::-1]
                
                    
                # Create an array with the names of the top features
                top_feature_names = np.array([df_name_features[index] for index in sorted_indices])
                top_feature_importances = np.array([feature_importances[index] for index in sorted_indices])
                top_feature_importances = top_feature_importances*100
                top_feature_importances = np.round(top_feature_importances,1)
                
                
                std_important_features = np.array([std[index] for index in sorted_indices])
                forest_importances = pd.Series(top_feature_importances, index=top_feature_names)
    
                '''
                Plot the feature importances usign MDI
                '''
                # fig, ax = plt.subplots()
                # forest_importances.plot.bar(yerr=std_important_features, ax=ax)
                # ax.set_title("Feature importances using MDI")
                # ax.set_ylabel("MDI (%)")
                # fig.tight_layout()
                # plt.savefig('MDI.png', dpi=300, bbox_inches='tight')
                # for i in range(len(top_feature_names[:11])):
                #     print(f"Feature '{top_feature_names[i]}': Importance {top_feature_importances[i]} %")
                
                # Iterate over each class
                target_names = ['control', 'withdraw', 'relapse']
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
                    
                    # Print the class label and its corresponding top-k features
                    # print("Class:", class_label)
                    # for i in range(len(top_feature_names_k[:11])):
                    #     print(f"Feature '{top_feature_names_k[i]}': Importance {top_feature_importances[i]} %")
                    
                    '''
                    Plot the feature importances for each class using MDI 
                    '''
                    
                    # Plot the feature importances for the current class
                    # fig, ax = plt.subplots()
                    # ax.bar(top_feature_names_k, top_feature_importances_k)
                    # ax.set_title(f"Feature Importances using MDI for Class {target_names[class_label]}")
                    # ax.set_ylabel("MDI (%)")
                    # plt.xticks(rotation=90)
                    # fig.tight_layout()
                    # plt.show()
                
                        
          
                
                # testing 
                y_pred = clf.predict_proba(X_test)
                pred_train = clf.predict(X_train)
                pred_test = clf.predict(X_test)
                
            
                # Evaluate the performance metrics
                # roc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
                # print("roc", roc)
                score = clf.score(X_test, y_test)
                conf = confusion_matrix(y_test, pred_test)
                report = classification_report(y_test, pred_test)
                
                '''
                print classification report for each iteration
                '''
                # print("classification report for iteration", i+1)
                # print(report)
    
                # Print the evaluation metrics
                # print(f"Iteration: {i+1}")
                # print("ROC AUC:", roc)
                # print("Confusion Matrix:")
                # for row in conf:
                #     print('\t'.join([str(cell) for cell in row]))
                # print("f1_scores", f1_scores)
                
                # print("Classification report for training:")
                # print(classification_report(y_train, pred_train, target_names=target_names))
                # print("Classification report for test:")
                # print(classification_report(y_test, pred_test, target_names=target_names))
                accuracy = accuracy_score(y_test, pred_test)
    
                # Append the performance score to the list
                performance_scores.append(accuracy)
                
                '''
                code to make binary classification
                '''
                # for class_pair in [(0, 1), (0, 2), (1, 2)]:
                #     # Filter the training data for the current class pair
                #     y_train_pair = np.array(y_train)
                #     train_indices = np.where((y_train_pair == class_pair[0]) | (y_train_pair == class_pair[1]))[0]
                #     X_train_pair = X_train[train_indices]
                #     y_train_pair_2 = y_train_pair[train_indices]
    
                #     # Fit the random forest classifier to the training data for the current class pair
                #     clf.fit(X_train_pair, y_train_pair_2)
    
                #     # Filter the test data for the current class pair
                #     y_test_pair = np.array(y_test)
                #     test_indices = np.where((y_test_pair == class_pair[0]) | (y_test_pair == class_pair[1]))[0]
                #     X_test_pair = X_test[test_indices]
                #     y_test_pair_2 = y_test_pair[test_indices]
    
                #     # Make predictions for the current class pair
                #     y_pred_pair = clf.predict(X_test_pair)
    
                #     # Calculate the F1 score for the current class pair
                #     f1_pair = accuracy_score(y_test_pair_2, y_pred_pair)
                #     # Append the F1 score to the list
                #     f1_scores.append((class_pair, f1_pair))
                
        
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
            
        final_results[0].append([names[i], np.mean(ACC_i), np.std(ACC_i)])
        # print(brain_results[0])
        # print(final_results[0])
        # Extract and format the elements from brain_results and final_results
        brain_results_str = ", ".join(str(item) for item in brain_results[0])
        final_results_str = " ".join(str(item) for item in final_results[0])
        
        # Combine them into a single string with new lines separating the rows
        final_result_classification = f"{brain_results_str}\n{final_results_str}"
   
        return final_result_classification                           
# write_to_csv(save_name, num_iter, final_results[0])




# List of filenames
'''
opioids experiment
'''
filenames = ["features_ac.csv", "features_dm.csv", "features_lat.csv", "features_vm.csv", "features_pc.csv"]
# Loop through each filename
for filename in filenames: #for opioids experiment 
    # Construct the full path
    path = os.path.join(current_dir, "r", filename)
    print(f"Processing: {filename}")
    # Call the function with the constructed path
    final_result_classification = experiment_classification(path)
    print(final_result_classification)
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
    final_result_classification = experiment_classification(path)
    print(final_result_classification)


