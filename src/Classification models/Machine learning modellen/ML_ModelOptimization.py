"""
This function is used for hyperparameter tuning and feature selection
"""

# %% Import packages
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r'C:\Users\f.hiemstra\Documents\ICK_sleep_project\ICK_sleep_project\src')
from mrmr import mrmr_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector
import time
from datetime import datetime

# File paths
results_data_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_2_Hyperparameter tuning'
model_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_6_OptimizationSet'
tuning_results_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_2_Hyperparameter tuning'

# Set classifiers & EEG channels
classifiers = ['DT']#, 'SVM', 'DT'] # Classifiers to tune
#eeg_channels = #['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                #'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                #'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
eeg_channels = ['EEG F3-C3:']
sample_number = 50000
channel_method = 'EEG F3-C3' # all_channels, or selected channel

feature_data = pd.DataFrame([])
print(datetime.now(), ': Start loading')
for feature_data_file_name in os.listdir(feature_data_file_path):
    if feature_data_file_name.endswith('.csv'):
        feature_data = feature_data.append(pd.read_csv(
            os.path.join(feature_data_file_path, feature_data_file_name),
            sep=',',
            decimal='.'))
        print(datetime.now(), ': %s loaded' % feature_data_file_name)
feature_data = feature_data.sample(sample_number) # Shuffle data

for number_of_stages in [3]:
    number_of_stages_key = 'Three_state_labels'  # Three_state_labels or Four_state_labels
    if number_of_stages == 4:
        feature_data = feature_data[feature_data['Age category'].isin(['6-12 months', '1-3 years', '5-9 years', \
                                                    '9-13 years', '13-18 years'])]
        number_of_stages_key = 'Four_state_labels'

    eog_features_list = [col for col in feature_data if col.startswith('EOG ROC-LOC:')]
    emg_features_list = [col for col in feature_data if col.startswith('EMG EMG Chin:')]
    age_features_list = list(pd.get_dummies(feature_data['Age category'],
                                            prefix='age_group_').columns)

    for classifier in classifiers:
        start_time_classifier = time.time()

        df_eeg_channel = []
        df_mrmr_selected_features = []
        df_feature_subset_sfs_names = []
        df_param_grid = []
        df_optimal_hyperparameters = []

        count = 0 # To track progress

        # Loop over all EEG channels
        for eeg_channel in eeg_channels:
            count = count + 1

            """
            ##### EXTRACT LABELS AND FEATURES #####
            """
            ### Add labels and extract the feature
            labels_and_features = pd.DataFrame([])
            labels_and_features[number_of_stages_key] = feature_data[number_of_stages_key]
            labels_and_features[number_of_stages_key + '- linear'] = feature_data[number_of_stages_key + '- linear']
            labels_and_features['Patient key'] = feature_data['Patient key']
            labels_and_features['Age category'] = feature_data['Age category']
            labels_and_features[eog_features_list] = feature_data[eog_features_list]
            labels_and_features[emg_features_list] = feature_data[emg_features_list]
            labels_and_features[age_features_list] = feature_data[age_features_list]

            # EEG features
            eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]
            labels_and_features[eeg_features_list] = feature_data[eeg_features_list]

            labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it
            age_features = labels_and_features[age_features_list]
            labels_and_features = labels_and_features.drop(columns=age_features_list)

            ### Extract labels, features and groups
            labels_cat = labels_and_features[number_of_stages_key]
            labels_lin = labels_and_features[number_of_stages_key + '- linear']
            groups = labels_and_features['Patient key']
            features = labels_and_features.iloc[:, 4:]

            """
            ##### FIRST FEATURE SELECTION #####
            """
            start_time = time.time()
            n_features_to_select = 20

            ### Minimum redundancy maximum relevance
            mrmr_selected_features = mrmr_classif(features, labels_lin, K=n_features_to_select)
            feature_subset_mrmr = features[mrmr_selected_features]
            feature_subset_mrmr[age_features_list] = age_features
            print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
            print(mrmr_selected_features)

            """
            ##### SEQUENTIAL FORWARD FEATURE SELECTION + HYPERPARAMETER TUNING #####
            """
            steps_feature_selection = 2
            start_time = time.time()

            # Classifier variables
            if classifier == 'DT':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(criterion='gini')
                param_grid = {'sfs__k_features': range(2, n_features_to_select + 1, steps_feature_selection),
                              'sfs__estimator__max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18]}
                labels = labels_cat
                n_iter = int(0.25*10*8)

            elif classifier == 'SVM':
                from sklearn.svm import SVC
                clf = SVC(kernel='rbf')
                param_grid = {'sfs__k_features': range(2, n_features_to_select + 1, steps_feature_selection),
                              'sfs__estimator__gamma': np.logspace(-3, 1, 5),
                              'sfs__estimator__C': np.logspace(-2, 2, 5)}
                labels = labels_cat
                n_iter = int(0.10*5*5*10)

            elif classifier == "xgboost":
                from xgboost import XGBClassifier
                clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6, gamma=0)
                param_grid ={'sfs__k_features': range(2, n_features_to_select + 1, steps_feature_selection),
                             "sfs__estimator__learning_rate" : [0.05, 0.10, 0.15, 0.20],
                             "sfs__estimator__min_child_weight": [ 1, 3, 5, 7],
                             "sfs__estimator__colsample_bytree" : [0.5, 0.75, 1]
                            }
                labels = labels_cat
                n_iter = int(0.10*10*4*4*3)

            ### Start randomized grid search for hyperparameter tuning
            sfs = SequentialFeatureSelector(estimator=clf, forward=True, scoring='accuracy', cv=3, n_jobs=-1)
            pipe = Pipeline([('sfs', sfs),
                             ('clf', clf)])
            grid_search = RandomizedSearchCV(estimator=pipe, n_iter=n_iter, param_distributions=param_grid,
                                             scoring='accuracy', cv=3, n_jobs=-1, verbose=10)
            grid_search.fit(feature_subset_mrmr, labels)

            ### Extract best hyperparameters
            idx_feature_subset_2 = grid_search.best_estimator_.steps[0][1].k_feature_idx_
            feature_subset_sfs = feature_subset_mrmr.iloc[:, np.array(idx_feature_subset_2)]
            feature_subset_sfs_names = feature_subset_sfs.columns
            optimal_hyperparameters = grid_search.best_params_

            print("Hyperparameter tuning %s completed: %s hours" % (eeg_channel, (time.time() - start_time) / 3600))
            print(optimal_hyperparameters,
                  "\n SFS selected features: ", feature_subset_sfs_names)

            """" 
            ##### 6: SAVE TRAINING RESULTS #####
            """
            df_eeg_channel.append(eeg_channel)
            df_mrmr_selected_features.append(str(mrmr_selected_features))
            df_feature_subset_sfs_names.append(str(feature_subset_sfs_names))
            df_param_grid.append(str(param_grid))
            df_optimal_hyperparameters.append(str(optimal_hyperparameters))

            print("%s: %s of %s channels completed (%s): %s hours"
                  % (classifier, count, len(eeg_channels), eeg_channel, (time.time() - start_time) / 3600))

        # Write to pandas DataFrame
        dfModelResults = pd.DataFrame({'EEG Channel': df_eeg_channel,
                                       'mrmr Feature selection': df_mrmr_selected_features,
                                       'sfs Feature selection': df_feature_subset_sfs_names,
                                       'Hyperparameters': df_optimal_hyperparameters,
                                       'Parameter grid': df_param_grid})

        tuning_results_file_name = classifier + '_tuning_' + channel_method + '_%ssamples_' % sample_number + number_of_stages_key + '.xlsx'
        writer = pd.ExcelWriter(os.path.join(results_data_file_path, tuning_results_file_name))
        dfModelResults.to_excel(writer)
        writer.save()

        print('Feature selection + hyperparameter tuning', classifier, 'completed: %s hours' % (
                (time.time() - start_time_classifier) / 3600))
