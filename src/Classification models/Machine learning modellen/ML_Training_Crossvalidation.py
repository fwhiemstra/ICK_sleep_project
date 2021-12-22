"""
This function is used for training and crossvalidation of ML models
"""

# %% Import packages
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score, roc_auc_score
import time
import pickle
from datetime import datetime

classifiers = ['DT'] #SVM', 'xgboost', 'DT'] #['LDA', 'SVM', 'xgboost', 'DT'] #, 'SVM', 'RF', 'xgboost']
channel_method = 'EEG F3-C3'
sample_number = 50000

# File paths
results_data_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_3_Training & Crossvalidation results'
model_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_7_TrainingSet'
tuning_results_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_2_Hyperparameter tuning'

feature_data = pd.DataFrame([])
print(datetime.now(), ': Start loading')
for feature_data_file_name in os.listdir(feature_data_file_path):
    if feature_data_file_name.endswith('.csv'):
        feature_data = feature_data.append(pd.read_csv(
            os.path.join(feature_data_file_path, feature_data_file_name),
            sep=',',
            decimal='.'))
        print(datetime.now(), ': %s loaded' % feature_data_file_name)
feature_data = feature_data.sample(50000) # Shuffle data

"""
feature_data = pd.read_excel(os.path.join(r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing',
                                          '5000sampled_feature_data_0-18years_nonscaled.xlsx'))
"""

for number_of_stages in [3, 4]:
    number_of_stages_key = 'Three_state_labels'  # Three_state_labels or Four_state_labels
    if number_of_stages == 4:
        feature_data = feature_data[feature_data['Age category'].isin(['6-12 months', '1-3 years', '5-9 years', \
                                                    '9-13 years', '13-18 years'])]
        number_of_stages_key = 'Four_state_labels'

    for classifier in classifiers:
        start_time_classifier = time.time()

        tuning_results_file_name = classifier + '_tuning_' + channel_method + '_%ssamples_' % sample_number + number_of_stages_key + '.xlsx'
        hyperparametertuning_file = pd.read_excel(os.path.join(tuning_results_file_path, tuning_results_file_name))

        eeg_channel_all = []
        optimal_hyperparameters_all = []
        feature_subset_sfs_names_all = []
        train_auc_posterior_all = []
        train_accuracy_all = []
        train_cohens_kappa_all = []
        cv_auc_posterior_mean_all = []
        cv_auc_posterior_std_all = []
        cv_accuracy_mean_all = []
        cv_accuracy_std_all = []
        cv_cohens_kappa_mean_all = []
        cv_cohens_kappa_std_all = []

        count = 0 # To track progress
        i = 0
        # Loop over all EEG channels
        for eeg_channel in hyperparametertuning_file['EEG Channel']:
            start_time_channel = time.time()
            count = count + 1

            eeg_channel_all.append(eeg_channel)
            optimal_hyperparameters = eval(hyperparametertuning_file['Hyperparameters'][i])
            feature_subset_sfs_names = eval(hyperparametertuning_file['sfs Feature selection'][i].replace('Index(', '').replace(",\n      dtype='object')", '').replace("\n", ''))

            """
            ##### EXTRACT LABELS AND FEATURES #####
            """
            ### Add labels and extract the feature
            labels_and_features = pd.DataFrame([])
            labels_and_features[number_of_stages_key] = feature_data[number_of_stages_key + '- linear']
            labels_and_features['Patient key'] = feature_data['Patient key']
            labels_and_features[feature_subset_sfs_names] = feature_data[feature_subset_sfs_names]
            labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it

            ### Extract labels, features and groups
            labels = labels_and_features[number_of_stages_key]
            groups = labels_and_features['Patient key']
            features = labels_and_features.iloc[:, 2:]

            start_time = time.time()
            if classifier == 'DT':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(criterion="gini",
                                             max_depth=optimal_hyperparameters.get('sfs__estimator__max_depth'))

            elif classifier == 'SVM':
                from sklearn.svm import SVC
                clf = SVC(kernel='rbf',
                          C=optimal_hyperparameters.get('sfs__estimator__C'),
                          gamma=optimal_hyperparameters.get('sfs__estimator__gamma'),
                          probability=True)

            elif classifier == 'xgboost':
                from xgboost import XGBClassifier
                clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6, gamma=0,
                                    use_label_encoder=False,
                                    learning_rate=optimal_hyperparameters.get('sfs__estimator__learning_rate'),
                                    min_child_weight=optimal_hyperparameters.get('sfs__estimator__min_child_weight'),
                                    colsample_bytree=optimal_hyperparameters.get('sfs__estimator__colsample_bytree'))

            """
            ##### TRAIN MODEL + OBTAIN TRAIN SCORES #####
            """
            clf.fit(features, labels)
            if channel_method != 'all_channels':
                model_file_name = '%s_Model_%s_%s.sav' % (classifier, channel_method, number_of_stages_key)
                pickle.dump(clf, open(os.path.join(model_file_path, model_file_name), 'wb'))

            labels_pred = clf.predict(features)
            labels_pred_prob = clf.predict_proba(features)
            train_accuracy = accuracy_score(labels, labels_pred)
            train_cohens_kappa = cohen_kappa_score(labels, labels_pred)
            train_auc = roc_auc_score(labels, labels_pred_prob, multi_class='ovr')

            """
            ##### CROSSVALIDATION SCORES #####
            """
            auc_cv_score = cross_val_score(clf, features, labels, groups=groups,
                                           cv=5, scoring='roc_auc_ovr', verbose=10, n_jobs=-1)
            acc_cv_score = cross_val_score(clf, features, labels, groups=groups,
                                           cv=5, scoring='accuracy', verbose=10, n_jobs=-1)
            cohens_kappa_cv_score = cross_val_score(clf, features, labels, groups=groups,
                                                    cv=5, scoring=make_scorer(cohen_kappa_score), verbose=10, n_jobs=-1)

            print("%s: %s completed: %s hours" % (classifier, eeg_channel, (time.time() - start_time_channel) / 3600))

            """" 
            ##### 6: SAVE TRAINING RESULTS #####
            """

            optimal_hyperparameters_all.append(optimal_hyperparameters)
            feature_subset_sfs_names_all.append(feature_subset_sfs_names)
            train_auc_posterior_all.append(train_auc)
            train_accuracy_all.append(train_accuracy)
            train_cohens_kappa_all.append(train_cohens_kappa)
            cv_auc_posterior_mean_all.append(np.mean(auc_cv_score))
            cv_auc_posterior_std_all.append(np.std(auc_cv_score))
            cv_accuracy_mean_all.append(np.mean(acc_cv_score))
            cv_accuracy_std_all.append(np.std(acc_cv_score))
            cv_cohens_kappa_mean_all.append(np.mean(cohens_kappa_cv_score))
            cv_cohens_kappa_std_all.append(np.std(cohens_kappa_cv_score))

            i = i + 1

            print("%s: %s of %s channels completed (%s): %s hours"
                  % (classifier, count, len(hyperparametertuning_file['EEG Channel']), eeg_channel, (time.time() - start_time) / 3600))

        # Write to pandas DataFrame
        dfModelResults = pd.DataFrame({'EEG channel': eeg_channel_all,
                                       'Hyperparameters': optimal_hyperparameters_all,
                                       'Feature subset (SFS)': feature_subset_sfs_names_all,
                                       'Train AUC': train_auc_posterior_all,
                                       'Train accuracy': train_accuracy_all,
                                       'Train Cohens kappa': train_cohens_kappa_all,
                                       'CV AUC mean': cv_auc_posterior_mean_all,
                                       'CV AUC std': cv_auc_posterior_std_all,
                                       'CV Accuracy mean': cv_accuracy_mean_all,
                                       'CV Accuracy std': cv_accuracy_std_all,
                                       'CV Cohens kappa mean': cv_cohens_kappa_mean_all,
                                       'CV Cohens kappa std': cv_cohens_kappa_std_all
                                       })

        model_results_file_name = classifier + '_Model_Performance_' + channel_method + '_%ssamples_' % sample_number + number_of_stages_key + '.xlsx'
            #time.strftime("%Y%m%d_%H%M%S") + classifier + '_Model_Performance_AllChannels_' + '.xlsx'
        writer = pd.ExcelWriter(os.path.join(results_data_file_path, model_results_file_name))
        dfModelResults.to_excel(writer)
        writer.save()

        print('Training & cross-validation', classifier, 'completed: %s hours' % (
                (time.time() - start_time_classifier) / 3600))
