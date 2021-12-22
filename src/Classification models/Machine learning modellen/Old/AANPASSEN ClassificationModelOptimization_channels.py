"""
This file is used to compare various eeg channels within one classifier.
July 2021, Floor Hiemstra
"""

# %% Import packages
import os
import numpy as np
import pandas as pd
from src.Functions import add_new_state_labels
from mrmr import mrmr_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector
import time

#%% Set variables
classifier = 'DT'
classifiers = ['xgboost']

# File paths and names
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_2_PSG_FeatureData_SCALED'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
model_result_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment'

load_data = 1 # 0 is data is already loaded
data = 'sampled'

# Set data parameters - if data = 'all'
include_age_categories = ['1-3 years']
number_of_stages = 3  # 2 = [Wake, Sleep], 3 = [Wake, NSWS, SWS], 4 = [Wake, REM, NSWS, SWS], 5 = [Wake, REM, N1, N2, N3]
remove_artifacts = 1

# Set data parameter - if data = 'sampled'
sampled_data_file_name = 'testFeatureData_1000_all_ages.xlsx'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'

"""
##### 1.0: LOAD FEATURE DATA #####
"""
if load_data == 1:
    start_time = time.time()
    if data == 'all':
        # List feature data file names
        patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
        keys = []
        for age_category in include_age_categories:
            keys.extend(patientData[patientData['Age category'] == age_category]['Key'])
        feature_data_file_names = [key + '_FeatureData_scaled.xlsx' for key in keys]
        count = 0

        ### Read feature data files in list and store as one dataframe
        allFeatureData = []
        for feature_data_file_name in feature_data_file_names:
            count = count + 1
            if feature_data_file_name in os.listdir(feature_data_file_path):
                featureData_raw = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

                # Add new sleep stage labels
                featureData_new_labels, number_of_stages_key = add_new_state_labels(featureData_raw, number_of_stages)
                featureData = featureData_new_labels

                # Remove artefacts
                if remove_artifacts == 1:
                    featureData_clean = featureData_new_labels[featureData['Artifact'] == 0]
                    featureData = featureData_clean

                # Create dataframes
                if feature_data_file_name == feature_data_file_names[0]:
                    allFeatureData = featureData
                else:
                    allFeatureData = allFeatureData.append(featureData)

                print("%s of %s files completed (%s)" % (count, len(feature_data_file_names),feature_data_file_name))

    elif data == 'sampled':
        """
        ##### 1.1: LOAD FEATURE DATA (SAMPLE SET) #####
        """
        allFeatureData = pd.read_excel(os.path.join(sampled_data_file_path,
                                         sampled_data_file_name))
        if number_of_stages == 3:
            number_of_stages_key = 'Three_state_labels'
        elif number_of_stages == 4:
            number_of_stages_key = 'Four_state_labels'

    print("Loading feature data completed: %s hours" % ((time.time() - start_time)/3600))

allFeatureData = pd.concat([allFeatureData, pd.get_dummies(allFeatureData['Age category'],
                                                           prefix='age_group_')], axis=1)  # Age category
age_features_list = list(pd.get_dummies(allFeatureData['Age category'],
                                        prefix='age_group_').columns)
eog_features_list = [col for col in allFeatureData if col.startswith('EOG ROC-LOC:')]
emg_features_list = [col for col in allFeatureData if col.startswith('EMG EMG Chin:')]

for classifier in classifiers:
    """
    ##### 2: EXTRACT LABELS AND FEATURES #####
    """
    ### Extract labels from
    labels_cat = allFeatureData[number_of_stages_key]
    labels_lin = allFeatureData[number_of_stages_key + '- linear']
    #eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                    #'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                    #'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
    eeg_channels = ['EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                    'EEG C3-A2:', 'EEG O1-A2:']

    df_eeg_channel = []
    df_mrmr_selected_features = []
    df_feature_subset_sfs_names = []
    df_optimal_hyperparameters = []
    df_auc_cv_mean = []
    df_auc_cv_std = []
    df_acc_cv_mean = []
    df_acc_cv_std = []
    df_cohens_kappa_cv_mean = []
    df_cohens_kappa_cv_std = []

    count = 0 # To track progress

    # Loop over all EEG channels
    for eeg_channel in eeg_channels:
        count = count + 1
        ### Extract features
        # EOG, EMG + age features
        features = pd.DataFrame([])
        features[eog_features_list] = allFeatureData[eog_features_list]  # EOG features
        features[emg_features_list] = allFeatureData[emg_features_list]  # EMG features

        # EEG features
        eeg_features_list = [col for col in allFeatureData if col.startswith(eeg_channel)]
        features[eeg_features_list] = allFeatureData[eeg_features_list]

        """
        ##### 3: FIRST FEATURE SELECTION #####
        """
        start_time = time.time()
        n_features_to_select = int(0.3 * features.shape[1])

        ### Minimum redundancy maximum relevance
        mrmr_selected_features = mrmr_classif(features, labels_lin, K=n_features_to_select)
        feature_subset_mrmr = features[mrmr_selected_features]
        feature_subset_mrmr[age_features_list] = allFeatureData[age_features_list]
        print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
        print(mrmr_selected_features)

        """
        ##### 4: SEQUENTIAL FORWARD FEATURE SELECTION + HYPERPARAMETER TUNING #####
        """
        steps_feature_selection = 3

        start_time = time.time()

        # Classifier variables
        if classifier == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(criterion='gini')
            param_grid = {'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection),
                          'sfs__estimator__max_depth': [2, 3, 4, 5, 6, 7, 8]}
            labels = labels_cat
            n_iter = 10

        elif classifier == 'LR':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(penalty='l2')
            param_grid = {'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection),
                          'sfs__estimator__C': np.logspace(-3, 3, 7)}
            labels = labels_lin
            n_iter = 10

        elif classifier == 'SVM':
            from sklearn.svm import SVC
            clf = SVC(gamma='scale')
            param_grid = {'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection),
                          'sfs__estimator__kernel': ['rbf', 'linear'],
                          'sfs__estimator__C': np.logspace(-3, 3, 7)}
            labels = labels_cat
            n_iter = 10

        elif classifier == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier()
            param_grid = {'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection),
                          'sfs__estimator__n_neighbors': [10, 20, 30, 40, 50, 60, 70]}
            labels = labels_cat
            n_iter = 10

        elif classifier == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
            param_grid = {'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection)}
            labels = labels_cat
            n_iter = 10

        elif classifier == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(max_depth=5)
            param_grid = {'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection),
                          'sfs__estimator__n_estimators': [50, 100, 150],
                          'sfs__estimator__bootstrap': [True, False]}
            labels = labels_cat
            n_iter = 10

        elif classifier == "xgboost":
            from xgboost import XGBClassifier
            clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6, gamma=0)
            param_grid ={'sfs__k_features': range(1, n_features_to_select + 1, steps_feature_selection),
                         "sfs__estimator__learning_rate" : [0.05, 0.10, 0.15, 0.20] ,
                         "sfs__estimator__min_child_weight": [ 1, 3, 5, 7],
                         "sfs__estimator__colsample_bytree" : [0.5 , 0.75, 1]
                        }
            labels = labels_cat
            n_iter = 10

        ### Start randomized grid search for hyperparameter tuning
        sfs = SequentialFeatureSelector(estimator=clf, forward=True, scoring='accuracy', cv=3)
        pipe = Pipeline([('sfs', sfs),
                         ('clf', clf)])
        grid_search = RandomizedSearchCV(estimator=pipe, n_iter=n_iter, param_distributions=param_grid,
                                         scoring='accuracy', cv=3)
        grid_search.fit(feature_subset_mrmr, labels)

        ### Extract best hyperparameters
        idx_feature_subset_2 = grid_search.best_estimator_.steps[0][1].k_feature_idx_
        feature_subset_sfs = feature_subset_mrmr.iloc[:, np.array(idx_feature_subset_2)]
        feature_subset_sfs_names = feature_subset_sfs.columns
        optimal_hyperparameters = grid_search.best_params_

        print("Hyperparameter tuning %s completed: %s hours" % (eeg_channel, (time.time() - start_time) / 3600))
        print(optimal_hyperparameters,
              "\n SFS selected features: ", feature_subset_sfs_names)

        """
        ##### 5: INTERNAL CROSS-VALIDATION #####
        """
        start_time = time.time()
        if classifier == 'DT':
            clf = DecisionTreeClassifier(criterion="gini",
                                         max_depth=optimal_hyperparameters.get('sfs__estimator__max_depth'))

        elif classifier == 'LR':
            clf = LogisticRegression(penalty='l2',
                                     C=optimal_hyperparameters.get('sfs__estimator__C'))

        elif classifier == 'SVM':
            clf = SVC(kernel=optimal_hyperparameters.get('sfs__estimator__kernel'),
                      C=optimal_hyperparameters.get('sfs__estimator__C'),
                      gamma='scale',
                      probability=True)

        elif classifier == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=optimal_hyperparameters.get('sfs__estimator__n_neighbors'))

        elif classifier == 'LDA':
            clf = LinearDiscriminantAnalysis()

        elif classifier == 'RF':
            clf = RandomForestClassifier(max_depth=5, n_estimators=optimal_hyperparameters.get('sfs__estimator__n_estimators'),
                                         bootstrap=optimal_hyperparameters.get('sfs__estimator__bootstrap'))

        elif classifier == 'xgboost':
            clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6, gamma=0,
                                learning_rate=optimal_hyperparameters.get('sfs__estimator__learning_rate'),
                                min_child_weight=optimal_hyperparameters.get('sfs__estimator__min_child_weight'),
                                colsample_bytree=optimal_hyperparameters.get('sfs__estimator__colsample_bytree'))

        auc_cv_score = cross_val_score(clf, feature_subset_sfs, labels, groups=allFeatureData['Patient key'],
                                   cv=10, scoring='roc_auc_ovo')
        acc_cv_score = cross_val_score(clf, feature_subset_sfs, labels, groups=allFeatureData['Patient key'],
                                   cv=10, scoring='accuracy')
        cohens_kappa_cv_score = cross_val_score(clf, feature_subset_sfs, labels, groups=allFeatureData['Patient key'],
                                   cv=10, scoring=make_scorer(cohen_kappa_score))

        print("Cross-validation completed: {:.2f} hours".format((time.time() - start_time) / 3600),
              "\n AUC mean: {:.2f}, std: {:.2f}".format(np.mean(auc_cv_score), np.std(auc_cv_score)),
              "\n Accuracy mean: {:.2f}, std: {:.2f}".format(np.mean(acc_cv_score), np.std(acc_cv_score)),
              "\n Cohen's kappa mean: {:.2f}, std: {:.2f}".format(np.mean(cohens_kappa_cv_score),
                                                                  np.std(cohens_kappa_cv_score)),
              )

        """" 
        ##### 6: SAVE TRAINING RESULTS #####
        """
        df_eeg_channel.append(eeg_channel)
        df_mrmr_selected_features.append(str(mrmr_selected_features))
        df_feature_subset_sfs_names.append(str(feature_subset_sfs_names))
        df_optimal_hyperparameters.append(str(optimal_hyperparameters))
        df_auc_cv_mean.append(np.mean(auc_cv_score))
        df_auc_cv_std.append(np.std(auc_cv_score))
        df_acc_cv_mean.append(np.mean(acc_cv_score))
        df_acc_cv_std.append(np.std(acc_cv_score))
        df_cohens_kappa_cv_mean.append(np.mean(cohens_kappa_cv_score))
        df_cohens_kappa_cv_std.append(np.std(cohens_kappa_cv_score))

        print("%s of %s channels completed (%s)" % (count, len(eeg_channels), eeg_channel))

    # Write to pandas DataFrame
    dfModelInformation = pd.DataFrame({'Data': include_age_categories,
                                   'Artifact': remove_artifacts,
                                   'Number of sleep stages': number_of_stages_key,
                                   'Model': classifier,
                                   'Parameter grid': str(param_grid),
                                   'Feature data': feature_data_file_path})

    dfModelResults = pd.DataFrame({'EEG Channel': df_eeg_channel,
                                   'mrmr Feature selection': df_mrmr_selected_features,
                                   'sfs Feature selection': df_feature_subset_sfs_names,
                                   'Hyperparameters': df_optimal_hyperparameters,
                                   'AUC CV score (mean)': df_auc_cv_mean,
                                   'AUC CV score (std)': df_auc_cv_std,
                                   'Accuracy CV score (mean)': df_acc_cv_mean,
                                   'Accuracy CV score (std)': df_acc_cv_std,
                                   'Cohens kappa CV score (mean)': df_cohens_kappa_cv_mean,
                                   'Cohens kappa CV score (std)': df_cohens_kappa_cv_std})

    model_results_file_name = time.strftime("%Y%m%d_%H%M%S") + classifier + '_' + 'Model_results_' + '.xlsx'
    writer = pd.ExcelWriter(os.path.join(model_result_file_path, model_results_file_name))

    dfModelInformation.to_excel(writer, sheet_name='Model information')
    dfModelResults.to_excel(writer, sheet_name='Model Results')
    writer.save()

    print('Feature selection + hyperparameter tuning', classifier, 'completed: %s hours' % (
            (time.time() - start_time) / 3600))