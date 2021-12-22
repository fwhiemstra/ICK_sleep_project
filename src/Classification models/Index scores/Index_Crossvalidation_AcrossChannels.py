"""
This functions evaluates the cross-validaiton performance of index-based measures per group of patients accross EEG channels
June 2021, Floor Hiemstra
"""

import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\src')
from Functions import load_and_prepare_raw_feature_data_multiple, \
    compute_feature_posterior_prob, auc_feature_calculation_posterior, find_optimal_index_thresholds
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
from datetime import datetime

"""
##### SET DATA VARIABLES #####
"""
# NOTE: This function only works for features that are positively correlated with
# the sleep stages, thus increase during Wake and decrease during SWS.

### Data settings
load_data = 1 # 0 if data is already loaded
data_type = 'new'  # 'new' or 'sampled'

### Classification settings
eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
number_of_stages = 4
number_of_stages_key = 'Four_state_labels'
feature_name = 'Gamma/delta ratio'
feature_name_file = 'GammaDeltaRatio'
plot_roc = 0

### Result file settings
results_data_file_path = r'E:\ICK slaap project\7_IndexBasedClassification'
results_data_file_name = 'CVScores_%s_%s_ALL_6months-18years_%s.xlsx' % (eeg_channels[0].replace(':', ''), feature_name_file, number_of_stages_key)

### Variables, if data == 'new'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years']

if number_of_stages == 4:
    include_age_categories = ['6-12 months', '1-3 years', '3-5 years', '5-9 years',
                              '9-13 years', '13-18 years']
remove_artifacts = 1

### Variables, if data == 'sampled'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
sampled_data_file_name = '1000sampled_feature_data_0-18years_nonscaled.xlsx'

"""
##### 1. LOAD FEATURE DATA #####
"""
if load_data == 1 and data_type == 'new':
    feature_data = load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                                             feature_data_file_path, include_age_categories,
                                                             remove_artifacts)
elif load_data == 1 and data_type == 'sampled':
    feature_data = pd.read_excel(os.path.join(sampled_data_file_path, sampled_data_file_name))
feature_data = feature_data.sample(frac=1) # Shuffle data

"""
##### 2: EXTRACT LABELS AND FEATURES #####
"""
### Create output array
eeg_channels_all = []
auc_posterior_all_mean = []
auc_posterior_all_std = []
auc_fvalue_all_mean = []
auc_fvalue_all_std = []
acc_max_acc_all_mean = []
acc_max_acc_all_std = []
cohens_max_acc_kappa_all_mean = []
cohens_max_acc_kappa_all_std = []
acc_max_tpr_all_mean = []
acc_max_tpr_all_std = []
cohens_max_tpr_kappa_all_mean = []
cohens_max_tpr_kappa_all_std = []
wake_threshold_acc_all = []
wake_threshold_tpr_all = []
sws_threshold_acc_all = []
sws_threshold_tpr_all = []
nsws_threshold_acc_all = []
nsws_threshold_tpr_all = []

start_time = time.time()
for eeg_channel in eeg_channels:
    print(datetime.now(), ': %s started' % eeg_channel)

    start_time_channel = time.time()

    ### Add labels and extract the feature
    labels_and_features = pd.DataFrame([])
    labels_and_features['Sleep stage labels (cat)'] = feature_data[number_of_stages_key]
    labels_and_features['Patient key'] = feature_data['Patient key']
    labels_and_features['Age category'] = feature_data['Age category']

    if feature_name_file == 'GammaDeltaRatio':
        labels_and_features[eeg_channel + ' ' + feature_name] = feature_data[eeg_channel + ' ' + feature_name]
    elif feature_name_file == 'GammaTheta+DeltaRatio':
        feature1 = feature_data[eeg_channel + ' ' + 'Abs Gamma power']
        feature2 = feature_data[eeg_channel + ' ' + 'Abs Delta power']
        feature3 = feature_data[eeg_channel + ' ' + 'Abs Theta power']
        labels_and_features[eeg_channel + ' ' + feature_name] = (feature1) / (feature2 + feature3)

    labels_and_features = labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

    index_score = labels_and_features[eeg_channel + ' ' + feature_name]
    sleep_stage_labels = labels_and_features['Sleep stage labels (cat)']
    groups = labels_and_features['Patient key']

    auc_posterior_cv_score = []
    auc_fvalue_cv_score = []
    acc_max_acc_cv_score = []
    cohens_max_acc_kappa_cv_score = []
    acc_max_tpr_cv_score = []
    cohens_max_tpr_kappa_cv_score = []
    wake_threshold_acc_cv = []
    wake_threshold_tpr_cv = []
    sws_threshold_acc_cv = []
    sws_threshold_tpr_cv = []
    nsws_threshold_acc_cv = []
    nsws_threshold_tpr_cv = []
    count = 0

    ### Start cross-validation
    n_folds = 5
    for train_index, test_index in GroupKFold(n_splits=n_folds).split(index_score,
                                                             sleep_stage_labels,
                                                             groups=groups):
        count = count + 1
        """
        ##### DETERMINE OPTIMAL THRESHOLDS ON TRAINING SET ##### 
        """
        optimal_thresholds_max_acc, optimal_thresholds_max_tpr, AUCs_fvalue = find_optimal_index_thresholds(index_score.iloc[train_index],
                                                                 sleep_stage_labels.iloc[train_index], number_of_stages, plot_roc=plot_roc)

        """" 
        ##### 4: EVALUATE THRESHOLDS ON TEST SET #####
        """
        # First with thresholds with accuracy maximized
        optimal_thresholds = optimal_thresholds_max_acc
        labels_pred = []
        if number_of_stages == 3:
            for i in range(len(index_score.iloc[test_index])):
                if index_score.iloc[test_index].iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif index_score.iloc[test_index].iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')
        elif number_of_stages == 4:
            for i in range(len(index_score)):
                if index_score.iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif (index_score.iloc[i] > optimal_thresholds[2]["NSWS"]) & (
                        index_score.iloc[i] < optimal_thresholds[0]["Wake"]):
                    labels_pred.append('REM')
                elif index_score.iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')

        acc_max_acc = accuracy_score(sleep_stage_labels.iloc[test_index], labels_pred)
        cohens_kappa_max_acc = cohen_kappa_score(sleep_stage_labels.iloc[test_index], labels_pred)

        # Next with thresholds with TPR maximized
        optimal_thresholds = optimal_thresholds_max_tpr
        labels_pred = []
        if number_of_stages == 3:
            for i in range(len(index_score.iloc[test_index])):
                if index_score.iloc[test_index].iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif index_score.iloc[test_index].iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')
        elif number_of_stages == 4:
            for i in range(len(index_score.iloc[test_index])):
                if index_score.iloc[test_index].iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif (index_score.iloc[i] > optimal_thresholds[2]["NSWS"]) & (
                            index_score.iloc[i] < optimal_thresholds[0]["Wake"]):
                    labels_pred.append('REM')
                elif index_score.iloc[test_index].iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')

        acc_max_tpr = accuracy_score(sleep_stage_labels.iloc[test_index], labels_pred)
        cohens_kappa_max_tpr = cohen_kappa_score(sleep_stage_labels.iloc[test_index], labels_pred)

        """
        ##### CALCULATE AUCs WITH POSTERIOR METHOD #####
        """
        np.seterr(divide='ignore', invalid='ignore')  # To suppress RunTimeWarning
        all_posteriors, x_kde, all_priors = compute_feature_posterior_prob(index_score.iloc[train_index],
                        sleep_stage_labels.iloc[train_index], number_of_stages)
        auc_posterior, tpr, fpr = auc_feature_calculation_posterior(index_score.iloc[test_index],
                        sleep_stage_labels.iloc[test_index], all_posteriors, x_kde, number_of_stages,
                                                          plot_roc=plot_roc)
        np.seterr(divide='warn', invalid='warn')

        """
        ##### ADD CV RESULTS TO ARRAY #####
        """
        auc_posterior_cv_score.append(auc_posterior)
        if number_of_stages == 3:
            auc_fvalue_cv_score.append(np.mean([float(AUCs_fvalue[0].get('Wake')), float(AUCs_fvalue[1].get('SWS'))]))
            wake_threshold_acc_cv.append(optimal_thresholds_max_acc[0].get('Wake'))
            wake_threshold_tpr_cv.append(optimal_thresholds_max_tpr[0].get('Wake'))
            sws_threshold_acc_cv.append(optimal_thresholds_max_acc[1].get('SWS'))
            sws_threshold_tpr_cv.append(optimal_thresholds_max_tpr[1].get('SWS'))

        elif number_of_stages == 4:
            auc_fvalue_cv_score.append(np.mean([float(AUCs_fvalue[0].get('Wake')), float(AUCs_fvalue[2].get('NSWS')),
                                                float(AUCs_fvalue[1].get('SWS'))]))
            wake_threshold_acc_cv.append(optimal_thresholds_max_acc[0].get('Wake'))
            wake_threshold_tpr_cv.append(optimal_thresholds_max_tpr[0].get('Wake'))
            nsws_threshold_acc_cv.append(optimal_thresholds_max_acc[2].get('NSWS'))
            nsws_threshold_tpr_cv.append(optimal_thresholds_max_tpr[2].get('NSWS'))
            sws_threshold_acc_cv.append(optimal_thresholds_max_acc[1].get('SWS'))
            sws_threshold_tpr_cv.append(optimal_thresholds_max_tpr[1].get('SWS'))

        acc_max_acc_cv_score.append(acc_max_acc)
        cohens_max_acc_kappa_cv_score.append(cohens_kappa_max_acc)
        acc_max_tpr_cv_score.append(acc_max_tpr)
        cohens_max_tpr_kappa_cv_score.append(cohens_kappa_max_tpr)

        print('%s: Fold %s of %s completed' % (eeg_channel, count, n_folds))

    eeg_channels_all.append(eeg_channel)
    auc_posterior_all_mean.append(np.mean(auc_posterior_cv_score))
    auc_posterior_all_std.append(np.std(auc_posterior_cv_score))
    auc_fvalue_all_mean.append(np.mean(auc_fvalue_cv_score))
    auc_fvalue_all_std.append(np.std(auc_fvalue_cv_score))
    acc_max_acc_all_mean.append(np.mean(acc_max_acc_cv_score))
    acc_max_acc_all_std.append(np.std(acc_max_acc_cv_score))
    cohens_max_acc_kappa_all_mean.append(np.mean(cohens_max_acc_kappa_cv_score))
    cohens_max_acc_kappa_all_std.append(np.std(cohens_max_acc_kappa_cv_score))
    acc_max_tpr_all_mean.append(np.mean(acc_max_tpr_cv_score))
    acc_max_tpr_all_std.append(np.std(acc_max_tpr_cv_score))
    cohens_max_tpr_kappa_all_mean.append(np.mean(cohens_max_tpr_kappa_cv_score))
    cohens_max_tpr_kappa_all_std.append(np.std(cohens_max_tpr_kappa_cv_score))
    wake_threshold_acc_all.append([wake_threshold_acc_cv])
    wake_threshold_tpr_all.append([wake_threshold_tpr_cv])
    sws_threshold_acc_all.append([sws_threshold_acc_cv])
    sws_threshold_tpr_all.append([sws_threshold_tpr_cv])
    if number_of_stages == 4:
        nsws_threshold_acc_all.append([nsws_threshold_acc_cv])
        nsws_threshold_tpr_all.append([sws_threshold_tpr_cv])

    print(datetime.now(), eeg_channel, 'completed in %s hours' % ((time.time() - start_time_channel) / 3600))

"""
##### 6: SAVE DATA SETTINGS, THRESHOLDS, AUC, ACCURACY, COHEN'S KAPPA FOR EACH FOLD 
"""
dfIndexResults = pd.DataFrame({'EEG channel': eeg_channels_all,
                   'AUC posterior - mean': auc_posterior_all_mean,
                   'AUC posterior - std': auc_posterior_all_std,
                   'AUC fvalue - mean': auc_fvalue_all_mean,
                   'AUC fvalue - std': auc_fvalue_all_std,
                   'Cohens kappa - max accuracy - mean': cohens_max_acc_kappa_all_mean,
                   'Cohens kappa - max accuracy - std': cohens_max_acc_kappa_all_std,
                   'Accuracy - max accuracy - mean': acc_max_acc_all_mean,
                   'Accuracy - max accuracy - std': acc_max_acc_all_std,
                   'Cohens kappa - max tpr - mean': cohens_max_tpr_kappa_all_mean,
                   'Cohens kappa - max tpr - std': cohens_max_tpr_kappa_all_std,
                   'Accuracy - max tpr - mean': acc_max_tpr_all_mean,
                   'Accuracy - max tpr - std': acc_max_tpr_all_std
                   })
if number_of_stages == 3:
    dfThresholds = pd.DataFrame({'Threshold Wake - max acc': wake_threshold_acc_all,
                                 'Threshold SWS - max acc': sws_threshold_acc_all,
                                 'Threshold Wake - max tpr': wake_threshold_tpr_all,
                                 'Threshold SWS - max tpr': sws_threshold_tpr_all,
                                 })
elif number_of_stages == 4:
    dfThresholds = pd.DataFrame({'Threshold Wake - max acc': wake_threshold_acc_all,
                                 'Threshold NSWS - max acc': nsws_threshold_acc_all,
                                 'Threshold SWS - max acc': sws_threshold_acc_all,
                                 'Threshold Wake - max tpr': wake_threshold_tpr_all,
                                 'Threshold NSWS - max tpr': nsws_threshold_tpr_all,
                                 'Threshold SWS - max tpr': sws_threshold_tpr_all,
                                 })
dfIndexResults_all = dfIndexResults.join(dfThresholds)

"""
##### WRITE RESULTS TO EXCEL FILE #####
"""
writer = pd.ExcelWriter(os.path.join(results_data_file_path, results_data_file_name))
dfIndexResults_all.to_excel(writer, sheet_name='Results')
writer.save()

print('Cross-validation training completed in %s hours' % ((time.time() - start_time) / 3600))
