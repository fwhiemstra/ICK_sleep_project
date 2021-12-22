"""
This functions evaluates the training performance of index-based measures per individual patient from one EEG channel
June 2021, Floor Hiemstra
"""

import os
import numpy as np
import pandas as pd
from Functions import load_and_prepare_raw_feature_data_single, \
    compute_feature_posterior_prob, auc_feature_calculation_posterior, find_optimal_index_thresholds
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
import time

"""
##### SET DATA VARIABLES #####
"""
# NOTE: This function only works for features that are positively correlated with
# the sleep stages, thus increase during Wake and decrease during SWS.

### Classification settings
number_of_stages = 4
number_of_stages_key = 'Four_state_labels'
feature_name = 'Gamma/delta ratio'
feature_name_file = 'GammaDeltaRatio'
eeg_channel = 'EEG F3-A2:'
plot_roc = 0

### Result file settings
results_data_file_path = r'E:\ICK slaap project\7_IndexBasedClassification'
results_data_file_name = 'TrainScores_%s_%s_Individuals_%s.xlsx' % (feature_name_file, eeg_channel.replace(':', ''), number_of_stages_key)

### Data settings
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
remove_artifacts = 1

"""
##### LOAD FEATURE DATA #####
"""
# List feature data file names
feature_data_file_names = os.listdir(feature_data_file_path)

if number_of_stages == 4:
    patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
    patientData = patientData[patientData['Age category'] != '0-2 months']
    patientData = patientData[patientData['Age category'] != '2-6 months']
    feature_data_file_keys = patientData['Key']
    feature_data_file_names = [key + '_FeatureData.xlsx' for key in feature_data_file_keys]

#feature_data_file_names = feature_data_file_names[0:4]

### Create output array
keys_all = []
age_categories_all = []
auc_posterior_all = []
auc_fvalue_all = []
acc_max_acc_all = []
cohens_max_acc_kappa_all = []
acc_max_tpr_all = []
cohens_max_tpr_kappa_all = []
wake_threshold_acc_all = []
wake_threshold_tpr_all = []
sws_threshold_acc_all = []
sws_threshold_tpr_all = []
nsws_threshold_acc_all = []
nsws_threshold_tpr_all = []
dfContingencyTables = pd.DataFrame([])
tpr_all = []
fpr_all = []

for feature_data_file_name in feature_data_file_names:
    if feature_data_file_name.endswith('.xlsx'):
        feature_data = load_and_prepare_raw_feature_data_single(feature_data_file_path, feature_data_file_name,
                                      remove_artifacts)
        feature_data = feature_data.sample(frac=1) # Shuffle data

        """
        ##### INDEX PERFORMANCE PER CHANNEL #####
        """

        start_time = time.time()
        ### Add labels and extract the feature
        labels_and_features = pd.DataFrame([])
        labels_and_features['Sleep stage labels (cat)'] = feature_data[number_of_stages_key]
        labels_and_features['Patient key'] = feature_data['Patient key']
        labels_and_features['Age category'] = feature_data['Age category']

        labels_and_features[eeg_channel + ' ' + feature_name] = feature_data[eeg_channel + ' ' + feature_name]
        #feature1 = feature_data[eeg_channel + ' ' + 'Abs Gamma power']
        #feature2 = feature_data[eeg_channel + ' ' + 'Abs Beta power']
        #feature3 = feature_data[eeg_channel + ' ' + 'Abs Delta power']
        #feature4 = feature_data[eeg_channel + ' ' + 'Abs Theta power']
        #labels_and_features[eeg_channel + ' ' + feature_name] = (feature1 + feature2)/(feature3 + feature4)

        labels_and_features = labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

        index_score = labels_and_features[eeg_channel + ' ' + feature_name]
        sleep_stage_labels = labels_and_features['Sleep stage labels (cat)']

        """
        ##### DETERMINE OPTIMAL THRESHOLDS ON TRAINING SET ##### 
        """
        optimal_thresholds_max_acc, optimal_thresholds_max_tpr, AUCs_fvalue = find_optimal_index_thresholds(
            index_score, sleep_stage_labels, number_of_stages, plot_roc=plot_roc)

        """" 
        ##### EVALUATE THRESHOLDS ON TEST SET #####
        """
        # First with thresholds with accuracy maximized
        optimal_thresholds = optimal_thresholds_max_acc
        labels_pred = []
        if number_of_stages == 3:
            for i in range(len(index_score)):
                if index_score.iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif index_score.iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')
        elif number_of_stages == 4:
            for i in range(len(index_score)):
                if index_score.iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif optimal_thresholds[0]["Wake"] < index_score.iloc[i] < optimal_thresholds[2][
                    "NSWS"]:
                    labels_pred.append('REM')
                elif index_score.iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')

        acc_max_acc = accuracy_score(sleep_stage_labels, labels_pred)
        cohens_kappa_max_acc = cohen_kappa_score(sleep_stage_labels, labels_pred)

        # Next with thresholds with TPR maximized
        optimal_thresholds = optimal_thresholds_max_tpr
        labels_pred = []
        if number_of_stages == 3:
            for i in range(len(index_score)):
                if index_score.iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif index_score.iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')
        elif number_of_stages == 4:
            for i in range(len(index_score)):
                if index_score.iloc[i] > optimal_thresholds[0]["Wake"]:
                    labels_pred.append('Wake')
                elif optimal_thresholds[0]["Wake"] > index_score.iloc[i] > optimal_thresholds[2][
                    "NSWS"]:
                    labels_pred.append('REM')
                elif index_score.iloc[i] < optimal_thresholds[1]["SWS"]:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')

        acc_max_tpr = accuracy_score(sleep_stage_labels, labels_pred)
        cohens_kappa_max_tpr = cohen_kappa_score(sleep_stage_labels, labels_pred)

        """
        ##### CALCULATE AUCs WITH POSTERIOR METHOD #####
        """
        np.seterr(divide='ignore', invalid='ignore') # To suppress RunTimeWarning
        all_posteriors, x_kde, all_priors = compute_feature_posterior_prob(index_score,
                                                                           sleep_stage_labels,
                                                                           number_of_stages)
        auc_posterior, tpr, fpr = auc_feature_calculation_posterior(index_score,
                                                                    sleep_stage_labels, all_posteriors,
                                                                    x_kde, number_of_stages,
                                                                    plot_roc=plot_roc)
        np.seterr(divide='warn', invalid='warn')

        """
        ##### CONTINGENCY TABLES 
        """

        df = pd.DataFrame({'True sleep stage': sleep_stage_labels,
                                     'Predicted sleep stage (IDOS)': labels_pred})
        contingency_table = pd.crosstab(df['True sleep stage'], df['Predicted sleep stage (IDOS)'])
        contingency_table['Patient key'] = feature_data['Patient key'].iloc[0]

        """
        ##### ADD RESULTS PER CHANNEL TO ARRAYS #####
        """

        keys_all.append(feature_data['Patient key'].iloc[0])
        age_categories_all.append(feature_data['Age category'].iloc[0])
        auc_posterior_all.append(auc_posterior)
        if number_of_stages == 3:
            auc_fvalue_all.append(np.mean([float(AUCs_fvalue[0].get('Wake')), float(AUCs_fvalue[1].get('SWS'))]))
            wake_threshold_acc_all.append(optimal_thresholds_max_acc[0].get('Wake'))
            wake_threshold_tpr_all.append(optimal_thresholds_max_tpr[0].get('Wake'))
            sws_threshold_acc_all.append(optimal_thresholds_max_acc[1].get('SWS'))
            sws_threshold_tpr_all.append(optimal_thresholds_max_tpr[1].get('SWS'))
        elif number_of_stages == 4:
            auc_fvalue_all.append(np.mean([float(AUCs_fvalue[0].get('Wake')), float(AUCs_fvalue[2].get('NSWS')),
                                                float(AUCs_fvalue[1].get('SWS'))]))
            wake_threshold_acc_all.append(optimal_thresholds_max_acc[0].get('Wake'))
            wake_threshold_tpr_all.append(optimal_thresholds_max_tpr[0].get('Wake'))
            nsws_threshold_acc_all.append(optimal_thresholds_max_acc[2].get('NSWS'))
            nsws_threshold_tpr_all.append(optimal_thresholds_max_tpr[2].get('NSWS'))
            sws_threshold_acc_all.append(optimal_thresholds_max_acc[1].get('SWS'))
            sws_threshold_tpr_all.append(optimal_thresholds_max_tpr[1].get('SWS'))

        acc_max_acc_all.append(acc_max_acc)
        cohens_max_acc_kappa_all.append(cohens_kappa_max_acc)

        acc_max_tpr_all.append(acc_max_tpr)
        cohens_max_tpr_kappa_all.append(cohens_kappa_max_tpr)

        tpr_all.append([tpr])
        fpr_all.append([fpr])

        dfContingencyTables = dfContingencyTables.append(contingency_table)

        print(feature_data_file_name, 'completed.')

"""
##### STORE RESULT ARRAYS IN DATAFRAME
"""
dfIndexResults = pd.DataFrame({'Patient key': keys_all,
                            'Age category': age_categories_all,
                           'AUC posterior': auc_posterior_all,
                           'AUC fvalue': auc_fvalue_all,
                           'Cohens kappa - max accuracy': cohens_max_acc_kappa_all,
                           'Accuracy - max accuracy': acc_max_acc_all,
                           'Cohens kappa - max tpr': cohens_max_tpr_kappa_all,
                           'Accuracy - max tpr': acc_max_tpr_all,
                           'TPR': tpr_all,
                           'FPR': fpr_all
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
dfContingencyTables.to_excel(writer, sheet_name='Contingency Tables')
writer.save()

print('Index score training completed in %s hours' % ((time.time() - start_time) / 3600))


