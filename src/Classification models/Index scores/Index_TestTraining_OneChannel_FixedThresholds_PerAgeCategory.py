"""
This functions evaluates the training performance of index-based measures per individual patient from one EEG channel
June 2021, Floor Hiemstra
"""

import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\src')
from Functions import load_and_prepare_raw_feature_data_multiple, \
    compute_feature_posterior_prob, auc_feature_calculation_posterior, find_optimal_index_thresholds
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time

"""
##### SET DATA VARIABLES #####
"""
# NOTE: This function only works for features that are positively correlated with
# the sleep stages, thus increase during Wake and decrease during SWS.

### Classification settings
load_data = 1  # 0 if data is already loaded
number_of_stages = 3
number_of_stages_key = 'Three_state_labels'
feature_name = 'Gamma/delta ratio'
feature_name_file = 'GammaTheta+DeltaRatio'
eeg_channel = 'EEG F3-C3:'
plot_roc = 0

### Result file settings
results_data_file_path = r'E:\ICK slaap project\7_IndexBasedClassification\7_4_Performance per age group'
results_data_file_name = 'TrainScores_%s_%s_FixedThreshold_PerAgeCategory_%s.xlsx' % (feature_name_file, eeg_channel.replace(':', ''), number_of_stages_key)

### Data settings
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
remove_artifacts = 1

include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years']
if number_of_stages == 4:
    include_age_categories = ['6-12 months', '1-3 years', '3-5 years', '5-9 years',
                              '9-13 years', '13-18 years']

### Create output array
age_categories_all = []
acc_all = []
cohens_kappa_all = []
acc_max_tpr_all = []
dfContingencyTables = pd.DataFrame([])

if feature_name_file == 'GammaDeltaRatio':
    if number_of_stages == 3:
        wake_threshold = 0.031400108
        sws_threshold = 0.003966432
    elif number_of_stages == 4:
        wake_threshold =  0.027481012 # 0.02835651
        sws_threshold = 0.003966432
        nsws_threshold =  0.024273302
elif feature_name_file == 'GammaTheta+DeltaRatio':
    if number_of_stages == 3:
        wake_threshold = 0.022055037
        sws_threshold = 0.003190831
    elif number_of_stages == 4:
        wake_threshold = 0.022055037
        sws_threshold = 0.003190831
        nsws_threshold = 0.019693002

for age_category in include_age_categories:
    if load_data == 1:
        feature_data = load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                                                  feature_data_file_path, [age_category],
                                                                  remove_artifacts)
        feature_data = feature_data.sample(frac=1) # Shuffle data

        """
        ##### INDEX PERFORMANCE PER CHANNEL #####
        """

        start_time = time.time()

        ### Add labels and extract the feature
        data = feature_data
        labels_and_features = pd.DataFrame([])
        labels_and_features['Sleep stage labels (cat)'] = data[number_of_stages_key]
        labels_and_features['Patient key'] = data['Patient key']
        labels_and_features['Age category'] = data['Age category']

        if feature_name_file == 'GammaDeltaRatio':
            labels_and_features[eeg_channel + ' ' + feature_name] = data[eeg_channel + ' ' + feature_name]
        elif feature_name_file == 'GammaTheta+DeltaRatio':
            feature1 = data[eeg_channel + ' ' + 'Abs Gamma power']
            feature2 = data[eeg_channel + ' ' + 'Abs Theta power']
            feature3 = data[eeg_channel + ' ' + 'Abs Delta power']
            labels_and_features[eeg_channel + ' ' + feature_name] = (feature1) / (feature2 + feature3)

        labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it

        index_score = labels_and_features[eeg_channel + ' ' + feature_name]
        sleep_stage_labels = labels_and_features['Sleep stage labels (cat)']

        """" 
        ##### EVALUATE THRESHOLDS ON TEST SET #####
        """
        labels_pred = []
        if number_of_stages == 3:
            for i in range(len(index_score)):
                if index_score.iloc[i] > wake_threshold:
                    labels_pred.append('Wake')
                elif index_score.iloc[i] < sws_threshold:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')
        elif number_of_stages == 4:
            for i in range(len(index_score)):
                if index_score.iloc[i] > wake_threshold:
                    labels_pred.append('Wake')
                elif (index_score.iloc[i] > nsws_threshold) & (index_score.iloc[i] < wake_threshold):
                    labels_pred.append('REM')
                elif index_score.iloc[i] < sws_threshold:
                    labels_pred.append('SWS')
                else:
                    labels_pred.append('NSWS')
                # elif (index_score.iloc[i] > sws_threshold) & (index_score.iloc[i] < nsws_threshold):
                # labels_pred.append('NSWS')

        accuracy = accuracy_score(sleep_stage_labels, labels_pred)
        cohens_kappa = cohen_kappa_score(sleep_stage_labels, labels_pred)

        """
        ##### CONTINGENCY TABLES 
        """
        df = pd.DataFrame({'True sleep stage': sleep_stage_labels,
                                     'Predicted sleep stage (IDOS)': labels_pred})
        contingency_table = pd.crosstab(df['True sleep stage'], df['Predicted sleep stage (IDOS)'])
        contingency_table['Age category'] = feature_data['Age category'].iloc[0]

        """
        ##### ADD RESULTS PER CHANNEL TO ARRAYS #####
        """
        age_categories_all.append(feature_data['Age category'].iloc[0])
        acc_all.append(accuracy)
        cohens_kappa_all.append(cohens_kappa)

        dfContingencyTables = dfContingencyTables.append(contingency_table)

        print(age_category, 'completed.')

"""
##### STORE RESULT ARRAYS IN DATAFRAME
"""
dfIndexResults = pd.DataFrame({'Age category': age_categories_all,
                           'Accuracy': acc_all,
                           'Cohens kappa': cohens_kappa_all,
                           })

"""
##### WRITE RESULTS TO EXCEL FILE #####
"""
writer = pd.ExcelWriter(os.path.join(results_data_file_path, results_data_file_name))
dfIndexResults.to_excel(writer, sheet_name='Results')
dfContingencyTables.to_excel(writer, sheet_name='Contingency Tables')
writer.save()

print('Index score training completed in %s hours' % ((time.time() - start_time) / 3600))


