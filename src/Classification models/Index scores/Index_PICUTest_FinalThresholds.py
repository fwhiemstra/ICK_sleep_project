"""
This functions evaluates the test performance of index-based measures on the PICU data set
June 2021, Floor Hiemstra
"""

import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\src')
from Functions import load_and_prepare_raw_feature_data_single, \
    load_and_prepare_raw_feature_data_multiple, compute_feature_posterior_prob, \
    auc_feature_calculation_posterior, find_optimal_index_thresholds
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
from datetime import datetime

"""
##### SET DATA VARIABLES #####
"""
# NOTE: This function only works for features that are positively correlated with
# the sleep stages, thus increase during Wake and decrease during SWS.

### Classification settings
load_picu_data = 1  # 0 if data is already loaded
calculate_posteriors = 0 # control PSG posteriors
load_psg_data = 0
number_of_stages = 3
number_of_stages_key = 'Four_state_labels'
feature_name = 'Gamma/delta ratio'
feature_name_file = 'GammaDeltaRatio' # GammaDeltaRatio or GammaTheta+DeltaRatio
eeg_channel = 'EEG F3-C3:'
plot_roc = 0

### Result file settings
results_data_file_path = r'E:\ICK slaap project\7_IndexBasedClassification\7_5_Performance on PICU data'
results_data_file_name = 'PICU_TestScores_%s_%s_FinalThreshold_%s.xlsx' % (feature_name_file, eeg_channel.replace(':', ''), number_of_stages_key)
posteriors_file_path = r'E:\ICK slaap project\7_IndexBasedClassification\7_1_Training & Crossvalidation results'

### Data settings
picu_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
remove_artifacts = 1
picu_files = ['PICU001_FeatureData.xlsx', 'PICU002_FeatureData.xlsx', \
              'PICU003_FeatureData.xlsx', 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx',\
              'PICU005_FeatureData.xlsx', 'PICU006_FeatureData.xlsx',
              'PICU007_FeatureData - EEG F4-C4.xlsx', 'PICU008_FeatureData.xlsx',
              'PICU009_FeatureData - EEG F4-C4.xlsx', 'PICU010_FeatureData.xlsx']
#picu_files = ['PICU010_FeatureData.xlsx']

if feature_name_file == 'GammaDeltaRatio':
    if number_of_stages == 3:
        wake_threshold = 0.031400108
        sws_threshold = 0.003966432
    elif number_of_stages == 4:
        wake_threshold = 0.027481012 # 0.02835651
        sws_threshold = 0.003966432
        nsws_threshold = 0.024273302
elif feature_name_file == 'GammaTheta+DeltaRatio':
    if number_of_stages == 3:
        wake_threshold = 0.022055037
        sws_threshold = 0.003190831
    elif number_of_stages == 4:
        wake_threshold = 0.022055037
        sws_threshold = 0.003190831
        nsws_threshold = 0.019693002

"""
##### Load PICU feature data #####
"""
### Load PICU feature data
count = 0
if load_picu_data == 1:
    print(datetime.now(), ': Start loading')
    for feature_data_file_name in picu_files:
        if feature_data_file_name.endswith('.xlsx'):
            feature_data_single = load_and_prepare_raw_feature_data_single(picu_data_file_path, feature_data_file_name,
                                                                    remove_artifacts=1)
            #feature_data_single = feature_data_single.sample(frac=1)  # Shuffle data
            print(datetime.now(), ': Loading of %s finished' % feature_data_file_name)
            if count == 0:
                feature_data = feature_data_single
            else:
                feature_data = feature_data.append(feature_data_single)
            count = count + 1

    #feature_data = feature_data.sample(frac=1) # Shuffle data
    #original_feature_data = feature_data.copy()
    #feature_data = feature_data.reset_index()
    #del feature_data['index']

    ### Deal with different channels (F3-C4, F4-C3)
    ##column_names = feature_data.columns[278:329]
    #idx = np.where((feature_data['Patient key'] == 'PICU004') | (feature_data['Patient key'] == 'PICU007')) [0]
    #for i in idx:
        #new_values = feature_data.values[i, 686:737]
        #feature_data.iloc[i, 278:329] = new_values
    #feature_data = feature_data.drop(feature_data.columns[686:737], axis=1)

"""
##### Calculate or load posteriors on control PSG data #####
"""

if calculate_posteriors == 1:
    if load_psg_data == 1:
        PSG_feature_data = load_and_prepare_raw_feature_data_multiple(
            r'E:\ICK slaap project\1_Patient_data',
            'PSG_PatientData.xlsx',
            r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData',
            ['6-12 months', '1-3 years', '3-5 years', '5-9 years',
             '9-13 years', '13-18 years'],
            remove_artifacts=1)

    ### Add labels and extract the feature
    psg_labels_and_features = pd.DataFrame([])
    psg_labels_and_features['Sleep stage labels (cat)'] = PSG_feature_data[number_of_stages_key]

    if feature_name_file == 'GammaDeltaRatio':
        psg_labels_and_features[eeg_channel + ' ' + feature_name] = PSG_feature_data[eeg_channel + ' ' + feature_name]
    elif feature_name_file == 'GammaTheta+DeltaRatio':
        feature1 = PSG_feature_data[eeg_channel + ' ' + 'Abs Gamma power']
        feature2 = PSG_feature_data[eeg_channel + ' ' + 'Abs Theta power']
        feature3 = PSG_feature_data[eeg_channel + ' ' + 'Abs Delta power']
        psg_labels_and_features[eeg_channel + ' ' + feature_name] = (feature1)/(feature2 + feature3)

    psg_labels_and_features = psg_labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

    psg_index_score = psg_labels_and_features[eeg_channel + ' ' + feature_name]
    psg_sleep_stage_labels = psg_labels_and_features['Sleep stage labels (cat)']

    np.seterr(divide='ignore', invalid='ignore')  # To suppress RunTimeWarning
    all_posteriors, x_kde, all_priors = compute_feature_posterior_prob(psg_index_score,
                                                                       psg_sleep_stage_labels,
                                                                       number_of_stages)
    if number_of_stages == 3:
        dfPosteriors = pd.DataFrame({'x_kde': x_kde[:, 0],
                                   'wake_posteriors': all_posteriors[0].get('Wake'),
                                   'nsws_posteriors': all_posteriors[1].get('NSWS'),
                                   'sws_posteriors': all_posteriors[2].get('SWS')})
    elif number_of_stages == 4:
        dfPosteriors = pd.DataFrame({'x_kde': x_kde[:, 0],
                                   'wake_posteriors': all_posteriors[0].get('Wake'),
                                   'rem_posteriors': all_posteriors[1].get('REM'),
                                   'nsws_posteriors': all_posteriors[2].get('NSWS'),
                                   'sws_posteriors': all_posteriors[3].get('SWS')})
    writer = pd.ExcelWriter(os.path.join(posteriors_file_path,
                                         'Posteriors_'+ feature_name_file + '_' + number_of_stages_key + '.xlsx'))
    dfPosteriors.to_excel(writer)
    writer.save()

else:
    posteriors_file = pd.read_excel(os.path.join(posteriors_file_path,
                                         'Posteriors_'+ feature_name_file + '_' + number_of_stages_key +'.xlsx'))
    x_kde = posteriors_file['x_kde']

    all_posteriors = []
    all_posteriors.append({'Wake': np.array(posteriors_file['wake_posteriors'])})
    if number_of_stages == 3:
        all_posteriors.append({'NSWS': np.array(posteriors_file['nsws_posteriors'])})
        all_posteriors.append({'SWS': np.array(posteriors_file['sws_posteriors'])})
    elif number_of_stages == 4:
        all_posteriors.append({'REM': np.array(posteriors_file['rem_posteriors'])})
        all_posteriors.append({'NSWS': np.array(posteriors_file['nsws_posteriors'])})
        all_posteriors.append({'SWS': np.array(posteriors_file['sws_posteriors'])})

"""
##### START TESTING INDEX PERFORMANCE #####
"""
keys_all = []
auc_all = []
acc_all = []
cohens_kappa_all = []
dfContingencyTables = pd.DataFrame([])

count = 0
#test = []
for file in picu_files:
    if file == 'PICU004_FeatureData - EEG F4-C3.xlsx' or file == 'PICU007_FeatureData - EEG F4-C3.xlsx':
        file = file.replace(' - EEG F4-C3', '')
    elif file == 'PICU007_FeatureData - EEG F4-C4.xlsx' or file == 'PICU009_FeatureData - EEG F4-C4.xlsx':
        file = file.replace(' - EEG F4-C4', '')
    elif file == 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx':
        file = file.replace('_ArtLabelsAdjusted', '')
    #test.append(file.replace('_FeatureData.xlsx', ''))
    data = feature_data[feature_data['Patient key'] == file.replace('_FeatureData.xlsx', '')]

    keys_all.append(file.replace('_FeatureData.xlsx', ''))

    start_time = time.time()

    ### Add labels and extract the feature
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
        labels_and_features[eeg_channel + ' ' + feature_name] = (feature1)/(feature2 + feature3)

    labels_and_features = labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

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
            #elif (index_score.iloc[i] > sws_threshold) & (index_score.iloc[i] < nsws_threshold):
                #labels_pred.append('NSWS')

    accuracy = accuracy_score(sleep_stage_labels, labels_pred)
    acc_all.append(accuracy)
    cohens_kappa = cohen_kappa_score(sleep_stage_labels, labels_pred)
    cohens_kappa_all.append(cohens_kappa)

    """
    ##### AUC POSTERIOR#####
    """
    try:
        auc_posterior, tpr, fpr = auc_feature_calculation_posterior(index_score,
                                                                    sleep_stage_labels, all_posteriors, x_kde,
                                                                    number_of_stages,
                                                                    plot_roc=plot_roc)
    except:
        auc_posterior = 0

    auc_all.append(auc_posterior)
    np.seterr(divide='warn', invalid='warn')

    """
    ##### CONTINGENCY TABLES #####
    """

    df = pd.DataFrame({'True sleep stage': sleep_stage_labels,
                                 'Predicted sleep stage (IDOS)': labels_pred})
    contingency_table = pd.crosstab(df['True sleep stage'], df['Predicted sleep stage (IDOS)'])
    contingency_table['Age category'] = feature_data['Patient key'].iloc[0]

    dfContingencyTables = dfContingencyTables.append(contingency_table)

    count = count + 1

# Now for all data
data = feature_data
keys_all.append('All PICU files')

"""
##### INDEX PERFORMANCE #####
"""

start_time = time.time()
### Add labels and extract the feature
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

accuracy = accuracy_score(sleep_stage_labels, labels_pred)
acc_all.append(accuracy)
cohens_kappa = cohen_kappa_score(sleep_stage_labels, labels_pred)
cohens_kappa_all.append(cohens_kappa)

"""
##### AUC POSTERIOR#####
"""
if count == 0:
    ### Add labels and extract the feature
    psg_labels_and_features = pd.DataFrame([])
    psg_labels_and_features['Sleep stage labels (cat)'] = feature_data[number_of_stages_key]

    if feature_name_file == 'GammaDeltaRatio':
        psg_labels_and_features[eeg_channel + ' ' + feature_name] = feature_data[eeg_channel + ' ' + feature_name]
    elif feature_name_file == 'GammaTheta+DeltaRatio':
        feature1 = feature_data[eeg_channel + ' ' + 'Abs Gamma power']
        feature2 = feature_data[eeg_channel + ' ' + 'Abs Theta power']
        feature3 = feature_data[eeg_channel + ' ' + 'Abs Delta power']
        psg_labels_and_features[eeg_channel + ' ' + feature_name] = (feature1) / (feature2 + feature3)

    psg_labels_and_features = psg_labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it

    psg_index_score = psg_labels_and_features[eeg_channel + ' ' + feature_name]
    psg_sleep_stage_labels = psg_labels_and_features['Sleep stage labels (cat)']

    np.seterr(divide='ignore', invalid='ignore')  # To suppress RunTimeWarning
    all_posteriors, x_kde, all_priors = compute_feature_posterior_prob(psg_index_score,
                                                                       psg_sleep_stage_labels,
                                                                       number_of_stages)

auc_posterior, tpr, fpr = auc_feature_calculation_posterior(index_score,
                                                            sleep_stage_labels, all_posteriors, x_kde,
                                                            number_of_stages,
                                                            plot_roc=plot_roc)
auc_all.append(auc_posterior)
np.seterr(divide='warn', invalid='warn')

"""
##### CONTINGENCY TABLES #####
"""

df = pd.DataFrame({'True sleep stage': sleep_stage_labels,
                   'Predicted sleep stage (IDOS)': labels_pred})
contingency_table = pd.crosstab(df['True sleep stage'], df['Predicted sleep stage (IDOS)'])
contingency_table['Age category'] = 'All'

dfContingencyTables = dfContingencyTables.append(contingency_table)

"""
##### STORE RESULT ARRAYS IN DATAFRAME
"""
dfIndexResults = pd.DataFrame({'Patient key': keys_all,
                        'AUC posterior': auc_all,
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


