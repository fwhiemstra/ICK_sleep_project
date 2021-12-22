"""
This function is used to evaluate the model performance on the PICU data
"""

import os
import pandas as pd
import sys
sys.path.append(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\src')
from Functions import load_and_prepare_raw_feature_data_single, \
    load_and_prepare_raw_feature_data_multiple, compute_feature_posterior_prob, \
    auc_feature_calculation_posterior, find_optimal_index_thresholds
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
import time
from datetime import datetime
import pickle

number_of_stages_key = 'Four_state_labels'
load_picu_data = 0
classifier = 'xgboost'
channel = 'EEG F3-C3'
model_file = '%s_Model_%s_%s.sav' % (classifier, channel, number_of_stages_key)
picu_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_4_PICU_FeatureData_SCALED'
model_files_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'
results_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_6_Performance on PICU data'
clf = pickle.load(open(os.path.join(model_files_path, model_file), 'rb'))
results_file_name = 'PICU_results_%s_%s_%s.xlsx' % (classifier, channel, number_of_stages_key)

### Select files for each channel
if channel == 'EEG F3-C4':
    picu_files = ['PICU001_FeatureData_scaled.xlsx', 'PICU002_FeatureData_scaled.xlsx',
                  'PICU003_FeatureData_scaled.xlsx', 'PICU004_FeatureData - EEG F4-C3_scaled.xlsx',
                  'PICU006_FeatureData_scaled.xlsx', 'PICU007_FeatureData - EEG F4-C3_scaled.xlsx']
elif channel == 'EEG F3-C3':
    picu_files = ['PICU001_FeatureData_scaled.xlsx', 'PICU002_FeatureData_scaled.xlsx',
                  'PICU003_FeatureData_scaled.xlsx', 'PICU004_FeatureData_ArtLabelsAdjusted_scaled.xlsx',
                  'PICU005_FeatureData_scaled.xlsx', 'PICU006_FeatureData_scaled.xlsx',
                  'PICU007_FeatureData - EEG F4-C4_scaled.xlsx', 'PICU008_FeatureData_scaled.xlsx',
                  'PICU009_FeatureData - EEG F4-C4_scaled.xlsx', 'PICU010_FeatureData_scaled.xlsx']

### Set features for each model
if model_file == 'DT_Model_EEG F3-C4_Three_state_labels.sav':
    feature_names = ['EEG F3-C4: DWT cD5 mean', 'EEG F3-C4: Rel Gamma power',
                    'EEG F3-C4: Interquartile range', 'EEG F3-C4: DWT cD1 std',
                    'EEG F3-C4: Zero crossing rate', 'EEG F3-C4: Gamma/theta ratio',
                    'EEG F3-C4: DWT cA mean', 'EEG F3-C4: DWT cD4 std',
                    'EEG F3-C4: DWT cD3 mean', 'EEG F3-C4: DWT cD1 mean',
                    'EEG F3-C4: Gamma/delta ratio', 'EEG F3-C4: DWT cD3 std',
                    'age_group__0-2 months', 'age_group__1-3 years',
                    'age_group__13-18 years', 'age_group__2-6 months',
                    'age_group__3-5 years', 'age_group__5-9 years',
                    'age_group__6-12 months', 'age_group__9-13 years']
elif model_file == 'DT_Model_EEG F3-C4_Four_state_labels.sav':
    feature_names = ['EEG F3-C4: Abs Beta power', 'EEG F3-C4: DWT cD5 std',
                   'EEG F3-C4: Interquartile range', 'EEG F3-C4: Rel Gamma power',
                   'EEG F3-C4: Abs mean amplitude', 'EEG F3-C4: Zero crossing rate',
                   'EEG F3-C4: DWT cA std', 'EEG F3-C4: DWT cD4 std',
                   'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C4: Abs Delta power',
                   'EEG F3-C4: Spectral edge', 'age_group__1-3 years',
                   'age_group__13-18 years', 'age_group__5-9 years',
                   'age_group__6-12 months', 'age_group__9-13 years']

if model_file == 'DT_Model_EEG F3-C3_Three_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
       'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: Zero crossing rate',
       'EEG F3-C3: DWT cD4 mean', 'EEG F3-C3: DWT cA mean',
       'EEG F3-C3: Gamma/theta ratio', 'EEG F3-C3: Hjorth mobility',
       'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: DWT cA std',
       'EEG F3-C3: DWT cD3 mean', 'EEG F3-C3: Gamma/delta ratio',
       'age_group__0-2 months', 'age_group__1-3 years',
       'age_group__13-18 years', 'age_group__2-6 months',
       'age_group__3-5 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']

elif model_file == 'DT_Model_EEG F3-C3_Four_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
       'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: Abs mean amplitude',
       'EEG F3-C3: Zero crossing rate',
       'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: Hjorth mobility',
       'EEG F3-C3: DWT cA std', 'EEG F3-C3: DWT cD4 mean',
       'EEG F3-C3: DWT cD1 mean', 'EEG F3-C3: Total power',
       'age_group__1-3 years', 'age_group__13-18 years',
       'age_group__5-9 years', 'age_group__6-12 months',
       'age_group__9-13 years']

elif model_file == 'SVM_Model_EEG F3-C3_Three_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD5 mean',
       'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: Abs mean amplitude',
       'EEG F3-C3: Signal sum', 'EEG F3-C3: DWT cD5 std',
       'EEG F3-C3: DWT cD4 mean', 'EEG F3-C3: Gamma/theta ratio',
       'EEG F3-C3: DWT cA mean', 'EEG F3-C3: DWT cA std',
       'EEG F3-C3: DWT cD4 std', 'EEG F3-C3: Gamma/delta ratio',
       'age_group__0-2 months', 'age_group__1-3 years',
       'age_group__13-18 years', 'age_group__2-6 months',
       'age_group__3-5 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']
elif model_file == 'SVM_Model_EEG F3-C3_Four_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
                   'EEG F3-C3: DWT cD5 mean', 'EEG F3-C3: Abs mean amplitude',
                   'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: DWT cA mean',
                   'EEG F3-C3: Zero crossing rate',
                   'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: DWT cD1 mean',
                   'EEG F3-C3: DWT cA std', 'EEG F3-C3: Hjorth mobility',
                   'EEG F3-C3: DWT cD4 mean', 'EEG F3-C3: Spectral edge',
                   'EEG F3-C3: Spectral entropy', 'EEG F3-C3: Abs Sigma power',
                   'EEG F3-C3: Abs Delta power', 'age_group__1-3 years',
                   'age_group__13-18 years', 'age_group__6-12 months',
                   'age_group__9-13 years']
elif model_file == 'xgboost_Model_EEG F3-C3_Three_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
                   'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: DWT cD5 mean',
                   'EEG F3-C3: DWT cA mean', 'EEG F3-C3: Gamma/theta ratio',
                   'EEG F3-C3: DWT cD5 std', 'EEG F3-C3: Hjorth mobility',
                   'EEG F3-C3: DWT cA std', 'EEG F3-C3: DWT cD1 mean',
                   'EEG F3-C3: DWT cD4 std', 'EEG F3-C3: DWT cD3 mean',
                   'EEG F3-C3: Total power', 'age_group__0-2 months',
                   'age_group__13-18 years', 'age_group__2-6 months',
                   'age_group__3-5 years', 'age_group__5-9 years',
                   'age_group__6-12 months', 'age_group__9-13 years']
elif model_file == 'xgboost_Model_EEG F3-C3_Four_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
       'EEG F3-C3: Abs mean amplitude', 'EEG F3-C3: DWT cA mean',
       'EEG F3-C3: DWT cD5 mean', 'EEG F3-C3: Rel Gamma power',
       'EEG F3-C3: Zero crossing rate', 'EEG F3-C3: DWT cD5 std',
       'EEG F3-C3: DWT cA std', 'EEG F3-C3: Abs Sigma power',
       'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: DWT cD4 mean',
       'EEG F3-C3: DWT cD3 mean', 'EEG F3-C3: Total power',
       'age_group__13-18 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']

### Load PICU feature data
count = 0
if load_picu_data == 1:
    print(datetime.now(), ': Start loading')
    for feature_data_file_name in picu_files:
        if feature_data_file_name.endswith('.xlsx'):
            feature_data_single = load_and_prepare_raw_feature_data_single(picu_data_file_path, feature_data_file_name,
                                                                    remove_artifacts=1)
            feature_data_single = feature_data_single.sample(frac=1)  # Shuffle data
            print(datetime.now(), ': Loading of %s finished' % feature_data_file_name)
            if count == 0:
                feature_data = feature_data_single
            else:
                feature_data = feature_data.append(feature_data_single)
            count = count + 1


### Start testing performance

keys_all = []
auc_all = []
acc_all = []
cohens_kappa_all = []
dfContingencyTables = pd.DataFrame([])
test = []
for file in picu_files:
    if file == 'PICU004_FeatureData - EEG F4-C3_scaled.xlsx' or file == 'PICU007_FeatureData - EEG F4-C3_scaled.xlsx.xlsx':
        file = file.replace(' - EEG F4-C3', '')
    elif file == 'PICU007_FeatureData - EEG F4-C4_scaled.xlsx' or file == 'PICU009_FeatureData - EEG F4-C4_scaled.xlsx':
        file = file.replace(' - EEG F4-C4', '')
    elif file == 'PICU004_FeatureData_ArtLabelsAdjusted_scaled.xlsx':
        file = file.replace('_ArtLabelsAdjusted', '')
    test.append(file.replace('_FeatureData_scaled.xlsx', ''))

    data = feature_data[feature_data['Patient key'] == file.replace('_FeatureData_scaled.xlsx', '')]

    keys_all.append(file.replace('_FeatureData_scaled.xlsx', ''))

    print(datetime.now(), ': %s started' % file)

    ### Add labels and extract the feature
    labels_and_features = pd.DataFrame([])
    labels_and_features[number_of_stages_key] = data[number_of_stages_key + '- linear']
    labels_and_features[feature_names] = data[feature_names]
    labels_and_features = labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

    features = labels_and_features[feature_names]
    labels = labels_and_features[number_of_stages_key]

    labels_pred = clf.predict(features)
    labels_pred_prob = clf.predict_proba(features)

    try:
        test_accuracy = accuracy_score(labels, labels_pred)
        test_cohens_kappa = cohen_kappa_score(labels, labels_pred)
        test_auc = roc_auc_score(labels, labels_pred_prob, multi_class='ovr')

        ### CONTINGENCY TABLES
        df = pd.DataFrame({'True sleep stage': labels,
                                     'Predicted sleep stage (ML)': labels_pred})
        contingency_table = pd.crosstab(df['True sleep stage'], df['Predicted sleep stage (ML)'])
        contingency_table['Key'] = data['Patient key'].iloc[0]

        dfContingencyTables = dfContingencyTables.append(contingency_table)

    except:
        print('%s failed (not enough classes)' % file)
        test_accuracy = 0
        test_cohens_kappa = 0
        test_auc = 0

    auc_all.append(test_auc)
    acc_all.append(test_accuracy)
    cohens_kappa_all.append(test_cohens_kappa)

    print(datetime.now(), ': %s completed' % file)

### Now for all PICU feature data
data = feature_data
keys_all.append('All')

print(datetime.now(), ': All PICU data started')

### Add labels and extract the feature
labels_and_features = pd.DataFrame([])
labels_and_features[number_of_stages_key] = data[number_of_stages_key + '- linear']
labels_and_features[feature_names] = data[feature_names]
labels_and_features = labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

features = labels_and_features[feature_names]
labels = labels_and_features[number_of_stages_key]

labels_pred = clf.predict(features)
labels_pred_prob = clf.predict_proba(features)
test_accuracy = accuracy_score(labels, labels_pred)
test_cohens_kappa = cohen_kappa_score(labels, labels_pred)
test_auc = roc_auc_score(labels, labels_pred_prob, multi_class='ovr')

auc_all.append(test_auc)
acc_all.append(test_accuracy)
cohens_kappa_all.append(test_cohens_kappa)

"""
##### CONTINGENCY TABLES #####
"""

df = pd.DataFrame({'True sleep stage': labels,
                   'Predicted sleep stage (ML)': labels_pred})
contingency_table = pd.crosstab(df['True sleep stage'], df['Predicted sleep stage (ML)'])
contingency_table['Key'] = 'All PICU data'

dfContingencyTables = dfContingencyTables.append(contingency_table)

print(datetime.now(), ': All PICU data completed')

"""
##### STORE RESULT ARRAYS IN DATAFRAME
"""
dfResults = pd.DataFrame({'Patient key': keys_all,
                        'AUC posterior': auc_all,
                        'Accuracy': acc_all,
                       'Cohens kappa': cohens_kappa_all,
                       })

"""
##### WRITE RESULTS TO EXCEL FILE #####
"""
writer = pd.ExcelWriter(os.path.join(results_file_path, results_file_name))
dfResults.to_excel(writer, sheet_name='Results')
dfContingencyTables.to_excel(writer, sheet_name='Contingency Tables')
writer.save()

print('PICU testing of %s %s completed' % (classifier, number_of_stages_key))


