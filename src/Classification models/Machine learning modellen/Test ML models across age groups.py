"""
This function is used to evaluate the model performance across the various age groups
"""

import os
import pickle
from datetime import datetime
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score

number_of_stages_key = 'Three_state_labels'
classifier = 'DT'
channel = 'EEG F3-C3'
model_file = '%s_Model_%s_%s.sav' % (classifier, channel, number_of_stages_key)
training_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_7_TrainingSet'
model_files_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'
age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years',
                  '3-5 years', '5-9 years', '9-13 years', '13-18 years']
results_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_5_Performance per age group'

feature_data = pd.DataFrame([])
print(datetime.now(), ': Start loading')
for feature_data_file_name in os.listdir(training_data_file_path):
    if feature_data_file_name.endswith('.csv'):
        feature_data = feature_data.append(pd.read_csv(
            os.path.join(training_data_file_path, feature_data_file_name),
            sep=',',
            decimal='.'))
        print(datetime.now(), ': %s loaded' % feature_data_file_name)
feature_data = feature_data.sample(40000) # Shuffle data

clf = pickle.load(open(os.path.join(model_files_path, model_file), 'rb'))

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

age_category_list = []
accuracy_list = []
auc_list = []
cohens_kappa_list = []

for age_category in age_categories:
    test_data = feature_data[feature_data['Age category'] == age_category]
    labels_and_features = pd.DataFrame([])
    labels_and_features[number_of_stages_key] = test_data[number_of_stages_key + '- linear']
    labels_and_features[feature_names] = test_data[feature_names]
    labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it

    ### Extract labels, features and groups
    labels = labels_and_features[number_of_stages_key]
    features = labels_and_features.iloc[:, 1:]

    labels_pred = clf.predict(features)
    labels_pred_prob = clf.predict_proba(features)
    test_accuracy = accuracy_score(labels, labels_pred)
    test_cohens_kappa = cohen_kappa_score(labels, labels_pred)
    test_auc = roc_auc_score(labels, labels_pred_prob, multi_class='ovr')

    age_category_list.append(age_category)
    accuracy_list.append(test_accuracy)
    auc_list.append(test_auc)
    cohens_kappa_list.append(test_cohens_kappa)

test_data = feature_data
labels_and_features = pd.DataFrame([])
labels_and_features[number_of_stages_key] = test_data[number_of_stages_key + '- linear']
labels_and_features[feature_names] = test_data[feature_names]
labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it

### Extract labels, features and groups
labels = labels_and_features[number_of_stages_key]
features = labels_and_features.iloc[:, 1:]

labels_pred = clf.predict(features)
labels_pred_prob = clf.predict_proba(features)
test_accuracy = accuracy_score(labels, labels_pred)
test_cohens_kappa = cohen_kappa_score(labels, labels_pred)
test_auc = roc_auc_score(labels, labels_pred_prob, multi_class='ovr')

age_category_list.append('0-18 years')
accuracy_list.append(test_accuracy)
auc_list.append(test_auc)
cohens_kappa_list.append(test_cohens_kappa)

dfResults = pd.DataFrame({'Age category': age_category_list,
                          'Accuracy': accuracy_list,
                          'AUC': auc_list,
                          'Cohens kappa': cohens_kappa_list})

model_results_file_name = classifier + '_PerformancePerAgeGroup_' + channel + number_of_stages_key + '.xlsx'
writer = pd.ExcelWriter(os.path.join(results_file_path, model_results_file_name))
dfResults.to_excel(writer)
writer.save()