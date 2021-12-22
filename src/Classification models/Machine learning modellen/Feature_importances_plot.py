"""
This script is used to plot features importances of the machine learning models
"""
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from datetime import datetime
import numpy as np

model_files_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'

plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

#%% Feature importances for decision tree models
plt.close('all')
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
axs[0, 0].xaxis.set_tick_params(which='both', labelbottom=True)
axs[0, 1].xaxis.set_tick_params(which='both', labelbottom=True)

plt.suptitle('Feature importances')

# Three-state classification
model_file = 'DT_Model_EEG F3-C3_Three_state_labels.sav'
clf = pickle.load(open(os.path.join(model_files_path, model_file), 'rb'))
importances  = clf.feature_importances_
feature_names = ['Interquartile range', 'DWT cD1 SD',
       'Rel. gamma power', 'Zero crossing rate',
       'DWT cD4 mean', 'DWT cA mean',
       'Gamma/theta ratio', 'Hjorth mobility',
       'EMG Chin: Abs. mean amplitude', 'DWT cA SD',
       'DWT cD3 mean', 'Gamma/delta ratio',
       'Age 0-2 months', 'Age 1-3 years',
       'Age 13-18 years', 'Age 2-6 months',
       'Age 3-5 years', 'Age 5-9 years',
       'Age 6-12 months', 'Age 9-13 years']
feat_importances = pd.Series(importances, feature_names)
feat_importances = feat_importances.nsmallest(20) # sort scores
axs[0, 0].set_title("Three-state classification")
axs[0, 0].barh(range(len(feat_importances.index)), feat_importances.values, color='olivedrab', alpha=0.8)
axs[0, 0].set_yticks(range(len(feat_importances.index)))
axs[0, 0].set_yticklabels(feat_importances.index)
axs[0, 0].set_xlabel('Feature importance score')
#plt.setp(axs[0, 0].get_xticklabels(), visible=True)

# Four-state classification
model_file_four = 'DT_Model_EEG F3-C3_Four_state_labels.sav'
clf = pickle.load(open(os.path.join(model_files_path, model_file_four), 'rb'))
importances = clf.feature_importances_
importances = np.append(importances, [0, 0, 0, 0])
feature_names = ['Interquartile range', 'DWT cD1 SD',
       'Rel. gamma power', 'Abs. mean amplitude',
       'Zero crossing rate',
       'EMG Chin: Abs. mean amplitude', 'Hjorth mobility',
       'DWT cA SD', 'DWT cD4 mean',
       'DWT cD1 mean', 'Total power',
       'Age 1-3 years', 'Age 13-18 years',
       'Age 5-9 years', 'Age _6-12 months',
       'Age 9-13 years', '', '', '', '']

feat_importances = pd.Series(importances, feature_names)
feat_importances = feat_importances.nsmallest(20) # sort scores
axs[0, 1].set_title("Four-state classification")
axs[0, 1].barh(range(len(feat_importances.index)), feat_importances.values, color='darkcyan', alpha=0.8)
axs[0, 1].set_yticks(range(len(feat_importances.index)))
axs[0, 1].set_yticklabels(feat_importances.index)
axs[0, 1].set_xlabel('Feature importance score')

plt.tight_layout()


###%% Feature importances for xgboost models
#fig, axs = plt.subplots(2, 2, figsize=(12, 4))
#plt.suptitle('Feature importances - XGBoost')
# Three-state classification
model_file_three = 'xgboost_Model_EEG F3-C3_Three_state_labels.sav'
clf = pickle.load(open(os.path.join(model_files_path, model_file_three), 'rb'))
importances  = clf.feature_importances_
feature_names = ['Interquartile range', 'DWT cD1 SD',
                   'Rel. gamma power', 'DWT cD5 mean',
                   'DWT cA mean', 'Gamma/theta ratio',
                   'DWT cD5 SD', 'Hjorth mobility',
                   'DWT cA SD', 'DWT cD1 mean',
                   'DWT cD4 SD', 'DWT cD3 mean',
                   'Total power', 'Age 0-2 months',
                   'Age 13-18 years', 'Age 2-6 months',
                   'Age 3-5 years', 'Age 5-9 years',
                   'Age 6-12 months', 'Age 9-13 years']
feat_importances = pd.Series(importances, feature_names)
feat_importances = feat_importances.nsmallest(20) # sort scores
axs[1, 0].set_title("Three-state classification")
axs[1, 0].barh(range(len(feat_importances.index)), feat_importances.values, color='olivedrab', alpha=0.8)
axs[1, 0].set_yticks(range(len(feat_importances.index)))
axs[1, 0].set_yticklabels(feat_importances.index)
axs[1, 0].set_xlabel('Feature importance score')

# Four-state classification
model_file_four = 'xgboost_Model_EEG F3-C3_Four_state_labels.sav'
clf = pickle.load(open(os.path.join(model_files_path, model_file_four), 'rb'))
importances = clf.feature_importances_
importances = np.append(importances, [0, 0])
feature_names = ['Interquartile range', 'DWT cD1 SD',
       'Abs. mean amplitude', 'DWT cA mean',
       'DWT cD5 mean', 'Rel Gamma power',
       'Zero crossing rate', 'DWT cD5 SD',
       'DWT cA SD', 'Abs. Sigma power',
       'EMG Chin: Abs. mean amplitude', 'DWT cD4 mean',
       'DWT cD3 mean', 'Total power',
       'Age 13-18 years', 'Age 5-9 years',
       'Age 6-12 months', 'Age 9-13 years', '', '']

feat_importances = pd.Series(importances, feature_names)
feat_importances = feat_importances.nsmallest(20) # sort scores
axs[1, 1].set_title("Four-state classification")
axs[1, 1].barh(range(len(feat_importances.index)), feat_importances.values, color='darkcyan', alpha=0.8)
axs[1, 1].set_yticks(range(len(feat_importances.index)))
axs[1, 1].set_yticklabels(feat_importances.index)
axs[1, 1].set_xlabel('Feature importance score')
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

plt.savefig(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Thesis report\Figuren thesis report\Feature importances - aligned.png', dpi=1000)

"""
#%% Old DT models

model_file = 'DT_Model_EEG F3-C4_Three_state_labels.sav'
clf = pickle.load(open(os.path.join(model_files_path, model_file), 'rb'))
importances  = clf.feature_importances_
feature_names = ['DWT cD5 mean', 'Rel. Gamma power',
       'Interquartile range', 'DWT cD1 std',
       'Zero crossing rate', 'Gamma/theta ratio',
       'DWT cA mean', 'DWT cD4 std',
       'DWT cD3 mean', 'DWT cD1 mean',
       'Gamma/delta ratio', 'DWT cD3 std',
       'Age 0-2 months', 'Age 1-3 years',
       'Age 13-18 years', 'Age 2-6 months',
       'Age 3-5 years', 'Age 5-9 years',
       'Age 6-12 months', 'Age 9-13 years']
feat_importances = pd.Series(importances, feature_names)
feat_importances = feat_importances.nsmallest(20) # sort scores
axs[0, 0].set_title("Three-state classification")
axs[0, 0].barh(range(len(feat_importances.index)), feat_importances.values, color='olivedrab', alpha=0.8)
axs[0, 0].set_yticks(range(len(feat_importances.index)))
axs[0, 0].set_yticklabels(feat_importances.index)
axs[0, 0].set_xlabel('Feature importance score')
#plt.setp(axs[0, 0].get_xticklabels(), visible=True)

# Four-state classification
model_file_four = 'DT_Model_EEG F3-C4_Four_state_labels.sav'
clf = pickle.load(open(os.path.join(model_files_path, model_file_four), 'rb'))
importances = clf.feature_importances_
importances = np.append(importances, [0, 0, 0, 0])
feature_names = ['Abs. Beta power', 'DWT cD5 std',
       'Interquartile range', 'Rel. Gamma power',
       'Abs. mean amplitude', 'Zero crossing rate',
       'DWT cA std', 'DWT cD4 std',
       'EMG Chin: Abs. mean amplitude', 'Abs. Delta power',
       'Spectral edge', 'Age 1-3 years',
       'Age 13-18 years', 'Age 5-9 years',
       'Age 6-12 months', 'Age 9-13 years', '', '', '', '']

feat_importances = pd.Series(importances, feature_names)
feat_importances = feat_importances.nsmallest(20) # sort scores
axs[0, 1].set_title("Four-state classification")
axs[0, 1].barh(range(len(feat_importances.index)), feat_importances.values, color='darkcyan', alpha=0.8)
axs[0, 1].set_yticks(range(len(feat_importances.index)))
axs[0, 1].set_yticklabels(feat_importances.index)
axs[0, 1].set_xlabel('Feature importance score')

plt.tight_layout()
"""
#%% Feature importances for SVM models




"""
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plt.suptitle('Feature importances - SVM')
# Load training data
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_7_TrainingSet'
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
# Three-state classification
number_of_stages_key = 'Three_state_labels'
model_file = 'SVM_Model_EEG F3-C3_%s.sav' % number_of_stages_key
clf = pickle.load(open(os.path.join(model_files_path, model_file), 'rb'))
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
### Add labels and extract the feature
labels_and_features = pd.DataFrame([])
labels_and_features[number_of_stages_key] = feature_data[number_of_stages_key + '- linear']
labels_and_features[feature_names] = feature_data[feature_names]
labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it
labels = labels_and_features[number_of_stages_key]
features = labels_and_features[feature_names]

permutation_score = permutation_importance(clf, features, labels, n_repeats=30, random_state=0)

#%% Four-state classification
model_file == 'SVM_Model_EEG F3-C3_Four_state_labels.sav'
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

"""


