"""
This functions evaluates the training performance of index-based measures per group of patients accross EEG channels
June 2021, Floor Hiemstra
"""

import os
import numpy as np
import pandas as pd
from Functions import load_and_prepare_raw_feature_data_multiple, \
    compute_feature_posterior_prob, auc_feature_calculation_posterior, find_optimal_index_thresholds
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr

"""
##### SET DATA VARIABLES #####
"""
# NOTE: This function only works for features that are positively correlated with
# the sleep stages, thus increase during Wake and decrease during SWS.

load_data = 1  # 0 if data is already loaded
data_type = 'sampled'  # 'new' or 'sampled'

eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']

### Variables, if data == 'new'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years']
remove_artifacts = 1

### Variables, if data == 'sampled'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
sampled_data_file_name = '1000sampled_feature_data_0-18years_nonscaled.xlsx'

"""
##### LOAD FEATURE DATA #####
"""
if load_data == 1 and data_type == 'new':
    feature_data = load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                                              feature_data_file_path, include_age_categories,
                                                              remove_artifacts)
elif load_data == 1 and data_type == 'sampled':
    feature_data = pd.read_excel(os.path.join(sampled_data_file_path, sampled_data_file_name))

feature_data = feature_data.sample(frac=1)

#%%
"""
##### SPEARMAN CORRELATION #####
"""
number_of_stages_key = 'Three_state_labels- linear'
eeg_channel = 'EEG F3-C4'
eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]
features = feature_data[eeg_features_list]
sleep_stage_labels = feature_data[number_of_stages_key]
corr_list = []
for feature_name in eeg_features_list:
    feature = feature_data[feature_name]
    corr, p = spearmanr(sleep_stage_labels, feature, nan_policy='omit')
    corr_list.append(corr)

correlations = pd.Series(data = corr_list, index=eeg_features_list)

