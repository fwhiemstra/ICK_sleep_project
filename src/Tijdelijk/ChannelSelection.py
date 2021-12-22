"""
Channel selection method
"""

import os
import numpy as np
import pandas as pd
import Functions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#%% Set variables
# File paths and names
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
feature_evaluation_file_path = r'E:\ICK slaap project\5_FeatureEvaluation\Channel selection'

# Function variables
age_category = '2-6 months'
number_of_stages = 3 # 2 = [Wake, Sleep], 3 = [Wake, NSWS, SWS], 4 = [Wake, REM, NSWS, SWS], 5 = [Wake, REM, N1, N2, N3]
remove_artifacts = 1

# Initiate function
patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
keys = patientData[patientData['Age category'] == age_category]['Key']
feature_data_file_names = [key + '_FeatureData.xlsx' for key in keys]

### Channel evaluation
allFeatureData = []
file_keys = []
dfChannelEvaluation = pd.DataFrame([])

for feature_data_file_name in feature_data_file_names:
    if feature_data_file_name in os.listdir(feature_data_file_path):
        featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
        file_key = feature_data_file_name.replace('_FeatureData.xlsx', '')
        file_keys.append(file_key)

        ### Add new sleep stage labels
        featureData_new, number_of_stages_key = Functions.add_new_state_labels(featureData, number_of_stages)

        ###  Remove artefacts
        if remove_artifacts == 1:
            featureData_clean = featureData_new[featureData['Artifact'] == 0]

        ### Loop over alle features
        data = featureData_clean

        ### Create dataframes
        if feature_data_file_name == feature_data_file_names[0]:
            dfChannelEvaluation = pd.DataFrame([])
            allFeatureData = data
        else:
            allFeatureData = allFeatureData.append(data)

        ### Channel evaluation
        channel_total = []
        channel = []

        for i in np.arange(13, 871, 39):
            channel.append(data.columns[i].replace(': Total power', ''))
            feature_data_subset = data[data.columns[i:i + 38]]
            test = SelectKBest(score_func=f_classif, k=10)
            fit = test.fit(feature_data_subset, data[number_of_stages_key])
            features_selected = feature_data_subset.columns[fit.get_support()]
            ANOVA_of_selected_features = fit.scores_[fit.get_support()]
            channel_total.append(np.sum(ANOVA_of_selected_features))
        df = pd.DataFrame({'Channel': channel, 'Channel score': channel_total})

        idx_top_channels = sorted(range(len(channel_total)), key=lambda i: channel_total[i])[-5:]
        top_channels_names = df['Channel'][idx_top_channels]
        top_channels_scores = df['Channel score'][np.array(idx_top_channels)]

        dfChannelEvaluation[file_key + ' top channels'] = np.array(top_channels_names)
        dfChannelEvaluation[file_key] = np.array(top_channels_scores)

### For all feature data
channel_total = []
channel = []
file_key = age_category
data = allFeatureData

for i in np.arange(13, 871, 39):
    channel.append(data.columns[i].replace(': Total power', ''))
    feature_data_subset = data[data.columns[i:i + 38]]
    test = SelectKBest(score_func=f_classif, k=10)
    fit = test.fit(feature_data_subset, data[number_of_stages_key])
    features_selected = feature_data_subset.columns[fit.get_support()]
    ANOVA_of_selected_features = fit.scores_[fit.get_support()]
    channel_total.append(np.sum(ANOVA_of_selected_features))
df = pd.DataFrame({'Channel': channel, 'Channel score': channel_total})

idx_top_channels = sorted(range(len(channel_total)), key=lambda i: channel_total[i])[-5:]
top_channels_names = df['Channel'][idx_top_channels]
top_channels_scores = df['Channel score'][np.array(idx_top_channels)]

dfChannelEvaluation[file_key + ' top channels'] = np.array(top_channels_names)
dfChannelEvaluation[file_key] = np.array(top_channels_scores)

### Write to excel file
channel_selection_file_name = 'ChannelSelection_' + age_category + '_' + number_of_stages_key +'.xlsx'
writer = pd.ExcelWriter(os.path.join(feature_evaluation_file_path, channel_selection_file_name))
dfChannelEvaluation.to_excel(writer)
writer.save()

print('Channel evaluation of age_category ' + age_category + ' completed')
