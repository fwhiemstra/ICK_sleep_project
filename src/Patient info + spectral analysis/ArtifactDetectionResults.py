"""
This file is used to evaluate the artifact detection results.
Floor Hiemstra, June 2021
"""

import os
import pandas as pd
import numpy as np

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
results_file_path = r'E:\ICK slaap project\10_Patient+Data_Characteristics'
artifact_results_file_name_per_recording = 'PICU_ArtifactResults_per_recording.xlsx'
artifact_results_file_name_per_age_category = 'PICU_ArtifactResults_per_age_category.xlsx'

# feature_data_file_name = 'PSG001_FeatureData.xlsx'
# featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
epoch_size = 30  # seconds

# %% Artifact detection results per recording
key = []

age_category = []
n_impedance_artifacts = []
n_EOGLOC_artifacts = []
n_EOGROC_artifacts = []
n_EMG_artifacts = []
n_EEGF3_artifacts = []
n_EEGF4_artifacts = []
n_EEGC3_artifacts = []
n_EEGC4_artifacts = []
n_EEGO1_artifacts = []
n_EEGO2_artifacts = []
n_EEGA1_artifacts = []
n_EEGA2_artifacts = []
total_number_of_epochs = []

for feature_data_file_name in os.listdir(feature_data_file_path):
    if feature_data_file_name.endswith(".xlsx"):
        featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

        # General information
        key.append(featureData["Patient key"][0])
        age_category.append(featureData["Age category"][0])

        # Number of epochs
        n_impedance_artifacts.append(len(featureData[featureData["Impedance artifact"] == 1]))
        n_EOGLOC_artifacts.append(len(featureData[featureData["EOG ROC - Artifact"] == 1]))
        n_EOGROC_artifacts.append(len(featureData[featureData["EOG LOC - Artifact"] == 1]))
        n_EMG_artifacts.append(len(featureData[featureData["EMG EMG Chin - Artifact"] == 1]))
        n_EEGF3_artifacts.append(len(featureData[featureData["EEG F3 - Artifact"] == 1]))
        n_EEGF4_artifacts.append(len(featureData[featureData["EEG F4 - Artifact"] == 1]))
        n_EEGC3_artifacts.append(len(featureData[featureData["EEG C3 - Artifact"] == 1]))
        n_EEGC4_artifacts.append(len(featureData[featureData["EEG C4 - Artifact"] == 1]))
        n_EEGO1_artifacts.append(len(featureData[featureData["EEG O1 - Artifact"] == 1]))
        n_EEGO2_artifacts.append(len(featureData[featureData["EEG O2 - Artifact"] == 1]))
        n_EEGA1_artifacts.append(len(featureData[featureData["EEG A1 - Artifact"] == 1]))
        n_EEGA2_artifacts.append(len(featureData[featureData["EEG A2 - Artifact"] == 1]))
        total_number_of_epochs.append(len(featureData))
        print('Artifact detection evaluation of ', feature_data_file_name, ' completed')

# Convert to dataframe
dfArtifactResults = pd.DataFrame({'Patient key': key,
                                  'Age category': age_category,
                                  'Total number of epochs': total_number_of_epochs,
                                  'n_impedance_artifacts': n_impedance_artifacts ,
                                  'n_EOGLOC_artifacts': n_EOGLOC_artifacts,
                                  'n_EOGROC_artifacts': n_EOGROC_artifacts,
                                  'n_EMG_artifacts': n_EMG_artifacts,
                                  'n_EEGF3_artifacts': n_EEGF3_artifacts,
                                  'n_EEGF4_artifacts': n_EEGF4_artifacts,
                                  'n_EEGC3_artifacts': n_EEGC3_artifacts,
                                  'n_EEGC4_artifacts': n_EEGC4_artifacts,
                                  'n_EEGO1_artifacts': n_EEGO1_artifacts,
                                  'n_EEGO2_artifacts': n_EEGO2_artifacts,
                                  'n_EEGA1_artifacts': n_EEGA1_artifacts,
                                  'n_EEGA2_artifacts': n_EEGA2_artifacts,
                                  })

writer = pd.ExcelWriter(os.path.join(results_file_path, artifact_results_file_name_per_recording))
dfArtifactResults .to_excel(writer)
writer.save()

# %% Artifact detection results per age category
dfArtifactResults = pd.read_excel(os.path.join(results_file_path, artifact_results_file_name_per_recording))
dfArtifactResults_per_age_category = pd.DataFrame([])

age_categories = ['all']#, 'all>2 months', 'all>6 months', '0-2 months', '2-6 months', '6-12 months', '1-3 years',
                  #'3-5 years', '5-9 years', '9-13 years', '13-18 years']

for age_category in age_categories:
    if age_category == 'all':
        dfArtifactResults_sub = dfArtifactResults
    elif age_category == 'all>2 months':
        dfArtifactResults_sub = dfArtifactResults[dfArtifactResults['Age category'].isin(
            ['2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years'])]
    elif age_category == 'all>6 months':
        dfArtifactResults_sub = dfArtifactResults[dfArtifactResults['Age category'].isin(
            ['6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years'])]
    else:
        dfArtifactResults_sub = dfArtifactResults[dfArtifactResults['Age category'] == age_category]

    total_high_amplitude_artifacts = np.sum(np.array(dfArtifactResults_sub.iloc[:, 3:]))
    total_epochs = np.sum(dfArtifactResults_sub['Total number of epochs'])
    seriesArtifactResults_per_age_category = \
        pd.Series({'Impedance artifacts (total number)': np.sum(dfArtifactResults_sub['n_impedance_artifacts']),
                   'High amplitude artifacts (total number)': total_high_amplitude_artifacts,
                   'Total number of epochs': total_epochs,

                   'EOG ROC total': np.sum(dfArtifactResults_sub['n_EOGROC_artifacts']),
                   'EOG ROC prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EOGROC_artifacts']) / total_high_amplitude_artifacts,
                   'EOG ROC prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EOGROC_artifacts']) / total_epochs,

                   'EOG LOC total': np.sum(dfArtifactResults_sub['n_EOGLOC_artifacts']),
                   'EOG LOC prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EOGLOC_artifacts']) / total_high_amplitude_artifacts,
                   'EOG LOC prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EOGLOC_artifacts']) / total_epochs,

                   'EMG Chin total': np.sum(dfArtifactResults_sub['n_EMG_artifacts']),
                   'EMG Chin prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EMG_artifacts']) / total_high_amplitude_artifacts,
                   'EMG Chin prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EMG_artifacts']) / total_epochs,

                   'EEG F3 total': np.sum(dfArtifactResults_sub['n_EEGF3_artifacts']),
                   'EEG F3 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGF3_artifacts']) / total_high_amplitude_artifacts,
                   'EEG F3 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGF3_artifacts']) / total_epochs,

                   'EEG F4 total': np.sum(dfArtifactResults_sub['n_EEGF4_artifacts']),
                   'EEG F4 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGF4_artifacts']) / total_high_amplitude_artifacts,
                   'EEG F4 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGF4_artifacts']) / total_epochs,

                   'EEG C3 total': np.sum(dfArtifactResults_sub['n_EEGC3_artifacts']),
                   'EEG C3 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGC3_artifacts']) / total_high_amplitude_artifacts,
                   'EEG C3 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGC3_artifacts']) / total_epochs,

                   'EEG C4 total': np.sum(dfArtifactResults_sub['n_EEGC4_artifacts']),
                   'EEG C4 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGC4_artifacts']) / total_high_amplitude_artifacts,
                   'EEG C4 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGC4_artifacts']) / total_epochs,

                   'EEG O1 total': np.sum(dfArtifactResults_sub['n_EEGO1_artifacts']),
                   'EEG O1 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGO1_artifacts']) / total_high_amplitude_artifacts,
                   'EEG O1 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGO1_artifacts']) / total_epochs,

                   'EEG O2 total': np.sum(dfArtifactResults_sub['n_EEGO2_artifacts']),
                   'EEG O2 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGO2_artifacts']) / total_high_amplitude_artifacts,
                   'EEG O2 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGO2_artifacts']) / total_epochs,

                   'EEG A1 total': np.sum(dfArtifactResults_sub['n_EEGA1_artifacts']),
                   'EEG A1 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGA1_artifacts']) / total_high_amplitude_artifacts,
                   'EEG A1 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGA1_artifacts']) / total_epochs,

                   'EEG A2 total': np.sum(dfArtifactResults_sub['n_EEGA2_artifacts']),
                   'EEG A2 prc of impedance artifacts': np.sum(
                       dfArtifactResults_sub['n_EEGA2_artifacts']) / total_high_amplitude_artifacts,
                   'EEG A2 prc of epochs': np.sum(
                       dfArtifactResults_sub['n_EEGA2_artifacts']) / total_epochs,
                   })

    dfArtifactResults_per_age_category[age_category] = seriesArtifactResults_per_age_category

writer = pd.ExcelWriter(os.path.join(results_file_path, artifact_results_file_name_per_age_category))
dfArtifactResults_per_age_category.to_excel(writer)
writer.save()

