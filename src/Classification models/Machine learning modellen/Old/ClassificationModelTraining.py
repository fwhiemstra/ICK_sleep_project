"""
This file is used for classification model training
Floor Hiemstra, July 2021
"""

# %% Import packages
import os
import pandas as pd
from src.Functions import add_new_state_labels
import pickle

#%% Set variables
# Select classifier
classifier = 'DT'

# File paths and names for training data
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
model_result_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment'

# Set data parameters
include_age_categories = ['13-18 years']
number_of_stages = 3  # 2 = [Wake, Sleep], 3 = [Wake, NSWS, SWS], 4 = [Wake, REM, NSWS, SWS], 5 = [Wake, REM, N1, N2, N3]
remove_artifacts = 1

#%%
"""
##### 1: LOAD TRAINING DATA #####
"""
# List feature data file names
patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
keys = []
for age_category in include_age_categories:
    keys.extend(patientData[patientData['Age category'] == age_category]['Key'])
feature_data_file_names = [key + '_FeatureData.xlsx' for key in keys]
feature_data_file_names = feature_data_file_names[0]
### Read feature data files in list and store as one dataframe
allFeatureData = []
for feature_data_file_name in feature_data_file_names:
    if feature_data_file_name in os.listdir(feature_data_file_path):
        featureData_raw = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

        # Add new sleep stage labels
        featureData_new_labels, number_of_stages_key = add_new_state_labels(featureData_raw, number_of_stages)
        featureData = featureData_new_labels

        # Remove artefacts
        if remove_artifacts == 1:
            featureData_clean = featureData_new_labels[featureData['Artifact'] == 0]
            featureData = featureData_clean

        # Create dataframes
        if feature_data_file_name == feature_data_file_names[0]:
            allFeatureData = featureData
        else:
            allFeatureData = allFeatureData.append(featureData)

labels_cat = allFeatureData[number_of_stages_key]
labels_lin = allFeatureData[number_of_stages_key + '- linear']

#%%
"""
##### 2: TRAIN CLASSIFIER #####
"""
if classifier == 'DT':
    from sklearn.tree import DecisionTreeClassifier
    optimal_feature_subset = ['EEG F3: DWT cD1 std', 'EEG F3: Signal sum',
       'EEG F3: Abs mean amplitude', 'EEG F3: Rel Gamma power',
       'EEG F3: Alpha/theta ratio']
    optimal_max_depth = 3
    clf = DecisionTreeClassifier(criterion='gini', max_depth=optimal_max_depth)
    labels = labels_cat
    features = allFeatureData[optimal_feature_subset]

clf.fit(features, labels)

#%%
"""
##### 3: SAVE CLASSIFIER #####
"""
model_file_name = classifier + '_model.sav'
pickle_dump_dict = {"fitted_model": clf, 'parameters': optimal_feature_subset}
pickle.dump(pickle_dump_dict, open(os.path.join(model_result_file_path, model_file_name), 'wb'))


