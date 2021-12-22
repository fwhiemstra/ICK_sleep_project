"""
Feature scaling. This file reads feature files, scales features and writes the scaled features to a new file
Floor Hiemstra, July 2021
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
scaled_feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_4_PICU_FeatureData_SCALED'
feature_data_file_names = os.listdir(feature_data_file_path)
feature_data_file_names = ['PICU007_FeatureData - EEG F4-C4.xlsx']

for feature_data_file_name in feature_data_file_names:
    try:
        featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name), index_col=0)
        scaler = StandardScaler()
        scaled_featureData = featureData.copy()
        if feature_data_file_name.startswith('PICU'):
            column_nr = 16
        elif feature_data_file_name.startswith('PSG'):
            column_nr = 18
        for i in range(0, len(featureData.iloc[:, column_nr:].columns)):
            scaled_featureData.iloc[:, i+column_nr] = scaler.fit_transform(np.array(featureData.iloc[:, i+column_nr]).reshape(-1, 1))

        scaled_feature_file_name = feature_data_file_name.replace('.xlsx', '_scaled.xlsx')

        writer = pd.ExcelWriter(os.path.join(scaled_feature_data_file_path, scaled_feature_file_name))
        scaled_featureData.to_excel(writer)
        writer.save()

        print('Feature scaling of ' + feature_data_file_name + ' completed.')

    except:
        print('Feature scaling of ' + feature_data_file_name + ' failed.')


