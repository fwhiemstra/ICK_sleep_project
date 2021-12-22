"""
This file is used to calculate sleep statistics.
Floor Hiemstra, June 2021
"""

import os
import pandas as pd
import numpy as np

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
sleep_statistics_file_path = r'E:\ICK slaap project\10_Patient+Data_Characteristics'

# feature_data_file_name = 'PSG001_FeatureData.xlsx'
# featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
epoch_size = 30  # seconds

# %% Sleep statistics per recording
sleep_statistics_file_name_per_recording = 'PICU_SleepStatisctics_per_recording.xlsx'
key = []
age_category = []
gender = []
cognitive_impairment = []
total_length_hours = []
total_length_epochs = []
total_sleep_time_hours = []
total_sleep_time_epochs = []
rem_epochs_all = []
n1_epochs_all = []
n2_epochs_all = []
n3_epochs_all = []
n4_epochs_all = []
n_epochs_all = []
total_sleep_epochs_all = []
prc_rem = []
prc_n1 = []
prc_n2 = []
prc_n3 = []
prc_n4 = []
prc_n = []
sleep_efficiency = []

for feature_data_file_name in os.listdir(feature_data_file_path):
    if feature_data_file_name.endswith(".xlsx"):
        featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

        # General information
        key.append(featureData["Patient key"][0])
        age_category.append(featureData["Age category"][0])
        #gender.append(featureData["Gender"][0])
        #cognitive_impairment.append(featureData["Cognitive impairment"][0])

        # Number of epochs
        rem_epochs = len(featureData[featureData["Sleep stage labels"] == 'Sleep stage R'])
        rem_epochs_all.append(rem_epochs)
        n1_epochs = len(featureData[featureData["Sleep stage labels"] == 'Sleep stage N1'])
        n1_epochs_all.append(n1_epochs)
        n2_epochs = len(featureData[featureData["Sleep stage labels"] == 'Sleep stage N2'])
        n2_epochs_all.append(n2_epochs)
        n3_epochs = len(featureData[featureData["Sleep stage labels"] == 'Sleep stage N3'])
        n3_epochs_all.append(n3_epochs)
        n4_epochs = len(featureData[featureData["Sleep stage labels"] == 'Sleep stage N4'])
        n4_epochs_all.append(n4_epochs)
        n_epochs = len(featureData[featureData["Sleep stage labels"] == 'Sleep stage N'])
        n_epochs_all.append(n_epochs)
        total_sleep_epochs = rem_epochs + n1_epochs + n2_epochs + n3_epochs + n4_epochs + n_epochs
        total_sleep_epochs_all.append(total_sleep_epochs)

        # Length of PSG recording [hours]
        total_length_hours.append(len(featureData["Sleep stage labels"]) * epoch_size / 3600)
        total_length_epochs.append(len(featureData["Sleep stage labels"]))

        # Total sleep time [hours]
        total_sleep_time_hours.append(total_sleep_epochs * epoch_size / 3600)
        total_sleep_time_epochs.append(total_sleep_epochs)

        # % per sleep stage
        prc_rem.append(100*rem_epochs / total_sleep_epochs)
        prc_n1.append(100*n1_epochs / total_sleep_epochs)
        prc_n2.append(100*n2_epochs / total_sleep_epochs)
        prc_n3.append(100*n3_epochs / total_sleep_epochs)
        prc_n4.append(100*n4_epochs / total_sleep_epochs)
        prc_n.append(100*n_epochs / total_sleep_epochs)

        # Sleep efficiency [%]
        sleep_efficiency.append(total_sleep_epochs/len(featureData["Sleep stage labels"]))

        print('Sleep statistic calculation of ', feature_data_file_name, ' completed')

# Convert to dataframe
dfSleepStatistics = pd.DataFrame({'Patient key': key,
                                  'Age category': age_category,
                                  #'Gender': gender,
                                  #'Cognitive impairment': cognitive_impairment,
                                  'PSG recording length [hours]': total_length_hours,
                                  'PSG recording length [epochs]': total_length_epochs,
                                  'Total sleep time [hours]': total_sleep_time_hours,
                                  'Total sleep time [epochs]': total_sleep_time_epochs,
                                  '%REM': prc_rem,
                                  'REM epochs': rem_epochs_all,
                                  '%N1': prc_n1,
                                  'N1 epochs': n1_epochs_all,
                                  '%N2': prc_n2,
                                  'N2 epochs': n2_epochs_all,
                                  '%N3': prc_n3,
                                  'N3 epochs': n3_epochs_all,
                                  '%N4': prc_n4,
                                  'N4 epochs': n4_epochs_all,
                                  '%N': prc_n,
                                  'N epochs': n_epochs_all,
                                  'Sleep efficiency': sleep_efficiency
                                  })

writer = pd.ExcelWriter(os.path.join(sleep_statistics_file_path, sleep_statistics_file_name_per_recording))
dfSleepStatistics.to_excel(writer)
writer.save()

#%% Sleep statistics per age category
sleep_statistics_file_name_per_category = 'PICU_SleepStatistics_ALL.xlsx'
dfSleepStatistics = pd.read_excel(os.path.join(sleep_statistics_file_path, sleep_statistics_file_name_per_recording))
dfSleepStatistics_per_age_category = pd.DataFrame([])

age_categories = ['all']#, 'all>2 months', 'all>6 months', '0-2 months', '2-6 months', '6-12 months', '1-3 years',
                  #'3-5 years', '5-9 years', '9-13 years', '13-18 years']
for age_category in age_categories:
    if age_category == 'all':
        dfSleepStatistics_sub = dfSleepStatistics
    elif age_category == 'all>2 months':
        dfSleepStatistics_sub = dfSleepStatistics[dfSleepStatistics['Age category'].isin(['2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years'])]
    elif age_category == 'all>6 months':
        dfSleepStatistics_sub = dfSleepStatistics[dfSleepStatistics['Age category'].isin(['6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years'])]
    else:
        dfSleepStatistics_sub = dfSleepStatistics[dfSleepStatistics['Age category'] == age_category]

    mean_recording_length_hours = np.mean(dfSleepStatistics_sub['PSG recording length [hours]'])
    std_recording_length_hours = np.std(dfSleepStatistics_sub['PSG recording length [hours]'])
    mean_total_sleep_time_hours = np.mean(dfSleepStatistics_sub['Total sleep time [hours]'])
    std_total_sleep_time_hours = np.std(dfSleepStatistics_sub['Total sleep time [hours]'])

    seriesSleepStatistics_per_age_category = \
        pd.Series({'total_recoding_length_hours': np.sum(dfSleepStatistics_sub['PSG recording length [hours]']),
                        'total_recoding_length_hs': str(int(np.sum(dfSleepStatistics_sub['PSG recording length [hours]']))) + ':' + str(round((np.sum(dfSleepStatistics_sub['PSG recording length [hours]'])*60) % 60)),
                        'total_recording_length_epochs' : np.sum(dfSleepStatistics_sub['PSG recording length [epochs]']),

                        'mean_recording_length_hours': mean_recording_length_hours,
                        'std_recording_length_hours': std_recording_length_hours,
                        'mean_recordings_length_hs': str(int(mean_recording_length_hours)) + ':' + str(round((mean_recording_length_hours*60) % 60)),
                        'std_recording_length_hs': str(int(std_recording_length_hours)) + ':' + str(round((std_recording_length_hours*60) % 60)),
                        'mean_recording_length_epochs': np.mean(dfSleepStatistics_sub['PSG recording length [epochs]']),
                        'std_recording_length_epochs': np.std(dfSleepStatistics_sub['PSG recording length [epochs]']),

                        'mean_total_sleep_time_hours': mean_total_sleep_time_hours,
                        'std_total_sleep_time_hours': std_total_sleep_time_hours,
                        'mean_total_sleep_time_hs': str(int(mean_total_sleep_time_hours)) + ':' + str(round((mean_total_sleep_time_hours*60) % 60)),
                        'STD_total_sleep_time_hs': str(int(std_total_sleep_time_hours)) + ':' + str(round((std_total_sleep_time_hours * 60) % 60)),
                        'mean_total_sleep_time_epochs': np.mean(dfSleepStatistics_sub['Total sleep time [epochs]']),
                        'std_total_sleep_time_epochs': np.std(dfSleepStatistics_sub['Total sleep time [epochs]']),

                        'mean_rem': np.mean(dfSleepStatistics_sub['%REM']),
                        'std_rem': np.std(dfSleepStatistics_sub['%REM']),
                        'total_rem_epochs': np.sum(dfSleepStatistics_sub['REM epochs']),

                        'mean_n1': np.mean(dfSleepStatistics_sub['%N1']),
                        'std_n1': np.std(dfSleepStatistics_sub['%N1']),
                        'total_n1_epochs': np.sum(dfSleepStatistics_sub['N1 epochs']),

                        'mean_n2': np.mean(dfSleepStatistics_sub['%N2']),
                        'std_n2': np.std(dfSleepStatistics_sub['%N2']),
                        'total_n2_epochs': np.sum(dfSleepStatistics_sub['N2 epochs']),

                        'mean_n3': np.mean(dfSleepStatistics_sub['%N3']),
                        'std_n3': np.std(dfSleepStatistics_sub['%N3']),
                        'total_n3_epochs': np.sum(dfSleepStatistics_sub['N3 epochs']),

                        'mean_n': np.mean(dfSleepStatistics_sub['%N']),
                        'std_n': np.std(dfSleepStatistics_sub['%N']),
                        'total_n_epochs': np.sum(dfSleepStatistics_sub['N epochs'])})

    dfSleepStatistics_per_age_category[age_category] = seriesSleepStatistics_per_age_category

writer = pd.ExcelWriter(os.path.join(sleep_statistics_file_path, sleep_statistics_file_name_per_category))
dfSleepStatistics_per_age_category.to_excel(writer)
writer.save()
