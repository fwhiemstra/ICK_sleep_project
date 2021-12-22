"""
Calculate features and create feature data sheet per recording
Feature datasheet contains:
- Sleep stage labels
- Artifact labels
- Features for all EOG, EMG and EEG leads
Floor Hiemstra, May 2021
"""

import os
import pyedflib
import src.Functions
import pandas as pd
import numpy as np

preprocessed_file_path = r'E:\ICK slaap project\3_Preprocessed_data'
# Set action to one of the following options: 'ALL' or 'SINGLE' to convert all files in the
# directory or to process a single file.
action = 'SINGLE'
file_name = 'PICU004p.edf'  # Leave empty ([]) if all files in directory should be anonymized
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PICU_PatientData.xlsx'
nr_signal = 0  # signal to use as example signal (to determine sample rate and signal length)
epoch_size_s = 30  # epoch size in seconds

patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))

if action == 'ALL':
    file_list = os.listdir(preprocessed_file_path)  # Process all files in directory
elif action == 'SINGLE':
    file_list = [file_name]

for file in file_list:
    try:
        feature_data_file_name = file.replace('p.edf', '_FeatureData.xlsx')
        if file.endswith(".edf") or file.endswith(".EDF"):
            edf_file = os.path.join(preprocessed_file_path, file)
            signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

            # Sleep stage labelling
            fs = signal_headers[nr_signal].get("sample_rate")
            epoch_size = epoch_size_s * fs
            annotations = header["annotations"]
            signal_length = len(signals[nr_signal])
            sleep_stage_labels = src.Functions.stage_labelling(annotations, signal_length, fs, epoch_size)

            # Artefact labelling
            dfArtifact = src.Functions.artifact_labelling(signal_headers, signals, fs, epoch_size)

            # Feature calculation
            dfFeatureData = src.Functions.FeatureCalculation(signals, signal_headers, fs, epoch_size)

            # Create dataframe and fill will sleep stage and artifact labels
            key = file.replace('p.edf', '')
            if feature_data_file_name.startswith('PSG'):
                dfPatientInfo_SleepLabels = pd.DataFrame({'Patient key': key,
                                              'Age category': patientData["Age category"][np.argmax(patientData["Key"] == key)],
                                              'Gender': patientData["Gender"][np.argmax(patientData["Key"] == key)],
                                              'Cognitive impairment': patientData["Potential neurophysiological EEG interference by cognitive impairment"][np.argmax(patientData["Key"] == key)],
                                              'Epilepsy': patientData["Epilepsy"][np.argmax(patientData["Key"] == key)],
                                              'Sleep stage labels': sleep_stage_labels,
                                              })
            if feature_data_file_name.startswith('PICU'):
                dfPatientInfo_SleepLabels = pd.DataFrame({'Patient key': key,
                                                          'Age category': patientData["Age category"][
                                                              np.argmax(patientData["Key"] == key)],
                                                          'Gender': patientData["Gender"][
                                                              np.argmax(patientData["Key"] == key)],
                                                          'Sleep stage labels': sleep_stage_labels,
                                                          })

            dfPatientInfo_SleepLabels_Artifacts = dfPatientInfo_SleepLabels.join(dfArtifact)
            dfFeatureDataAll = dfPatientInfo_SleepLabels_Artifacts.join(dfFeatureData)

            # Remove infinity values
            dfFeatureDataAll.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Write to excel
            writer = pd.ExcelWriter(os.path.join(feature_data_file_path, feature_data_file_name))
            dfFeatureDataAll.to_excel(writer)
            writer.save()

            print('Conversion of', feature_data_file_name, 'sheet completed')

        else:
            print(file, ' is not an EDF file')

    except:
        print(file + ': Error! in feature calculation')
        raise
