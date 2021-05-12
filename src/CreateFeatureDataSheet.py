"""

Floor Hiemstra, May 2021
"""

import os
import pyedflib
import Functions

file_name =  'EDFtest2p.edf'
file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\3_Preprocessed_data'
#file_name = [] # Leave empty ([]) if all files in directory should be anonymized
#file_path = r'E:\ICK slaap project\PSG\2_Raw_data'
#new_file_path = r'E:\ICK slaap project\PSG\3_Preprocessed_data'

nr_signal = 0 # signal to use as example signal (to determine sample rate and signal length)
epoch_size_s = 30 # epoch size in seconds
unipolar_eegsignals = ['EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1', 'EEG O2', 'EEG A1', 'EEG A2']
include_signals = ['EOG ROC', 'EOG LOC', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4',
                   'EEG O1', 'EEG O2', 'EEG A1', 'EEG A2', 'EOG ROC-LOC', 'EEG F3-F4', 'EEG F3-C3',
                   'EEG F3-C4', 'EEG F3-O1', 'EEG F3-O2', 'EEG F3-A1', 'EEG F3-A2', 'EEG C3-C4',
                   'EEG C3-O1', 'EEG C3-O2', 'EEG C3-A1', 'EEG C3-A2', 'EEG O1-O2', 'EEG O1-A1',
                   'EEG O1-A2']


# Set action to one of the following options: 'ALL' or 'SINGLE' to convert all files in the
# directory or to process a single file.
action = 'SINGLE'

if action == 'ALL':
    # Process all files in directory
    for file in os.listdir(file_path):
         file_name = os.fsdecode(file)
         if file_name.endswith(".edf") or file_name.endswith(".EDF"):
            print('Info volgt')

         else:
             print(file_name, ' is not an EDF file')

elif action == 'SINGLE':
    edf_file = os.path.join(file_path, file_name)
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

    # Sleep stage labelling
    fs = signal_headers[nr_signal].get("sample_rate")
    epoch_size = epoch_size_s * fs
    annotations = header["annotations"]
    signal_length = len(signals[nr_signal])
    sleep_stage_labels = Functions.stage_labelling(annotations, signal_length, fs, epoch_size)

    # Artefact labelling
    impArtifact_labels, movArtifact_labels, artifact_labels = Functions.artifact_labelling(signal_headers, signals, unipolar_eegsignals, fs, epoch_size)

    # Feature calculation


    # Create dataframe and write to excel
