# testPre_processing.py
# This file is used to test the pre-processing functions

import pyedflib
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#%% Load file
file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\2_Raw_data'
file_name = '1410419_26112020.EDF'
new_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\3_Preprocessed_data'
edf_file = os.path.join(file_path, file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)
print('Finished')

# %% Define new file name
file, ext = os.path.splitext(file_name)
new_file_name = file + 'p' + ext
new_file = os.path.join(new_file_path, new_file_name)

# %% Remove patient identifying information from file
del header['technician'], header['patientname'], header['patientcode'], header['recording_additional'], header[
    'patient_additional'], header['equipment'], header['admincode'], header['gender'], header['birthdate']

#%% Remove signals
correction = 0 # add correction to prevent list index out of range after deleting from list
for i in range(0, len(signal_headers)):
    i = i - correction
    if signal_headers[i]["label"] in {'ECG', 'Resp Thorax', 'Resp Abdomen', 'EMG Tib R', 'EMG Tib L',
                                      'EMG Tib. R', 'EMG Tib. L', 'SaO2 SaO2', 'SaO2 Heartrate',
                                      'SaO2 Pulse', 'SaO2 Status', 'Resp Comb. flow', 'EEG Fp1',
                                      'EEG Fp2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5',
                                      'EEG T6', 'EEG P3', 'EEG P4', 'EEG Pz', 'EEG Cz', 'SaO2 PCO2',
                                      'Sound Snore', 'Resp Nasal flow', 'BodyPos Bodypos.', 'EEG Fz',
                                      'Device Battery'}:
        del signal_headers[i], signals[i]
        correction = correction + 1

#%% Add signals
# Make list of labels to find index of signals
signal_labels = []
for i in range(0, len(signal_headers)):
    signal_labels.append(signal_headers[i]["label"])

# EOG ROC-LOC signal
new_signal = signals[signal_labels.index('EOG ROC')] - signals[signal_labels.index('EOG LOC')]
new_signal_header = pyedflib.highlevel.make_signal_header('EOG ROC-LOC', dimension='uV', sample_rate=250,
                                                          physical_min=-5000, physical_max=5000,
                                                          digital_min=-65535,
                                                          digital_max=65535, transducer='',
                                                          prefiler='')
signals.append(new_signal)
signal_headers.append(new_signal_header)

## EEG signals
eegsignal_leads = [['EEG F3-F4', 'EEG F3', 'EEG F4'],
                ['EEG F3-C3', 'EEG F3', 'EEG C3'],
                ['EEG F3-C4', 'EEG F3', 'EEG C4'],
                ['EEG F3-O1', 'EEG F3', 'EEG O1'],
                ['EEG F3-O2', 'EEG F3', 'EEG O2'],
                ['EEG F3-A1', 'EEG F3', 'EEG A1'],
                ['EEG F3-A2', 'EEG F3', 'EEG A2'],
                ['EEG C3-C4', 'EEG C3', 'EEG C4'],
                ['EEG C3-O1', 'EEG C3', 'EEG O1'],
                ['EEG C3-O2', 'EEG C3', 'EEG O2'],
                ['EEG C3-A1', 'EEG C3', 'EEG A1'],
                ['EEG C3-A2', 'EEG C3', 'EEG A2'],
                ['EEG O1-O2', 'EEG O1', 'EEG O2'],
                ['EEG O1-A1', 'EEG O1', 'EEG A1'],
                ['EEG O1-A2', 'EEG O1', 'EEG A2']]

# EEG header information
sample_rate = 250
physical_min = -5000
physical_max = 5000
digital_min = -65535
digital_max = 65535
prefilter = ''
transducer = ''

# EEG frequency filter
n = 16 # Filter order
fco_low = 0.5 # lower cutoff frequency
fco_up = 48 # upper cutoff frequency
fs = signal_headers[signal_labels.index("EEG F3")].get("sample_rate")
sos = signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)

for i in range(0, len(eegsignal_leads)):
    new_signal = signals[signal_labels.index(eegsignal_leads[i][1])] - signals[signal_labels.index(eegsignal_leads[i][2])]
    new_signal_header = pyedflib.highlevel.make_signal_header(eegsignal_leads[i][0], dimension='uV', sample_rate=sample_rate,
                       physical_min=physical_min, physical_max=physical_max, digital_min=digital_min,
                       digital_max=digital_max, transducer=transducer, prefiler=prefilter)

    # Frequency filter EEG lead signals
    #new_signal = signal.sosfilt(sos, new_signal)

    # Add new signals + header to signals, signal_headers list
    signals.append(new_signal)
    signal_headers.append(new_signal_header)

#%% Filter monopolar EEG signals
#unipolar_eeg = ['EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1', 'EEG O2']

#for i in range(0, len(unipolar_eeg)):
    ## signals[signal_labels.index(unipolar_eeg(i))] = signal.sosfilt(sos, signals[signal_labels.index(unipolar_eeg(i))])

#%% Convert duration in seconds in annotations from byte to float type
for i in range(0, len(header['annotations'])):
    header['annotations'][i][1] = float(header['annotations'][i][1])

#%% Write new edf file
pyedflib.highlevel.write_edf(new_file, signals, signal_headers, header, digital=True)
print('Preprocessing of', new_file_name, 'completed')



