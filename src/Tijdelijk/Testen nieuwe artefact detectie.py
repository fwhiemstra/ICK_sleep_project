import pyedflib
import os
import math
import numpy as np
import pandas as pd

#%% Load EDF file
preprocessed_file_path = r'E:\ICK slaap project\3_Preprocessed_data'
edf_file_name = 'PSG072p.edf'
edf_file = os.path.join(preprocessed_file_path, edf_file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)
fs = signal_headers[4].get("sample_rate")
epoch_size = 30 * fs

#%% Function
def artifact_labelling(signal_headers, signals, fs, epoch_size):
    """
    Detects artifacts in the signals and adds labels to the epochs in which an artifact
    is detected. Impedance artifacts are labelled as 'Impedance artifact', high-amplitude
    artifacts (movement, 50-Hz interference) are scored as '[EEG channel containing the
    artifact] -'artifact'.

    :param signal_headers:
        list - containing dictionaries with signal header (information)
        (output from pyedflib.highlevel.read_edf)
    :param signals:
        ndarray - containing raw signals
        (output from pyedflib.highlevel.read_edf)
    :param fs:
        int - sample frequency (= signal.headers[0].get("sample_rate"))
    :param epoch_size:
        int - epoch_size in samples (=epoch_size_s*fs)

    :return: dfArtifact
        pd DataFrame - Dataframe containing the artifact labels for each epoch. 1: artifact,
        0: no artifact
    """

    # Function variables
    ampl_thres_eeg = 450
    ampl_thres_eog = 1000
    ampl_thres_emg = 1000
    unipolar_psg_signals = ['EOG ROC', 'EOG LOC', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1', 'EEG O2', 'EEG A1', 'EEG A2']

    ## Step 1: Window all unipolar signals
    signal_labels = []
    epochs_unipolar = []
    for i in range(0, len(signal_headers)):  # Create list with signal labels
        signal_labels.append(signal_headers[i].get("label"))

    for i in unipolar_psg_signals:
        y = signals[signal_labels.index(i)]
        epochs = []
        for j in range(math.floor(len(y) / epoch_size) - 1):
            k = j * epoch_size
            epochs.append(y[k: k + epoch_size])
        epochs_unipolar.append(epochs)
        # epochs_unipolar contains 8 lists (each list is a signal) with each [number of epochs]
        # lists containing an array with epoch signal

    ## Step 2: Detect impedance artifacts
    impedance_artifact_labels = np.zeros(len(epochs_unipolar[0]))

    # Impedance artifacts
    for signal in range(0, len(epochs_unipolar)):  # loop over each signal
        for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
            count = 0
            for j in range(0, len(epochs_unipolar[signal][i]),
                           int(fs / 2)):  # check every half second if sample is equal to zero
                if epochs_unipolar[signal][i][j] == 0:
                    count = count + 1
                    if count > 4:  # if sample is equal to zero for more than 2 seconds
                        impedance_artifact_labels[i] = 1
                        break  # skip if one impedance period of 2 seconds is present
                    if impedance_artifact_labels[i] == 1: break  # skip if one impedance period is present in the epoch
                else:
                    if impedance_artifact_labels[i] == 1: break  # skip if one impedance period is present in the epoch
                    count = 0

    ### Step 3: Detect high amplitude artifacts
    high_amplitude_artifact_labels = np.zeros((len(epochs_unipolar[0]), len(epochs_unipolar)))

    for signal in range(0, len(epochs_unipolar)):  # loop over each signal
        if signal <= 1:
            ampl_thres = ampl_thres_eog
        elif signal == 2:
            ampl_thres = ampl_thres_emg
        elif signal > 2:
            ampl_thres = ampl_thres_eeg
        for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
            if np.mean(np.absolute(epochs_unipolar[signal][i])) > ampl_thres or \
                    np.mean(np.absolute(epochs_unipolar[signal][i])) < -ampl_thres:
                high_amplitude_artifact_labels[i, signal] = 1

    ### Step 4: Create dataframe with artifact labels
    high_amplitude_artifact_column_names = []
    for item in range(0, len(unipolar_psg_signals)):
        high_amplitude_artifact_column_names.append(unipolar_psg_signals[item] + ' - Artifact')

    dfArtifact = pd.DataFrame({'Impedance artifact': impedance_artifact_labels,
                               high_amplitude_artifact_column_names[0]: high_amplitude_artifact_labels[:, 0],
                               high_amplitude_artifact_column_names[1]: high_amplitude_artifact_labels[:, 1],
                               high_amplitude_artifact_column_names[2]: high_amplitude_artifact_labels[:, 2],
                               high_amplitude_artifact_column_names[3]: high_amplitude_artifact_labels[:, 3],
                               high_amplitude_artifact_column_names[4]: high_amplitude_artifact_labels[:, 4],
                               high_amplitude_artifact_column_names[5]: high_amplitude_artifact_labels[:, 5],
                               high_amplitude_artifact_column_names[6]: high_amplitude_artifact_labels[:, 6],
                               high_amplitude_artifact_column_names[7]: high_amplitude_artifact_labels[:, 7],
                               })

    return dfArtifact

#%%
dfArtifact = artifact_labelling(signal_headers, signals, fs, epoch_size)

#%% Evaluate EOG and EMG artifact
preprocessed_file_path = r'E:\ICK slaap project\3_Preprocessed_data'
edf_file_name = 'PSG026p.edf'
edf_file = os.path.join(preprocessed_file_path, edf_file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)
fs = signal_headers[4].get("sample_rate")
epoch_size = 30 * fs

signal_labels = []
epochs_unipolar = []
for i in range(0, len(signal_headers)):  # Create list with signal labels
    signal_labels.append(signal_headers[i].get("label"))

for i in ['EOG ROC', 'EOG LOC', 'EMG EMG Chin']:
    y = signals[signal_labels.index(i)]
    epochs = []
    for j in range(math.floor(len(y) / epoch_size) - 1):
        k = j * epoch_size
        epochs.append(y[k: k + epoch_size])
    epochs_unipolar.append(epochs)
    # epochs_unipolar contains 8 lists (each list is a signal) with each [number of epochs]
    # lists containing an array with epoch signal

import matplotlib.pyplot as plt

signal = 0
mean_abs_amplitude = []
for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
    mean_abs_amplitude.append(np.mean(np.absolute(epochs_unipolar[signal][i])))
plt.figure(1)
plt.plot(mean_abs_amplitude)

signal = 1
mean_abs_amplitude = []
for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
    mean_abs_amplitude.append(np.mean(np.absolute(epochs_unipolar[signal][i])))
plt.figure(2)
plt.plot(mean_abs_amplitude)
