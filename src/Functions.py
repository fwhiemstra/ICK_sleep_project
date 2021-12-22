"""
This file contains the functions used in the ICK sleep project.
"""

import pyedflib
import os
import numpy as np
import scipy
from scipy import signal
import math
import pandas as pd
import matplotlib.pyplot as plt
import eeglib
import pywt
import time

def edf_preprocessing(file_name, file_path, new_file_path):
    """ Preprocess an EDF file by:
        - Anonymization (i.e. removing all header information that is patient specific)
        - Removal of unnecessary signals
        - Addition of EEG leads
        (- Bandpass filtering of EEG signals) --> gebeurt in volgende stap

        Parameters
        ----------
        file_name: str
            Filename of the to be preprocessed EDF file
        file_path: str
            File path of the to be preprocessed EDF file
        new_file_path: str
            File path of the preprocessed EDF file

        Returns
        -------
        New anonymous and preprocessed EDF file is created in [new_file_path].
        The new EDF file has the same extension as the input file and is named: file_name + p
        (for example: PSG001 --> PSG001p).
        If succesful, the following statement is printed: 'Preprocessing of [new_file_name] completed')

        """

    # Read edf file
    edf_file = os.path.join(file_path, file_name)
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

    # Define new file name
    file, ext = os.path.splitext(file_name)
    new_file_name = file + 'p' + ext
    new_file = os.path.join(new_file_path, new_file_name)

    # Remove patient identifying information from file
    del header['technician'], header['patientname'], header['patientcode'], header['recording_additional'], header[
        'patient_additional'], header['equipment'], header['admincode'], header['gender'], header['birthdate']

    # Remove signals
    correction = 0  # add correction to prevent list index out of range after deleting from list
    for i in range(0, len(signal_headers)):
        if signal_headers[i]["label"] == 'EEG M1':
            signal_headers[i]["label"] = 'EEG A1'
        elif signal_headers[i]["label"] == 'EEG M2':
            signal_headers[i]["label"] = 'EEG A2'
        elif signal_headers[i]["label"] == 'EOG E1':
            signal_headers[i]["label"] = 'EOG ROC'
        elif signal_headers[i]["label"] == 'EOG E2':
            signal_headers[i]["label"] = 'EOG LOC'
        elif signal_headers[i]["label"] == 'EMG Chin1':
            signal_headers[i]["label"] = 'EMG EMG Chin'

        i = i - correction
        if signal_headers[i]["label"] not in {'EOG ROC', 'EOG LOC', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3',
                                              'EEG C4', 'EEG O1', 'EEG O2', 'EEG A1', 'EEG A2'}:
            del signal_headers[i], signals[i]
            correction = correction + 1

    # Add signals
    # Make list of labels to find index of signals
    signal_labels = []
    for i in range(0, len(signal_headers)):
        signal_labels.append(signal_headers[i]["label"])

    # EOG ROC-LOC signal
    new_signal = signals[signal_labels.index('EOG ROC')] - signals[signal_labels.index('EOG LOC')]
    new_signal_header = pyedflib.highlevel.make_signal_header('EOG ROC-LOC', dimension='uV',
                                                              sample_rate=signal_headers[0].get('sample_rate'),
                                                              physical_min=-5000, physical_max=5000,
                                                              digital_min=-65535,
                                                              digital_max=65535, transducer='',
                                                              prefiler='')
    signals.append(new_signal)
    signal_headers.append(new_signal_header)

    # EEG signals
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
    sample_rate = signal_headers[3].get('sample_rate')
    physical_min = -5000
    physical_max = 5000
    digital_min = -65535
    digital_max = 65535
    prefilter = ''
    transducer = ''

    # EEG frequency filter
    # n = 16  # Filter order
    # fco_low = 0.5  # lower cutoff frequency
    # fco_up = 48  # upper cutoff frequency
    # fs = signal_headers[signal_labels.index("EEG F3")].get("sample_rate")
    # sos = signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)

    for i in range(0, len(eegsignal_leads)):
        new_signal = signals[signal_labels.index(eegsignal_leads[i][1])] - signals[
            signal_labels.index(eegsignal_leads[i][2])]
        new_signal_header = pyedflib.highlevel.make_signal_header(eegsignal_leads[i][0], dimension='uV',
                                                                  sample_rate=sample_rate,
                                                                  physical_min=physical_min, physical_max=physical_max,
                                                                  digital_min=digital_min,
                                                                  digital_max=digital_max, transducer=transducer,
                                                                  prefiler=prefilter)

        # Frequency filter EEG lead signals
        # new_signal = signal.sosfilt(sos, new_signal)

        # Add new signals + header to signals, signal_headers list
        signals.append(new_signal)
        signal_headers.append(new_signal_header)

    # Filter monopolar EEG signals
    # unipolar_eeg = ['EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1', 'EEG O2']
    # for i in unipolar_EEG:
    # signals[signal_labels.index(i)] = signal.sosfilt(sos, signals[signal_labels.index(i)])

    # Convert duration in seconds in annotations from byte to float type
    for i in range(0, len(header['annotations'])):
        header['annotations'][i][1] = float(header['annotations'][i][1])

    # Write new edf file
    pyedflib.highlevel.write_edf(new_file, signals, signal_headers, header, digital=True)

    print('Preprocessing of', new_file_name, 'completed')

def stage_labelling(annotations, signal_length, fs, epoch_size):
    """
    Adds sleep stage labels from visually scored hypnogram to epochs

    :param annotations:
        list - annotations from EDF file header (= header["annotations"]) with
        [time, duration, annotations]
    :param signal_length:
        int - length of signal in EDF file (= len(signals[0]))
    :param fs:
        int - sample frequency (= signal.headers[0].get("sample_rate"))
    :param epoch_size:
        int - epoch_size in samples (=epoch_size_s*fs)

    :return: sleep_stage_labels
        list - sleep stage labels per epoch of visually scored hypnogram

    """

    # Extract raw hypnogram data
    raw_hyp_data = [[0, 'Sleep stage W']] # The first epoch is wake stage (assumption)

    for i in range(len(annotations)):
        hyp_annotation = annotations[i]
        if hyp_annotation[2] == 'Sleep stage W' or hyp_annotation[2] == 'Sleep stage R' or \
                hyp_annotation[2] == 'Sleep stage N1' or hyp_annotation[2] == 'Sleep stage N2' or \
                hyp_annotation[2] == 'Sleep stage N3' or hyp_annotation[2] == 'Sleep stage N4' or \
                hyp_annotation[2] == 'Sleep stage N':
            raw_hyp_data.append([hyp_annotation[0], hyp_annotation[2]])

    raw_hyp_data.append([signal_length / fs, raw_hyp_data[-1][1]])  # Add hyp data to last time stamp \
    # with a sleep stage equal to the sleep stage of the last annotation (assumption)

    # Find hyp_data
    sleep_stage_labels = []
    for i in range(0, math.floor(signal_length / epoch_size) - 1):
        t = i * 30
        for j in range(0, len(raw_hyp_data) - 1):
            if raw_hyp_data[j][0] <= t < raw_hyp_data[j + 1][0]:
                sleep_stage_labels.append(raw_hyp_data[j][1])

    #if math.floor(signal_length / epoch_size) - 1 == len(sleep_stage_labels):
        #print('Epoch labelling succesfully completed')
    #else:
        #raise ValueError('Epoch size is not equal to hyp_data size')

    return sleep_stage_labels

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
                               high_amplitude_artifact_column_names[8]: high_amplitude_artifact_labels[:, 8],
                               high_amplitude_artifact_column_names[9]: high_amplitude_artifact_labels[:, 9],
                               high_amplitude_artifact_column_names[10]: high_amplitude_artifact_labels[:, 10],
                               })

    return dfArtifact

def windowing(signal, epoch_size):
    """
    Cuts signal [y] into epochs/windows with size [epoch_size]

    :param signal: array - one raw signal
    :param epoch_size: int - size of epoch in samples (= seconds * sample frequency)

    :return: epochs: list with epochs of the signal as array
    """

    epochs = []
    for i in range(math.floor(len(signal) / epoch_size) - 1):
        k = i * epoch_size
        epochs.append(signal[k: k + epoch_size])
    return epochs

def bandpower(y, fs, nfft, noverlap, fmin, fmax):
    """
    Calculation of the signal bandpower

    :param y: signal
    :param fs: sample frequency
    :param fmin: lower cutoff frequency of bandpower range
    :param fmax: upper cutoff frequency of bandpower range

    :return: bandpower
    """

    f, Pxx = scipy.signal.welch(y, fs=fs, window='hann', noverlap=noverlap, nfft=nfft) # Default window, nfft settings
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def spectralFeatures(y, fs):
    """
    Calculation of spectral edge frequency, median frequency, mean frequency, spectral kurtosis, spectral skewness and
    spectral entropy
    :param y: signal
    :param fs: sample frequency

    :return: spectral_edge, median_freq, mean_freq, spectral_kurtosis, spectral_skewness, spectral_entropy
    """

    f, Pxx = f, Pxx = scipy.signal.periodogram(y, fs=fs, window='hann')

    # Spectral edge
    percent = 0.95 # percentage (range 0 -1) of total power that is located under the spectral edge frequency
    ind_max = np.argmax(np.cumsum(Pxx) > (percent*np.trapz(Pxx)))
    spectral_edge = f[ind_max]

    # Median frequency
    ind_max = np.argmax(np.cumsum(Pxx) > (0.5 * np.trapz(Pxx)))
    median_freq = f[ind_max]

    # Mean frequency
    ind_max = np.argmax(np.cumsum(Pxx) > np.mean(Pxx))
    mean_freq = f[ind_max]

    # Spectral kurtosis
    spectral_kurtosis = scipy.stats.kurtosis(Pxx)

    # Spectral skewness
    spectral_skewness = scipy.stats.skew(Pxx)

    # Spectral entropy
    pk, bins = np.histogram(Pxx, len(Pxx))
    spectral_entropy = scipy.stats.entropy(pk)

    return spectral_edge, median_freq, mean_freq, spectral_kurtosis, spectral_skewness, spectral_entropy

def hjorth_features(y):
    """
    Calculation of the Hjorth features (activity, mobility and complexity)
    :param y: signal

    :return: activity, mobility, complexity
    """

    first_deriv = np.diff(y)
    second_deriv = np.diff(y, 2)

    var_zero = np.mean(y ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return activity, mobility, complexity

def NaNfill(len):
    """
    Fill array with NaN's
    :param len: length of array to fill
    :return: array with NaN's
    """
    nan_array = np.zeros(len)
    nan_array[:] = np.nan

    return nan_array

def FeatureCalculation(signals, signal_headers, fs, epoch_size):
    """
    Calculation of features for all EEG leads, EMG and EOG (after bandpass filtering)

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

    :return: dfFeatureData
        dataframe - containing sleep stage labels, artifact labels and features

    """

    dfFeatureData = pd.DataFrame()

    # Function variables:
    include_signals = ['EOG ROC-LOC', 'EMG EMG Chin',  # EOG + EMG channel
                       'EEG F3', 'EEG C3', 'EEG O1', 'EEG A1',  # Unipolar channels
                       'EEG F3-C3', 'EEG F3-C4', 'EEG F3-O2', 'EEG F3-A2', 'EEG C3-C4', 'EEG C3-O2', 'EEG C3-A2',
                       'EEG O1-O2', 'EEG O1-A2'  # Bipolar channels
                       ]

    # Extract signal
    signal_labels = []
    for i in range(0, len(signal_headers)):
        signal_labels.append(signal_headers[i]["label"])

    ### EOG features ###
    signal_label = include_signals[0]
    signal = signals[signal_labels.index(include_signals[2])]

    # Filter EOG signal with lowpass filter (30 Hz)
    n = 16  # Bandpass filter order
    fco_up = 30  # Lowpass filter upper cutoff frequency
    sos = scipy.signal.butter(n, fco_up, btype='lowpass', analog=False, output='sos', fs=fs)
    y = scipy.signal.sosfilt(sos, signal)  # filter signal

    # Cut signal into epochs
    epochs = windowing(y, epoch_size)

    # Fill feature arrays with zeros
    eog_bandpower_rem = NaNfill(len(epochs))
    eog_bandpower_rem2 = NaNfill(len(epochs))
    eog_bandpower_sem = NaNfill(len(epochs))
    eog_variance = NaNfill(len(epochs))

    for i in range(0, len(epochs)):
        eog_bandpower_rem[i] = bandpower(epochs[i], fs, 15 * fs, 0.5, 0.35, 0.5)
        eog_bandpower_rem2[i] = bandpower(epochs[i], fs, 15 * fs, 0.5, 0.35, 2)
        eog_bandpower_sem[i] = bandpower(epochs[i], fs, 15 * fs, 0.5, 0, 0.35)
        eog_variance[i] = np.var(epochs[i])
    dfFeatureData[signal_label + ': Bandpower REM (0.35-0.5)'] = eog_bandpower_rem
    dfFeatureData[signal_label + ': Bandpower REM (0.5-2)'] = eog_bandpower_rem2
    dfFeatureData[signal_label + ': Bandpower SEM (0-0.35)'] = eog_bandpower_sem
    dfFeatureData[signal_label + ': Variance'] = eog_variance

    ### EMG features ###
    # EMG features
    signal_label = include_signals[1]
    signal = signals[signal_labels.index(include_signals[3])]

    # Filter EMG signal with bandpass filter (5-40 Hz)
    n = 16  # Bandpass filter order
    fco_low = 5  # Bandpass filter lower cutoff frequency
    fco_up = 40  # Bandpass filter upper cutoff frequency
    sos = scipy.signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)
    y = scipy.signal.sosfilt(sos, signal)  # filter signal

    # Cut signal into epochs
    epochs = windowing(y, epoch_size)

    # Fill feature arrays with zeros
    emg_energy = NaNfill(len(epochs))
    emg_mean_abs_ampl = NaNfill(len(epochs))

    for i in range(0, len(epochs)):
        yy = np.float64(epochs[i])
        emg_energy[i] = np.float64(sum(yy * yy) / len(yy))
        emg_mean_abs_ampl[i] = np.mean(np.absolute(epochs[i]))

    dfFeatureData[signal_label + ': Energy'] = emg_energy
    dfFeatureData[signal_label + ': Mean absolute amplitude'] = emg_mean_abs_ampl

    ### EEG features ###
    nfft = 2 * fs
    noverlap = 0.5

    for signal_label in include_signals[2:]:
        # signal_label = include_signals[2]
        nr = signal_labels.index(signal_label)

        # Filter EEG signal with bandpass filter (0.5-48 Hz)
        n = 16  # Bandpass filter order
        fco_low = 0.5  # Bandpass filter lower cutoff frequency
        fco_up = 48  # Bandpass filter upper cutoff frequency
        sos = scipy.signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)
        y = scipy.signal.sosfilt(sos, signals[nr])  # filter signal

        # Cut signal into epochs
        epochs = windowing(y, epoch_size)

        # Fill feature arrays with zeros - Time domain features
        abs_mean_ampl = NaNfill(len(epochs))
        variance = NaNfill(len(epochs))
        zero_crossing_rate = NaNfill(len(epochs))
        interquartile_range = NaNfill(len(epochs))
        signal_sum = NaNfill(len(epochs))
        energy = NaNfill(len(epochs))
        kurtosis = NaNfill(len(epochs))
        skewness = NaNfill(len(epochs))
        entropy = NaNfill(len(epochs))
        activity = NaNfill(len(epochs))
        mobility = NaNfill(len(epochs))
        complexity = NaNfill(len(epochs))
        higuchi_fd = NaNfill(len(epochs))
        dfa = NaNfill(len(epochs))

        # Fill feature arrays with zeros - Frequency domain features
        total_power = NaNfill(len(epochs))
        abs_beta_power = NaNfill(len(epochs))
        abs_sigma_power = NaNfill(len(epochs))
        abs_alpha_power = NaNfill(len(epochs))
        abs_theta_power = NaNfill(len(epochs))
        abs_delta_power = NaNfill(len(epochs))
        abs_gamma_power = NaNfill(len(epochs))
        rel_beta_power = NaNfill(len(epochs))
        rel_sigma_power = NaNfill(len(epochs))
        rel_alpha_power = NaNfill(len(epochs))
        rel_theta_power = NaNfill(len(epochs))
        rel_delta_power = NaNfill(len(epochs))
        rel_gamma_power = NaNfill(len(epochs))
        gamma_delta_ratio = NaNfill(len(epochs))
        gamma_theta_ratio = NaNfill(len(epochs))
        beta_delta_ratio = NaNfill(len(epochs))
        beta_theta_ratio = NaNfill(len(epochs))
        alpha_delta_ratio = NaNfill(len(epochs))
        alpha_theta_ratio = NaNfill(len(epochs))
        spectral_edge = NaNfill(len(epochs))
        median_freq = NaNfill(len(epochs))
        mean_freq = NaNfill(len(epochs))
        spectral_kurtosis = NaNfill(len(epochs))
        spectral_skewness = NaNfill(len(epochs))
        spectral_entropy = NaNfill(len(epochs))

        # Fill feature arrays with zeros - Tim-frequency domain features
        cA_mean = NaNfill(len(epochs))
        cA_std = NaNfill(len(epochs))
        cD5_mean = NaNfill(len(epochs))
        cD5_std = NaNfill(len(epochs))
        cD4_mean = NaNfill(len(epochs))
        cD4_std = NaNfill(len(epochs))
        cD3_mean = NaNfill(len(epochs))
        cD3_std = NaNfill(len(epochs))
        cD2_mean = NaNfill(len(epochs))
        cD2_std = NaNfill(len(epochs))
        cD1_mean = NaNfill(len(epochs))
        cD1_std = NaNfill(len(epochs))

        # Feature calculation for each epoch
        for i in range(0, len(epochs)):
            # Time domain features
            abs_mean_ampl[i] = np.mean(np.absolute(epochs[i]))
            variance[i] = np.var(epochs[i])
            zero_crossing_rate[i] = (np.diff(np.sign(epochs[i])) != 0).sum()
            interquartile_range[i] = scipy.stats.iqr(epochs[i])
            signal_sum[i] = sum(np.absolute(epochs[i]))
            energy[i] = sum(epochs[i] * epochs[i]) / len(epochs[i])
            kurtosis[i] = scipy.stats.kurtosis(epochs[i])
            skewness[i] = scipy.stats.skew(epochs[i])
            pk, bins = np.histogram(epochs[i], len(epochs[i]))
            entropy[i] = scipy.stats.entropy(pk)
            activity[i], mobility[i], complexity[i] = hjorth_features(epochs[i])
            higuchi_fd[i] = eeglib.features.HFD(epochs[i])
            dfa[i] = eeglib.features.DFA(epochs[i])

            # Frequency domain features
            total_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 0.5, 48)
            abs_beta_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 12, 30)
            abs_alpha_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 7, 12)
            abs_sigma_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 11, 15)
            abs_theta_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 4, 7)
            abs_delta_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 0.5, 4)
            abs_gamma_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 30, 48)
            rel_beta_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 12, 30) / total_power[i] * 100
            rel_alpha_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 7, 12) / total_power[i] * 100
            rel_sigma_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 11, 15) / total_power[i] * 100
            rel_theta_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 4, 7) / total_power[i] * 100
            rel_delta_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 0.5, 4) / total_power[i] * 100
            rel_gamma_power[i] = bandpower(epochs[i], fs, nfft, noverlap, 30, 48) / total_power[i] * 100
            gamma_delta_ratio[i] = abs_gamma_power[i] / abs_delta_power[i]
            gamma_theta_ratio[i] = abs_gamma_power[i] / abs_theta_power[i]
            beta_delta_ratio[i] = abs_beta_power[i] / abs_delta_power[i]
            beta_theta_ratio[i] = abs_beta_power[i] / abs_theta_power[i]
            alpha_delta_ratio[i] = abs_alpha_power[i] / abs_delta_power[i]
            alpha_theta_ratio[i] = abs_alpha_power[i] / abs_theta_power[i]
            spectral_edge[i], median_freq[i], mean_freq[i], spectral_kurtosis[i], spectral_skewness[i], \
            spectral_entropy[i] \
                = spectralFeatures(epochs[i], fs)

            # Time-frequency domain features
            # Downsample signal
            if fs == 250:
                downsample_factor = 2.5
            elif fs == 256:
                downsample_factor = 2.56
            y_downsampled = scipy.signal.resample(epochs[i], round(len(epochs[i]) / downsample_factor))

            # Discrete wavelet transform
            cA, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(y_downsampled, 'db4', level=5)

            cA_mean[i] = np.mean(np.abs(cA))
            cA_std[i] = np.std(cA)
            cD5_mean[i] = np.mean(np.abs(cD5))
            cD5_std[i] = np.std(cD5)
            cD4_mean[i] = np.mean(np.abs(cD4))
            cD4_std[i] = np.std(cD4)
            cD3_mean[i] = np.mean(np.abs(cD3))
            cD3_std[i] = np.std(cD3)
            cD2_mean[i] = np.mean(np.abs(cD2))
            cD2_std[i] = np.std(cD2)
            cD1_mean[i] = np.mean(np.abs(cD1))
            cD1_std[i] = np.std(cD1)

        # Time domain features
        dfFeatureData[signal_label + ': Abs mean amplitude'] = abs_mean_ampl
        dfFeatureData[signal_label + ': Variance'] = variance
        dfFeatureData[signal_label + ': Zero crossing rate'] = zero_crossing_rate
        dfFeatureData[signal_label + ': Interquartile range'] = interquartile_range
        dfFeatureData[signal_label + ': Signal sum'] = signal_sum
        dfFeatureData[signal_label + ': Energy'] = energy
        dfFeatureData[signal_label + ': Kurtosis'] = kurtosis
        dfFeatureData[signal_label + ': Skewness'] = skewness
        dfFeatureData[signal_label + ': Entropy'] = entropy
        dfFeatureData[signal_label + ': Hjorth activity'] = activity
        dfFeatureData[signal_label + ': Hjorth mobility'] = mobility
        dfFeatureData[signal_label + ': Hjorth complexity'] = complexity
        dfFeatureData[signal_label + ': Higuchi Fractional Dimension'] = higuchi_fd
        dfFeatureData[signal_label + ': Detrended Fluctuation Analysis'] = dfa

        # Frequency domain features
        dfFeatureData[signal_label + ': Total power'] = total_power
        dfFeatureData[signal_label + ': Abs Beta power'] = abs_beta_power
        dfFeatureData[signal_label + ': Abs Alpha power'] = abs_alpha_power
        dfFeatureData[signal_label + ': Abs Sigma power'] = abs_sigma_power
        dfFeatureData[signal_label + ': Abs Theta power'] = abs_theta_power
        dfFeatureData[signal_label + ': Abs Delta power'] = abs_delta_power
        dfFeatureData[signal_label + ': Abs Gamma power'] = abs_gamma_power
        dfFeatureData[signal_label + ': Rel Beta power'] = rel_beta_power
        dfFeatureData[signal_label + ': Rel Alpha power'] = rel_alpha_power
        dfFeatureData[signal_label + ': Rel Sigma power'] = rel_sigma_power
        dfFeatureData[signal_label + ': Rel Theta power'] = rel_theta_power
        dfFeatureData[signal_label + ': Rel Delta power'] = rel_delta_power
        dfFeatureData[signal_label + ': Rel Gamma power'] = rel_gamma_power
        dfFeatureData[signal_label + ': Gamma/delta ratio'] = gamma_delta_ratio
        dfFeatureData[signal_label + ': Gamma/theta ratio'] = gamma_theta_ratio
        dfFeatureData[signal_label + ': Beta/delta ratio'] = beta_delta_ratio
        dfFeatureData[signal_label + ': Beta/theta ratio'] = beta_theta_ratio
        dfFeatureData[signal_label + ': Alpha/delta ratio'] = alpha_delta_ratio
        dfFeatureData[signal_label + ': Alpha/theta ratio'] = alpha_theta_ratio
        dfFeatureData[signal_label + ': Spectral edge'] = spectral_edge
        dfFeatureData[signal_label + ': Median freq'] = median_freq
        dfFeatureData[signal_label + ': Mean freq'] = mean_freq
        dfFeatureData[signal_label + ': Spectral kurtosis'] = spectral_kurtosis
        dfFeatureData[signal_label + ': Spectral skewness'] = spectral_skewness
        dfFeatureData[signal_label + ': Spectral entropy'] = spectral_entropy

        # Time-frequency domain features
        dfFeatureData[signal_label + ': DWT cA mean'] = cA_mean
        dfFeatureData[signal_label + ': DWT cA std'] = cA_std
        dfFeatureData[signal_label + ': DWT cD5 mean'] = cD5_mean
        dfFeatureData[signal_label + ': DWT cD5 std'] = cD5_std
        dfFeatureData[signal_label + ': DWT cD4 mean'] = cD4_mean
        dfFeatureData[signal_label + ': DWT cD4 std'] = cD4_std
        dfFeatureData[signal_label + ': DWT cD3 mean'] = cD3_mean
        dfFeatureData[signal_label + ': DWT cD3 std'] = cD3_std
        dfFeatureData[signal_label + ': DWT cD2 mean'] = cD2_mean
        dfFeatureData[signal_label + ': DWT cD2 std'] = cD2_std
        dfFeatureData[signal_label + ': DWT cD1 mean'] = cD1_mean
        dfFeatureData[signal_label + ': DWT cD1 std'] = cD1_std

    return dfFeatureData

def add_new_state_labels(featureData, number_of_stages):
    """

    :param featureData:
    :param number_of_stages:
    :return:
    """
    new_state_labels = []
    new_state_labels_linear = []

    if number_of_stages == 2:
        number_of_stages_key = 'Two_state_labels'
        for epoch in featureData.iterrows():
            if epoch[1]['Sleep stage labels'] == 'Sleep stage W':
                new_state_labels.append('Wake')
                new_state_labels_linear.append(1)
            else:
                new_state_labels.append('Sleep')
                new_state_labels_linear.append(0)
        featureData[number_of_stages_key] = new_state_labels
        featureData[number_of_stages_key + '- linear'] = new_state_labels_linear
        featureData_new = featureData

    elif number_of_stages == 3:
        number_of_stages_key = 'Three_state_labels'
        for epoch in featureData.iterrows():
            if epoch[1]['Sleep stage labels'] == 'Sleep stage W':
                new_state_labels.append('Wake')
                new_state_labels_linear.append(2)
            elif epoch[1]['Sleep stage labels'] == 'Sleep stage R' or epoch[1][
                'Sleep stage labels'] == 'Sleep stage N1' or \
                    epoch[1]['Sleep stage labels'] == 'Sleep stage N2':
                new_state_labels.append('NSWS')
                new_state_labels_linear.append(1)
            else:
                new_state_labels.append('SWS')
                new_state_labels_linear.append(0)

        featureData[number_of_stages_key] = new_state_labels
        featureData[number_of_stages_key + '- linear'] = new_state_labels_linear
        featureData_new = featureData

    elif number_of_stages == 4:
        number_of_stages_key = 'Four_state_labels'
        for epoch in featureData.iterrows():
            if epoch[1]['Sleep stage labels'] == 'Sleep stage W':
                new_state_labels.append('Wake')
                new_state_labels_linear.append(3)
            elif epoch[1]['Sleep stage labels'] == 'Sleep stage R':
                new_state_labels.append('REM')
                new_state_labels_linear.append(2)
            elif epoch[1]['Sleep stage labels'] == 'Sleep stage R' or epoch[1][
                'Sleep stage labels'] == 'Sleep stage N1' or \
                    epoch[1]['Sleep stage labels'] == 'Sleep stage N2':
                new_state_labels.append('NSWS')
                new_state_labels_linear.append(1)
            else:
                new_state_labels.append('SWS')
                new_state_labels_linear.append(0)

        featureData[number_of_stages_key] = new_state_labels
        featureData[number_of_stages_key + '- linear'] = new_state_labels_linear
        featureData_new = featureData

    elif number_of_stages == 5:
        number_of_stages_key = 'Five_state_labels'
        new_state_labels = featureData['Sleep stage labels']
        order = ['Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
        new_state_labels_linear = [order.index(x) for x in featureData["Sleep stage labels"]]

        featureData[number_of_stages_key] = new_state_labels
        featureData[number_of_stages_key + '- linear'] = new_state_labels_linear
        featureData_new = featureData

    return featureData_new, number_of_stages_key


def load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                      feature_data_file_path, include_age_categories,
                                      remove_artifacts):
    """
    This function loads and prepares multiple raw feature data files for input in the classification models by
    removing artifacts from the file, adding sleep stage labels for two, three and four state classi-
    fication and adding age category as dummy variables.
    July 2021, Floor Hiemstra

    :param patient_data_file_path:
        str - file path to patient data file
    :param patient_data_file_name:
        str - file name of patient data file
    :param feature_data_file_path:
        str - file path to feature data files
    :param include_age_categories:
        str - age categories to be loaded
    :param remove_artifacts:
        int - 1: remove epochs with artifacts, 0: do not remove epochs with artifacts

    :return: allFeatureData_final:
        Dataframe containing all feature data files with:
        - if remove_artifacts = 1: artifact removal
        - Addition of sleep stage labels for two, three and four sleep stage classification

    """
    ### 1.1 Load raw feature data ###
    start_time = time.time()

    # List feature data file names
    patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
    keys = []
    for age_category in include_age_categories:
        keys.extend(patientData[patientData['Age category'] == age_category]['Key'])
    feature_data_file_names = [key + '_FeatureData.xlsx' for key in keys], [key + '_FeatureData_scaled.xlsx' for key in keys]
    feature_data_file_names = feature_data_file_names[0] + feature_data_file_names[1]

    # Read feature data files in list and store as one dataframe
    allFeatureData = []
    count = 0
    for feature_data_file_name in feature_data_file_names:
        if feature_data_file_name in os.listdir(feature_data_file_path) and feature_data_file_name.endswith('.xlsx'):
            count = count + 1
            featureData_raw = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

            # Create dataframes
            if feature_data_file_name == feature_data_file_names[0]:
                allFeatureData_raw = featureData_raw
            else:
                allFeatureData_raw = allFeatureData_raw.append(featureData_raw)

            print(
                "Loading %s of %s files completed (%s)" % (count, int(len(feature_data_file_names)/2), feature_data_file_name))

    print("Loading feature data completed: %s hours" % ((time.time() - start_time) / 3600))

    ### 1.2 Remove artifacts ###
    if remove_artifacts == 1:
        # Remove impedance artifacts
        allFeatureData_clean = allFeatureData_raw[allFeatureData_raw['Impedance artifact'] == 0]

        # Remove high amplitude artifacts
        unipolar_psg_signals = ['EOG ROC', 'EOG LOC', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1',
                                'EEG O2', 'EEG A1', 'EEG A2']
        for signal in unipolar_psg_signals:  # Loop over all signals that have artifact labels
            count = count + 1
            if count <= 1:  # To determine the letters of the signal the column should contain
                x = 3  # f.e. for signal=EOG ROC --> the column name should contain the last three letters (ROC)
            elif count == 2:
                x = 8
            elif count > 2:
                x = 2

            for column in allFeatureData_clean.iloc[:, 19:].columns:
                if signal[-x:] in column:
                    allFeatureData_clean = allFeatureData_clean.copy()
                    mask = allFeatureData_clean[signal + ' - Artifact'] == 1
                    allFeatureData_clean.loc[mask, column] = np.NaN

    else:
        allFeatureData_clean = allFeatureData_raw.copy()

    print("Artifact removal completed")

    ### 1.3 Add new sleep stage labels ###
    allFeatureData_clean_newstages, number_of_stages_key_two = add_new_state_labels(allFeatureData_clean, 2)
    allFeatureData_clean_newstages, number_of_stages_key_three = add_new_state_labels(allFeatureData_clean_newstages, 3)
    allFeatureData_clean_newstages, number_of_stages_key_four = add_new_state_labels(allFeatureData_clean_newstages, 4)
    print("New sleep stages added")

    ### 1.4 Add age category as dummy variables
    allFeatureData_final = pd.concat(
        [allFeatureData_clean_newstages, pd.get_dummies(allFeatureData_clean_newstages['Age category'],
                                                        prefix='age_group_')], axis=1)  # Age category
    print("Dummy variables added")
    print("Loading and preparation of raw feature data completed")

    return allFeatureData_final

def load_and_prepare_raw_feature_data_single(feature_data_file_path, feature_data_file_name,
                                      remove_artifacts):
    """
    This function loads and prepares a single raw feature data file for input in the classification models by
    removing artifacts from the file, adding sleep stage labels for two, three and four state classi-
    fication and adding age category as dummy variables.
    July 2021, Floor Hiemstra

    :param patient_data_file_path:
        str - file path to patient data file
    :param patient_data_file_name:
        str - file name of patient data file
    :param feature_data_file_path:
        str - file path to feature data files
    :param include_age_categories:
        str - age categories to be loaded
    :param remove_artifacts:
        int - 1: remove epochs with artifacts, 0: do not remove epochs with artifacts

    :return: allFeatureData_final:
        Dataframe containing all feature data files with:
        - if remove_artifacts = 1: artifact removal
        - Addition of sleep stage labels for two, three and four sleep stage classification

    """
    ### 1.1 Load raw feature data ###
    allFeatureData_raw = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

    ### 1.2 Remove artifacts ###
    if remove_artifacts == 1:
        # Remove impedance artifacts
        allFeatureData_clean = allFeatureData_raw[allFeatureData_raw['Impedance artifact'] == 0]

        # Remove high amplitude artifacts
        count = 0
        unipolar_psg_signals = ['EOG ROC', 'EOG LOC', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1',
                                'EEG O2', 'EEG A1', 'EEG A2']
        for signal in unipolar_psg_signals:  # Loop over all signals that have artifact labels
            count = count + 1
            if count <= 1:  # To determine the letters of the signal the column should contain
                x = 3  # f.e. for signal=EOG ROC --> the column name should contain the last three letters (ROC)
            elif count == 2:
                x = 8
            elif count > 2:
                x = 2

            for column in allFeatureData_clean.iloc[:, 19:].columns:
                if signal[-x:] in column:
                    allFeatureData_clean = allFeatureData_clean.copy()
                    mask = allFeatureData_clean[signal + ' - Artifact'] == 1
                    allFeatureData_clean.loc[mask, column] = np.NaN

    else:
        allFeatureData_clean = allFeatureData_raw.copy()

    print("Artifact removal completed")

    ### 1.3 Add new sleep stage labels ###
    allFeatureData_clean_newstages, number_of_stages_key_two = add_new_state_labels(allFeatureData_clean, 2)
    allFeatureData_clean_newstages, number_of_stages_key_three = add_new_state_labels(allFeatureData_clean_newstages, 3)
    allFeatureData_clean_newstages, number_of_stages_key_four = add_new_state_labels(allFeatureData_clean_newstages, 4)
    print("New sleep stages added")

    ### 1.4 Add age category as dummy variables
    dummy_variables_names = ['age_group__0-2 months', 'age_group__1-3 years',
       'age_group__13-18 years', 'age_group__2-6 months',
       'age_group__3-5 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']
    dfDummyVariables = pd.DataFrame(
        data=np.zeros((len(allFeatureData_clean_newstages), len(dummy_variables_names))), columns=dummy_variables_names)
    dfDummyVariable_file = pd.get_dummies(allFeatureData_clean_newstages['Age category'],
                                                        prefix='age_group_')
    allFeatureData_final = allFeatureData_clean_newstages.join(dfDummyVariables)
    allFeatureData_final[dfDummyVariable_file.columns] = dfDummyVariable_file.values
    print("Dummy variables added")

    print("Loading and preparation of raw feature data completed")

    return allFeatureData_final

def compute_feature_posterior_prob(feature, labels, number_of_stages=3):
    """
    This function computes the posterior probabilities for various feature
    values for each of the class labels

    :param feature: series - feature values to compute the posterior for
    :param labels: series - labels corresponding to the feature values
    :param number_of_stages: 3 or 4 (only works for 3 or 4 stage classification)
    :return:
    - all_posteriors: list with dictionary for each class with posteriors
    - x_kde: corresponding feature values to posteriors
    - all_priors: list with dictionary priors (class proportions)
    """
    from sklearn.neighbors import KernelDensity

    all_kde = []
    all_priors = []

    if number_of_stages == 3:
        stages = ['Wake', 'NSWS', 'SWS']
    elif number_of_stages == 4:
        stages = ['Wake', 'REM', 'NSWS', 'SWS']

    for stage in stages:
        stage_feature = feature[labels == stage]
        prior = len(labels[labels == stage])/len(labels)

        ### Kernel density estimation
        x_kde = np.linspace(np.min(feature), np.max(feature), 100000)[:, np.newaxis]
        iqr = scipy.stats.iqr(stage_feature, rng=(5, 95))
        kernel_bandwidth = iqr/10
        kde_model = KernelDensity(kernel='gaussian', bandwidth=kernel_bandwidth).fit(np.array(stage_feature).reshape(-1, 1))
        kde = np.exp(kde_model.score_samples(x_kde))
        all_kde.append({stage: kde})
        all_priors.append({stage: prior})

    ### Determine posteriors
    all_posteriors = []
    if number_of_stages == 3:
        denominator = all_kde[0].get('Wake')*all_priors[0].get('Wake') + all_kde[1].get('NSWS')*all_priors[1].get('NSWS') \
              + all_kde[2].get('SWS')*all_priors[2].get('SWS')
    elif number_of_stages == 4:
        denominator = all_kde[0].get('Wake') * all_priors[0].get('Wake') \
                      + all_kde[1].get('REM') * all_priors[1].get('REM') \
                      + all_kde[2].get('NSWS') * all_priors[2].get('NSWS') \
                      + all_kde[3].get('SWS') * all_priors[3].get('SWS')

    for (stage, nr) in zip(stages, range(0, number_of_stages)):
        posterior = all_kde[nr].get(stage)*all_priors[nr].get(stage) / denominator
        all_posteriors.append({stage: posterior})

    return all_posteriors, x_kde, all_priors

def auc_feature_calculation_posterior(feature, labels, all_posteriors, x_kde, number_of_stages=3, plot_roc=0):
    from sklearn.metrics import roc_auc_score, roc_curve
    """
    This function calculate the auc using the posterior probabilities 

    :param feature: series - feature values to compute the posterior for
    :param labels: series - labels corresponding to the feature values
        :return:
    :param all_posteriors: list with dictionary for each class with posteriors
    :param x_kde: corresponding feature values to posteriors
    :param number_of_stages: 3 or 4 (only works for 3 or 4 stage classification)
        - all_posteriors: list with dictionary for each class with posteriors
    :param plot_roc: 1: ROC curve plot is shown
    :return: auc, tpr, fpr 
    """

    # Find posterior probabilities for feature value
    y_pred = np.zeros((len(feature), number_of_stages))
    y_true = np.zeros((len(feature), number_of_stages))
    for i in range(0, len(feature)):
        idx = np.argmin(np.abs(x_kde - feature.iloc[i]))
        if number_of_stages == 3:
            stages = ['Wake', 'NSWS', 'SWS']
        elif number_of_stages == 4:
            stages = ['Wake', 'REM', 'NSWS', 'SWS']

        for (stage, nr) in zip(stages, range(0, number_of_stages+1)):
            y_pred[i, nr] = all_posteriors[nr].get(stage)[idx]
            if labels.iloc[i] == stage:
                y_true[i, nr] = 1

    # Remove NaNs (as result of zero division error in posterior calculation)
    idx = np.where(np.isnan(y_pred).any(axis=1))
    y_pred = np.delete(y_pred, idx, axis=0)
    y_true = np.delete(y_true, idx, axis=0)

    auc = roc_auc_score(y_true, y_pred, multi_class='ovr')

    fpr = dict()
    tpr = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())

    if plot_roc == 1:

        plt.figure(1, figsize=(7, 6), dpi=100)
        plt.title("ROC curve - posterior")
        plt.plot(fpr["micro"], tpr["micro"], label='AUC=%s' % auc)
        plt.legend()
        plt.minorticks_on()
        plt.grid(b=True, which='both', linestyle='--')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.ylim([0, 1])
        plt.xlim([0, 1])

    return auc, tpr, fpr

def find_optimal_index_thresholds(feature, labels, number_of_stages, plot_roc=0):
    """

    :param feature:
    :param labels:
    :param number_of_stages:
    :return:
    """

    if plot_roc == 1:
        plt.figure(100, figsize=(7, 6), dpi=100)
        plt.title("ROC curve - feature value thresholds")
        plt.minorticks_on()
        plt.grid(b=True, which='both', linestyle='--')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.ylim([0, 1])
        plt.xlim([0, 1])

    optimal_thresholds_max_acc = []
    optimal_thresholds_max_tpr = []
    AUCs_fvalue = []

    for stage in ['Wake', 'SWS']:
        label_true = []
        label_false = []
        thresholds = np.linspace(np.min(feature), np.max(feature), 10000)
        TPR = np.zeros(len(thresholds))
        FPR = np.zeros(len(thresholds))
        roc_accuracies = np.zeros(len(thresholds))

        for i in range(0, len(feature)):
            if labels.iloc[i] == stage:
                label_true.append(feature.iloc[i])
            if labels.iloc[i] != stage:
                label_false.append(feature.iloc[i])

        for i in range(0, len(thresholds)):
            x = [element for element in label_true if element >= thresholds[i]]
            TP = len(x)
            x = [element for element in label_true if element < thresholds[i]]
            FN = len(x)
            x = [element for element in label_false if element >= thresholds[i]]
            FP = len(x)
            x = [element for element in label_false if element < thresholds[i]]
            TN = len(x)

            # to prevent zeroDivisionError
            if TP + FN == 0:
                TPR[i] = 0
            elif FP + TN == 0:
                FPR[i] = 0
            else:
                TPR[i] = TP / (TP + FN)
                FPR[i] = FP / (FP + TN)
                roc_accuracies[i] = (TP + TN) / (TP + TN + FP + FN)

        AUC = abs(np.trapz(TPR, FPR))
        if AUC < 0.5:
            AUC = 1 - AUC
            optimal_threshold_max_acc = thresholds[np.argmin(roc_accuracies)]
            optimal_threshold_max_tpr = thresholds[np.argmin(TPR-FPR)]
            if plot_roc == 1:
                plt.plot(TPR, FPR , label='%s, AUC=%s' % (stage, AUC))
                plt.legend()
        else:
            optimal_threshold_max_tpr = thresholds[np.argmax(TPR-FPR)]
            optimal_threshold_max_acc = thresholds[np.argmax(roc_accuracies)]
            if plot_roc == 1:
                plt.plot(FPR, TPR, label='%s, AUC=%s' % (stage, AUC))
                plt.legend()

        AUCs_fvalue.append({stage: "{:.2f}".format(AUC)})  # 2 decimals
        optimal_thresholds_max_acc.append({stage: optimal_threshold_max_acc})
        optimal_thresholds_max_tpr.append({stage: optimal_threshold_max_tpr})

    if number_of_stages == 4:
        stage = 'NSWS'
        # Threshold for REM/NSWS discrimination is added
        label_true = []
        label_false = []
        #thresholds = np.linspace(np.min(feature), np.max(feature), 10000)
        thresholds = np.linspace(optimal_thresholds_max_acc[1].get('SWS'), optimal_thresholds_max_acc[0].get('Wake'), 10000)
        TPR = np.zeros(len(thresholds))
        FPR = np.zeros(len(thresholds))
        roc_accuracies = np.zeros(len(thresholds))

        for i in range(0, len(feature)):
            if labels.iloc[i] == 'REM':
                label_true.append(feature.iloc[i])
            if labels.iloc[i] != 'REM':
                label_false.append(feature.iloc[i])

        for i in range(0, len(thresholds)):
            x = [element for element in label_true if
                 thresholds[i] <= element <= optimal_thresholds_max_acc[0].get('Wake')]
            TP = len(x)

            x = [element for element in label_false if
                 thresholds[i] <= element <= optimal_thresholds_max_acc[0].get('Wake')]
            FP = len(x)

            x1 = [element for element in label_true if element < thresholds[i]]
            x2 = [element for element in label_true if element > optimal_thresholds_max_acc[0].get('Wake')]
            FN = len(x1) + len(x2)

            x1 = [element for element in label_false if element < thresholds[i]]
            x2 = [element for element in label_false if element > optimal_thresholds_max_acc[0].get('Wake')]
            TN = len(x1) + len(x2)

            # to prevent zeroDivisionError
            if TP + FN == 0:
                TPR[i] = 0
            elif FP + TN == 0:
                FPR[i] = 0
            else:
                TPR[i] = TP / (TP + FN)
                FPR[i] = FP / (FP + TN)
                roc_accuracies[i] = (TP + TN) / (TP + TN + FP + FN)

        AUC = abs(np.trapz(TPR, FPR))
        if AUC < 0.5:
            AUC = 1 - AUC
            #optimal_threshold_max_acc = thresholds[np.argmin(roc_accuracies)]
            optimal_threshold_max_acc = thresholds[np.argmin(TPR - FPR)]
            optimal_threshold_max_tpr = thresholds[np.argmin(TPR - FPR)]
            if plot_roc == 1:
                plt.plot(TPR, FPR, label='%s, AUC=%s' % (stage, AUC))
                plt.legend()

        else:
            optimal_threshold_max_tpr = thresholds[np.argmax(TPR - FPR)]
            optimal_threshold_max_acc = thresholds[np.argmax(TPR - FPR)]
            #optimal_threshold_max_acc = thresholds[np.argmax(roc_accuracies)]
            if plot_roc == 1:
                plt.plot(FPR, TPR, label='%s, AUC=%s' % (stage, AUC))
                plt.legend()

        AUCs_fvalue.append({stage: "{:.2f}".format(AUC)})  # 2 decimals
        optimal_thresholds_max_acc.append({stage: optimal_threshold_max_acc})
        optimal_thresholds_max_tpr.append({stage: optimal_threshold_max_tpr})

    return optimal_thresholds_max_acc, optimal_thresholds_max_tpr, AUCs_fvalue