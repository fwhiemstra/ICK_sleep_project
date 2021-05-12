"""
This file contains the functions used in the ICK sleep project.
"""

import pyedflib
import os
import numpy as np
import scipy
import math


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

    # Add signals
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
    sample_rate = 250
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
    raw_hyp_data = []
    for i in range(len(annotations)):
        hyp_annotation = annotations[i]
        if i == 0:  # The first epoch is wake stage (assumption)
            raw_hyp_data.append([0, 'Sleep stage W'])
        if hyp_annotation[2] == 'Sleep stage W' or hyp_annotation[2] == 'Sleep stage R' or \
                hyp_annotation[2] == 'Sleep stage N1' or hyp_annotation[2] == 'Sleep stage N2' or \
                hyp_annotation[2] == 'Sleep stage N3' or hyp_annotation[2] == 'Sleep stage N4':
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

    if math.floor(signal_length / epoch_size) - 1 == len(sleep_stage_labels):
        print('Epoch labelling succesfully completed')
    else:
        raise ValueError('Epoch size is not equal to hyp_data size')

    return sleep_stage_labels

def artifact_labelling(signal_headers, signals, unipolar_eegsignals, fs, epoch_size):
    """
    Detects artifacts in the signals and adds labels to the epochs in which an artifact
    is detected as 'movement artifact', 'impedance artifact' or 'general artifact'.

    :param signal_headers:
        list - containing dictionaries with signal header (information)
        (output from pyedflib.highlevel.read_edf)
    :param signals:
        ndarray - containing raw signals
        (output from pyedflib.highlevel.read_edf)
    :param unipolar_eegsignals:
        list - containing strings with the labels of unipolar EEG signals (artifact detection is
        only in unipolar EEG signals)
    :param fs:
        int - sample frequency (= signal.headers[0].get("sample_rate"))
    :param epoch_size:
        int - epoch_size in samples (=epoch_size_s*fs)

    :return: impArtifact_labels
        ndarray - for each epoch/index 1 (impedance artifact) or 0 (no impedance artifact)
    :return: movArtifact_labels
        ndarray - for each epoch/index 1 (movement artifact) or 0 (no movement artifact)
    :return: artifact_labels
        ndarray - for each epoch/index 1 (impedance and/or movement artifact) or 0
        (no impedance and/or movement artifact)
    """

    # Function variables
    ampl_thres = 450

    ## Step 1: Window all unipolar signals
    signal_labels = []
    epochs_unipolar = []
    for i in range(0, len(signal_headers)): # Create list with signal labels
        signal_labels.append(signal_headers[i].get("label"))

    for i in unipolar_eegsignals:
        y = signals[signal_labels.index(i)]
        epochs = []
        for j in range(math.floor(len(y) / epoch_size) - 1):
            k = j * epoch_size
            epochs.append(y[k: k + epoch_size])
        epochs_unipolar.append(epochs)
        # epochs_unipolar contains 8 lists (each list is a signal) with each [number of epochs]
        # lists containing an array with epoch signal

    ## Step 2: Detect impedance + movement artifact + label
    impArtifact_labels = np.zeros(len(epochs_unipolar[0]))
    movArtifact_labels = np.zeros(len(epochs_unipolar[0]))
    artifact_labels = np.zeros(len(epochs_unipolar[0]))

    # Impedance artifacts
    for signal in range(0, len(epochs_unipolar)):  # loop over each signal
        for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
            count = 0
            for j in range(0, len(epochs_unipolar[signal][i]),
                           int(fs / 2)):  # check every half second if sample is equal to zero
                if epochs_unipolar[signal][i][j] == 0:
                    count = count + 1
                    if count > 4:  # if sample is equal to zero for more than 2 seconds
                        impArtifact_labels[i] = 1
                        break  # skip if one impedance period of 2 seconds is present
                    if impArtifact_labels[i] == 1: break # skip if one impedance period is present in the epoch
                else:
                    if impArtifact_labels[i] == 1: break # skip if one impedance period is present in the epoch
                    count = 0
                if impArtifact_labels[i] == 1: break # skip if one impedance period is present in one epoch of one signal

    # Movement artifacts
    for signal in range(0, len(epochs_unipolar)):  # loop over each signal
        for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
            if movArtifact_labels[i] == 1:  # als er al een bewegingsartefact is gedetecteerd in een vorig signaal, sla over
                break
            if np.mean(np.absolute(epochs_unipolar[signal][i])) > ampl_thres or \
                    np.mean(np.absolute(epochs_unipolar[signal][i])) < -ampl_thres:
                movArtifact_labels[i] = 1

    # Artifact label (impedance + movement artifact)
    for i in range(0, len(artifact_labels)):
        if impArtifact_labels[i] == 1 or movArtifact_labels[i] == 1:
            artifact_labels[i] = 1

    return impArtifact_labels, movArtifact_labels, artifact_labels



def windowing(include_signals, signals, signal_headers, epoch_size):
    """

    :param y:
    :param epoch_size:
    :return:
    """

    y = signals[1]
    epochs = []
    for i in range(math.floor(len(y) / epoch_size) - 1):
        k = i * epoch_size
        epochs.append(y[k: k + epoch_size])
    return epochs


def bandpower(y, fs, fmin, fmax):
    """

    :param y:
    :param fs:
    :param fmin:
    :param fmax:
    :return:
    """
    f, Pxx = scipy.signal.periodogram(y, fs=fs)  # Default window, nfft, overlap settings
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def spectralEdge(y, fs, f_edge):
    """

    :param y:
    :param fs:
    :param f_edge:
    :return:
    """
    f, Pxx = scipy.signal.periodogram(y, fs=fs)  # Default window, nfft (= 256), overlap settings (= 128)
    ind_max = np.argmax(f > f_edge) - 1
    return np.trapz(Pxx[0: ind_max], f[0: ind_max])


def hjorth(a):
    """"
    """
    first_deriv = np.diff(a)
    second_deriv = np.diff(a, 2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity
