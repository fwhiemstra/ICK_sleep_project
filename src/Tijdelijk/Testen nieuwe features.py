import os
import pyedflib
from src.Functions import windowing, NaNfill, bandpower, hjorth_features, spectralFeatures
import scipy
import numpy as np
import pandas as pd
import eeglib
import pywt

#%% Load file
file_name = 'PSG014p.edf'
file_path = r'E:\ICK slaap project\3_Preprocessed_data'

edf_file = os.path.join(file_path, file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

fs = signal_headers[0].get('sample_rate')
epoch_size = 30*fs

#%% FUNCTION
dfFeatureData = pd.DataFrame()

# Function variables:
include_signals = ['EOG ROC-LOC', 'EMG EMG Chin', # EOG + EMG channel
                   'EEG F3', 'EEG C3', 'EEG O1', 'EEG A1', # Unipolar channels
                   'EEG F3-C3', 'EEG F3-C4', 'EEG F3-O2', 'EEG F3-A2', 'EEG C3-C4', 'EEG C3-O2', 'EEG C3-A2',
                        'EEG O1-O2', 'EEG O1-A2' # Bipolar channels
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
    eog_bandpower_rem[i] = bandpower(epochs[i], fs, 15*fs, 0.5, 0.35, 0.5)
    eog_bandpower_rem2[i] = bandpower(epochs[i], fs, 15*fs, 0.5, 0.35, 2)
    eog_bandpower_sem[i] = bandpower(epochs[i], fs, 15*fs, 0.5, 0, 0.35)
    eog_variance[i] = np.var(epochs[i])
dfFeatureData[signal_label + ': Bandpower REM (0.35-0.5)'] = eog_bandpower_rem
dfFeatureData[signal_label + ': Bandpower REM (0.5-2)'] = eog_bandpower_rem2
dfFeatureData[signal_label + ': Bandpower SEM'] = eog_bandpower_sem
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
nfft = 2*fs
noverlap = 0.5

for signal_label in include_signals[2:]:
#signal_label = include_signals[2]
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
            =spectralFeatures(epochs[i], fs, nfft, noverlap)

        # Time-frequency domain features
        # Downsample signal
        if fs == 250:
            downsample_factor = 2.5
        elif fs == 256:
            downsample_factor = 2.56
        y_downsampled = scipy.signal.resample(epochs[i], round(len(epochs[i]) / downsample_factor))
        # Discrete wavelet transform
        cA, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(y_downsampled, 'db4', level=5)

        cA_mean[i] = np.mean(cA)
        cA_std[i] = np.std(cA)
        cD5_mean[i] = np.mean(cD5)
        cD5_std[i] = np.std(cD5)
        cD4_mean[i] = np.mean(cD4)
        cD4_std[i] = np.std(cD4)
        cD3_mean[i] = np.mean(cD3)
        cD3_std[i] = np.std(cD3)
        cD2_mean[i] = np.mean(cD2)
        cD2_std[i] = np.std(cD2)
        cD1_mean[i] = np.mean(cD1)
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
    dfFeatureData[signal_label + ': DWT cD4 std'] = cD4_mean
    dfFeatureData[signal_label + ': DWT cD4 std'] = cD4_std
    dfFeatureData[signal_label + ': DWT cD3 std'] = cD3_mean
    dfFeatureData[signal_label + ': DWT cD3 std'] = cD3_std
    dfFeatureData[signal_label + ': DWT cD2 std'] = cD2_mean
    dfFeatureData[signal_label + ': DWT cD2 std'] = cD2_std
    dfFeatureData[signal_label + ': DWT cD1 std'] = cD1_mean
    dfFeatureData[signal_label + ': DWT cD1 std'] = cD1_std





