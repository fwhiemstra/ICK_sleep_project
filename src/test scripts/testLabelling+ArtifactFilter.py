# Import packages
import os
import pandas as pd
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import math
import pandas

#%% Import file
#file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\3_Preprocessed_data'
#file_name =  'EDFtest1p.edf'
file_path = r'E:\ICK slaap project\PSG\3_Preprocessed_data'
file_name = 'PSG013p.edf'
nr_signal = 0 # signal to use as example signal (to determine sample rate and signal length)
epoch_size_s = 30 # window_size in seconds
unipolar_eegsignals = ['EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1', 'EEG O2', 'EEG A1', 'EEG A2']
include_signals = ['EOG ROC', 'EOG LOC', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4',
                   'EEG O1', 'EEG O2', 'EEG A1', 'EEG A2', 'EOG ROC-LOC', 'EEG F3-F4', 'EEG F3-C3',
                   'EEG F3-C4', 'EEG F3-O1', 'EEG F3-O2', 'EEG F3-A1', 'EEG F3-A2', 'EEG C3-C4',
                   'EEG C3-O1', 'EEG C3-O2', 'EEG C3-A1', 'EEG C3-A2', 'EEG O1-O2', 'EEG O1-A1',
                   'EEG O1-A2']

edf_file = os.path.join(file_path, file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

#%% Sleep stage labelling
# Input variables
fs = signal_headers[nr_signal].get("sample_rate")
epoch_size = epoch_size_s * fs
annotations = header["annotations"]
signal_length = len(signals[nr_signal])

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
sleep_stage_labels= []
for i in range(0, math.floor(signal_length / epoch_size) - 1):
    t = i * 30
    for j in range(0, len(raw_hyp_data) - 1):
        if raw_hyp_data[j][0] <= t < raw_hyp_data[j + 1][0]:
            sleep_stage_labels.append(raw_hyp_data[j][1])

if math.floor(signal_length / epoch_size) - 1 == len(sleep_stage_labels):
    print('Epoch labelling succesfully completed')
else:
    raise ValueError('Epoch size is not equal to hyp_data size')

#%% Artefact labelling
# Input variables: signal_headers, signals, unipolar_EEG signals, epoch_size
# Function variables: ampl_thres = 450
# Return: impArtifact_labels, movArtifact_labels, artifact_labels

## Step 1: Window all unipolar signals
signal_labels = []
epochs_unipolar = []
for i in range(0, len(signal_headers)):
    signal_labels.append(signal_headers[i].get("label"))
for i in unipolar_eegsignals:
    y = signals[signal_labels.index(i)]
    epochs = []
    for j in range(math.floor(len(y) / epoch_size) - 1):
        k = j * epoch_size
        epochs.append(y[k: k + epoch_size])
    epochs_unipolar.append(epochs)

## Step 2: Detect impedance + movement artifact + label
impArtifact_labels = np.zeros(len(epochs_unipolar[0]))
movArtifact_labels = np.zeros(len(epochs_unipolar[0]))
artifact_labels = np.zeros(len(epochs_unipolar[0]))

# Impedance artifacts
for signal in range(0, len(epochs_unipolar)): # loop over each signal
    for i in range(0, len(epochs_unipolar[signal])): # loop over each epoch in signal
        count = 0
        for j in range(0, len(epochs_unipolar[signal][i]), int(fs/2)): # check every half second if sample is equal to zero
            if epochs_unipolar[signal][i][j] == 0:
                count = count + 1
                if count > 4: # if more than 2 seconds
                    impArtifact_labels[i] = 1
                    break # skip if one impedance period of 2 seconds is present
                if impArtifact_labels[i] == 1: break
            else:
                if impArtifact_labels[i] == 1: break
                count = 0
            if impArtifact_labels[i] == 1: break

# Movement artifacts
for signal in range(0, len(epochs_unipolar)):  # loop over each signal
    for i in range(0, len(epochs_unipolar[signal])):  # loop over each epoch in signal
        if movArtifact_labels[i] == 1: # als er al een bewegingsartefact is gedetecteerd in een vorig signaal, sla over
            break
        if np.mean(np.absolute(epochs_unipolar[signal][i])) > ampl_thres or \
            np.mean(np.absolute(epochs_unipolar[signal][i])) < -ampl_thres:
            movArtifact_labels[i] = 1

# Artifact label
for i in range(0, len(artifact_labels)):
    if impArtifact_labels[i] == 1 or movArtifact_labels[i] == 1:
        artifact_labels[i] = 1

#%% Test and evaluate impedance artefact filter
artifact_indices = []
artifact_labels = np.zeros(len(epochs_unipolar[0]))
plot_count = 0
for signal in range(0, len(epochs_unipolar)): # loop over each signal
    for i in range(0, len(epochs_unipolar[signal])): # loop over each epoch in signal
        count = 0
        for j in range(0, len(epochs_unipolar[signal][i]), int(fs/2)): # check every half second if sample is equal to zero
            if epochs_unipolar[signal][i][j] == 0:
                count = count + 1
                if count > 4: # if more than 2 seconds
                    plot_count = plot_count + 1
                    plt.plot(epochs_unipolar[signal][i]+1000*plot_count)
                    plt.grid(which='both')
                    plt.title('Impedance artefact epochs')
                    plt.show()
                    artifact_indices.append(['Signal index: '+ str(signal) + ', Epoch index: '+ str(i)])
                    artifact_labels[i] = 1
                    break # skip if one impedance period of 2 seconds is present
                if artifact_labels[i] == 1: break
            else:
                if artifact_labels[i] == 1: break
                count = 0
            if artifact_labels[i] == 1: break

#%% Test and evaluate movement artifact filter
#%% Plot
import matplotlib.pylab as pylab

fs = 250

def bandpower(y, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(y, fs=fs) # Default window, nfft, overlap settings
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

abs_lowFreq_power = np.zeros(len(epochs_unipolar[0]))
rel_lowFreq_power = np.zeros(len(epochs_unipolar[0]))
abs_amplMean = np.zeros(len(epochs_unipolar[0]))
variance = np.zeros(len(epochs_unipolar[0]))

signal = 5
for i in range(0, len(epochs_unipolar[signal])): # loop over each epoch in signal
        abs_lowFreq_power[i] = bandpower(epochs_unipolar[signal][i], fs, 0, 0.5)
        abs_amplMean[i] = np.mean(np.absolute(epochs_unipolar[signal][i]))
        variance[i] = np.var(epochs_unipolar[signal][i])

        if bandpower(epochs_unipolar[signal][i], fs, 0, 48) == 0:
            rel_lowFreq_power[i] = 1
        else:
            rel_lowFreq_power[i] = bandpower(epochs_unipolar[signal][i], fs, 0, 0.5)/bandpower(epochs_unipolar[signal][i], fs, 0, 48)

# Plot
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small'}

pylab.rcParams.update(params)
order = ['Sleep stage N4', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
sleep_stage_labels_plot = [order.index(x) for x in sleep_stage_labels]

fig, axs = plt.subplots(5) # , sharex=True : align x axis subplots

fig.suptitle(file_name)
axs[0].plot(abs_lowFreq_power, label='Absolute low frequency power')
axs[0].set_title('Absolute low frequency power, mean = {:.2f}'.format(np.mean(abs_lowFreq_power), fontsize=8))
axs[0].grid()
axs[0].set_ylim([0, 50000000])

axs[1].plot(rel_lowFreq_power, label='Relative low frequency power')
axs[1].set_title('Relative low frequency power, mean = {:.2f}'.format(np.mean(rel_lowFreq_power)))
axs[1].grid()
axs[1].set_ylim([0, 1])

axs[2].plot(abs_amplMean, label='Mean absolute amplitude')
axs[2].set_title('Mean absolute amplitude, mean = {:.2f}'.format(np.mean(abs_amplMean)))
axs[2].grid()
axs[2].set_ylim([0, 10000])

axs[3].plot(variance, label='Variance')
axs[3].grid()
axs[3].set_title('Variance, mean = {:.2f}'.format(np.mean(variance)))
axs[2].set_ylim([0, 7500])

axs[4].step(range(0, len(sleep_stage_labels)), sleep_stage_labels_plot, where='post')
axs[4].set_yticks(range(len(order))) # set number of labels y axis
axs[4].set_yticklabels(order) # add labels to y axis
axs[4].set_title('Hypnogram')
axs[4].set_ylim([0, 6])
axs[4].grid()
plt.tight_layout(h_pad=0.1) # prevents overlapping of subtitles and xlabels

#%% Test and evaluate movement artifact filter
artifact_indices = []
artifact_labels = np.zeros(len(epochs_unipolar[0]))
artifact_indices2 = []
artifact_labels2 = np.zeros(len(epochs_unipolar[0]))
plot_count = 0
plot_count2 = 0

ampl_thres = 450
pow_thres = 1000000
for signal in range(0, len(epochs_unipolar)): # loop over each signal
    for i in range(0, len(epochs_unipolar[signal])): # loop over each epoch in signal
        if artifact_labels[i] == 1: # als er al een bewegingsartefact is gedetecteerd in een vorig signaal, sla over
            break
        if np.mean(np.absolute(epochs_unipolar[signal][i])) > ampl_thres or \
            np.mean(np.absolute(epochs_unipolar[signal][i])) < -ampl_thres:
            artifact_labels[i] = 1
            artifact_indices.append(['Signal index: ' + str(signal) + ', Epoch index: ' + str(i)])
            plot_count = plot_count + 1
            plt.figure(1)
            plt.plot(epochs_unipolar[signal][i] + 1000 * plot_count)
            plt.grid(which='both')
            plt.title('Movement artefact epochs (Amplitude based) - ' + file_name)
            plt.show()
        if bandpower(epochs_unipolar[signal][i], fs, 0, 0.5) > pow_thres:
            artifact_labels2[i] = 1
            artifact_indices2.append(['Signal index: ' + str(signal) + ', Epoch index: ' + str(i)])
            plot_count2 = plot_count2 + 1
            plt.figure(2)
            plt.plot(epochs_unipolar[signal][i] + 1000 * plot_count2)
            plt.grid(which='both')
            plt.title('Movement artefact epochs (Signal power based) - ' + file_name)
            plt.show()

#%% Feature calculation

#%% Functions
def bandpower(y, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(y, fs=fs) # Default window, nfft, overlap settings
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def spectralEdge(y, fs, f_edge):
    f, Pxx = scipy.signal.periodogram(y, fs=fs) # Default window, nfft (= 256), overlap settings (= 128)
    ind_max = np.argmax(f > f_edge) - 1
    return np.trapz(Pxx[0: ind_max], f[0: ind_max])

def hjorth(a):
    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity

def windowing(y, epoch_size):
    epochs = []
    for i in range(math.floor(len(y)/epoch_size)-1):
        k = i*epoch_size
        epochs.append(y[k: k+epoch_size])
    return epochs
#%% Select signal
which_signal = 'EEG C3-C4' # Fill in to choose signal

# Extract signal
signal_labels = []
for i in range(0, len(signal_headers)):
    signal_labels.append(signal_headers[i]["label"])
y_raw = signals[signal_labels.index(which_signal)]
fs = signal_headers[signal_labels.index(which_signal)].get("sample_rate")

#%% Apply frequency filter to signals
n = 16 # Filter order
fco_low = 0.5 # lower cutoff frequency
fco_up = 48 # upper cutoff frequency
sos = signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)
y = signal.sosfilt(sos, y_raw) # filter signal


#%% Cut signal into epochs
epoch_size_s = 30 # window_size in seconds
epoch_size = epoch_size_s * fs

epochs = windowing(y, epoch_size)

#%% Label each artifact

#%% Calculate features
# Fill feature arrays with zeros
total_power = np.zeros(len(epochs))
gamma_power = np.zeros(len(epochs))
beta_power = np.zeros(len(epochs))
alpha_power = np.zeros(len(epochs))
theta_power = np.zeros(len(epochs))
delta_power = np.zeros(len(epochs))
gamma_delta_ratio = np.zeros(len(epochs))
beta_theta_ratio = np.zeros(len(epochs))
theta_delta_ratio = np.zeros(len(epochs))
beta_delta_ratio = np.zeros(len(epochs))
sef95 = np.zeros(len(epochs))
activity = np.zeros(len(epochs))
mobility = np.zeros(len(epochs))
complexity = np.zeros(len(epochs))
variance = np.zeros(len(epochs))

# Feature calculation for each epoch
for i in range(0, len(epochs)):
    total_power[i] = bandpower(epochs[i], fs, 0.5, 48)
    gamma_power[i] = bandpower(epochs[i], fs, 30, 48)/total_power[i]*100
    beta_power[i] = bandpower(epochs[i], fs, 12, 30)/total_power[i]*100
    alpha_power[i] = bandpower(epochs[i], fs, 7, 12)/total_power[i]*100
    theta_power[i] = bandpower(epochs[i], fs, 4, 7)/total_power[i]*100
    delta_power[i] = bandpower(epochs[i], fs, 0.5, 4)/total_power[i]*100
    gamma_delta_ratio[i] = gamma_power[i]/delta_power[i]
    beta_theta_ratio[i] = beta_power[i]/theta_power[i]
    theta_delta_ratio[i] = theta_power[i]/delta_power[i]
    beta_delta_ratio[i] = beta_power[i]/delta_power[i]
    sef95[i] = spectralEdge(epochs[i], fs, 95)
    activity[i], mobility[i], complexity[i] = hjorth(epochs[i])
    variance[i] = np.var(epochs[i])

#%% Create feature dataframe + convert to excel
dfFeatureData = pd.DataFrame({'Sleep stages': hyp_data,
                              which_signal+': Total power': total_power,
                              which_signal+': Gamma power': gamma_power,
                              which_signal + ': Beta power': beta_power,
                              which_signal + ': Alpha power': alpha_power,
                              which_signal + ': Delta power': delta_power,
                              which_signal + ': Theta power': theta_power,
                              which_signal + ': Delta power': delta_power,
                              which_signal + ': Gamma/Delta ratio': gamma_delta_ratio,
                              which_signal + ': Beta/Theta ratio': beta_theta_ratio,
                              which_signal + ': Theta/Delta ratio': theta_delta_ratio,
                              which_signal + ': Beta/Delta ratio': beta_delta_ratio,
                              which_signal + ': Spectral edge frequency': sef95,
                              which_signal + ': Activity': activity,
                              which_signal + ': Mobility': mobility,
                              which_signal + ': Complexity': complexity,
                              which_signal + ': Variance': variance
                              })

#%% Convert to excel file
xlsx_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\4_Feature_data'
xlsx_file_name = file_name.replace('p.edf','_FeatureData.xlsx')

writer = pd.ExcelWriter(os.path.join(xlsx_file_path, xlsx_file_name))
dfFeatureData.to_excel(writer)
writer.save()

#%% Testen artifact filtering
which_signal = 'EEG C3-C4' # Fill in to choose signal

# Extract signal
signal_labels_new = []
for i in range(0, len(signal_headers)):
    signal_labels_new.append(signal_headers[i]["label"])
y = signals[signal_labels_new.index(which_signal)]
fs = signal_headers[signal_labels_new.index(which_signal)].get("sample_rate")

for i in range(0, len(epochs)):
    for j in range(0, len(epochs[i]), fs/2):
        count = 0
        if epochs[i][j] == 0:
            count = count + 1
            if count > 4: print("Impedance artifact")
        else:
            count = 0


