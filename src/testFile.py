#%%
%reset
# Import packages
import os
import matplotlib
import pandas as pd
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import math
import pandas
#from src.process_raw_data.EDFfiles import load_edf_file
#from main import print_hi
#from settings import RAW_DATA, INPUT_DATA

# https://pyedflib.readthedocs.io/en/latest/ref/highlevel.html#pyedflib.highlevel.anonymize_edf

# %reset # to remove all variables
# del a: to remove 1 variable
#%matplotlib # to open interactive matlab-like plot
#plt.isinteractive(), plt.ion(), plt.ioff()

#%% Load + import file
# Select file
data_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\2_raw_data'
# file_name = '1410419_26112020.EDF'
# file_name = '1305609_11112020.EDF'
# file_name = '1438646_26112020.EDF'
# file_name = 'EDFtest1_anonymized.edf'
# file_name = 'EDFtest2_anonymized.edf'
file_name = 'EDFtest3.edf'
# file_name = 'EDFtest3_anonymized.edf'

edf_file = os.path.join(data_path, file_name)

# Import file
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file)
annotations = pyedflib.highlevel.read_edf_header(edf_file, read_annotations=True)

#%% Lead selection C3/C4
nr = 16
signal_c3 = signals[nr]
nr = 17
signal_c4 = signals[nr]
fs = signal_headers[nr].get("sample_rate")
x = np.linspace(0, len(signal_c3)/fs, num=len(signal_c3))

signal_c3toc4 = signal_c3-signal_c4
plot = 0
if plot == 1:
    label = 'C3/C4'
    plt.plot(x, signal_c3toc4)
    plt.ylabel("EEG Amplitude")
    plt.xlabel("Time [seconds]")
    plt.title(label)
    plt.grid()
    plt.xlim([30, 60])
    plt.show()

#%% Bandpass filtering
n = 16 # filter order
fco_low = 0.5 # lower cutoff frequency
fco_up = 48 # upper cutoff frequency
fs = signal_headers[nr].get("sample_rate") # sampling rate
sos = signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)
y = signal.sosfilt(sos, signal_c3toc4)
plot = 0

if plot == 1:
    x = np.linspace(0, len(y)/fs, num=len(y))
    label = "C3/C4 filtered"
    plt.figure('C3/C4 filtered')
    plt.plot(x, signal_c3toc4,label='unfiltered')
    plt.plot(x, y, label='bp filtered')
    plt.ylabel("EEG Amplitude")
    plt.xlabel("Time [seconds]")
    plt.title(label)
    plt.grid()
    plt.legend(framealpha=1, frameon=True);
    plt.show()

#%% Windowing
def windowing(y, epoch_size):
    epochs = []
    for i in range(math.floor(len(y)/epoch_size)-1):
        k = i*epoch_size
        epochs.append(y[k: k+epoch_size])
    return epochs

epoch_size_s = 30 # window_size in seconds
epoch_size = epoch_size_s * fs

epochs = windowing(y, epoch_size)
#epochs = np.array_split(y, len(y)/epoch_size)

#%% Calculate features

# Define feature calculation functions
def bandpower(y, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(y, fs=fs) # Default window, nfft, overlap settings
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def spectralEdge(y, fs, f_edge):
    f, Pxx = scipy.signal.periodogram(y, fs=fs) # Default window, nfft (= 256), overlap settings (= 128)
    ind_max = np.argmax(f > f_edge) - 1
    return np.trapz(Pxx[0: ind_max], f[0: ind_max])

# Fill feature arrays with zeros
total_power = np.zeros(len(epochs))
alpha = np.zeros(len(epochs))
beta = np.zeros(len(epochs))
theta = np.zeros(len(epochs))
delta = np.zeros(len(epochs))
gamma = np.zeros(len(epochs))
sef95 = np.zeros(len(epochs))
idos = np.zeros(len(epochs))

# Feature calculation for each epoch
for i in range(0, len(epochs)):
    total_power[i] = bandpower(epochs[i], fs, 0.5, 48)
    alpha[i] = bandpower(epochs[i], fs, 7, 12)/total_power[i]*100
    beta[i] = bandpower(epochs[i], fs, 12, 30)/total_power[i]*100
    theta[i] = bandpower(epochs[i], fs, 4, 7)/total_power[i]*100
    delta[i] = bandpower(epochs[i], fs, 0.5, 4)/total_power[i]*100
    gamma[i] = bandpower(epochs[i], fs, 30, 48)/total_power[i]*100
    idos[i] = gamma[i]/delta[i]
    sef95[i] = spectralEdge(epochs[i], fs, 95)

#%% Extract hypnograms
hyp_stagesLabel = []
hyp_t = []
hyp_stagesNum = []
for i in range(len(annotations.get('annotations'))):
    hyp_annotation = annotations.get('annotations')[i]
    if i == 0:
        hyp_t.append(0)
        hyp_stagesNum.append(5)
        hyp_stagesLabel.append('Sleep stage W')
    if hyp_annotation[2] == 'Sleep stage W':
        hyp_t.append(hyp_annotation[0])
        hyp_stagesLabel.append(hyp_annotation[2])
        hyp_stagesNum.append(5)
    if hyp_annotation[2] == 'Sleep stage R':
        hyp_t.append(hyp_annotation[0])
        hyp_stagesLabel.append(hyp_annotation[2])
        hyp_stagesNum.append(4)
    if hyp_annotation[2] == 'Sleep stage N1':
        hyp_t.append(hyp_annotation[0])
        hyp_stagesLabel.append(hyp_annotation[2])
        hyp_stagesNum.append(3)
    if hyp_annotation[2] == 'Sleep stage N2':
        hyp_t.append(hyp_annotation[0])
        hyp_stagesLabel.append(hyp_annotation[2])
        hyp_stagesNum.append(2)
    if hyp_annotation[2] == 'Sleep stage N3':
        hyp_t.append(hyp_annotation[0])
        hyp_stagesLabel.append(hyp_annotation[2])
        hyp_stagesNum.append(1)
    if hyp_annotation[2] == 'Sleep stage N4':
        hyp_t.append(hyp_annotation[0])
        hyp_stagesLabel.append(hyp_annotation[2])
        hyp_stagesNum.append(0)
hyp_data_array = [hyp_t, hyp_stagesLabel, hyp_stagesLabel, hyp_stagesNum]

# Convert to dataframe
hyp_data = pd.DataFrame(hyp_data_array, index=['Timestamp', 'Sleep stage label', 'Sleep stage label 2', 'Sleep stage number']).T
# Sort zodat het goed komt te staan in hypnogram
order = ['Sleep stage N4', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
hyp_data['Sleep stage label'] = [order.index(x) for x in hyp_data['Sleep stage label']]

#%% Plot hypnogram
plt.yticks(range(len(order)), order)
plt.step(hyp_data['Timestamp'], hyp_data['Sleep stage label'], where='post', color='tab:cyan')
plt.grid(which='both')
plt.ylabel('Sleep stage')
plt.xlabel('Time [seconds from start]')
plt.title('Hypnogram')
plt.show()

#%% Plot features + hypnogram
fig, axs = plt.subplots(2) # , sharex=True : align x axis subplots
fig.suptitle('Recording 2')

xx = np.linspace(0, 40200, len(epochs)) # same time axis as hypnogram, seconds from start
# https://matplotlib.org/stable/gallery/color/color_by_yvalue.html
#axs[0].plot(xx[idos >= 0.001], idos[idos >= 0.001], color='tab:orange') #proberen dat lijntje andere kleur krijgt als het >/< threshold zit
#axs[0].plot(xx[idos < 0.001], idos[idos < 0.001], color='tab:cyan')
#axs[0].plot(xx, idos, color='tab:orange')
axs[0].plot(xx[idos >= 0.001], idos[idos >= 0.001], xx[idos < 0.001], idos[idos < 0.001])
axs[0].grid(which='minor') # minor grid version
axs[0].set_yscale('log') # log plot
axs[0].grid(which = 'both')
axs[0].set_title('IDOS index')
axs[0].set_ylabel('IDOS')
axs[0].set_xticklabels([]) # remove number at x axis
axs[0].set_xlim([0, 40000])

axs[1].set_yticks(range(len(order))) # set number of labels y axis
axs[1].set_yticklabels(order) # add labels to y axis
axs[1].step(hyp_data['Timestamp'], hyp_data['Sleep stage label'], where='post', color = 'tab:blue')
axs[1].grid(which = 'both')
axs[1].set_ylabel('Sleep stage')
axs[1].set_title('Hypnogram')
axs[1].set_xticklabels([])
axs[1].set_xlim([0, 40000])


#axs[2].plot(xx, gamma*10, label = 'gamma', color = 'tab:green')
#axs[2].plot(xx, delta, label = 'delta', color = 'tab:purple')
#axs[2].plot(xx, sef95/100, label = 'sef95', color = 'tab:cyan')
#axs[2].set_title('Gamma and delta power')
#axs[2].set_xlabel('Time')
#xs[2].set_ylabel('Band power')
#axs[2].legend()
#axs[2].grid(which = 'both')
#axs[2].set_xticklabels([])
#axs[2].set_xlim([0, 40000])

plt.tight_layout() # prevents overlapping of subtitles and xlabels
plt.show()

#%% Visualize features over time with hypnograms
fig, ax1 = plt.subplots()
ax2 = ax1.twinx() # secondary axis
plt.title('Hypnogram + IDOS - PSG 1')
plt.step(hyp_data[0], hyp_data[1], where='post')
plt.plot(xx, idos)
plt.grid(which = 'both')

#%% https://mne.tools/dev/auto_tutorials/sample-datasets/plot_sleep.html
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5613192/

#%%


#%% Plot features
#plt.plot(alpha, label = 'alpha')
#plt.plot(beta, label = 'beta')
#plt.plot(sef95/1000, label = 'sef95')
plt.plot(delta, label = 'delta')
plt.plot(1/delta*1000, label = '1/delta')

plt.plot(gamma*10, label = 'gamma')
#plt.plot(theta, label = 'theta')
plt.plot(idos*500, label = 'IDOS')
plt.legend()



#%% Spectrogram
Sxx, f, t, im = plt.specgram(y, NFFT=256, Fs=fs, noverlap=128) # window = hanning
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(top=48)
plt.title('Spectrogram')
plt.show()

#%% Convert time label from seconds to hh:mm:ss
import datetime
start_time = 70200 #19.30 uur
x_seconds = np.linspace(start_time,40200+start_time,len(epochs))
x_hhmmss = []
for i in range(len(x_seconds)):
    x_hhmmss.append(datetime.timedelta(seconds = x_seconds[i]))