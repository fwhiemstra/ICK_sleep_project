import pyedflib
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#%% Load file
file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\2_Raw_data'
file_name = '1305609_11112020edf'
edf_file = os.path.join(file_path, file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)
print('Finished')

#%% Testen EEG frequency filtering
which_signal = 'EEG C3-C4' # Fill in to choose signal

# Extract signal
signal_labels_new = []
for i in range(0, len(signal_headers)):
    signal_labels_new.append(signal_headers[i]["label"])
y = signals[signal_labels_new.index(which_signal)]
fs = signal_headers[signal_labels_new.index(which_signal)].get("sample_rate")

# Frequency filter
n = 16 # Filter order
fco_low = 0.5 # lower cutoff frequency
fco_up = 48 # upper cutoff frequency
sos = signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)
y_filt = signal.sosfilt(sos, y) # filter signal

# Frequency response
plot = 1
if plot == 1:
    w, h = signal.sosfreqz(sos, fs=fs)
    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w, db)
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [Hz]')
    plt.show()

# Filter effect
plot = 1
if plot == 1:
    plt.plot(y, label='Original EEG signal')
    plt.plot(y_filt, label='Filtered EEG signal')
    plt.grid('minor')
    plt.ylabel('EEG amplitude [uV]')
    plt.xlabel('Time')
    plt.title('Frequency filter effect EEG signal')
    plt.legend()
    plt.show()

#%% Testen artifact filtering
which_signal = 'EEG C3-C4' # Fill in to choose signal

# Extract signal
signal_labels_new = []
for i in range(0, len(signal_headers)):
    signal_labels_new.append(signal_headers[i]["label"])
y = signals[signal_labels_new.index(which_signal)]
fs = signal_headers[signal_labels_new.index(which_signal)].get("sample_rate")

if y > 5000: # for ... seconds
    y = 'artifact'
elif y = 0: # for ... seconds
    y = 'artifact'

