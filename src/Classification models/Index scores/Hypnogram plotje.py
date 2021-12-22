import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

### File + index settings
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
feature_data_file_name = 'PICU007_FeatureData.xlsx'
start_time = datetime.datetime(2021, 10, 21, 16, 00, 00)
save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Presentaties\Figuren'
save_plot_file_name = 'Example_hypnogram_PICU.png'

#feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
#feature_data_file_name = 'PICU007_FeatureData.xlsx'
#start_time = datetime.datetime(2021, 10, 21, 16, 0, 0)
#save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Presentaties\Figuren'
#save_plot_file_name = 'Example_hypnogram_PICU.png'

print(datetime.datetime.now(), ': %s started' % feature_data_file_name)

### Load data
feature_data = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
order = ['Sleep stage N', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
#order = ['Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']

sleep_stage_labels = np.array([order.index(x) for x in feature_data["Sleep stage labels"]])

for i in range(0, len(sleep_stage_labels)):
    if sleep_stage_labels[i] == 0:
        sleep_stage_labels[i] = 1

### Plot settings
plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

plt.close('all')
plt.figure(figsize=(10, 4))
# Add time stamp labels
end_time = start_time + datetime.timedelta(seconds=30*len(sleep_stage_labels))
t = []
for i in range(len(sleep_stage_labels)):
    t.append((start_time + datetime.timedelta(seconds=30*i)).strftime('%H:%M'))

ticks = [i for i in range(0, len(t), 120)]
timestamp_labels = [t[i] for i in ticks]

# Hypnogram
plt.title('Hypnogram')
plt.step(range(0, len(sleep_stage_labels)), sleep_stage_labels, where='post',
            color='darkcyan', linewidth=1)
plt.xticks(ticks=ticks, labels=timestamp_labels, rotation=90)
plt.yticks(ticks=range(1, 6), labels=['N3', 'N2', 'N1', 'REM', 'Wake'])
plt.xlabel('Time (hh:mm)')
plt.ylabel('Sleep stage')
plt.tight_layout()
plt.xlim([0, len(sleep_stage_labels)])
plt.savefig(os.path.join(save_plot_file_path, save_plot_file_name), dpi=1000)

