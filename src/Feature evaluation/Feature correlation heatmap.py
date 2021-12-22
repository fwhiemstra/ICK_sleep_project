import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
feature_data_file_name = '10000sampled_feature_data_0-18years_nonscaled.xlsx'
feature_data = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

#%%
eeg_channel = 'EEG F3-C3:'
eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]
feature_list = [name.replace(eeg_channel, '') for name in eeg_features_list]

feature_list = [' Abs. mean amplitude',
 ' Variance',
 ' Zero crossing rate',
 ' Interquartile range',
 ' Signal sum',
 ' Energy',
 ' Kurtosis',
 ' Skewness',
 ' Entropy',
 ' Hjorth activity',
 ' Hjorth mobility',
 ' Hjorth complexity',
 ' Higuchi Fractional Dimension',
 ' Detrended Fluctuation Analysis',
 ' Total power',
 ' Abs.beta power',
 ' Abs. alpha power',
 ' Abs. sigma power',
 ' Abs. theta power',
 ' Abs. delta power',
 ' Abs. gamma power',
 ' Rel. beta power',
 ' Rel. alpha power',
 ' Rel. sigma power',
 ' Rel. theta power',
 ' Rel. delta power',
 ' Rel. gamma power',
 ' Gamma/delta ratio',
 ' Gamma/theta ratio',
 ' Beta/delta ratio',
 ' Beta/theta ratio',
 ' Alpha/delta ratio',
 ' Alpha/theta ratio',
 ' Spectral edge',
 ' Median freq.',
 ' Mean freq.',
 ' Spectral kurtosis',
 ' Spectral skewness',
 ' Spectral entropy',
 ' DWT cA mean',
 ' DWT cA SD',
 ' DWT cD5 mean',
 ' DWT cD5 SD',
 ' DWT cD4 mean',
 ' DWT cD4 SD',
 ' DWT cD3 mean',
 ' DWT cD3 SD',
 ' DWT cD2 mean',
 ' DWT cD2 SD',
 ' DWT cD1 mean',
 ' DWT cD1 SD']

features = pd.DataFrame(feature_data[eeg_features_list].values, columns=feature_list)
#eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]

#%%
plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

plt.close('all')
plt.figure(figsize=(12, 12))
heatmap = sns.heatmap(features.corr(), vmin=-1, vmax=1, annot_kws={"size": 4}, fmt='.2f', cmap='BrBG', annot=True)
#heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.tight_layout()
plt.savefig(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Thesis report\Figuren thesis report\Feature heatmap.png')


