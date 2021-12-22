"""
This functions evaluates the training performance of index-based measures per group of patients accross EEG channels
June 2021, Floor Hiemstra
"""

import os
import numpy as np
import pandas as pd
from Functions import load_and_prepare_raw_feature_data_multiple, \
    compute_feature_posterior_prob, auc_feature_calculation_posterior, find_optimal_index_thresholds
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
"""
##### SET DATA VARIABLES #####
"""
# NOTE: This function only works for features that are positively correlated with
# the sleep stages, thus increase during Wake and decrease during SWS.

load_data = 1  # 0 if data is already loaded
data_type = 'new'  # 'new' or 'sampled'

### Variables, if data == 'new'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years']
remove_artifacts = 1

### Variables, if data == 'sampled'
age_category = '0-18years'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
sampled_data_file_name = '10000sampled_feature_data_%s_nonscaled.xlsx' % age_category

"""
##### LOAD FEATURE DATA #####
"""
if load_data == 1 and data_type == 'new':
    feature_data = load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                                              feature_data_file_path, include_age_categories,
                                                              remove_artifacts)
elif load_data == 1 and data_type == 'sampled':
    feature_data = pd.read_excel(os.path.join(sampled_data_file_path, sampled_data_file_name))

feature_data = feature_data.sample(frac=1)

#age_category = '13-18 years'
#feature_data_sub = feature_data[feature_data['Age category'] == age_category]
#%%
feature_data = feature_data[feature_data["Patient key"] != 'PSG046']
#%% Hue = ages
plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

plt.figure(figsize=(10,4))
feature = 'EEG F3-C4: Abs Theta power'
title = 'Theta power'
#feature = 'EEG F3-C4: Gamma/delta ratio'
#feature = 'EEG F3-C4: Gamma/(Theta+Delta) ratio'
hue = 'Age category'
plt.yscale('Log')
sns.set(style='whitegrid', rc={"grid.linewidth": 0.01})
sns.set_context(font_scale=1)
sns.boxplot(x=feature_data["Sleep stage labels"],
            y=feature_data[feature],
            #y=feature_data['EEG F3-C4: Abs Gamma power']/(feature_data['EEG F3-C4: Abs Theta power'] + feature_data['EEG F3-C4: Abs Delta power']),
            hue=feature_data[hue],
            hue_order=['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years'],
            order=['Sleep stage W', 'Sleep stage R', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage N'],
            #data=feature_data,
            #palette="Set2",
            palette="viridis",
            fliersize=0.2,
            linewidth=0.5)
plt.title(title)
plt.ylabel('Power spectral density')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Wake', 'REM', 'N1', 'N2', 'N3', 'N'])
plt.legend(loc='upper center', ncol=8, fontsize=8)
#plt.xticks(rotation=90)
plt.tight_layout()

#%% Subplots per age
plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

fig, axes = plt.subplots(2, 4, sharex='col', sharey='row')
feature = 'EEG F3-C4: Abs Delta power'
#feature = 'EEG F3-C4: Gamma/delta ratio'
#feature = 'EEG F3-C4: Gamma/(Theta+Delta) ratio'

plt.suptitle(feature)

count = 0
row = 0
for age_category in include_age_categories:
    data = feature_data[feature_data['Age category'] == age_category]
    if count > 3:
        count = 0
        row = 1
    axes[row, count].set_yscale('log')
    axes[row, count].set_title(age_category)
    axes[row, count].tick_params(axis='x', rotation=90)
    #axes[row, count].set_ylim([10e-5, 10e-1])

    sns.boxplot(x=data["Sleep stage labels"],
                y=data[feature],
                #y=data['EEG F3-C4: Abs Gamma power']/(data['EEG F3-C4: Abs Theta power'] + data['EEG F3-C4: Abs Delta power']),
                order=['Sleep stage W', 'Sleep stage R', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage N'],
                #data=feature_data,
                palette="Set2",
                fliersize=0.2,
                linewidth=0.5,
                ax=axes[row, count])
    axes[row, count].set_xlabel("")
    axes[row, count].set_ylabel("")
    axes[row, count].set_xticks(ticks=[0, 1, 2, 3, 4, 5])
    axes[row, count].set_xticklabels(labels=['Wake', 'REM', 'N1', 'N2', 'N3', 'N'])
    plt.grid(which='minor')
    if count == 0:
        axes[row, count].set_ylabel('Ratio')
    count = count+1

plt.tight_layout()
#plt.legend(loc='lower left')