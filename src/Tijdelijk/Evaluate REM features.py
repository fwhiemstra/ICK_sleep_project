# Import packages
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from Functions import load_and_prepare_raw_feature_data_multiple
import seaborn as sns

"""
##### SET DATA VARIABLES #####
"""

load_data = 1  # 0 if data is already loaded
data_type = 'sampled'  # 'new' or 'sampled'

### Variables, if data == 'new'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years']
remove_artifacts = 1

### Variables, if data == 'sampled'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
sampled_data_file_name = '5000sampled_feature_data_0-18years_nonscaled.xlsx'

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

"""
##### ADD REM SLEEP LABELS (REM, NO REM) #####
"""
sleep_stage_column_name = 'REM labels'
rem_stage_labels = []
for epoch in feature_data.iterrows():
    if epoch[1]['Sleep stage labels'] == 'Sleep stage R':
        rem_stage_labels.append('REM')
    else:
        rem_stage_labels.append('no REM')
feature_data[sleep_stage_column_name] = rem_stage_labels

#%%
"""
##### START FEATURE EVALUATION #####
"""
#feature = 'EMG EMG Chin: Energy'
#feature = 'EMG EMG Chin: Mean absolute amplitude'
#feature = 'EOG ROC-LOC: Bandpower REM (0.35-0.5)'
#feature = 'EOG ROC-LOC: Bandpower REM (0.5-2)'
#feature = 'EOG ROC-LOC: Bandpower SEM (0-0.35)'
feature = 'EOG ROC-LOC: Variance'

plt.figure()
plt.yscale('log')
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=0.9)
sns.boxplot(x='Sleep stage labels',
            y=feature,
            data=feature_data,
            order=['Sleep stage W', 'Sleep stage R', 'Sleep stage N', 'Sleep stage N1', 'Sleep stage N2',
                   'Sleep stage N3'],
            palette="Set3")
plt.title('Boxplot %s' % feature)

#%%
"""
##### START FEATURE EVALUATION #####
"""
#feature = 'EMG EMG Chin: Energy'
#feature = 'EMG EMG Chin: Mean absolute amplitude'
#feature = 'EOG ROC-LOC: Bandpower REM (0.35-0.5)'
#feature = 'EOG ROC-LOC: Bandpower REM (0.5-2)'
feature = 'EOG ROC-LOC: Bandpower SEM (0-0.35)'
#feature = 'EOG ROC-LOC: Variance'


featureData = feature_data.copy()
plt.figure()
plt.title('ROC curve %s)' % feature)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.yscale('linear')
plt.grid(which='both')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--')

AUCs = []
optimal_thresholds = []

label_true = []
label_false = []
thresholds = np.linspace(np.min(featureData[feature]), np.max(featureData[feature]), 10000000)
TPR = np.zeros(len(thresholds))
FPR = np.zeros(len(thresholds))

for i in range(0, len(featureData[sleep_stage_column_name])):
    if featureData.iloc[i][sleep_stage_column_name] == 'REM':
        label_true.append(featureData.iloc[i][feature])
    if featureData.iloc[i][sleep_stage_column_name] != 'REM':
        label_false.append(featureData.iloc[i][feature])

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

AUC = abs(np.trapz(TPR, FPR))
if AUC < 0.5:
    AUC = 1 - AUC
    optimal_idx = np.argmax(FPR - TPR)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(TPR, FPR, label='AUC = {:.2f}'.format(AUC))
    plt.legend()
else:
    optimal_idx = np.argmax(TPR - FPR)
    optimal_threshold_rem = thresholds[optimal_idx]
    plt.plot(FPR, TPR, label='AUC = {:.2f}'.format(AUC))
    plt.legend()

AUC_rem = "{:.2f}".format(AUC)  # 2 decimals


