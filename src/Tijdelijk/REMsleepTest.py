"""

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from Functions import load_and_prepare_raw_feature_data_multiple
import seaborn as sns

#%%
### Set variables
# File paths and names
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
results_plots_file_path = r'E:\ICK slaap project\6_IDOSevaluation'

# Function variables
age_category = '13-18 years'
show_boxplot = 1
save_boxplot = 1
show_ROC_plot = 1
save_ROC_plot = 1

### Initiate function
patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
feature = 'EOG ROC-LOC: Bandpower'
#feature = 'EMG EMG Chin: Energy'
ttl_feature = feature
keys = patientData[patientData['Age category'] == age_category]['Key']
feature_data_file_names = [key + '_FeatureData.xlsx' for key in keys]
#feature_data_file_names = feature_data_file_names[0:2]

file_keys = []
auc_wake = []
auc_sws = []
threshold_wake = []
threshold_sws = []
anova_fval = []
spearman_corr = []
cohens_kappa = []
contingency_tables = pd.DataFrame([])
accuracy = []

#%%
count = 0
for feature_data_file_name in feature_data_file_names:
    if feature_data_file_name in os.listdir(feature_data_file_path):
        ttl_feature = feature.replace(':', '_')

        plt.close('all')
        count = count + 1
        featureData = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

        file_key = feature_data_file_name.replace('_FeatureData.xlsx', '')

        ### Add REM sleep labels (REM, )
        sleep_stage_column_name = 'REM labels'
        rem_stage_labels = []
        for epoch in featureData.iterrows():
            if epoch[1]['Sleep stage labels'] == 'Sleep stage R':
                rem_stage_labels.append('REM')
            else:
                rem_stage_labels.append('no REM')

        featureData[sleep_stage_column_name] = rem_stage_labels
        featureData[sleep_stage_column_name + '- linear'] = rem_stage_labels

        ### Create ROCs, calculate AUCs and determine optimal thresholds
        if show_ROC_plot == 1:
            plt.figure(count)
            plt.title('ROC curve for ' + feature + ' (' + file_key + '/' + age_category + ')')
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
        thresholds = np.linspace(np.min(featureData[feature]), np.max(featureData[feature]), 10000)
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
            if show_ROC_plot == 1:
                plt.plot(TPR, FPR, label='AUC = {:.2f}'.format(AUC))
                plt.legend()
        else:
            optimal_idx = np.argmax(TPR - FPR)
            optimal_threshold_rem = thresholds[optimal_idx]
            if save_ROC_plot == 1:
                plt.plot(FPR, TPR, label='AUC = {:.2f}'.format(AUC))
                plt.legend()

        if save_ROC_plot == 1 and show_ROC_plot == 1:
            plot_file_name = 'REMdetection_ROC_' + ttl_feature + '_' + file_key + '_' + age_category + '.png'
            plt.savefig(os.path.join(results_plots_file_path, plot_file_name))

        AUC_rem = "{:.2f}".format(AUC)  # 2 decimals

    # Boxplot
    if show_boxplot == 1:
        plt.figure(count + 100)
        plt.yscale('log')
        sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
        sns.set_context("paper", font_scale=0.9)
        sns.boxplot(x='Sleep stage labels', y=feature, data=featureData,
                    palette="Set3")
        plt.title('Boxplot ' + feature + ' (' + file_key + '/' + age_category + ')')

    if save_boxplot == 1 and show_boxplot == 1:
        plot_file_name = 'REMdetection_boxplot_' + ttl_feature + '_' + file_key + '_' + age_category + '.png'
        plt.savefig(os.path.join(results_plots_file_path, plot_file_name))

    # Create allFeatureData file
    if feature_data_file_name == feature_data_file_names[0]:
        allFeatureData = featureData
    else:
        allFeatureData = allFeatureData.append(featureData)


# Now for all feature data
plt.close('all')
### Create ROCs, calculate AUCs and determine optimal thresholds
if show_ROC_plot == 1:
    plt.figure(count)
    plt.title('ROC curve for ' + feature + ' (' + age_category + ')')
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
thresholds = np.linspace(np.min(allFeatureData[feature]), np.max(allFeatureData[feature]), 10000)
TPR = np.zeros(len(thresholds))
FPR = np.zeros(len(thresholds))

for i in range(0, len(allFeatureData[sleep_stage_column_name])):
    if allFeatureData.iloc[i][sleep_stage_column_name] == 'REM':
        label_true.append(allFeatureData.iloc[i][feature])
    if allFeatureData.iloc[i][sleep_stage_column_name] != 'REM':
        label_false.append(allFeatureData.iloc[i][feature])

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
    if show_ROC_plot == 1:
        plt.plot(TPR, FPR, label='AUC = {:.2f}'.format(AUC))
        plt.legend()
else:
    optimal_idx = np.argmax(TPR - FPR)
    optimal_threshold_rem = thresholds[optimal_idx]
    if save_ROC_plot == 1:
        plt.plot(FPR, TPR, label='AUC = {:.2f}'.format(AUC))
        plt.legend()

if save_ROC_plot == 1 and show_ROC_plot == 1:
    plot_file_name = 'REMdetection_ROC_' + ttl_feature + '_' + '_' + age_category + '.png'
    plt.savefig(os.path.join(results_plots_file_path, plot_file_name))

AUC_rem = "{:.2f}".format(AUC)  # 2 decimals

# Boxplot
if show_boxplot == 1:
    plt.figure(count + 100)
    plt.yscale('log')
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
    sns.set_context("paper", font_scale=0.9)
    sns.boxplot(x='Sleep stage labels', y=feature, data=allFeatureData,
                palette="Set3")
    plt.title('Boxplot ' + feature + ' (' + age_category + ')')

if save_boxplot == 1 and show_boxplot == 1:
    plot_file_name = 'REMdetection_boxplot_' + ttl_feature + '_' + age_category + '.png'
    plt.savefig(os.path.join(results_plots_file_path, plot_file_name))