import pandas as pd
import matplotlib.pyplot as plt
import os

save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Thesis report\Figuren thesis report'
save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Presentaties\Figuren'

plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

classification_task = 'Four-state'
result_sheet_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Results'
result_sheet_file_name = 'Classification models - Results plots.xlsx'
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=classification_task)

if classification_task == 'Three-state':
    colors = ['olivedrab', 'yellowgreen']
elif classification_task == 'Four-state':
    colors = ['darkcyan', 'darkturquoise']

# Accuracy
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
plt.suptitle('Classification performance - %s classification' % classification_task)
result_sheet.plot(x='Unnamed: 0',
                  y=['CV Accuracy mean', 'Test Accuracy'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet['CV Accuracy std'].values, result_sheet['error'].values],
                  ax=axs[0])
axs[0].set_xlabel('')
axs[0].set_ylim([0, 1])
axs[0].get_legend().remove()
axs[0].set_title('Accuracy')
axs[0].minorticks_on()
axs[0].grid(b=True, which='both', linestyle='--', axis='y')
axs[0].set_xticklabels(labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)

result_sheet.plot(x='Unnamed: 0',
                  y=['CV AUC mean', 'Test AUC'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet['CV AUC std'].values, result_sheet['error'].values],
                  ax=axs[1])
axs[1].set_xlabel('')
axs[1].set_ylim([0, 1])
axs[1].get_legend().remove()
axs[1].set_title('AUC')
axs[1].minorticks_on()
axs[1].grid(b=True, which='both', linestyle='--', axis='y')
axs[1].set_xticklabels(labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)

result_sheet.plot(x='Unnamed: 0',
                  y=["CV Cohen's kappa mean", "Test Cohen's kappa"],
                  label=['Internal validation, CV score', 'External validation (PICU), test score'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet["CV Cohen's kappa std"].values, result_sheet['error'].values],
                  ax=axs[2])
axs[2].set_xlabel('')
axs[2].set_ylim([0, 1])
axs[2].set_title("Cohen's kappa")
axs[2].minorticks_on()
axs[2].grid(b=True, which='both', linestyle='--', axis='y')
axs[2].set_xticklabels(labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

plt.savefig(os.path.join(save_plot_file_path, 'ALL classification models - result plot - %s.png' % classification_task), dpi=1000)

#%% Only internal validation
plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

classification_task = 'Four-state'
result_sheet_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Results'
result_sheet_file_name = 'Classification models - Results plots.xlsx'
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=classification_task)

if classification_task == 'Three-state':
    colors = ['olivedrab', 'yellowgreen']
elif classification_task == 'Four-state':
    colors = ['darkcyan', 'darkturquoise']

# Accuracy
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
plt.suptitle('Classification performance - %s classification' % classification_task)
result_sheet.plot(x='Unnamed: 0',
                  y=['CV Accuracy mean'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet['CV Accuracy std'].values],
                  ax=axs[0])
axs[0].set_xlabel('')
axs[0].set_ylim([0, 1])
axs[0].get_legend().remove()
axs[0].set_title('Accuracy')
axs[0].minorticks_on()
axs[0].grid(b=True, which='both', linestyle='--', axis='y')
axs[0].set_xticklabels(labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)

result_sheet.plot(x='Unnamed: 0',
                  y=['CV AUC mean'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet['CV AUC std'].values],
                  ax=axs[1])
axs[1].set_xlabel('')
axs[1].set_ylim([0, 1])
axs[1].get_legend().remove()
axs[1].set_title('AUC')
axs[1].minorticks_on()
axs[1].grid(b=True, which='both', linestyle='--', axis='y')
axs[1].set_xticklabels(labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)

result_sheet.plot(x='Unnamed: 0',
                  y=["CV Cohen's kappa mean"],
                  #label=['Internal validation, CV score'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet["CV Cohen's kappa std"].values],
                  ax=axs[2])
axs[2].set_xlabel('')
axs[2].set_ylim([0, 1])
axs[2].get_legend().remove()
axs[2].set_title("Cohen's kappa")
axs[2].minorticks_on()
axs[2].grid(b=True, which='both', linestyle='--', axis='y')
axs[2].set_xticklabels(labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.savefig(os.path.join(save_plot_file_path, 'Classification results - internal validation - %s.png') % classification_task, dpi=1000)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Presentaties\Figuren'

plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

classification_task = 'Three-state'
result_sheet_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Results'
result_sheet_file_name = 'Classification models - Results plots.xlsx'
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=classification_task)

if classification_task == 'Three-state':
    colors = ['olivedrab', 'yellowgreen']
elif classification_task == 'Four-state':
    colors = ['darkcyan', 'darkturquoise']

# Accuracy
plt.figure(figsize=(6, 6))
result_sheet.plot(x='Unnamed: 0',
                  y=['CV Accuracy mean'],
                  kind='bar',
                  color=colors,
                  width=0.8,
                  alpha=0.8,
                  yerr=[result_sheet['CV Accuracy std'].values])
plt.xlabel('')
plt.ylim([0, 1])
plt.title('Accuracy')
plt.minorticks_on()
plt.grid(b=True, which='both', linestyle='--', axis='y')
plt.xticks(ticks=range(0, 5), labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.savefig(os.path.join(save_plot_file_path, 'Classification results - accuracy - %s.png'), dpi=1000)


#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

classification_task = 'Three-state'
result_sheet_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Results'
result_sheet_file_name = 'Classification models - Results plots.xlsx'
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=classification_task)

plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

if classification_task == 'Three-state':
    colors = ['olivedrab', 'yellowgreen']
elif classification_task == 'Four-state':
    colors = ['darkcyan', 'darkturquoise']

# Accuracy
plt.figure(figsize=(6, 6))
series=pd.Series(np.array(result_sheet['CV Accuracy mean']), np.array(result_sheet['Unnamed: 0']))
series.plot(kind='bar',
               color=['darkorange', 'darkorange', 'skyblue', 'skyblue', 'skyblue'],
               width=0.8,
               alpha=0.8,
               yerr=[result_sheet['CV Accuracy std'].values])
plt.xlabel('')
plt.ylim([0, 1])
plt.minorticks_on()
plt.grid(b=True, which='both', linestyle='--', axis='y')
plt.xticks(ticks=range(0, 5), labels=['Gamma/ \n delta', 'Gamma/ \n (theta+delta)', 'DT', 'SVM', 'XGBoost'], rotation=0)
plt.ylabel('Accuracy')
plt.tight_layout()
#plt.subplots_adjust(wspace=0.2)
#plt.savefig(os.path.join(save_plot_file_path, 'Classification results - accuracy - %s.png'), dpi=1000)

