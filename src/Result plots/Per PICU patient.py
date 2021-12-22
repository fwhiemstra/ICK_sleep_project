import pandas as pd
import matplotlib.pyplot as plt
import os

save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Thesis report\Figuren thesis report'
result_sheet_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Results'
result_sheet_file_name = 'Performance per PICU patient - Results plots.xlsx'
performance_metric = 'Accuracy'

plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

classification_task = 'Three-state-%s' % performance_metric
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=classification_task)
result_sheet = result_sheet.drop(labels='Unnamed: 0', axis=1)
result_sheet.reset_index(drop=True, inplace=True)
result_sheet.plot(y=['Gamma/delta', 'Gamma/(theta+delta)', 'DT', 'SVM', 'XGBoost'],
                  kind='bar',
                  color=['darkorange', 'orangered', 'skyblue', 'steelblue', 'navy'],
                  width=0.6,
                  alpha=0.8,
                  #yerr=[result_sheet['CV Accuracy std'].values, result_sheet['error'].values],
                  ax=axs[0])
axs[0].set_xticks(range(0, 10))
axs[0].set_xticklabels(['PICU Patient A', 'PICU Patient B', 'PICU Patient C', 'PICU Patient D',
                                 'PICU Patient E', 'PICU Patient F', 'PICU Patient G', 'PICU Patient H',
                                 'PICU Patient I', 'PICU Patient J'], rotation=90)
axs[0].minorticks_on()
axs[0].grid(b=True, which='both', linestyle='--', axis='y')
axs[0].set_ylim([0, 1])
axs[0].set_ylabel(performance_metric)
axs[0].set_title(classification_task.replace('-%s' % performance_metric, '')  + ' classification')
axs[0].get_legend().remove()

classification_task = 'Four-state-%s' % performance_metric
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=classification_task)
result_sheet = result_sheet.drop(labels='Unnamed: 0', axis=1)
result_sheet.reset_index(drop=True, inplace=True)
result_sheet.plot(y=['Gamma/delta', 'Gamma/(theta+delta)', 'DT', 'SVM', 'XGBoost'],
                  kind='bar',
                  color=['darkorange', 'orangered', 'skyblue', 'steelblue', 'navy'],
                  width=0.6,
                  alpha=0.8,
                  #yerr=[result_sheet['CV Accuracy std'].values, result_sheet['error'].values],
                  ax=axs[1])
axs[1].set_xticks(range(0, 10))
axs[1].set_xticklabels(['PICU Patient A', 'PICU Patient B', 'PICU Patient C', 'PICU Patient D',
                                 'PICU Patient E', 'PICU Patient F', 'PICU Patient G', 'PICU Patient H',
                                 'PICU Patient I', 'PICU Patient J'], rotation=90)
axs[1].minorticks_on()
axs[1].grid(b=True, which='both', linestyle='--', axis='y')
axs[1].set_ylim([0, 1])
axs[1].set_ylabel(performance_metric)
axs[1].set_title(classification_task.replace('-%s' % performance_metric, '')  + ' classification')
axs[1].get_legend().remove()
plt.tight_layout()

handles, labels = axs[1].get_legend_handles_labels()
fig.subplots_adjust(bottom=0.25)
fig.legend(handles, labels, loc='lower center', ncol=5)

plt.savefig(os.path.join(save_plot_file_path, 'Performance per PICU patient - result plot - %s.png' % performance_metric), dpi=1000)
