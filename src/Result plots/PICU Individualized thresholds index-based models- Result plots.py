import pandas as pd
import matplotlib.pyplot as plt
import os

result_sheet_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Results'
result_sheet_file_name = 'Individualized thresholds - 3-state - Results plots.xlsx'
save_plot_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\Thesis report\Figuren thesis report'

plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

index_model = 'GammaDelta'
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=index_model)
#plt.suptitle('Classification performance - %s classification' % classification_task)
result_sheet.plot(x='Unnamed: 0',
                  y=['test_auc_model', 'mean_auc_individual'],
                  label=['General thresholds (final model)', 'Individual thresholds'],
                  kind='bar',
                  color=['crimson', 'pink'],
                  width=0.8,
                  alpha=0.8,
                  ax=axs[0],
                  yerr=[result_sheet['test_error'].values, result_sheet['std_acc_individual'].values])
axs[0].set_xlabel('')
axs[0].set_ylabel('AUC')
axs[0].set_ylim([0, 1])
axs[0].minorticks_on()
axs[0].grid(b=True, which='both', linestyle='--', axis='y')
axs[0].set_xticklabels(labels=result_sheet['Unnamed: 0'], rotation=90)
axs[0].set_title("Gamma/delta")

index_model = 'GammaTheta+Delta'
result_sheet = pd.read_excel(os.path.join(result_sheet_file_path, result_sheet_file_name), sheet_name=index_model)
#plt.suptitle('Classification performance - %s classification' % classification_task)
result_sheet.plot(x='Unnamed: 0',
                  y=['test_auc_model', 'mean_auc_individual'],
                  label=['General thresholds (final model)', 'Individual thresholds'],
                  kind='bar',
                  color=['darkgreen', 'lightgreen'],
                  width=0.8,
                  alpha=0.8,
                  ax=axs[1],
                  yerr=[result_sheet['test_error'].values, result_sheet['std_acc_individual'].values])
axs[1].set_xlabel('')
axs[1].set_ylabel('AUC')
axs[1].set_ylim([0, 1])
axs[1].minorticks_on()
axs[1].grid(b=True, which='both', linestyle='--', axis='y')
axs[1].set_xticklabels(labels=result_sheet['Unnamed: 0'], rotation=90)
axs[1].set_title("Gamma/(theta+delta)")

plt.tight_layout()
plt.savefig(os.path.join(save_plot_file_path, 'PICU individualized thresholds.png'), dpi=1000)


