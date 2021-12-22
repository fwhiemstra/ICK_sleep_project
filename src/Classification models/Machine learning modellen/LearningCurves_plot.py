"""
This function is used to plot the learning curves
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

learning_curves_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_1_Learning curves\Training size 60000'
colors = ['blueviolet', 'orange', 'darkcyan', 'black', 'olivedrab', 'navy', 'crimson']
learning_curve_file_names = ['learning_curve_DT_Three_state_labels.xlsx',
                             'learning_curve_LR_Three_state_labels.xlsx',
                             'learning_curve_LDA_Three_state_labels.xlsx',
                             'learning_curve_KNN_Three_state_labels.xlsx',
                             'learning_curve_SVM_Three_state_labels.xlsx',
                             'learning_curve_RF_Three_state_labels.xlsx',
                             'learning_curve_xgboost_Three_state_labels.xlsx']
classifiers = ['DT', 'LR', 'LDA', 'KNN', 'SVM', 'RF', 'XGBoost']
#learning_curves_file_name = 'learning_curve_LR_Three_state_labels.xlsx'


plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)
plt.close('all')

plt.figure(1, figsize=(8, 7), dpi=100)
#plt.title('Learning curves (three state classification)', fontsize=14)
plt.minorticks_on()
plt.grid(b=True, which='both', linestyle='--')
plt.ylabel('Accuracy score')
plt.xlabel('Training set size')
plt.ylim([0.65, 1])
plt.xlim([0, 60000])

for color, learning_curve_file_name, classifier in zip(colors, learning_curve_file_names, classifiers):
    train_size_abs = pd.read_excel(os.path.join(learning_curves_file_path, learning_curve_file_name),
                                        sheet_name='train_size_abs', index_col=0)
    train_scores = pd.read_excel(os.path.join(learning_curves_file_path, learning_curve_file_name),
                                        sheet_name='train_scores', index_col=0)
    test_scores = pd.read_excel(os.path.join(learning_curves_file_path, learning_curve_file_name),
                                        sheet_name='test_scores', index_col=0)

    plt.plot(train_size_abs, np.mean(train_scores, axis=1), label=classifier,
             linewidth=1, color=color)
    plt.plot(train_size_abs, np.mean(test_scores, axis=1),
             linewidth=1, color=color, linestyle='dashed')
    plt.legend(ncol=2)


