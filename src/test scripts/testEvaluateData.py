import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import roc_curve, auc

#%%
xlsx_file_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\test\4_Feature_data'
xlsx_file_name = 'EDFtest2_FeatureData.xlsx'
FeatureData = pd.read_excel(os.path.join(xlsx_file_path, xlsx_file_name))

#%% Feature correlation
FeatureCorrelation = FeatureData.corr()
FeatureCorrelation = FeatureCorrelation.drop("Unnamed: 0", axis=1)
FeatureCorrelation = FeatureCorrelation.drop("Unnamed: 0", axis=0)
sns.set(font_scale=0.5)
ax = sns.heatmap(FeatureCorrelation, annot=True)
ax.figure.tight_layout()
plt.title("Correlation between features")
plt.show()

#%% Correlation with sleep stages as numerical variables
order = ['Sleep stage N4', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
# Add Sleep stages 2 om het plotje goed te krijgen
FeatureData["Sleep stages 2"] = [order.index(x) for x in FeatureData["Sleep stages"]]

corr = FeatureData.corr(method='pearson')
sns.set(font_scale=0.5)
ax = sns.heatmap(corr, annot=True)
ax.figure.tight_layout()
plt.title("Feature correlation")
plt.show()

corr["Sleep stages 2"]

#%% Individual Feature evaluation
select_feature = 'EEG C3-C4: Beta/Delta ratio'
feature = FeatureData[select_feature]

#%% Regression plot
x = pd.DataFrame.to_numpy(FeatureData["Sleep stages 2"]).reshape((-1, 1))
y = pd.DataFrame.to_numpy(FeatureData[select_feature])
#x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
#y = np.array([5, 20, 14, 32, 22, 38])
model = LinearRegression().fit(x, y)
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)

plt.figure(100)
plt.scatter(x, y, s=1, marker='x')
plt.plot(x, y_pred, color='tab:orange')
#plt.grid(which='both')
plt.title('Scatter plot: {}'.format(select_feature), fontsize=14)
plt.ylabel("Feature value")
plt.xlabel("Sleep stages")
#plt.xticks(['Wake', 'REM', 'N1', 'N2', 'N3'])
plt.show()

#%% Visualization with hypnogram
order = ['Sleep stage N4', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
# Add Sleep stages 2 om het plotje goed te krijgen
FeatureData["Sleep stages 2"] = [order.index(x) for x in FeatureData["Sleep stages"]]

fig, axs = plt.subplots(2)

axs[0].step(FeatureData["Unnamed: 0"], FeatureData["Sleep stages 2"], where='post', color='tab:blue')
axs[0].set_yticks(range(len(order))) # set number of labels y axis
axs[0].set_yticklabels(order) # add labels to y axis
axs[0].grid(which='minor')
axs[0].set_ylabel('Sleep stage')
axs[0].set_xticklabels([]) # remove number at x axis
axs[0].set_title('Hypnogram')

axs[1].plot(feature, color='tab:orange')
axs[1].grid(which='minor')
axs[1].set_xticklabels([]) # remove number at x axis
axs[1].set_ylabel('Feature value')
axs[1].set_title(select_feature)

#%% Visualization per cluster
plt.figure(100)
plt.title("Feature scatter plot of "+select_feature)
plt.scatter(FeatureData["Sleep stages"], FeatureData[select_feature], s=1, marker='x')
plt.show()

#%% Correlation with sleep stages as categorical variables

#%% Area under the curve
select_feature = 'EEG C3-C4: Gamma/Delta ratio'
feature = FeatureData[select_feature]

sleep_stage_labels = ['Sleep stage W', 'Sleep stage R', 'Sleep stage N1','Sleep stage N2',
                      'Sleep stage N3']

plt.figure(1)
plt.title('ROC curve for ' + select_feature)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid(which='both')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--')

for stage in sleep_stage_labels:
    label_true = []
    label_false = []
    thresholds = np.linspace(np.min(feature), np.max(feature), 1000)
    TPR = np.zeros(len(thresholds))
    FPR = np.zeros(len(thresholds))

    for i in range(0, len(FeatureData["Sleep stages"])):
        if FeatureData["Sleep stages"][i] == stage: label_true.append(feature[i])
        if FeatureData["Sleep stages"][i] != stage: label_false.append(feature[i])

    for i in range(0, len(thresholds)):
        x = [element for element in label_true if element >= thresholds[i]]
        TP = len(x)
        x = [element for element in label_true if element < thresholds[i]]
        FN = len(x)
        x = [element for element in label_false if element >= thresholds[i]]
        FP = len(x)
        x = [element for element in label_false if element < thresholds[i]]
        TN = len(x)

        TPR[i] = TP/(TP+FN)
        FPR[i] = FP/(FP+TN)

    AUC = abs(np.trapz(TPR,FPR))
    print('AUC = ', AUC)
    if AUC < 0.5:
        AUC = 1 - AUC
        plt.plot(TPR, FPR, label=stage + ': AUC = {:.2f}'.format(AUC))
        plt.legend()
    else:
        plt.plot(FPR, TPR, label=stage + ': AUC = {:.2f}'.format(AUC))
        plt.legend()

#%%
order = ['Sleep stage N4', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
FeatureData["Sleep stages 2"] = [order.index(x) for x in FeatureData["Sleep stages"]]
y = FeatureData["Sleep stages 2"]
xFeatures = FeatureData.drop(["Unnamed: 0", "Sleep stages", "Sleep stages 2"], axis=1)

#%% ANOVA F-value
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(xFeatures, y)

np.set_printoptions(precision=3)
for i in range(0, len(fit.scores_)):
    print(['ANOVA F-value: ' + xFeatures.columns[i] + ' : ' + str(fit.scores_[i])])
# summarize selected features
#features = fit.transform(xFeatures)
#print(features[0:5,:])

#%% Feature importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(xFeatures, y)
for i in range(0, len(model.feature_importances_)):
    print(['Feature importance (decTrees): ' + xFeatures.columns[i] + ' : ' +
           str(model.feature_importances_[i])])

#%%


#%%
