"""
This function is used to evaluate the added value of DWT features
"""

#%% Import modules
from src.Functions import load_and_prepare_raw_feature_data_multiple
import os
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector
import time
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector

#%% Set data variables
load_data = 1 # 0 if data is already loaded
data_type = 'sampled' # 'raw' or 'sampled'
channel_number = 8
number_of_stages_key = 'Four_state_labels'

### Variables, if data == 'new'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_2_PSG_FeatureData_SCALED'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
include_age_categories = ['2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years']
remove_artifacts = 1

### Variables, if data == 'sampled'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
sampled_data_file_name = '5000sampled_feature_data_0-18years.xlsx'

#%%
"""
##### 1. LOAD FEATURE DATA #####
"""
if load_data == 1 and data_type == 'new':
    feature_data = load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                                             feature_data_file_path, include_age_categories,
                                                             remove_artifacts)
elif load_data == 1 and data_type == 'sampled':
    feature_data = pd.read_excel(os.path.join(sampled_data_file_path, sampled_data_file_name))

#%%
"""
##### 2: EXTRACT LABELS AND FEATURES #####
"""

### Select EEG channel
eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
eeg_channel = eeg_channels[channel_number]

### Extract features
features = pd.DataFrame([])
# EOG features
eog_features_list = [col for col in feature_data if col.startswith('EOG ROC-LOC:')]
features[eog_features_list] = feature_data[eog_features_list]  # EOG features
# EMG features
emg_features_list = [col for col in feature_data if col.startswith('EMG EMG Chin:')]
features[emg_features_list] = feature_data[emg_features_list]  # EMG features
# EEG features
eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]
features[eeg_features_list] = feature_data[eeg_features_list]

# Add labels to feature sheet so that these epochs are also dropped with the NaNs
features['labels cat'] = feature_data[number_of_stages_key]
features['labels lin'] = feature_data[number_of_stages_key + '- linear']
features['Patient key'] = feature_data['Patient key']
features = features.dropna(axis=0)
labels_cat = features['labels cat'].copy()
labels_lin = features['labels lin'].copy()
patient_keys = features['Patient key'].copy()
del features['labels cat'], features['labels lin'], features['Patient key']

features_with = features.copy()
features_without = features_with.iloc[:, 0:45]

#%%
""" 
##### mRMR feature selection, SFS feature selection + cross-validation#####
"""
# Select features + classifier
use_features = features_with
classifier = 'xgboost'

### Classifiers
if classifier == 'DT':
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', max_depth=6)
    labels = labels_cat
elif classifier == 'LR':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', C=1.0)
    labels = labels_lin
elif classifier == 'SVM':
    from sklearn.svm import SVC
    clf = SVC(gamma='scale', kernel='rbf', C=1.0, probability=True)
    labels = labels_cat
elif classifier == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=50)
    labels = labels_cat
elif classifier == 'LDA':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    labels = labels_cat
elif classifier == 'RF':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=6, n_estimators=100, bootstrap=True)
    labels = labels_cat
elif classifier == "xgboost":
    from xgboost import XGBClassifier
    clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6,
                        gamma=0, learning_rate=0.3, colsample_bytree=1)
    labels = labels_cat

"""
##### mRMR feature selection #####
"""
start_time = time.time()
n_features_to_select = 20

### Minimum redundancy maximum relevance
mrmr_selected_features = mrmr_classif(use_features, labels_lin, K=n_features_to_select)
feature_subset_mrmr = use_features[mrmr_selected_features]
print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
print(mrmr_selected_features)

# Add age features
age_features_list = list(pd.get_dummies(feature_data['Age category'], prefix='age_group_').columns)
feature_subset_mrmr[age_features_list] = feature_data[age_features_list]

"""
##### Sequential Forward Feature Selection #####
"""
#sfs = SequentialFeatureSelector(clf, n_features_to_select=15, cv=5, scoring='accuracy')
#sfs_features = sfs.fit_transform(feature_subset_mrmr, labels)
#print(mrmr_selected_features)
sfs_features = feature_subset_mrmr

"""
##### Cross-validation results #####
"""
auc_cv_score = cross_val_score(clf, sfs_features, labels, groups=patient_keys,
                               cv=10, scoring='roc_auc_ovo')
acc_cv_score = cross_val_score(clf, sfs_features, labels, groups=patient_keys,
                               cv=10, scoring='accuracy')
cohens_kappa_cv_score = cross_val_score(clf, sfs_features, labels, groups=patient_keys,
                                        cv=10, scoring=make_scorer(cohen_kappa_score))

print("Cross-validation %s" % classifier,
      "\n AUC mean: {:.2f}, std: {:.2f}".format(np.mean(auc_cv_score), np.std(auc_cv_score)),
      "\n Accuracy mean: {:.2f}, std: {:.2f}".format(np.mean(acc_cv_score), np.std(acc_cv_score)),
      "\n Cohen's kappa mean: {:.2f}, std: {:.2f}".format(np.mean(cohens_kappa_cv_score),
                                                          np.std(cohens_kappa_cv_score)),
      )

#%% Feature importances
""" 
##### mRMR feature selection, SFS feature selection + cross-validation#####
"""
# Select features + classifier
use_features = features_with
classifier = 'LR'

### Classifiers
if classifier == 'DT':
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', max_depth=6)
    labels = labels_cat
elif classifier == 'LR':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', C=1.0)
    labels = labels_lin
elif classifier == 'SVM':
    from sklearn.svm import SVC
    clf = SVC(gamma='scale', kernel='rbf', C=1.0, probability=True)
    labels = labels_cat
elif classifier == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=50)
    labels = labels_cat
elif classifier == 'LDA':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    labels = labels_cat
elif classifier == 'RF':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=6, n_estimators=100, bootstrap=True)
    labels = labels_cat
elif classifier == "xgboost":
    from xgboost import XGBClassifier
    clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6,
                        gamma=0, learning_rate=0.3, colsample_bytree=1)
    labels = labels_cat

"""
##### mRMR feature selection #####
"""
start_time = time.time()
n_features_to_select = 20

### Minimum redundancy maximum relevance
mrmr_selected_features = mrmr_classif(use_features, labels_lin, K=n_features_to_select)
feature_subset_mrmr = use_features[mrmr_selected_features]
print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
print(mrmr_selected_features)

# Add age features
age_features_list = list(pd.get_dummies(feature_data['Age category'], prefix='age_group_').columns)
feature_subset_mrmr[age_features_list] = feature_data[age_features_list]
sfs_features = feature_subset_mrmr

### Feature importances
clf.fit(sfs_features, labels)
#importances = clf.feature_importances_
importances = clf.coef_[0]
feature_names = sfs_features.columns

clf_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
clf_importances.plot.bar(ax=ax)
ax.set_title("Feature importances %s" % classifier)
ax.set_ylabel("Feature importance")
fig.tight_layout()

#%%
"""
##### EXPLORE CORRELATION BETWEEN FEATURES #####
"""
feature1 = features['EEG C3-C4: DWT cD1 std']
feature2 = features['EEG C3-C4: Abs Gamma power']

corr1, p = spearmanr(feature1, feature2, nan_policy='omit')
corr2, p = pearsonr(feature1, feature2)

#corr = np.corrcoef(np.array(feature1), np.array(feature2))
print('Spearman %.2f' % corr1,
      '\n Pearson %.2f' % corr2)

#corr, p = spearmanr(features, features, nan_policy='omit')
#corr = features.corr()
#sns.set(font_scale=0.5)
#ax = sns.heatmap(corr, annot=True)
#ax.figure.tight_layout()
#plt.title("Correlation between features")
#plt.show()