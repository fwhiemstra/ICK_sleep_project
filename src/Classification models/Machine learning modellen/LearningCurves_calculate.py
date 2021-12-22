"""
This function is used to create learning curves
"""

#%% Import modules
from Functions import load_and_prepare_raw_feature_data_multiple
import os
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.model_selection import learning_curve
import time
import matplotlib.pyplot as plt

#%% Set data variables
load_data = 1 # 0 if data is already loaded
data_type = 'new' # 'new' or 'sampled'
channel_number = 8
number_of_stages_key = 'Three_state_labels'

### Variables, if data == 'new'
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_2_PSG_FeatureData_SCALED'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years']
remove_artifacts = 1

### Variables, if data == 'sampled'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
sampled_data_file_name = '10000sampled_feature_data_0-18years.xlsx'

### Learning curve variables
learning_curve_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_1_Learning curves'

"""
##### 1. LOAD FEATURE DATA #####
"""
if load_data == 1 and data_type == 'new':
    feature_data = load_and_prepare_raw_feature_data_multiple(patient_data_file_path, patient_data_file_name,
                                                             feature_data_file_path, include_age_categories,
                                                             remove_artifacts)
elif load_data == 1 and data_type == 'sampled':
    feature_data = pd.read_excel(os.path.join(sampled_data_file_path, sampled_data_file_name))


"""
##### 2: EXTRACT LABELS AND FEATURES #####
"""
number_of_stages_key = 'Three_state_labels'

### Select EEG channel
eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
eeg_channel = eeg_channels[channel_number]

### Add labels
labels_and_features = pd.DataFrame([])
labels_and_features['Sleep stage labels (cat)'] = feature_data[number_of_stages_key]
labels_and_features['Sleep stage labels (lin)'] = feature_data[number_of_stages_key + '- linear']
labels_and_features['Patient key'] = feature_data['Patient key']
labels_and_features['Age category'] = feature_data['Age category']

### Add features
# EOG features
eog_features_list = [col for col in feature_data if col.startswith('EOG ROC-LOC:')]
labels_and_features[eog_features_list] = feature_data[eog_features_list]  # EOG features
# EMG features
emg_features_list = [col for col in feature_data if col.startswith('EMG EMG Chin:')]
labels_and_features[emg_features_list] = feature_data[emg_features_list]  # EMG features
# EEG features
eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]
labels_and_features[eeg_features_list] = feature_data[eeg_features_list]

### Add age features
age_features_list = list(pd.get_dummies(feature_data['Age category'], prefix='age_group_').columns)
labels_and_features[age_features_list] = feature_data[age_features_list]

labels_and_features = labels_and_features.dropna(axis=0) # Drop rows with NaNs in it

age_features = labels_and_features[age_features_list] # Extract age features with removed rows with NaNs to later add again
labels_and_features = labels_and_features.drop(labels_and_features[age_features_list], axis=1)

"""
##### mRMR feature selection #####
"""
start_time = time.time()
n_features_to_select = 20

### Minimum redundancy maximum relevance
mrmr_selected_features = mrmr_classif(labels_and_features.iloc[:, 4:], labels_and_features['Sleep stage labels (lin)'],
                                      K=n_features_to_select)
feature_subset_mrmr = labels_and_features.iloc[:, 4:][mrmr_selected_features]
print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
print(mrmr_selected_features)

### Change labels_and_features
labels_and_features = labels_and_features.copy()
labels_and_features = labels_and_features.drop(labels_and_features.iloc[:, 4:], axis=1)
labels_and_features[mrmr_selected_features] = feature_subset_mrmr
labels_and_features[age_features_list] = age_features # Add age features again (with removed rows with NaNs)

labels_and_features = labels_and_features.sample(frac=1) # Shuffle data before cross-validation splits

"""
##### Create learning curve #####
"""

train_sizes = np.linspace(0.01, 0.3, 25)

for classifier in ['DT', 'LDA', 'LR', 'KNN', 'SVM', 'RF', 'xgboost']:
    start_time = time.time()

    ### Classifiers
    if classifier == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='gini', max_depth=6)
        labels = labels_and_features['Sleep stage labels (cat)']
    elif classifier == 'LR':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2', C=1)
        labels = labels_and_features['Sleep stage labels (lin)']
    elif classifier == 'SVM':
        from sklearn.svm import SVC
        clf = SVC(gamma='scale', kernel='rbf', C=10, probability=True)
        labels = labels_and_features['Sleep stage labels (cat)']
    elif classifier == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=40)
        labels = labels_and_features['Sleep stage labels (cat)']
    elif classifier == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis()
        labels = labels_and_features['Sleep stage labels (cat)']
    elif classifier == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=6, n_estimators=75, bootstrap=True)
        labels = labels_and_features['Sleep stage labels (cat)']
    elif classifier == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6,
                            gamma=0, learning_rate=0.1, colsample_bytree=1, n_estimators=100)
        labels = labels_and_features['Sleep stage labels (cat)']

    train_size_abs, train_scores, test_scores, fit_times, score_times = \
        learning_curve(clf, X=labels_and_features.iloc[:, 4:], y=labels, shuffle=True, random_state=2,
                       groups=labels_and_features['Patient key'], train_sizes=train_sizes,
                       cv=5, return_times=True)

    # Dataframe
    dfTrain_sizes = pd.DataFrame({'train_size_abs': train_size_abs})
    dfTrain_scores = pd.DataFrame(train_scores)
    dfTest_scores = pd.DataFrame(test_scores)
    dfFit_times = pd.DataFrame(fit_times)
    dfScore_times = pd.DataFrame(score_times)

    learning_curve_file_name = 'learning_curve_%s_%s.xlsx' % (classifier, number_of_stages_key)
    writer = pd.ExcelWriter(os.path.join(learning_curve_file_path, learning_curve_file_name))
    dfTrain_sizes.to_excel(writer, sheet_name='train_size_abs')
    dfTrain_scores.to_excel(writer, sheet_name='train_scores')
    dfTest_scores.to_excel(writer, sheet_name='test_scores')
    dfFit_times.to_excel(writer, sheet_name='fit_times')
    dfScore_times.to_excel(writer, sheet_name='score_times')

    writer.save()

    print('GOODMORNING! Learning curve calculation of %s completed in %s hours' % (classifier, (time.time()-start_time)/3600))

#%%
plt.title('Learning curve %s' % classifier)
plt.plot(train_size_abs, np.mean(train_scores, axis=1), label='Train score' )
plt.plot(train_size_abs, np.mean(test_scores, axis=1), label='Test score')
plt.ylabel('Classification accuracy')
plt.xlabel('Number of training samples')
plt.legend()

