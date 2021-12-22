"""
This file is used to compare test and play with classification models
"""

# %% Import packages
import os
#import keras.models
import numpy as np
import pandas as pd
#from src.Functions import add_new_state_labels
from mrmr import mrmr_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector
import time

#%% Set variables
# File paths and names
feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_2_PSG_FeatureData_SCALED'
patient_data_file_path = r'E:\ICK slaap project\1_Patient_data'
patient_data_file_name = 'PSG_PatientData.xlsx'
model_result_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment'

load_data = 1 # 0 is data is already loaded
data = 'all'

# Set data parameters - if data = 'all'
include_age_categories = ['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years', '9-13 years', '13-18 years']
number_of_stages = 3  # 2 = [Wake, Sleep], 3 = [Wake, NSWS, SWS], 4 = [Wake, REM, NSWS, SWS], 5 = [Wake, REM, N1, N2, N3]
remove_artifacts = 1

# Set data parameter - if data = 'sampled'
sampled_data_file_name = 'testFeatureData_1000_all_ages.xlsx'
sampled_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'

#%%
"""
##### 1.0: LOAD FEATURE DATA #####
"""
if load_data == 1:
    start_time = time.time()
    if data == 'all':
        # List feature data file names
        patientData = pd.read_excel(os.path.join(patient_data_file_path, patient_data_file_name))
        keys = []
        for age_category in include_age_categories:
            keys.extend(patientData[patientData['Age category'] == age_category]['Key'])
        feature_data_file_names = [key + '_FeatureData_scaled.xlsx' for key in keys]
        count = 0

        ### Read feature data files in list and store as one dataframe
        allFeatureData = []
        for feature_data_file_name in feature_data_file_names:
            count = count + 1
            if feature_data_file_name in os.listdir(feature_data_file_path):
                featureData_raw = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))

                # Add new sleep stage labels
                featureData_new_labels, number_of_stages_key = add_new_state_labels(featureData_raw, number_of_stages)
                featureData = featureData_new_labels

                # Remove artefacts
                if remove_artifacts == 1:
                    featureData_clean = featureData_new_labels[featureData['Artifact'] == 0]
                    featureData = featureData_clean

                # Create dataframes
                if feature_data_file_name == feature_data_file_names[0]:
                    allFeatureData = featureData
                else:
                    allFeatureData = allFeatureData.append(featureData)

                print("%s of %s files completed (%s)" % (count, len(feature_data_file_names),feature_data_file_name))

    elif data == 'sampled':
        """
        ##### 1.1: LOAD FEATURE DATA (SAMPLE SET) #####
        """
        allFeatureData = pd.read_excel(os.path.join(sampled_data_file_path,
                                         sampled_data_file_name))
        if number_of_stages == 3:
            number_of_stages_key = 'Three_state_labels'
        elif number_of_stages == 4:
            number_of_stages_key = 'Four_state_labels'

    print("Loading feature data completed: %s hours" % ((time.time() - start_time)/3600))

#%%
"""
##### 2: EXTRACT LABELS AND FEATURES #####
"""
### Extract labels from
labels_cat = allFeatureData[number_of_stages_key]
labels_lin = allFeatureData[number_of_stages_key + '- linear']
eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
eeg_channel = eeg_channels[4]
count = 0

### Extract features
# EOG, EMG + age features
eog_features_list = [col for col in allFeatureData if col.startswith('EOG ROC-LOC:')]
emg_features_list = [col for col in allFeatureData if col.startswith('EMG EMG Chin:')]
features = pd.DataFrame([])
allFeatureData = pd.concat([allFeatureData, pd.get_dummies(allFeatureData['Age category'],
                                                           prefix='age_group_')], axis=1)  # Age category
age_features_list = list(pd.get_dummies(allFeatureData['Age category'],
                                        prefix='age_group_').columns)
features[eog_features_list] = allFeatureData[eog_features_list]  # EOG features
features[emg_features_list] = allFeatureData[emg_features_list]  # EMG features

# EEG features
eeg_features_list = [col for col in allFeatureData if col.startswith(eeg_channel)]
features[eeg_features_list] = allFeatureData[eeg_features_list]

"""
##### 3: FIRST FEATURE SELECTION #####
"""
start_time = time.time()
n_features_to_select = int(0.3 * features.shape[1])

### Minimum redundancy maximum relevance
mrmr_selected_features = mrmr_classif(features, labels_lin, K=n_features_to_select)
feature_subset_mrmr = features[mrmr_selected_features]
feature_subset_mrmr[age_features_list] = allFeatureData[age_features_list]
print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
print(mrmr_selected_features)

#%% Test classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

feature_subset_sfs = feature_subset_mrmr
labels = labels_cat
clf = DecisionTreeClassifier(criterion='gini', max_depth = 6)
#clf = LogisticRegression(penalty='l2', C=1.00000000e-04)
#clf = SVC(kernel='rbf', probability=True)
#clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6, gamma=0,
                    #learning_rate=0.01,
                    #min_child_weight=1,
                    #colsample_bytree=0.7)

auc_cv_score = cross_val_score(clf, feature_subset_sfs, labels, groups=allFeatureData['Patient key'],
                           cv=10, scoring='roc_auc_ovo')
acc_cv_score = cross_val_score(clf, feature_subset_sfs, labels, groups=allFeatureData['Patient key'],
                           cv=10, scoring='accuracy')
cohens_kappa_cv_score = cross_val_score(clf, feature_subset_sfs, labels, groups=allFeatureData['Patient key'],
                           cv=10, scoring=make_scorer(cohen_kappa_score))

print("Cross-validation completed: {:.2f} hours".format((time.time() - start_time) / 3600),
      "\n AUC mean: {:.2f}, std: {:.2f}".format(np.mean(auc_cv_score), np.std(auc_cv_score)),
      "\n Accuracy mean: {:.2f}, std: {:.2f}".format(np.mean(acc_cv_score), np.std(acc_cv_score)),
      "\n Cohen's kappa mean: {:.2f}, std: {:.2f}".format(np.mean(cohens_kappa_cv_score), np.std(cohens_kappa_cv_score)),
      )

#%% Hidden Markov Model
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

labels = labels_lin
X_train, X_test, y_train, y_test = train_test_split(feature_subset_mrmr, labels, test_size=0.33, random_state=42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full", init_params="mcs")
#model.startprob_ = np.array([0.6, 0.3, 0.1])
#model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            #[0.3, 0.5, 0.2],
                            #[0.3, 0.3, 0.4]])
model.fit(X_train, y_train)
model.decode(X_train)
y_hat = model.predict(X_test)
y_hat_prob = model.predict_proba(X_test)

#%%
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_hat)

print(acc)

#%% LSTM
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import LSTM
# prepare data for lstm
from sklearn.preprocessing import MinMaxScaler
from src.Functions import add_new_state_labels

#%% Load train and test data and add new sleep stages
#train_data_raw = pd.read_excel(os.path.join(r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData', 'PSG011_FeatureData.xlsx'))
#test_data_raw = pd.read_excel(os.path.join(r'E:\ICK slaap project\4_FeatureData\4_1_PSG_FeatureData', 'PSG013_FeatureData.xlsx'))
train_data_raw = allFeatureData[0:8077]
test_data_raw = allFeatureData[8078:]
train_data_new_stages, number_of_stages_key = add_new_state_labels(train_data_raw, 3)
test_data_new_stages, number_of_stages_key = add_new_state_labels(test_data_raw, 3)

#%% Extract features, scale and reshape data
scaler = MinMaxScaler(feature_range=(0, 1))

train_data = train_data_new_stages[eeg_features_list]
train_data[number_of_stages_key + '- linear'] = train_data_new_stages[number_of_stages_key + '- linear']
train_data = scaler.fit_transform(train_data)

test_data = test_data_new_stages[eeg_features_list]
test_data[number_of_stages_key + '- linear'] = test_data_new_stages[number_of_stages_key + '- linear']
test_data = scaler.fit_transform(test_data)

train_data = np.array(train_data)
test_data = np.array(test_data)

#%% Reshape input sequences
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = 10 #number of epochs to look back
X_train, y_train = split_sequences(train_data, n_steps)
X_test, y_test = split_sequences(test_data, n_steps)
n_features = X_train.shape[2]

#%% Define LSTM model and train on training data
model = tf.keras.models.Sequential()
model.add(LSTM(10, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

#%% Predict output on test data
X_input = X_test
y_true = y_test
y_pred = model.predict(X_input, batch_size=1, verbose=2)
y_pred_classes = model.predict_classes(X_input, batch_size=1, verbose=2)
#y_pred_probability = model.predict_proba(X_input, batch_size=1, verbose=2)
print(y_pred)

#%% Evaluate LSTM performance
y_pred_label = []
for i in range(0, len(y_pred)):
    if y_pred[i] <= 0.25:
        y_pred_label.append(0)
    elif 0.25 < y_pred[i] < 0.75:
        y_pred_label.append(1)
    elif y_pred[i] >= 0.75:
        y_pred_label.append(2)
y_true_int = y_true*2

from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error
acc = accuracy_score(y_true_int, y_pred_label)
cohens_kappa = cohen_kappa_score(y_true_int, y_pred_label)
mse = mean_squared_error(y_true, y_pred)
print('Accuracy:', acc,
      '\n Cohen kappa: ', cohens_kappa,
      '\n MSE: ', mse)

#%% Compare with simple classifiers
train_features = train_data_new_stages[eeg_features_list]
train_labels = train_data_new_stages[number_of_stages_key + '- linear']
test_features = test_data_new_stages[eeg_features_list]
test_labels = test_data_new_stages[number_of_stages_key + '- linear']

classifier = 'SVM'

if classifier == 'LR':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', C=0.1)
elif classifier == 'DT':
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
elif classifier == 'SVM':
    from sklearn.svm import SVC
    clf = SVC(gamma='scale', kernel='linear', C=0.01)

clf.fit(train_features, train_labels)
#%%
input_labels = test_labels
pred_labels = clf.predict(test_features)

from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error
acc = accuracy_score(input_labels, pred_labels)
cohens_kappa = cohen_kappa_score(input_labels, pred_labels)
print('Accuracy:', acc,
      '\n Cohen kappa: ', cohens_kappa)








"""
"""
#%% TRAIN & TEST MODEL ON INDIVIDUALS
channel_number = 8
eeg_channels = ['EEG F3:', 'EEG C3:', 'EEG O1:', 'EEG A1:', 'EEG F3-C3:',
                'EEG F3-C4:', 'EEG F3-O2:', 'EEG F3-A2:', 'EEG C3-C4:',
                'EEG C3-O2:', 'EEG C3-A2:', 'EEG O1-O2:', 'EEG O1-A2:']
eeg_channel = eeg_channels[channel_number]

### Extract features
# EOG, EMG + age features
eog_features_list = [col for col in allFeatureData if col.startswith('EOG ROC-LOC:')]
emg_features_list = [col for col in allFeatureData if col.startswith('EMG EMG Chin:')]
allFeatureData_dummies = pd.concat([allFeatureData, pd.get_dummies(allFeatureData['Age category'],
                                                           prefix='age_group_')], axis=1)  # Age category
age_features_list = list(pd.get_dummies(allFeatureData_dummies['Age category'],
                                        prefix='age_group_').columns)
eeg_features_list = [col for col in allFeatureData if col.startswith(eeg_channel)]

#%% Training data
train_features = pd.DataFrame([])
train_features[eog_features_list] = allFeatureData[eog_features_list]  # EOG features
train_features[emg_features_list] = allFeatureData[emg_features_list]  # EMG features
train_features[eeg_features_list] = allFeatureData[eeg_features_list]
train_labels_cat = allFeatureData[number_of_stages_key]
train_labels_lin = allFeatureData[number_of_stages_key + '- linear']

### Reduce feature subset wit mrmr
from mrmr import mrmr_classif
from src.Functions import add_new_state_labels
n_features_to_select=20
mrmr_selected_features = mrmr_classif(train_features, train_labels_lin, K=n_features_to_select)
train_feature_subset = train_features[mrmr_selected_features]
train_feature_subset[age_features_list] = allFeatureData_dummies[age_features_list]
print("mRMR feature selection completed: %s hours" % ((time.time() - start_time) / 3600))
print(mrmr_selected_features)

### Train ML model
classifier = 'LR'

if classifier == 'LR':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', C=10)

clf.fit(train_feature_subset, train_labels_lin)

# Training results
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
pred_train_labels = clf.predict(train_feature_subset)
pred_train_labels_prob = clf.predict_proba(train_feature_subset)
train_acc = accuracy_score(train_labels_lin, pred_train_labels)
train_cohen_kappa = cohen_kappa_score(train_labels_lin, pred_train_labels)
train_auc = roc_auc_score(pd.get_dummies(train_labels_lin), pd.DataFrame(pred_train_labels_prob), multi_class='ovo')

print("\n Train AUC: {:.2f}".format(train_auc),
      "\n Train Accuracy: {:.2f}".format(train_acc),
      "\n Train Cohen's kappa: {:.2f}".format(train_cohen_kappa))

#%% Test results
remove_artifact = 1
test_data = pd.read_excel(os.path.join(feature_data_file_path, 'PSG101_FeatureData_scaled.xlsx'))
if remove_artifacts == 1:
    test_data = test_data[test_data['Artifact'] == 0]
test_data, number_of_stages_key = add_new_state_labels(test_data, 3)
test_feature_subset = test_data[mrmr_selected_features]
test_feature_subset[age_features_list] = 0
test_age_category = test_data['Age category'].iloc[0]
test_feature_subset['age_group__%s' % test_age_category] = 1
test_labels_cat = test_data[number_of_stages_key]
test_labels_lin = test_data[number_of_stages_key + '- linear']

# Predict and calculate performance metrics
pred_test_labels = clf.predict(test_feature_subset)
pred_test_labels_prob = clf.predict_proba(test_feature_subset)
test_acc = accuracy_score(test_labels_lin, pred_test_labels)
test_cohen_kappa = cohen_kappa_score(test_labels_lin, pred_test_labels)
test_auc = roc_auc_score(pd.get_dummies(test_labels_lin), pd.DataFrame(pred_test_labels_prob), multi_class='ovo')

print("\n Test AUC: {:.2f}".format(test_auc),
      "\n Test Accuracy: {:.2f}".format(test_acc),
      "\n Test Cohen's kappa: {:.2f}".format(test_cohen_kappa))