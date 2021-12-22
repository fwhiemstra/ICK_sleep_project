import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_5_Sampled_FeatureData_for_testing'
feature_data_file_name = '10000sampled_feature_data_0-18years_nonscaled.xlsx'

feature_data = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
number_of_stages_key = 'Three_state_labels'

eog_features_list = [col for col in feature_data if col.startswith('EOG ROC-LOC:')]
emg_features_list = [col for col in feature_data if col.startswith('EMG EMG Chin:')]
age_features_list = list(pd.get_dummies(feature_data['Age category'],
                                        prefix='age_group_').columns)

"""
##### EXTRACT LABELS AND FEATURES #####
"""
### Add labels and extract the feature
labels_and_features = pd.DataFrame([])
labels_and_features[number_of_stages_key] = feature_data[number_of_stages_key]
labels_and_features[number_of_stages_key + '- linear'] = feature_data[number_of_stages_key + '- linear']
labels_and_features['Patient key'] = feature_data['Patient key']
labels_and_features['Age category'] = feature_data['Age category']
labels_and_features[eog_features_list] = feature_data[eog_features_list]
labels_and_features[emg_features_list] = feature_data[emg_features_list]
labels_and_features[age_features_list] = feature_data[age_features_list]
# EEG features
eeg_features_list = [col for col in feature_data if col.startswith(eeg_channel)]
labels_and_features[eeg_features_list] = feature_data[eeg_features_list]

labels_and_features = labels_and_features.dropna(axis=0)  # Drop rows with NaNs in it
age_features = labels_and_features[age_features_list]
labels_and_features = labels_and_features.drop(columns=age_features_list)

### Extract labels, features and groups
labels_cat = labels_and_features[number_of_stages_key]
labels_lin = labels_and_features[number_of_stages_key + '- linear']
groups = labels_and_features['Patient key']
features = labels_and_features.iloc[:, 4:]

from xgboost import XGBClassifier
#clf = XGBClassifier(objective='reg:linear', eval_metric='mlogloss', max_depth=6, gamma=0)

clf = DecisionTreeClassifier(criterion='gini')
clf.fit(features, labels_cat)
clf.score(features, labels_cat)

#%%
import pickle
file_name = r'E:\ICK slaap project\DTclf.sav'
pickle.dump(clf,  open(file_name, 'wb'))

#%%
model_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'
model_file = 'DT_Model_EEG F3-C4_Four_state_labels.sav'
loaded_model = pickle.load(open(r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_3_Training & Crossvalidation results\Trained models per channel\DT_Model_EEG A1.sav', 'rb'))
result = loaded_model.score(features, labels_cat)
print(result)

#%%
import pandas as pd
from datetime import datetime
import os

file_path = r'E:\ICK slaap project\4_FeatureData\4_7_TrainingSet'

for file_name in os.listdir(file_path):
    if file_name.endswith('.xlsx'):
        read_file = pd.read_excel(os.path.join(file_path, file_name))
        read_file.to_csv(os.path.join(file_path, file_name.replace('.xlsx', '.csv')), index=None, header=True)
        print(datetime.now(), ': %s converted' % file_name)

