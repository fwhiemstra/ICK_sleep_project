import pickle
import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append(r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\src')
from Functions import add_new_state_labels
import numpy as np

number_of_stages_key = 'Three_state_labels'
number_of_stages = 3
classifier = 'SVM'
channel = 'EEG F3-C3'
picu_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_4_PICU_FeatureData_SCALED'
model_files_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_4_Final models'
results_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_6_Performance on PICU data'
save_plots_file_path = r'E:\ICK slaap project\6_ClassificationModelDevelopment\6_7_Hypnogram plots'

model_file = '%s_Model_%s_%s.sav' % (classifier, channel, number_of_stages_key)
clf = pickle.load(open(os.path.join(model_files_path, model_file), 'rb'))
results_file_name = 'PICU_results_%s_%s_%s.xlsx' % (classifier, channel, number_of_stages_key)
results_sheet = pd.read_excel(os.path.join(results_file_path, results_file_name))

picu_files = ['PICU001_FeatureData_scaled.xlsx', 'PICU002_FeatureData_scaled.xlsx',
              'PICU003_FeatureData_scaled.xlsx', 'PICU004_FeatureData_ArtLabelsAdjusted_scaled.xlsx',
              'PICU005_FeatureData_scaled.xlsx', 'PICU006_FeatureData_scaled.xlsx',
              'PICU007_FeatureData - EEG F4-C4_scaled.xlsx', 'PICU008_FeatureData_scaled.xlsx',
              'PICU009_FeatureData - EEG F4-C4_scaled.xlsx', 'PICU010_FeatureData_scaled.xlsx']

### Set features for each model
if model_file == 'DT_Model_EEG F3-C4_Three_state_labels.sav':
    feature_names = ['EEG F3-C4: DWT cD5 mean', 'EEG F3-C4: Rel Gamma power',
                    'EEG F3-C4: Interquartile range', 'EEG F3-C4: DWT cD1 std',
                    'EEG F3-C4: Zero crossing rate', 'EEG F3-C4: Gamma/theta ratio',
                    'EEG F3-C4: DWT cA mean', 'EEG F3-C4: DWT cD4 std',
                    'EEG F3-C4: DWT cD3 mean', 'EEG F3-C4: DWT cD1 mean',
                    'EEG F3-C4: Gamma/delta ratio', 'EEG F3-C4: DWT cD3 std',
                    'age_group__0-2 months', 'age_group__1-3 years',
                    'age_group__13-18 years', 'age_group__2-6 months',
                    'age_group__3-5 years', 'age_group__5-9 years',
                    'age_group__6-12 months', 'age_group__9-13 years']
elif model_file == 'DT_Model_EEG F3-C4_Four_state_labels.sav':
    feature_names = ['EEG F3-C4: Abs Beta power', 'EEG F3-C4: DWT cD5 std',
                   'EEG F3-C4: Interquartile range', 'EEG F3-C4: Rel Gamma power',
                   'EEG F3-C4: Abs mean amplitude', 'EEG F3-C4: Zero crossing rate',
                   'EEG F3-C4: DWT cA std', 'EEG F3-C4: DWT cD4 std',
                   'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C4: Abs Delta power',
                   'EEG F3-C4: Spectral edge', 'age_group__1-3 years',
                   'age_group__13-18 years', 'age_group__5-9 years',
                   'age_group__6-12 months', 'age_group__9-13 years']

if model_file == 'DT_Model_EEG F3-C3_Three_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
       'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: Zero crossing rate',
       'EEG F3-C3: DWT cD4 mean', 'EEG F3-C3: DWT cA mean',
       'EEG F3-C3: Gamma/theta ratio', 'EEG F3-C3: Hjorth mobility',
       'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: DWT cA std',
       'EEG F3-C3: DWT cD3 mean', 'EEG F3-C3: Gamma/delta ratio',
       'age_group__0-2 months', 'age_group__1-3 years',
       'age_group__13-18 years', 'age_group__2-6 months',
       'age_group__3-5 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']

elif model_file == 'DT_Model_EEG F3-C3_Four_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
       'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: Abs mean amplitude',
       'EEG F3-C3: Zero crossing rate',
       'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: Hjorth mobility',
       'EEG F3-C3: DWT cA std', 'EEG F3-C3: DWT cD4 mean',
       'EEG F3-C3: DWT cD1 mean', 'EEG F3-C3: Total power',
       'age_group__1-3 years', 'age_group__13-18 years',
       'age_group__5-9 years', 'age_group__6-12 months',
       'age_group__9-13 years']

elif model_file == 'SVM_Model_EEG F3-C3_Three_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD5 mean',
       'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: Abs mean amplitude',
       'EEG F3-C3: Signal sum', 'EEG F3-C3: DWT cD5 std',
       'EEG F3-C3: DWT cD4 mean', 'EEG F3-C3: Gamma/theta ratio',
       'EEG F3-C3: DWT cA mean', 'EEG F3-C3: DWT cA std',
       'EEG F3-C3: DWT cD4 std', 'EEG F3-C3: Gamma/delta ratio',
       'age_group__0-2 months', 'age_group__1-3 years',
       'age_group__13-18 years', 'age_group__2-6 months',
       'age_group__3-5 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']
elif model_file == 'SVM_Model_EEG F3-C3_Four_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
                   'EEG F3-C3: DWT cD5 mean', 'EEG F3-C3: Abs mean amplitude',
                   'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: DWT cA mean',
                   'EEG F3-C3: Zero crossing rate',
                   'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: DWT cD1 mean',
                   'EEG F3-C3: DWT cA std', 'EEG F3-C3: Hjorth mobility',
                   'EEG F3-C3: DWT cD4 mean', 'EEG F3-C3: Spectral edge',
                   'EEG F3-C3: Spectral entropy', 'EEG F3-C3: Abs Sigma power',
                   'EEG F3-C3: Abs Delta power', 'age_group__1-3 years',
                   'age_group__13-18 years', 'age_group__6-12 months',
                   'age_group__9-13 years']
elif model_file == 'xgboost_Model_EEG F3-C3_Three_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
                   'EEG F3-C3: Rel Gamma power', 'EEG F3-C3: DWT cD5 mean',
                   'EEG F3-C3: DWT cA mean', 'EEG F3-C3: Gamma/theta ratio',
                   'EEG F3-C3: DWT cD5 std', 'EEG F3-C3: Hjorth mobility',
                   'EEG F3-C3: DWT cA std', 'EEG F3-C3: DWT cD1 mean',
                   'EEG F3-C3: DWT cD4 std', 'EEG F3-C3: DWT cD3 mean',
                   'EEG F3-C3: Total power', 'age_group__0-2 months',
                   'age_group__13-18 years', 'age_group__2-6 months',
                   'age_group__3-5 years', 'age_group__5-9 years',
                   'age_group__6-12 months', 'age_group__9-13 years']
elif model_file == 'xgboost_Model_EEG F3-C3_Four_state_labels.sav':
    feature_names = ['EEG F3-C3: Interquartile range', 'EEG F3-C3: DWT cD1 std',
       'EEG F3-C3: Abs mean amplitude', 'EEG F3-C3: DWT cA mean',
       'EEG F3-C3: DWT cD5 mean', 'EEG F3-C3: Rel Gamma power',
       'EEG F3-C3: Zero crossing rate', 'EEG F3-C3: DWT cD5 std',
       'EEG F3-C3: DWT cA std', 'EEG F3-C3: Abs Sigma power',
       'EMG EMG Chin: Mean absolute amplitude', 'EEG F3-C3: DWT cD4 mean',
       'EEG F3-C3: DWT cD3 mean', 'EEG F3-C3: Total power',
       'age_group__13-18 years', 'age_group__5-9 years',
       'age_group__6-12 months', 'age_group__9-13 years']

for feature_data_file_name in picu_files:
    print(datetime.datetime.now(), ': %s started' % feature_data_file_name)
    feature_data = pd.read_excel(os.path.join(picu_data_file_path, feature_data_file_name))

    ### Preprare data
    feature_data, a = add_new_state_labels(feature_data, 3)
    feature_data, a = add_new_state_labels(feature_data, 4)
    print(datetime.datetime.now(), ": New sleep stages added")
    dummy_variables_names = ['age_group__0-2 months', 'age_group__1-3 years',
                             'age_group__13-18 years', 'age_group__2-6 months',
                             'age_group__3-5 years', 'age_group__5-9 years',
                             'age_group__6-12 months', 'age_group__9-13 years']
    dfDummyVariables = pd.DataFrame(
        data=np.zeros((len(feature_data), len(dummy_variables_names))), columns=dummy_variables_names)
    dfDummyVariable_file = pd.get_dummies(feature_data['Age category'],
                                          prefix='age_group_')
    feature_data = feature_data.join(dfDummyVariables)
    feature_data[dfDummyVariable_file.columns] = dfDummyVariable_file.values
    print(datetime.datetime.now(), ": Dummy variables added")

    labels_true = feature_data[number_of_stages_key + '- linear']
    labels_pred = clf.predict(feature_data[feature_names])

    title_name = 'x'
    if feature_data_file_name == 'PICU001_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient A'
        # start_time = datetime.datetime(2021, 10, 21, 12, 14, 35)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU002_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient B'
        # start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU003_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient C'
        # start_time = datetime.datetime(2021, 10, 21, 9, 17, 28)
        start_time = datetime.datetime(2021, 10, 21, 9, 00, 00)

    elif feature_data_file_name == 'PICU004_FeatureData_ArtLabelsAdjusted_scaled.xlsx':
        title_name = 'PICU Patient D'
        # start_time = datetime.datetime(2021, 10, 21, 9, 2, 27)
        start_time = datetime.datetime(2021, 10, 21, 9, 0, 0)

    elif feature_data_file_name == 'PICU005_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient E'
        # start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 12, 0, 0)

    elif feature_data_file_name == 'PICU006_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient F'
        # start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 11, 0, 0)

    elif feature_data_file_name == 'PICU007_FeatureData - EEG F4-C4_scaled.xlsx':
        title_name = 'PICU Patient G'
        # start_time = datetime.datetime(2021, 10, 21, 15, 59, 53)
        start_time = datetime.datetime(2021, 10, 21, 16, 0, 0)

    elif feature_data_file_name == 'PICU008_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient H'
        # start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU009_FeatureData - EEG F4-C4_scaled.xlsx':
        title_name = 'PICU Patient I'
        # start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU010_FeatureData_scaled.xlsx':
        title_name = 'PICU Patient J'
        # start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 16, 00, 00)

    row_idx = results_sheet['Patient key'] == feature_data_file_name[0:7]
    accuracy = results_sheet[row_idx]['Accuracy']
    AUC = results_sheet[row_idx]['AUC posterior']
    cohens_kappa = results_sheet[row_idx]['Cohens kappa']

    # Hypnogram
    plt.rc('font', size=18, family='sans-serif')          # controls default text sizes
    plt.rc('axes', titlesize=10)     # fontsize of the axes title
    plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title
    plt.rc('axes', axisbelow=True)
    plt.close('all')
    end_time = start_time + datetime.timedelta(seconds=30 * len(labels_true))
    t = []
    for i in range(len(labels_true)):
        t.append((start_time + datetime.timedelta(seconds=30 * i)).strftime('%H:%M'))

    ticks = [i for i in range(0, len(t), 120)]
    timestamp_labels = [t[i] for i in ticks]
    plt.figure(figsize=(10, 4))
    #order = ['Sleep stage N', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
    if classifier == 'DT':
        color = 'crimson'
    elif classifier == 'SVM':
        color = 'darkcyan'
    elif classifier == 'xgboost':
        color = 'olivedrab'

    plt.step(range(0, len(labels_pred)), labels_pred, where='post',
                color=color, linewidth=0.5, linestyle='--', label='Predicted hypnogram')
    plt.step(range(0, len(labels_true)), labels_true, where='post',
                color='black', linewidth=1, label='Visually scored hypnogram')
    if number_of_stages == 3:
        plt.yticks(ticks=range(0, 3), labels=['SWS', 'NSWS', 'Wake'])
    if number_of_stages == 4:
        plt.yticks(ticks=range(0, 4), labels=['SWS', 'REM', 'NSWS', 'Wake'])
    plt.ylabel('Sleep stage labels')
    plt.xlabel('Time (hh:mm)')
    plt.xlim([0, len(labels_true)])
    plt.ylim(-0.1, number_of_stages-0.5)
    plt.xticks(ticks=ticks, labels=timestamp_labels, rotation=90)
    plt.tight_layout()
    plt.legend(loc='upper right')
    text = 'Acc.=%.2f, AUC=%.2f, $\kappa$=%.2f' % (accuracy, AUC, cohens_kappa)
    plt.text(25, 2.35, text, fontsize=6, fontstyle='italic')

    plt.savefig(os.path.join(save_plots_file_path, title_name.replace('PICU ', '') + '_%s_%s_%s' %
                             (classifier, number_of_stages_key, channel) + '.png'), dpi=1000)




