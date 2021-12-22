import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

### File + index settings
picu_files = ['PICU001_FeatureData.xlsx', 'PICU002_FeatureData.xlsx', \
              'PICU003_FeatureData.xlsx', 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx',\
              'PICU005_FeatureData.xlsx', 'PICU006_FeatureData.xlsx',
              'PICU007_FeatureData - EEG F4-C4.xlsx', 'PICU008_FeatureData.xlsx',
              'PICU009_FeatureData - EEG F4-C4.xlsx',
              'PICU010_FeatureData.xlsx']
#picu_files = ['PICU006_FeatureData.xlsx']
eeg_channel = 'EEG F3-C3:'

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
plot_file_path = r'E:\ICK slaap project\7_IndexBasedClassification\7_4_Hypnogram + Index plots'
index_measure = 'GammaTheta+DeltaRatio'

for feature_data_file_name in picu_files:
    print(datetime.datetime.now(), ': %s started' % feature_data_file_name)
    ### Load data
    feature_data = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
    order = ['Sleep stage N', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
    sleep_stage_labels = np.array([order.index(x) for x in feature_data["Sleep stage labels"]])
    title_name = 'x'

    if index_measure == 'GammaDeltaRatio':
        index = feature_data['%s Gamma/delta ratio' % eeg_channel]
        index_sws_threshold_fixed = np.array([0.003966432 for i in range(len(sleep_stage_labels))])
        index_wake_threshold_fixed = np.array([0.031400108 for i in range(len(sleep_stage_labels))])
        ylabel = 'Gamma/delta ratio'
    elif index_measure == 'GammaTheta+DeltaRatio':
        index = feature_data['%s Abs Gamma power' % eeg_channel] / (feature_data['%s Abs Theta power' % eeg_channel]+feature_data['% s Abs Delta power' % eeg_channel])
        index_sws_threshold_fixed = np.array([0.003190831 for i in range(len(sleep_stage_labels))])
        index_wake_threshold_fixed = np.array([0.022055037 for i in range(len(sleep_stage_labels))])
        ylabel = 'Gamma/(theta+delta) ratio'

    if feature_data_file_name == 'PICU001_FeatureData.xlsx':
        title_name = 'PICU Patient A'
        #start_time = datetime.datetime(2021, 10, 21, 12, 14, 35)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.599551857
            auc_fixed = 0.774181379
            cohens_kappa_fixed = 0.280830603

            accuracy_individual = 0.667413572
            auc_individual = 0.793501313
            cohens_kappa_individual = 0.341058189

            index_sws_threshold = np.array([0.001571331 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.013603255 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.624199744
            auc_fixed = 0.792455866
            cohens_kappa_fixed = 0.322805321

            accuracy_individual = 0.672215109
            auc_individual = 0.811963751
            cohens_kappa_individual = 0.354001144

            index_sws_threshold = np.array([0.001421386 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.011467565 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU002_FeatureData.xlsx':
        title_name = 'PICU Patient B'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.635666347
            auc_fixed = 0.712126901
            cohens_kappa_fixed = 0.186804709

            accuracy_individual = 0.678811122
            auc_individual = 0.74895707
            cohens_kappa_individual = 0.291092319

            index_sws_threshold = np.array([0.003238328 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.004048845 for i in range(len(sleep_stage_labels))])

        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.625119847
            auc_fixed = 0.715807173
            cohens_kappa_fixed = 0.200128274

            accuracy_individual = 0.699904123
            auc_individual = 0.751065066
            cohens_kappa_individual = 0.338350878

            index_sws_threshold = np.array([0.002808228 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.00302141 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU003_FeatureData.xlsx':
        title_name = 'PICU Patient C'
        #start_time = datetime.datetime(2021, 10, 21, 9, 17, 28)
        start_time = datetime.datetime(2021, 10, 21, 9, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.7240671
            auc_fixed = 0.807298269
            cohens_kappa_fixed = 0.518833212

            accuracy_individual = 0.785005135
            auc_individual = 0.830268006
            cohens_kappa_individual = 0.618286017

            index_sws_threshold = np.array([0.020578346 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.067838425 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.712427251
            auc_fixed = 0.802198272
            cohens_kappa_fixed = 0.494793073

            accuracy_individual = 0.786032181
            auc_individual = 0.829786924
            cohens_kappa_individual = 0.619889999

            index_sws_threshold = np.array([0.020581071 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.063612317 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx':
        title_name = 'PICU Patient D'
        #start_time = datetime.datetime(2021, 10, 21, 9, 2, 27)
        start_time = datetime.datetime(2021, 10, 21, 9, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.598386531
            auc_fixed = 0.803546365
            cohens_kappa_fixed = 0.302830371

            accuracy_individual = 0.871623992
            auc_individual = 0.85496989
            cohens_kappa_individual = 0.72877064

            index_sws_threshold = np.array([0.003850674 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.003850674 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.804993768
            auc_fixed = 0.803546365
            cohens_kappa_fixed = 0.316411169

            accuracy_individual = 0.855840056
            auc_individual = 0.857967116
            cohens_kappa_individual = 0.702735442

            index_sws_threshold = np.array([0.003421092 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.004062999 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU005_FeatureData.xlsx':
        title_name = 'PICU Patient E'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 12, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.477673936
            auc_fixed = 0.746071011
            cohens_kappa_fixed = 0.192478334

            accuracy_individual = 0.671512634
            auc_individual = 0.749012247
            cohens_kappa_individual = 0.378188877

            index_sws_threshold = np.array([0.007881359 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.011259639 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.442367601
            auc_fixed = 0.746557463
            cohens_kappa_fixed = 0.170242296

            accuracy_individual = 0.680512288
            auc_individual = 0.751352582
            cohens_kappa_individual = 0.389245738

            index_sws_threshold = np.array([0.007645318 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.010250887 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU006_FeatureData.xlsx':
        title_name = 'PICU Patient F'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 11, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.557377049
            auc_fixed = 0.815517195
            cohens_kappa_fixed = 0.367930399

            accuracy_individual = 0.656420765
            auc_individual = 0.819762721
            cohens_kappa_individual = 0.489743041

            index_sws_threshold = np.array([0.005545864 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.017482007 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.539959016
            auc_fixed = 0.822190388
            cohens_kappa_fixed = 0.346300964

            accuracy_individual = 0.660860656
            auc_individual = 0.825232057
            cohens_kappa_individual = 0.496703521

            index_sws_threshold = np.array([0.004994776 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.015065154 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU007_FeatureData - EEG F4-C4.xlsx':
        title_name = 'PICU Patient G'
        #start_time = datetime.datetime(2021, 10, 21, 15, 59, 53)
        start_time = datetime.datetime(2021, 10, 21, 16, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.598684211
            auc_fixed = 0.742333522
            cohens_kappa_fixed = 0.28195891

            accuracy_individual = 0.632739938
            auc_individual = 0.758150455
            cohens_kappa_individual = 0.318742596

            index_sws_threshold = np.array([0.002492049 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.047399398 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.585913313
            auc_fixed = 0.753813806
            cohens_kappa_fixed = 0.263433465

            accuracy_individual = 0.649380805
            auc_individual = 0.767906401
            cohens_kappa_individual = 0.352275057

            index_sws_threshold = np.array([0.002269427 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.038176179 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU008_FeatureData.xlsx':
        title_name = 'PICU Patient H'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.076207
            auc_fixed = 0
            cohens_kappa_fixed = -0.058442055

            accuracy_individual = 0.076206897
            auc_individual = 0.758150455
            cohens_kappa_individual = -0.000689873

            index_sws_threshold = np.array([0.000276361 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.000276361 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.076207
            auc_fixed = 0
            cohens_kappa_fixed = -0.060507881

            accuracy_individual = 0.076206897
            auc_individual = 0.767906401
            cohens_kappa_individual = -0.000689873

            index_sws_threshold = np.array([0.000270848 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.000270848 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU009_FeatureData - EEG F4-C4.xlsx':
        title_name = 'PICU Patient I'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.41
            auc_fixed = 0.5510582
            cohens_kappa_fixed = -0.117490819

            accuracy_individual = 0.299304952
            auc_individual = 0.688531036
            cohens_kappa_individual = -0.109625352

            #index_sws_threshold = np.array([0.0027841476566260125 for i in range(len(sleep_stage_labels))])
            #index_wake_threshold = np.array([0.0032092647675535622 for i in range(len(sleep_stage_labels))])
            index_sws_threshold = np.array([0.003140429 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.009362074 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.37
            auc_fixed = 0.544917124
            cohens_kappa_fixed = -0.140288391

            accuracy_individual = 0.298436142
            auc_individual = 0.69119155
            cohens_kappa_individual = -0.112892893

            index_sws_threshold = np.array([0.002974283 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.008873768 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU010_FeatureData.xlsx':
        title_name = 'PICU Patient J'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 16, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.714787496
            auc_fixed = 0.898388452
            cohens_kappa_fixed = 0.563510759

            accuracy_individual = 0.811029153
            auc_individual = 0.919411099
            cohens_kappa_individual = 0.693682772

            index_sws_threshold = np.array([0.002272441 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.014232048 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.770635757
            auc_fixed = 0.909717248
            cohens_kappa_fixed = 0.640886517

            accuracy_individual = 0.810677907
            auc_individual = 0.92037158
            cohens_kappa_individual = 0.694176261

            index_sws_threshold = np.array([0.002379061 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.013742176 for i in range(len(sleep_stage_labels))])

    # Plot settings
    plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.rc('axes', axisbelow=True)

    plt.close('all')
    fig, axs = plt.subplots(3, figsize=(10, 5))
    #plt.suptitle(title_name)
    # Add time stamp labels
    end_time = start_time + datetime.timedelta(seconds=30*len(sleep_stage_labels))
    t = []
    for i in range(len(sleep_stage_labels)):
        t.append((start_time + datetime.timedelta(seconds=30*i)).strftime('%H:%M'))

    ticks = [i for i in range(0, len(t), 120)]
    timestamp_labels = [t[i] for i in ticks]

    # Hypnogram
    axs[0].set_title('Hypnogram')
    axs[0].set_xticklabels([]) # remove number at x axis
    axs[0].set_xticks(ticks=ticks)
    axs[0].set_yticks(ticks=[0, 1, 2, 3, 4, 5])
    axs[0].set_yticklabels(labels=['N', 'N3', 'N2', 'N1', 'REM', 'Wake']) # remove number at x axis
    axs[0].set_ylabel('Sleep stage labels')
    axs[0].set_xlim([0, len(sleep_stage_labels)])
    axs[0].step(range(0, len(sleep_stage_labels)), sleep_stage_labels, where='post',
                color='olivedrab', linewidth=0.5)
    hypn_wake = np.array([4 for i in range(len(sleep_stage_labels))])
    hypn_wake[sleep_stage_labels > 4] = sleep_stage_labels[sleep_stage_labels > 4]
    axs[0].step(range(0, len(sleep_stage_labels)), hypn_wake, where='post',
                color='crimson', linewidth=0.5)
    axs[0].plot(np.array([4 for i in range(len(sleep_stage_labels))]),
                linewidth=0.7, color='white')
    axs[0].plot(np.array([4 for i in range(len(sleep_stage_labels))]), label='Wake threshold',
                linewidth=0.7, color='black', linestyle='--')
    hypn_sws = np.array([2 for i in range(len(sleep_stage_labels))])
    hypn_sws[sleep_stage_labels < 2] = sleep_stage_labels[sleep_stage_labels < 2]
    axs[0].step(range(0, len(sleep_stage_labels)), hypn_sws, where='post',
                color='navy', linewidth=0.5)
    axs[0].plot(np.array([2 for i in range(len(sleep_stage_labels))]),
                linewidth=0.7, color='white')
    axs[0].plot(np.array([2 for i in range(len(sleep_stage_labels))]),
                linewidth=0.7, color='black', linestyle='--')
    #axs[0].set_ylim([10e-4, 5])

    # Fixed thresholds
    axs[1].plot(index, color='olivedrab', linewidth=0.5, label='NSWS')
    axs[1].set_yscale('log')
    axs[1].set_xticklabels([]) # remove number at x axis
    axs[1].set_ylabel(ylabel)
    axs[1].set_title("General thresholds")
    #axs[1].set_title("")
    axs[1].set_xlim([0, len(sleep_stage_labels)])
    axs[1].set_xticks(ticks=ticks)

    index_wake = np.array([index_wake_threshold_fixed[0] for i in range(len(sleep_stage_labels))])
    index_wake[index > index_wake_threshold_fixed[0]] = index[index > index_wake_threshold_fixed[0]]
    axs[1].plot(index_wake, color='crimson', linewidth=0.5, label='Wake')
    axs[1].plot(index_wake_threshold_fixed,
                linewidth=0.7, color='white')
    axs[1].plot(index_wake_threshold_fixed,
                linewidth=0.7, color='black', linestyle='--')

    index_sws = np.array([index_sws_threshold_fixed[0] for i in range(len(sleep_stage_labels))])
    index_sws[index < index_sws_threshold_fixed[0]] = index[index < index_sws_threshold_fixed[0]]
    axs[1].plot(index_sws, color='navy', linewidth=0.5, label='SWS')
    axs[1].plot(index_sws_threshold_fixed,
                linewidth=0.7, color='white')
    axs[1].plot(index_sws_threshold_fixed, label='Index-score threshold',
                linewidth=0.7, color='black', linestyle='--')
    axs[1].set_ylim([10e-5, 7])
    axs[1].text(25, 2, 'Acc.=%.2f, AUC=%.2f, $\kappa$=%.2f' %
                (accuracy_fixed, auc_fixed, cohens_kappa_fixed),
                fontsize=6, fontstyle='italic')

    # Individualized thresholds
    axs[2].plot(index, color='olivedrab', linewidth=0.5, label='NSWS')
    axs[2].set_yscale('log')
    axs[2].set_ylabel(ylabel)
    axs[2].set_title("Individualized thresholds")
    axs[2].set_xlim([0, len(sleep_stage_labels)])

    index_wake = np.array([index_wake_threshold[0] for i in range(len(sleep_stage_labels))])
    index_wake[index > index_wake_threshold[0]] = index[index > index_wake_threshold[0]]
    axs[2].plot(index_wake, color='crimson', linewidth=0.5, label='Wake')
    axs[2].plot(index_wake_threshold,
                linewidth=0.7, color='white')
    axs[2].plot(index_wake_threshold,
                linewidth=0.7, color='black', linestyle='--')
    index_sws = np.array([index_sws_threshold[0] for i in range(len(sleep_stage_labels))])
    index_sws[index < index_sws_threshold[0]] = index[index < index_sws_threshold[0]]
    axs[2].plot(index_sws, color='navy', linewidth=0.5, label='SWS')
    axs[2].plot(index_sws_threshold,
                linewidth=0.7, color='white')
    axs[2].plot(index_sws_threshold,
                linewidth=0.7, color='black', linestyle='--', label='Index-score threshold',)
    axs[2].set_xlabel('Time (hours)')
    axs[2].set_xticks(ticks=ticks)
    axs[2].set_xticklabels(labels=timestamp_labels, rotation=90) # remove number at x axis
    axs[2].set_ylim([10e-5, 7])
    axs[2].text(25, 2, 'Acc.=%.2f, AUC=%.2f, $\kappa$=%.2f' %
                (accuracy_individual, auc_individual, cohens_kappa_individual),
                fontsize=6, fontstyle='italic')

    # Add legend
    handles, labels = axs[2].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.16, hspace=0.35)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    # Save plot
    plot_file_name = 'Patient' + title_name.replace('PICU Patient ', '') + '_%s' % eeg_channel.replace(':', '') \
                     + '_Fixed_+Individualized_thresholds_' + index_measure + '.png'
    plt.savefig(os.path.join(plot_file_path, plot_file_name), dpi=1000)

#%% With fixed thresholds only

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

### File + index settings
picu_files = ['PICU001_FeatureData.xlsx', 'PICU002_FeatureData.xlsx', \
              'PICU003_FeatureData.xlsx', 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx',\
              'PICU005_FeatureData.xlsx', 'PICU006_FeatureData.xlsx',
              'PICU007_FeatureData - EEG F4-C4.xlsx', 'PICU008_FeatureData.xlsx',
              'PICU009_FeatureData - EEG F4-C4.xlsx',
              'PICU010_FeatureData.xlsx']
picu_files = ['PICU003_FeatureData.xlsx']
eeg_channel = 'EEG F3-C3:'

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
plot_file_path = r'E:\ICK slaap project\7_IndexBasedClassification\7_4_Hypnogram + Index plots'
index_measure = 'GammaTheta+DeltaRatio'

for feature_data_file_name in picu_files:
    print(datetime.datetime.now(), ': %s started' % feature_data_file_name)
    ### Load data
    feature_data = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
    order = ['Sleep stage N', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
    sleep_stage_labels = np.array([order.index(x) for x in feature_data["Sleep stage labels"]])
    title_name = 'x'

    if index_measure == 'GammaDeltaRatio':
        index = feature_data['%s Gamma/delta ratio' % eeg_channel]
        index_sws_threshold_fixed = np.array([0.003966432 for i in range(len(sleep_stage_labels))])
        index_wake_threshold_fixed = np.array([0.031400108 for i in range(len(sleep_stage_labels))])
        ylabel = 'Gamma/delta ratio'
    elif index_measure == 'GammaTheta+DeltaRatio':
        index = feature_data['%s Abs Gamma power' % eeg_channel] / (feature_data['%s Abs Theta power' % eeg_channel]+feature_data['% s Abs Delta power' % eeg_channel])
        index_sws_threshold_fixed = np.array([0.003190831 for i in range(len(sleep_stage_labels))])
        index_wake_threshold_fixed = np.array([0.022055037 for i in range(len(sleep_stage_labels))])
        ylabel = 'Gamma/(theta+delta) ratio'

    if feature_data_file_name == 'PICU001_FeatureData.xlsx':
        title_name = 'PICU Patient A'
        #start_time = datetime.datetime(2021, 10, 21, 12, 14, 35)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.599551857
            auc_fixed = 0.774181379
            cohens_kappa_fixed = 0.280830603

            accuracy_individual = 0.667413572
            auc_individual = 0.793501313
            cohens_kappa_individual = 0.341058189

            index_sws_threshold = np.array([0.001571331 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.013603255 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.624199744
            auc_fixed = 0.792455866
            cohens_kappa_fixed = 0.322805321

            accuracy_individual = 0.672215109
            auc_individual = 0.811963751
            cohens_kappa_individual = 0.354001144

            index_sws_threshold = np.array([0.001421386 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.011467565 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU002_FeatureData.xlsx':
        title_name = 'PICU Patient B'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.635666347
            auc_fixed = 0.712126901
            cohens_kappa_fixed = 0.186804709

            accuracy_individual = 0.678811122
            auc_individual = 0.74895707
            cohens_kappa_individual = 0.291092319

            index_sws_threshold = np.array([0.003238328 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.004048845 for i in range(len(sleep_stage_labels))])

        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.625119847
            auc_fixed = 0.715807173
            cohens_kappa_fixed = 0.200128274

            accuracy_individual = 0.699904123
            auc_individual = 0.751065066
            cohens_kappa_individual = 0.338350878

            index_sws_threshold = np.array([0.002808228 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.00302141 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU003_FeatureData.xlsx':
        title_name = 'PICU Patient C'
        #start_time = datetime.datetime(2021, 10, 21, 9, 17, 28)
        start_time = datetime.datetime(2021, 10, 21, 9, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.7240671
            auc_fixed = 0.807298269
            cohens_kappa_fixed = 0.518833212

            accuracy_individual = 0.785005135
            auc_individual = 0.830268006
            cohens_kappa_individual = 0.618286017

            index_sws_threshold = np.array([0.020578346 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.067838425 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.712427251
            auc_fixed = 0.802198272
            cohens_kappa_fixed = 0.494793073

            accuracy_individual = 0.786032181
            auc_individual = 0.829786924
            cohens_kappa_individual = 0.619889999

            index_sws_threshold = np.array([0.020581071 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.063612317 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx':
        title_name = 'PICU Patient D'
        #start_time = datetime.datetime(2021, 10, 21, 9, 2, 27)
        start_time = datetime.datetime(2021, 10, 21, 9, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.598386531
            auc_fixed = 0.803546365
            cohens_kappa_fixed = 0.302830371

            accuracy_individual = 0.871623992
            auc_individual = 0.85496989
            cohens_kappa_individual = 0.72877064

            index_sws_threshold = np.array([0.003850674 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.003850674 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.804993768
            auc_fixed = 0.803546365
            cohens_kappa_fixed = 0.316411169

            accuracy_individual = 0.855840056
            auc_individual = 0.857967116
            cohens_kappa_individual = 0.702735442

            index_sws_threshold = np.array([0.003421092 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.004062999 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU005_FeatureData.xlsx':
        title_name = 'PICU Patient E'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 12, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.477673936
            auc_fixed = 0.746071011
            cohens_kappa_fixed = 0.192478334

            accuracy_individual = 0.671512634
            auc_individual = 0.749012247
            cohens_kappa_individual = 0.378188877

            index_sws_threshold = np.array([0.007881359 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.011259639 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.442367601
            auc_fixed = 0.746557463
            cohens_kappa_fixed = 0.170242296

            accuracy_individual = 0.680512288
            auc_individual = 0.751352582
            cohens_kappa_individual = 0.389245738

            index_sws_threshold = np.array([0.007645318 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.010250887 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU006_FeatureData.xlsx':
        title_name = 'PICU Patient F'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 11, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.557377049
            auc_fixed = 0.815517195
            cohens_kappa_fixed = 0.367930399

            accuracy_individual = 0.656420765
            auc_individual = 0.819762721
            cohens_kappa_individual = 0.489743041

            index_sws_threshold = np.array([0.005545864 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.017482007 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.539959016
            auc_fixed = 0.822190388
            cohens_kappa_fixed = 0.346300964

            accuracy_individual = 0.660860656
            auc_individual = 0.825232057
            cohens_kappa_individual = 0.496703521

            index_sws_threshold = np.array([0.004994776 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.015065154 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU007_FeatureData - EEG F4-C4.xlsx':
        title_name = 'PICU Patient G'
        #start_time = datetime.datetime(2021, 10, 21, 15, 59, 53)
        start_time = datetime.datetime(2021, 10, 21, 16, 0, 0)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.598684211
            auc_fixed = 0.742333522
            cohens_kappa_fixed = 0.28195891

            accuracy_individual = 0.632739938
            auc_individual = 0.758150455
            cohens_kappa_individual = 0.318742596

            index_sws_threshold = np.array([0.002492049 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.047399398 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.585913313
            auc_fixed = 0.753813806
            cohens_kappa_fixed = 0.263433465

            accuracy_individual = 0.649380805
            auc_individual = 0.767906401
            cohens_kappa_individual = 0.352275057

            index_sws_threshold = np.array([0.002269427 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.038176179 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU008_FeatureData.xlsx':
        title_name = 'PICU Patient H'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.076207
            auc_fixed = 'NaN'
            cohens_kappa_fixed = -0.058442055

            accuracy_individual = 0.076206897
            auc_individual = 0.758150455
            cohens_kappa_individual = -0.000689873

            index_sws_threshold = np.array([0.000276361 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.000276361 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.076207
            auc_fixed = 'NaN'
            cohens_kappa_fixed = -0.060507881

            accuracy_individual = 0.076206897
            auc_individual = 0.767906401
            cohens_kappa_individual = -0.000689873

            index_sws_threshold = np.array([0.000270848 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.000270848 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU009_FeatureData - EEG F4-C4.xlsx':
        title_name = 'PICU Patient I'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.41
            auc_fixed = 0.5510582
            cohens_kappa_fixed = -0.117490819

            accuracy_individual = 0.299304952
            auc_individual = 0.688531036
            cohens_kappa_individual = -0.109625352

            #index_sws_threshold = np.array([0.0027841476566260125 for i in range(len(sleep_stage_labels))])
            #index_wake_threshold = np.array([0.0032092647675535622 for i in range(len(sleep_stage_labels))])
            index_sws_threshold = np.array([0.003140429 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.009362074 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.37
            auc_fixed = 0.544917124
            cohens_kappa_fixed = -0.140288391

            accuracy_individual = 0.298436142
            auc_individual = 0.69119155
            cohens_kappa_individual = -0.112892893

            index_sws_threshold = np.array([0.002974283 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.008873768 for i in range(len(sleep_stage_labels))])

    elif feature_data_file_name == 'PICU010_FeatureData.xlsx':
        title_name = 'PICU Patient J'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 16, 00, 00)

        if index_measure == 'GammaDeltaRatio':
            accuracy_fixed = 0.714787496
            auc_fixed = 0.898388452
            cohens_kappa_fixed = 0.563510759

            accuracy_individual = 0.811029153
            auc_individual = 0.919411099
            cohens_kappa_individual = 0.693682772

            index_sws_threshold = np.array([0.002272441 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.014232048 for i in range(len(sleep_stage_labels))])
        elif index_measure == 'GammaTheta+DeltaRatio':
            accuracy_fixed = 0.770635757
            auc_fixed = 0.909717248
            cohens_kappa_fixed = 0.640886517

            accuracy_individual = 0.810677907
            auc_individual = 0.92037158
            cohens_kappa_individual = 0.694176261

            index_sws_threshold = np.array([0.002379061 for i in range(len(sleep_stage_labels))])
            index_wake_threshold = np.array([0.013742176 for i in range(len(sleep_stage_labels))])


    #index_sws_threshold_fixed = index_sws_threshold
    #index_wake_threshold_fixed = index_wake_threshold

    # Plot settings
    plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.rc('axes', axisbelow=True)

    plt.close('all')
    fig, axs = plt.subplots(2, figsize=(10, 4))
    #plt.suptitle(title_name)
    # Add time stamp labels
    end_time = start_time + datetime.timedelta(seconds=30*len(sleep_stage_labels))
    t = []
    for i in range(len(sleep_stage_labels)):
        t.append((start_time + datetime.timedelta(seconds=30*i)).strftime('%H:%M'))

    ticks = [i for i in range(0, len(t), 120)]
    timestamp_labels = [t[i] for i in ticks]

    # Hypnogram
    axs[0].set_title('Hypnogram')
    axs[0].set_xticklabels([]) # remove number at x axis
    axs[0].set_xticks(ticks=ticks)
    axs[0].set_yticks(ticks=[0, 1, 2, 3, 4, 5])
    axs[0].set_yticklabels(labels=['N', 'N3', 'N2', 'N1', 'REM', 'Wake']) # remove number at x axis
    axs[0].set_ylabel('Sleep stage labels')
    axs[0].set_xlim([0, len(sleep_stage_labels)])
    axs[0].step(range(0, len(sleep_stage_labels)), sleep_stage_labels, where='post',
                color='olivedrab', linewidth=0.5)
    hypn_wake = np.array([4 for i in range(len(sleep_stage_labels))])
    hypn_wake[sleep_stage_labels > 4] = sleep_stage_labels[sleep_stage_labels > 4]
    axs[0].step(range(0, len(sleep_stage_labels)), hypn_wake, where='post',
                color='crimson', linewidth=0.5)
    axs[0].plot(np.array([4 for i in range(len(sleep_stage_labels))]),
                linewidth=0.7, color='white')
    axs[0].plot(np.array([4 for i in range(len(sleep_stage_labels))]), label='Wake threshold',
                linewidth=0.7, color='black', linestyle='--')
    hypn_sws = np.array([2 for i in range(len(sleep_stage_labels))])
    hypn_sws[sleep_stage_labels < 2] = sleep_stage_labels[sleep_stage_labels < 2]
    axs[0].step(range(0, len(sleep_stage_labels)), hypn_sws, where='post',
                color='navy', linewidth=0.5)
    axs[0].plot(np.array([2 for i in range(len(sleep_stage_labels))]),
                linewidth=0.7, color='white')
    axs[0].plot(np.array([2 for i in range(len(sleep_stage_labels))]),
                linewidth=0.7, color='black', linestyle='--')
    #axs[0].set_ylim([10e-4, 5])

    # Fixed thresholds
    axs[1].plot(index, color='olivedrab', linewidth=0.5, label='NSWS')
    axs[1].set_yscale('log')
    axs[1].set_xticklabels([]) # remove number at x axis
    axs[1].set_ylabel(ylabel)
    axs[1].set_title("Index with thresholds")
    #axs[1].set_title("")
    axs[1].set_xlim([0, len(sleep_stage_labels)])
    axs[1].set_xticks(ticks=ticks)

    index_wake = np.array([index_wake_threshold_fixed[0] for i in range(len(sleep_stage_labels))])
    index_wake[index > index_wake_threshold_fixed[0]] = index[index > index_wake_threshold_fixed[0]]
    axs[1].plot(index_wake, color='crimson', linewidth=0.5, label='Wake')
    axs[1].plot(index_wake_threshold_fixed,
                linewidth=0.7, color='white')
    axs[1].plot(index_wake_threshold_fixed,
                linewidth=0.7, color='black', linestyle='--')

    index_sws = np.array([index_sws_threshold_fixed[0] for i in range(len(sleep_stage_labels))])
    index_sws[index < index_sws_threshold_fixed[0]] = index[index < index_sws_threshold_fixed[0]]
    axs[1].plot(index_sws, color='navy', linewidth=0.5, label='SWS')
    axs[1].plot(index_sws_threshold_fixed,
                linewidth=0.7, color='white')
    axs[1].plot(index_sws_threshold_fixed, label='Index-score threshold',
                linewidth=0.7, color='black', linestyle='--')
    axs[1].set_ylim([10e-5, 7])
    if feature_data_file_name == 'PICU008_FeatureData.xlsx':
        text = 'Acc.=%.2f, $\kappa$=%.2f' % (accuracy_fixed, cohens_kappa_fixed)
    else:
        text = 'Acc.=%.2f, AUC=%.2f, $\kappa$=%.2f' % (accuracy_fixed, auc_fixed, cohens_kappa_fixed)
    axs[1].text(25, 2, text, fontsize=6, fontstyle='italic')

    axs[1].set_xlabel('Time (hours)')
    axs[1].set_xticks(ticks=ticks)
    axs[1].set_xticklabels(labels=timestamp_labels, rotation=90) # remove number at x axis

    # Add legend
    handles, labels = axs[1].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.21, hspace=0.35)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    # Save plot
    plot_file_name = 'Patient' + title_name.replace('PICU Patient ', '') + '_%s' % eeg_channel.replace(':', '') \
                     + '_Fixed_thresholds_' + index_measure + '.png'
    plt.savefig(os.path.join(plot_file_path, plot_file_name), dpi=1000)

#%% With fixed thresholds only

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

### File + index settings
picu_files = ['PICU001_FeatureData.xlsx', 'PICU002_FeatureData.xlsx', \
              'PICU003_FeatureData.xlsx', 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx',\
              'PICU005_FeatureData.xlsx', 'PICU006_FeatureData.xlsx',
              'PICU007_FeatureData - EEG F4-C4.xlsx', 'PICU008_FeatureData.xlsx',
              'PICU009_FeatureData - EEG F4-C4.xlsx',
              'PICU010_FeatureData.xlsx']
picu_files = ['PICU003_FeatureData.xlsx']
eeg_channel = 'EEG F3-C3:'

feature_data_file_path = r'E:\ICK slaap project\4_FeatureData\4_3_PICU_FeatureData'
plot_file_path = r'E:\ICK slaap project\7_IndexBasedClassification\7_4_Hypnogram + Index plots'
index_measure = 'GammaTheta+DeltaRatio'

for feature_data_file_name in picu_files:
    print(datetime.datetime.now(), ': %s started' % feature_data_file_name)
    ### Load data
    feature_data = pd.read_excel(os.path.join(feature_data_file_path, feature_data_file_name))
    order = ['Sleep stage N', 'Sleep stage N3', 'Sleep stage N2', 'Sleep stage N1', 'Sleep stage R', 'Sleep stage W']
    sleep_stage_labels = np.array([order.index(x) for x in feature_data["Sleep stage labels"]])
    title_name = 'x'

    if index_measure == 'GammaDeltaRatio':
        index = feature_data['%s Gamma/delta ratio' % eeg_channel]
        ylabel = 'Gamma/delta ratio'
    elif index_measure == 'GammaTheta+DeltaRatio':
        index = feature_data['%s Abs Gamma power' % eeg_channel] / (feature_data['%s Abs Theta power' % eeg_channel]+feature_data['% s Abs Delta power' % eeg_channel])
        ylabel = 'Gamma/(theta+delta) ratio'

    if feature_data_file_name == 'PICU001_FeatureData.xlsx':
        title_name = 'PICU Patient A'
        #start_time = datetime.datetime(2021, 10, 21, 12, 14, 35)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU002_FeatureData.xlsx':
        title_name = 'PICU Patient B'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU003_FeatureData.xlsx':
        title_name = 'PICU Patient C'
        #start_time = datetime.datetime(2021, 10, 21, 9, 17, 28)
        start_time = datetime.datetime(2021, 10, 21, 9, 00, 00)

    elif feature_data_file_name == 'PICU004_FeatureData_ArtLabelsAdjusted.xlsx':
        title_name = 'PICU Patient D'
        #start_time = datetime.datetime(2021, 10, 21, 9, 2, 27)
        start_time = datetime.datetime(2021, 10, 21, 9, 0, 0)

    elif feature_data_file_name == 'PICU005_FeatureData.xlsx':
        title_name = 'PICU Patient E'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 12, 0, 0)

    elif feature_data_file_name == 'PICU006_FeatureData.xlsx':
        title_name = 'PICU Patient F'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 11, 0, 0)

    elif feature_data_file_name == 'PICU007_FeatureData - EEG F4-C4.xlsx':
        title_name = 'PICU Patient G'
        #start_time = datetime.datetime(2021, 10, 21, 15, 59, 53)
        start_time = datetime.datetime(2021, 10, 21, 16, 0, 0)

    elif feature_data_file_name == 'PICU008_FeatureData.xlsx':
        title_name = 'PICU Patient H'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

     elif feature_data_file_name == 'PICU009_FeatureData - EEG F4-C4.xlsx':
        title_name = 'PICU Patient I'
        #start_time = datetime.datetime(2021, 10, 21, 11, 32, 42)
        start_time = datetime.datetime(2021, 10, 21, 12, 00, 00)

    elif feature_data_file_name == 'PICU010_FeatureData.xlsx':
        title_name = 'PICU Patient J'
        #start_time = datetime.datetime(2021, 10, 21, 11, 25, 28)
        start_time = datetime.datetime(2021, 10, 21, 16, 00, 00)

    # Plot settings
    plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    plt.rc('axes', axisbelow=True)

    plt.close('all')
    fig, axs = plt.subplots(2, figsize=(10, 4))
    #plt.suptitle(title_name)
    # Add time stamp labels
    end_time = start_time + datetime.timedelta(seconds=30*len(sleep_stage_labels))
    t = []
    for i in range(len(sleep_stage_labels)):
        t.append((start_time + datetime.timedelta(seconds=30*i)).strftime('%H:%M'))

    ticks = [i for i in range(0, len(t), 120)]
    timestamp_labels = [t[i] for i in ticks]

    # Hypnogram
    axs[0].set_title('Hypnogram')
    axs[0].set_xticklabels([]) # remove number at x axis
    axs[0].set_xticks(ticks=ticks)
    axs[0].set_yticks(ticks=[0, 1, 2, 3, 4, 5])
    axs[0].set_yticklabels(labels=['N', 'N3', 'N2', 'N1', 'REM', 'Wake']) # remove number at x axis
    axs[0].set_ylabel('Sleep stage labels')
    axs[0].set_xlim([0, len(sleep_stage_labels)])
    axs[0].step(range(0, len(sleep_stage_labels)), sleep_stage_labels, where='post',
                color='darkcyan', linewidth=0.5)

    # Fixed thresholds
    axs[1].plot(index, color='black', linewidth=0.5, label='NSWS')
    axs[1].set_yscale('log')
    axs[1].set_xticklabels([]) # remove number at x axis
    axs[1].set_ylabel(ylabel)
    axs[1].set_title("Index")
    #axs[1].set_title("")
    axs[1].set_xlim([0, len(sleep_stage_labels)])
    axs[1].set_xticks(ticks=ticks)
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_xticks(ticks=ticks)
    axs[1].set_xticklabels(labels=timestamp_labels, rotation=90) # remove number at x axis
    axs[1].set_ylim([10e-5, 7])

    fig.subplots_adjust(bottom=0.15, hspace=0.35)

    # Save plot
    plot_file_name = 'Patient' + title_name.replace('PICU Patient ', '') + '_%s' % eeg_channel.replace(':', '') \
                     + '_wIndex_' + index_measure + '.png'
    plt.savefig(os.path.join(plot_file_path, plot_file_name), dpi=1000)