def feature_evaluation(data, number_of_stages, file_key, dfANOVA, dfSpearman, dfANOVA_top, dfSpearman_top):
    spearman_corr = []
    anova_fval = []

    for feature in data.columns[11:910]:
        ### AUC calculation

        ### ANOVA F-statitic
        if number_of_stages == 2:
            number_of_stages_key = 'Two_state_labels'
            statistic, pvalue = scipy.stats.f_oneway(data[data[number_of_stages_key] == 'Wake'][feature],
                                                     data[data[number_of_stages_key] == 'Sleep'][feature])
        elif number_of_stages == 3:
            number_of_stages_key = 'Three_state_labels'
            statistic, pvalue = scipy.stats.f_oneway(data[data[number_of_stages_key] == 'Wake'][feature],
                                                     data[data[number_of_stages_key] == 'NSWS'][feature],
                                                     data[data[number_of_stages_key] == 'SWS'][feature])
        elif number_of_stages == 4:
            number_of_stages_key = 'Four_state_labels'
            statistic, pvalue = scipy.stats.f_oneway(data[data[number_of_stages_key] == 'Wake'][feature],
                                                     data[data[number_of_stages_key] == 'REM'][feature],
                                                     data[data[number_of_stages_key] == 'NSWS'][feature],
                                                     data[data[number_of_stages_key] == 'SWS'][feature])
        elif number_of_stages == 5:
            number_of_stages_key = 'Five_state_labels'
            statistic, pvalue = scipy.stats.f_oneway(data[data[number_of_stages_key] == 'Wake'][feature],
                                                     data[data[number_of_stages_key] == 'REM'][feature],
                                                     data[data[number_of_stages_key]== 'N1'][feature],
                                                     data[data[number_of_stages_key] == 'N2'][feature],
                                                     data[data[number_of_stages_key] == 'N3'][feature])
        anova_fval.append(statistic)

        ### Spearman correlation
        coef, p = scipy.stats.spearmanr(data[number_of_stages_key + '- linear'], data[feature])
        spearman_corr.append(coef)

    dfANOVA[file_key] = anova_fval
    dfSpearman[file_key] = spearman_corr

    ### Extract top 10 values
    idx_top_anova = sorted(range(len(anova_fval)), key=lambda i: anova_fval[i])[-10:]
    top_anova_fnames = dfANOVA['Features'][idx_top_anova]
    top_anova_fvalues = dfANOVA[file_key][idx_top_anova]
    dfANOVA_top[file_key + ' Top features'] = np.array(top_anova_fnames)
    dfANOVA_top[file_key + ' Top values'] = np.array(top_anova_fvalues)

    idx_top_corr = sorted(range(len(spearman_corr)), key=lambda i: spearman_corr[i])[-10:]
    top_corr_fnames = dfSpearman['Features'][idx_top_corr]
    top_corr_fvalues = dfSpearman[file_key][idx_top_corr]
    dfSpearman_top[file_key + ' Top features'] = np.array(top_corr_fnames)
    dfSpearman_top[file_key + ' Top values'] = np.array(top_corr_fvalues)

    return dfANOVA, dfSpearman, dfANOVA_top, dfSpearman_top

def AUC_ROC_calculation(featureData, save_plot, ROC_plot_file_path, ROC_plot_file_name, feature, file_name, sleep_stage_column_name):
    """
    This function calculates the AUC and ROC curve
    :param featureData:
    :param save_plot: if 1, ROC curve is saved
    :param ROC_plot_file_path:
    :param ROC_plot_file_name:
    :param feature:
    :param file_name:
    :param sleep_stage_column_name:
    :return:
    """

    plt.figure(1)
    plt.title('ROC curve for ' + feature + ' (' + file_name + ')')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid(which='both')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), '--')

    sleep_stage_labels = featureData[sleep_stage_column_name].drop_duplicates()
    AUCs = []
    pd.DataFrame(AUCs)

    for stage in np.array(sleep_stage_labels):
        label_true = []
        label_false = []
        thresholds = np.linspace(np.min(featureData[feature]), np.max(featureData[feature]), 10000)
        TPR = np.zeros(len(thresholds))
        FPR = np.zeros(len(thresholds))

        for i in range(0, len(featureData[sleep_stage_column_name])):
            if featureData.iloc[i][sleep_stage_column_name] == stage:
                label_true.append(featureData.iloc[i][feature])
            if featureData.iloc[i][sleep_stage_column_name] != stage:
                label_false.append(featureData.iloc[i][feature])

        for i in range(0, len(thresholds)):
            x = [element for element in label_true if element >= thresholds[i]]
            TP = len(x)
            x = [element for element in label_true if element < thresholds[i]]
            FN = len(x)
            x = [element for element in label_false if element >= thresholds[i]]
            FP = len(x)
            x = [element for element in label_false if element < thresholds[i]]
            TN = len(x)

            # to prevent zeroDivisionError
            if TP + FN == 0:
                TPR[i] = 0
            elif FP + TN == 0:
                FPR[i] = 0
            else:
                TPR[i] = TP / (TP + FN)
                FPR[i] = FP / (FP + TN)

        AUC = abs(np.trapz(TPR, FPR))
        print('AUC = ', AUC)
        if AUC < 0.5:
            AUC = 1 - AUC
            plt.plot(TPR, FPR, label=stage + ': AUC = {:.2f}'.format(AUC))
            plt.legend()
        else:
            plt.plot(FPR, TPR, label=stage + ': AUC = {:.2f}'.format(AUC))
            plt.legend()

        if save_plot == 1:
            plt.savefig(os.path.join(ROC_plot_file_path, ROC_plot_file_name))

        AUC = "{:.2f}".format(AUC) # 2 decimals
        AUCs.append({stage: AUC})

    return AUCs

def picu_edf_preprocessing(file_name, file_path, new_file_path):
    """ Preprocess an EDF file by:
        - Anonymization (i.e. removing all header information that is patient specific)
        - Removal of unnecessary signals
        - Addition of EEG leads
        (- Bandpass filtering of EEG signals) --> gebeurt in volgende stap

        Parameters
        ----------
        file_name: str
            Filename of the to be preprocessed EDF file
        file_path: str
            File path of the to be preprocessed EDF file
        new_file_path: str
            File path of the preprocessed EDF file

        Returns
        -------
        New anonymous and preprocessed EDF file is created in [new_file_path].
        The new EDF file has the same extension as the input file and is named: file_name + p
        (for example: PSG001 --> PSG001p).
        If succesful, the following statement is printed: 'Preprocessing of [new_file_name] completed')

        """

    # Read edf file
    edf_file = os.path.join(file_path, file_name)
    signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

    # Define new file name
    file, ext = os.path.splitext(file_name)
    new_file_name = file + 'p' + ext
    new_file = os.path.join(new_file_path, new_file_name)

    # Remove patient identifying information from file
    del header['technician'], header['patientname'], header['patientcode'], header['recording_additional'], header[
        'patient_additional'], header['equipment'], header['admincode'], header['gender'], header['birthdate']

    # Remove signals
    correction = 0  # add correction to prevent list index out of range after deleting from list
    for i in range(0, len(signal_headers)):
        i = i - correction
        if signal_headers[i]["label"] not in {'EOG E1', 'EOG E2', 'EMG EMG Chin', 'EEG F3', 'EEG F4', 'EEG C3',
                                              'EEG C4', 'EEG O1', 'EEG O2', 'EEG M1', 'EEG M2'}:
            del signal_headers[i], signals[i]
            correction = correction + 1

    # Add signals
    # Make list of labels to find index of signals
    signal_labels = []
    for i in range(0, len(signal_headers)):
        signal_labels.append(signal_headers[i]["label"])

    # EOG ROC-LOC signal
    new_signal = signals[signal_labels.index('EOG E1')] - signals[signal_labels.index('EOG E2')]
    new_signal_header = pyedflib.highlevel.make_signal_header('EOG ROC-LOC', dimension='uV',
                                                              sample_rate=signal_headers[0].get('sample_rate'),
                                                              physical_min=-5000, physical_max=5000,
                                                              digital_min=-65535,
                                                              digital_max=65535, transducer='',
                                                              prefiler='')
    signals.append(new_signal)
    signal_headers.append(new_signal_header)

    # EEG signals
    eegsignal_leads = [['EEG F3-F4', 'EEG F3', 'EEG F4'],
                       ['EEG F3-C3', 'EEG F3', 'EEG C3'],
                       ['EEG F3-C4', 'EEG F3', 'EEG C4'],
                       ['EEG F3-O1', 'EEG F3', 'EEG O1'],
                       ['EEG F3-O2', 'EEG F3', 'EEG O2'],
                       ['EEG F3-A1', 'EEG F3', 'EEG M1'],
                       ['EEG F3-A2', 'EEG F3', 'EEG M2'],
                       ['EEG C3-C4', 'EEG C3', 'EEG C4'],
                       ['EEG C3-O1', 'EEG C3', 'EEG O1'],
                       ['EEG C3-O2', 'EEG C3', 'EEG O2'],
                       ['EEG C3-A1', 'EEG C3', 'EEG M1'],
                       ['EEG C3-A2', 'EEG C3', 'EEG M2'],
                       ['EEG O1-O2', 'EEG O1', 'EEG O2'],
                       ['EEG O1-A1', 'EEG O1', 'EEG M1'],
                       ['EEG O1-A2', 'EEG O1', 'EEG M2']]

    # EEG header information
    sample_rate = signal_headers[3].get('sample_rate')
    physical_min = -5000
    physical_max = 5000
    digital_min = -65535
    digital_max = 65535
    prefilter = ''
    transducer = ''

    # EEG frequency filter
    # n = 16  # Filter order
    # fco_low = 0.5  # lower cutoff frequency
    # fco_up = 48  # upper cutoff frequency
    # fs = signal_headers[signal_labels.index("EEG F3")].get("sample_rate")
    # sos = signal.butter(n, [fco_low, fco_up], btype='bandpass', analog=False, output='sos', fs=fs)

    for i in range(0, len(eegsignal_leads)):
        new_signal = signals[signal_labels.index(eegsignal_leads[i][1])] - signals[
            signal_labels.index(eegsignal_leads[i][2])]
        new_signal_header = pyedflib.highlevel.make_signal_header(eegsignal_leads[i][0], dimension='uV',
                                                                  sample_rate=sample_rate,
                                                                  physical_min=physical_min, physical_max=physical_max,
                                                                  digital_min=digital_min,
                                                                  digital_max=digital_max, transducer=transducer,
                                                                  prefiler=prefilter)

        # Frequency filter EEG lead signals
        # new_signal = signal.sosfilt(sos, new_signal)

        # Add new signals + header to signals, signal_headers list
        signals.append(new_signal)
        signal_headers.append(new_signal_header)

    # Filter monopolar EEG signals
    # unipolar_eeg = ['EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG O1', 'EEG O2']
    # for i in unipolar_EEG:
    # signals[signal_labels.index(i)] = signal.sosfilt(sos, signals[signal_labels.index(i)])

    # Convert duration in seconds in annotations from byte to float type
    for i in range(0, len(header['annotations'])):
        header['annotations'][i][1] = float(header['annotations'][i][1])

    # Write new edf file
    pyedflib.highlevel.write_edf(new_file, signals, signal_headers, header, digital=True)

    print('Preprocessing of', new_file_name, 'completed')