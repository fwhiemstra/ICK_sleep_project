# Anomymize the EDF files
# for file in ...

import os
import pyedflib
#%% Select file
data_path = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\ICK_sleep_project\data\1_raw_data\EDF_test_files'
# file_name = 'EDFtest1.edf'
#file_name = 'EDFtest2.edf'
file_name = 'EDFtest3.edf'

edf_file = os.path.join(data_path, file_name)

#%% Anomymizer
pyedflib.highlevel.anonymize_edf(edf_file, new_file=None, to_remove=['patientname', 'patientcode'], new_values=['xxx', ''])

#def edf_anomyzer:
