import os
import pyedflib

file_name = 'zzzPSG012.edf'
file_path = r'E:\ICK slaap project\PSG\2_Raw_data'

edf_file = os.path.join(file_path, file_name)
signals, signal_headers, header = pyedflib.highlevel.read_edf(edf_file, digital=True)

