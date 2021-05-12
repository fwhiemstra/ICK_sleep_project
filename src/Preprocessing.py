"""
This file is used to convert the EDF files to a processed file using the edf_preprocessing function.
Floor Hiemstra, April 2021
"""

# Import functions and packages
from Functions import edf_preprocessing
import os
import pyedflib
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#file_path = r'W:\Analyses\Floor\Data\PSG\2_Raw_data'
#new_file_path = r'W:\Analyses\Floor\Data\PSG\3_Preprocessed_data'
file_path = r'W:\Analyses\Floor\Data\PSG\2_Raw_data'
new_file_path = r'E:\ICK slaap project\PSG\3_Preprocessed_data'
file_name = 'PSG028.edf' # Leave empty ([]) if all files in directory should be anonymized

# Set action to one of the following options: 'ALL' or 'SINGLE' to convert all files in the
# directory or to convert a single file.
action = 'SINGLE'

if action == 'ALL':
    # Anonymize all files in directory
    for file in os.listdir(file_path):
         file_name = os.fsdecode(file)
         if file_name.endswith(".edf") or file_name.endswith(".EDF"):
             edf_preprocessing(file_name, file_path, new_file_path)
         else:
             print(file_name, ' is not an EDF file')
elif action == 'SINGLE':
    edf_preprocessing(file_name, file_path, new_file_path)
