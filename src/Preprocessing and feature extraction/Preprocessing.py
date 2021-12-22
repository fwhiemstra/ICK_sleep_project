"""
This file is used to convert the EDF files to a processed file using the edf_preprocessing function.
Floor Hiemstra, April 2021

ERRORS Solutions:
- If 'OSError: File is discontinuous and cannot be read' appears, convert file to edf C+ (EDFBrowser --> Tools).
"""

# Import functions and packages
from src.Functions import edf_preprocessing
import os

#raw_file_path = r'E:\ICK slaap project\2_Raw_data'
raw_file_path = r'W:\Analyses\Floor\Data\2_Raw_data'
preprocessed_file_path = r'E:\ICK slaap project\3_Preprocessed_data'
# Set action to one of the following options: 'ALL' or 'SINGLE' to convert all files in the
# directory or to convert a single file.
action = 'SINGLE'
file_name = 'PSG019.edf'  # Leave empty ([]) if all files in directory should be anonymized

if action == 'ALL':
    file_list = os.listdir(raw_file_path)
elif action == 'SINGLE':
    file_list = [file_name]

for file_name in file_list:
    try:
        if file_name.endswith(".edf") or file_name.endswith(".EDF"):
            edf_preprocessing(file_name, raw_file_path, preprocessed_file_path)
        else:
            print(file_name, ' is not an EDF file')
    except:
        print(file_name, ': Error in read EDF file (DD/ Disrupted file, incomplete electrode set)')

