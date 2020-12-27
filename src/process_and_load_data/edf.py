import pyedflib
from src.settings import RAW_DATA
import os

def load_edf_file(file_name):
    """
    :param file_name:
    :return: signals, lists in a list of values
    """
    signals, signal_headers, header = pyedflib.highlevel.read_edf(file_name)
    #annotations = pyedflib.highlevel.read_edf_header(file_name, read_annotations=True)
    return signals, signal_headers, header

def annonomize_edf_data(signals_headers):
    annominized_signal_headers = signals_headers
    return annominized_signal_headers


if __name__ == '__main__':
    # file_name  = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\EDF files\ins6.edf'
    file_name = os.path.join(RAW_DATA, 'EDF_test_files', '1305609_11112020.EDF')

    # file_name = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\EDF files\1410419_26112020.EDF'
    # file_name = r'C:\Users\floor\Documents\Studie\Master Technical Medicine\TM jaar 3\Afstuderen\EDF files\1438646_26112020.EDF'
    # signal = pyedflib.EdfReader(file_name)
    signals, signal_headers, header = load_edf_file(file_name)
    annonomize_edf_data(signal_headers)
