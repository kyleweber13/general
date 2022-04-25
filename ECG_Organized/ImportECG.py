import nwdata
import pandas as pd
import os
import pickle
from Filtering import filter_signal


def import_ecg_file(pathway):
    print(f"\nImporting {pathway}...")

    ecg = nwdata.NWData()
    ecg.import_edf(file_path=pathway, quiet=True)

    return ecg


def import_snr_pickle(pathway):

    print("\nImporting pickled SNR file...")

    f = open(pathway, 'rb')
    snr = pickle.load(f)
    f.close()

    return snr


class ECG:

    def __init__(self, edf_folder, ecg_fname, smital_filepath, bandpass=(.67, 40)):

        if os.path.exists(edf_folder + ecg_fname):
            self.ecg = import_ecg_file(edf_folder + ecg_fname)

            self.fs = self.ecg.signal_headers[self.ecg.get_signal_index("ECG")]['sample_rate']
            self.signal = self.ecg.signals[self.ecg.get_signal_index("ECG")]
            self.start_stamp = self.ecg.header['start_datetime']
            self.ts = pd.date_range(self.start_stamp, periods=len(self.signal), freq="{}ms".format(1000/self.fs))
            self.filt = filter_signal(data=self.signal, sample_f=self.fs, low_f=bandpass[0], high_f=bandpass[1],
                                      filter_order=5, filter_type='bandpass')

            self.temperature = self.ecg.signals[self.ecg.get_signal_index('Temperature')]
            self.temp_fs = self.ecg.signal_headers[self.ecg.get_signal_index('Temperature')]['sample_rate']

        if not os.path.exists(edf_folder + ecg_fname):
            print("File does not exist.")

            self.ecg = None
            self.fs = 1
            self.signal = []
            self.filt = []
            self.start_stamp = None
            self.ts = []
            self.temperature = None
            self.temp_fs = 1

        if os.path.exists(smital_filepath):
            self.snr = import_snr_pickle(smital_filepath)

        if not os.path.exists(smital_filepath):
            print("Smital file does not exist.")
            self.snr = [0] * len(self.signal)


if __name__ == 'main':
    data = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/", ecg_fname="OND09_0082_01_BF36_Chest.edf",
               smital_filepath="W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/OND09_0082_01_BF36_Chest.pickle")
