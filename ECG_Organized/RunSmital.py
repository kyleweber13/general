import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyedflib
import Filtering
import pandas as pd
import pickle
import nwecg.ecg_quality as ecg_quality
import nwdata
import os
from tqdm import tqdm
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")


class Smital:

    def __init__(self, ecg_signal, sample_rate, start_time, timestamps, epoch_len=1, quiet=True):
        self.ecg = ecg_signal
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.timestamps = timestamps
        self.epoch_len = epoch_len
        self.quiet = quiet

        if not self.quiet:
            print("\nRunning Smital et al. algorithm...")
        self.x, self.s, self.snr, self.annots, self.thresholds = ecg_quality.annotate_ecg_quality(signal=self.ecg,
                                                                                                  sample_rate=self.sample_rate)

        self.cat = self.annots.to_array()
        self.cat[self.cat < 0] = 0
        self.cat = np.array([i+1 for i in self.cat])

        # self.df_epoch = self.epoch_snr(snr_sig=self.snr, sample_rate=self.sample_rate, epoch_len=self.epoch_len, start_time=self.start_time)

        self.bouts = self.format_quality_bouts(df=self.annots._annotations, timestamps=self.timestamps)

    def epoch_snr(self, snr_sig, sample_rate, epoch_len, start_time):
        epoch_samples = int(epoch_len * sample_rate)

        snr = [np.mean(snr_sig[i:i + epoch_samples]) for i in np.arange(0, len(snr_sig), epoch_samples)]

        stamps = pd.date_range(start=start_time, periods=len(snr), freq=f"{epoch_len}S")

        df_out = pd.DataFrame({"timestamp": stamps, 'idx': np.arange(0, len(snr_sig), epoch_samples), 'snr': snr})

        return df_out

    def plot_results(self, x_values, ds_ratio=1):

        fig, ax = plt.subplots(3, sharex='col', figsize=(10, 6))

        # raw data
        ax[0].plot(x_values[::ds_ratio], self.ecg[::ds_ratio], color='black', label='raw')

        # wiener filtered
        ax[0].plot(x_values[::ds_ratio], self.s[::ds_ratio], color='dodgerblue', label='wwf')
        ax[0].legend()

        # snr
        ax[1].plot(x_values[::ds_ratio], self.snr[::ds_ratio], color='black', label='snr')
        ax[1].axhline(self.thresholds[0], color='red')
        ax[1].axhline(self.thresholds[1], color='green')
        ax[1].legend()
        ax[1].grid()
        ax[1].set_ylabel("dB")

        # quality category
        ax[2].plot(x_values[::ds_ratio], self.cat[::ds_ratio], color='limegreen', label='snr')
        ax[2].legend()
        ax[2].set_ylabel("Category")
        ax[2].grid()

        plt.tight_layout()

        return fig

    def format_quality_bouts(self, df, timestamps):

        if not self.quiet:
            print("\nBouting results...")

        data_out = []

        for row in df.itertuples():
            q = str(row.quality).split(".")[1]
            data_out.append([q, timestamps[row.start_idx], timestamps[row.end_idx]])

        df_out = pd.DataFrame(data_out, columns=['quality', 'start_timestamp', 'end_timestamp'])

        return df_out


def process_snr(out_dir, edf_folder, ecg_fname, window_len=3600, overlap_secs=60, quiet=True):

    print(f"-Loading {ecg_fname} for analysis...")
    ecg = nwdata.NWData()
    ecg.import_edf(file_path=edf_folder + ecg_fname)
    ecg_chn_inx = ecg.get_signal_index('ECG')

    fs = ecg.signal_headers[ecg_chn_inx]['sample_rate']
    ts = pd.date_range(start=ecg.header['start_datetime'], periods=len(ecg.signals[ecg_chn_inx]),
                       freq="{}ms".format(1000 / fs))

    epoch_samples = int(fs * window_len)
    data_len = len(ecg.signals[ecg_chn_inx])

    snr_all = np.array([])

    for i in tqdm(np.arange(0, data_len, epoch_samples)):
        #if not quiet:
        #    print("{} || {}%".format(ecg_fname, round(i / data_len * 100, 2)))

        # first epoch
        if i + epoch_samples <= data_len and i == 0:
            data = ecg.signals[ecg_chn_inx][0:int(i + epoch_samples + 4 * fs)]
            crop_start = 0
            crop_end = -int(4 * fs)

        # subsequent epochs
        if i + epoch_samples <= (data_len + 2 * fs) and i > 0:
            data = ecg.signals[ecg_chn_inx][int(i - overlap_secs * fs):int(i + epoch_samples + 4 * fs)]
            crop_start = int(overlap_secs * fs)
            crop_end = -int(4 * fs)

        # final epoch
        if i + epoch_samples > (data_len + 2 * fs):
            data = ecg.signals[ecg_chn_inx][int(i - overlap_secs * fs):-1]
            crop_start = int(overlap_secs * fs)
            crop_end = -1

        s = Smital(ecg_signal=data, sample_rate=fs, start_time=ecg.header['start_datetime'],
                   timestamps=ts, epoch_len=1, quiet=True)

        snr_all = np.append(snr_all, s.snr[crop_start:crop_end])

        del s

    pickle_fname = ecg_fname.split(".")[0] + ".pickle"

    pickle_file = open(out_dir + pickle_fname, 'wb')
    pickle.dump(snr_all, pickle_file)
    pickle_file.close()
    print(f"-Saved SNR file as pickle ({out_dir + pickle_fname}).")

    return snr_all


def run_folder(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
               proc_snr_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/",
               proc_bout_folder="C:/Users/ksweber/Desktop/Smital Bouts V0.1.5/",
               save_dir="C:/Users/ksweber/Desktop/Smital/", thresholds=(5, 25), skip_subjs=()):

    # Files ----------------------------

    # edf files
    ecg_fnames = [i for i in os.listdir(edf_folder) if "BF36" in i]  # all edf files
    all_subjs = [i.split("_")[1] for i in ecg_fnames]  # all subject IDs

    # already processed smital SNR time series files
    proc_smital_files = os.listdir(proc_snr_folder)
    proc_smital_ids = [i.split("_")[1] for i in proc_smital_files]

    # already processed smital SNR bout files
    proc_smital_bouts = [i for i in os.listdir(proc_bout_folder) if 'BF36' in i]
    proc_bout_ids = [i.split("_")[1] for i in proc_smital_bouts]

    """====================================== RUNS ====================================="""
    failed = []

    for subj in [i for i in all_subjs if i not in skip_subjs]:
        print(f" ===================== {subj} =====================")

        if subj in proc_smital_ids and subj in proc_bout_ids:
            print("-Subject has been processed. Skipping.")

        try:
            snr = None

            # Runs smital SNR algorithm if not already processed ----------------
            if subj not in proc_smital_ids:

                print("-Running Smital algorithm...")

                snr = process_snr(edf_folder=edf_folder, out_dir=save_dir,
                                  ecg_fname=f"OND09_{subj}_01_BF36_Chest.edf",
                                  window_len=3600, overlap_secs=60, quiet=False)
                print("     -Complete.")

            # Runs smital bouting algorithm if not already processed -------------
            if subj not in proc_bout_ids:
                f = pyedflib.EdfReader(edf_folder + f"OND09_{subj}_01_BF36_Chest.edf")
                fs = f.getSampleFrequency(0)
                f.close()

                # Loads pickled SNR file if snr not in memory from process_snr() call -------
                if snr is None:
                    print("-Loading pickled SNR file...")
                    snr_file = proc_snr_folder + f"OND09_{subj}_01_BF36_Chest" + ".pickle"
                    f = open(snr_file, 'rb')
                    snr = pickle.load(f)
                    f.close()
                    print("     -Loaded.")

                # Replaces unknown values for first and last 2 seconds of SNR data
                snr[0:int(2 * fs)] = -1
                snr[-int(2 * fs):] = -1

                print("-Finding signal quality bouts...")
                annotations = ecg_quality._annotate_SNR(rolling_snr=snr, signal_len=len(snr), thresholds=thresholds,
                                                        sample_rate=fs, shortest_time=5)
                df = pd.DataFrame(annotations.get_annotations())
                df['quality'] = [i.value for i in df['quality']]
                df['thresh'] = [thresholds] * df.shape[0]

                file_out = "{}{}_SmitalBouts.xlsx".format(save_dir, f"OND09_{subj}_01_BF36_Chest")
                df.to_excel(file_out, index=False)
                print(f"-Bout file saved ({file_out})")

        except:
            print(f" == {subj}: FILE FAILED ===")
            failed.append(subj)

    return failed


if __name__ == 'main':
    f = run_folder(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                   proc_snr_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/",
                   proc_bout_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_bouts/Smital Bouts V0.1.5/Thresholds 5 20/",
                   save_dir="W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_bouts/Smital Bouts V0.1.5/Thresholds 5 20/",
                   thresholds=(5, 20), skip_subjs=('0003'))
