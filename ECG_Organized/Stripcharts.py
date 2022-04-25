import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta as td
from ECG_Organized.ImportECG import *
from ECG_Organized.ImportTabular import *
from ECG_Organized.Filtering import filter_signal

""" =============================== IMPORT ==============================="""

subj = 'OND09_0060'
focus_arrs = ["Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block']

data = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
           ecg_fname=f"{subj}_01_BF36_Chest.edf", bandpass=(.67, 80),
           smital_filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/{subj}_01_BF36_Chest.pickle")
sig_f = filter_signal(data=data.signal, sample_f=data.fs, high_f=.1, filter_type='highpass')

# Cardiac Navigator data
df_all, df_sinus, df_cn = import_cn_file(pathway=f"C:/Users/ksweber/Desktop/CardiacNavigator/edf_cropped/CustomSettings/{subj}_Events.csv",
                                         sample_rate=data.fs, start_time=data.start_stamp, ecg_signal=data.signal,
                                         use_arrs=focus_arrs, timestamp_method='start')

df_beats = import_cn_beats_file(pathway=f'C:/Users/ksweber/Desktop/CardiacNavigator/edf_cropped/CustomSettings/{subj}_Beats.csv',
                                start_stamp=data.start_stamp, sample_rate=data.fs)

""" =============================== ACTUAL CODE ==============================="""

# plot_n_secs = 10

fig, ax = plt.subplots(5, figsize=(12, 8), sharex='col')
max_dur = df_cn.iloc[:5]['duration'].max()

for row in df_cn.iloc[:5].itertuples():

    e = [i / 1000 for i in data.signal[row.start_idx:row.end_idx]]
    bias = np.mean(e)
    e_zeroed = [i - bias for i in e]

    ax[row.Index].plot(np.arange(len(e_zeroed))/data.fs, e_zeroed, color='black')

    ax[row.Index].set_xticks(np.arange(0, max_dur, .2))
    ax[row.Index].set_xticks(np.arange(0, max_dur, .04), minor=True)

plt.tight_layout()
