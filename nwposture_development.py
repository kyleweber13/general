import pandas as pd

from nwposture_dev.nwposture import NWPosture
import nwdata
from nwpipeline import NWPipeline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
import pandas as pd
from datetime import timedelta as td
from scipy import signal
from Filtering import filter_signal
import pywt
import peakutils

""" ================================================ DATA IMPORT ================================================== """
folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/001_Pilot/Converted/"
chest_file = folder + "001_chest_AX6_6014664_Accelerometer.edf"
la_file = folder + "001_left ankle_AX6_6014408_Accelerometer.edf"
log_file = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/001_Pilot/EventLog.xlsx"

df_event = pd.read_excel(log_file, usecols=["Event", "EventShort", "Duration", "Type", "Start", "Stop"])
df_event["Event"] = df_event["Event"].fillna("Transition")
df_event["Duration"] = [(j-i).total_seconds() for i, j in zip(df_event["Start"], df_event["Stop"])]

chest_acc = nwdata.NWData()
chest_acc.import_edf(file_path=chest_file, quiet=False)
chest_acc_indexes = {"Acc_x": chest_acc.get_signal_index("Accelerometer x"),
                     "Acc_y": chest_acc.get_signal_index("Accelerometer y"),
                     "Acc_z": chest_acc.get_signal_index("Accelerometer z")}
chest_axes = {"X": "Up", "Y": "Left", "Z": "Anterior"}
chest_ts = pd.date_range(start=chest_acc.header["startdate"], periods=len(chest_acc.signals[0]),
                         freq="{}ms".format(1000/chest_acc.signal_headers[chest_acc.get_signal_index("Accelerometer x")]['sample_rate']))

la_acc = nwdata.NWData()
la_acc.import_edf(file_path=la_file, quiet=False)
la_acc_indexes = {"Acc_x": la_acc.get_signal_index("Accelerometer x"),
                  "Acc_y": la_acc.get_signal_index("Accelerometer y"),
                  "Acc_z": la_acc.get_signal_index("Accelerometer z")}
la_axes = {"X": "Up", "Y": "Anterior", "Z": "Right"}

ankle_ts = pd.date_range(start=la_acc.header["startdate"], periods=len(la_acc.signals[0]),
                         freq="{}ms".format(1000/la_acc.signal_headers[la_acc.get_signal_index("Accelerometer x")]['sample_rate']))

""" ============================================= FACTOR IN GAIT ================================================== """

df_gait = df_event.loc[df_event["EventShort"] == "Walking"]

gait_mask = np.zeros(int(len(chest_ts)/chest_acc.signal_headers[chest_acc.get_signal_index("Accelerometer x")]['sample_rate']))
for row in df_gait.itertuples():
    start = int((row.Start - chest_ts[0]).total_seconds())
    stop = int((row.Stop - chest_ts[0]).total_seconds())
    gait_mask[int(start):int(stop)] = 1

""" ========================================== POSTURE DETECTION ================================================== """

post = NWPosture(ankle_file_path=la_file, chest_file_path=chest_file)

# Reformatting posture output -------------------------
code_dict = {0: "prone", 1: "supine", 2: "side", 3: "sitting", 4: 'sit/stand', 5: 'dynamic', 6: "Walking", 7: "Standing"}
df = post.processing()
df['GaitMask'] = gait_mask
del gait_mask

# Replaces "sitting" with "sit/stand" code
df['posture'] = df['posture'].replace(to_replace=3, value=4)

# flags gait bouts at "Standing"
posture2 = []
for row in df.itertuples():
    if row.GaitMask == 0:
        posture2.append(row.posture)
    if row.GaitMask == 1:
        posture2.append(7)
df['posture2'] = posture2
del posture2

# Replaces coded values with strings
df['posture'] = [code_dict[i] for i in df['posture']]
df['posture2'] = [code_dict[i] for i in df['posture2']]

df = df.loc[df.index<df_event.iloc[-1]["Stop"]]

# GOLD STANDARD REFERENCE DATA ---------------------
n_secs = int(np.ceil((chest_ts[-1] - chest_ts[0]).total_seconds()))
gs = ["Nothing" for i in range(int((df_event.iloc[0]["Start"]-chest_ts[0]).total_seconds()))]
for row in df_event.itertuples():
    start = int((row.Start - chest_ts[0]).total_seconds())
    end = int((row.Stop - chest_ts[0]).total_seconds())
    for i in range(end-start):
        gs.append(row.EventShort)


def plot_results(lowpass_f=1.0):
    fig, ax = plt.subplots(5, figsize=(14, 8), sharex='col', gridspec_kw={"height_ratios": [1, 1, 1, 1, .25]})

    ax[0].plot(chest_ts, filter_signal(data=chest_acc.signals[chest_acc_indexes["Acc_x"]], filter_order=7,
                                       sample_f=100, low_f=lowpass_f, filter_type='lowpass'),
               color='black', label=chest_axes["X"])

    ax[0].plot(chest_ts, filter_signal(data=chest_acc.signals[chest_acc_indexes["Acc_y"]], filter_order=7,
                                       sample_f=100, low_f=lowpass_f, filter_type='lowpass'),
               color='red', label=chest_axes["Y"])

    ax[0].plot(chest_ts, filter_signal(data=chest_acc.signals[chest_acc_indexes["Acc_z"]], filter_order=7,
                                       sample_f=100, low_f=lowpass_f, filter_type='lowpass'),
               color='dodgerblue', label=chest_axes["Z"])

    ax[0].set_title(f"Chest accelerometer ({lowpass_f}Hz lowpass)")
    ax[0].legend(loc='lower right')

    ax[1].plot(ankle_ts, la_acc.signals[la_acc_indexes["Acc_x"]], color='black')
    ax[1].plot(ankle_ts, la_acc.signals[la_acc_indexes["Acc_y"]], color='red')
    ax[1].plot(ankle_ts, la_acc.signals[la_acc_indexes["Acc_z"]], color='dodgerblue')
    ax[1].set_title("Ankle accelerometer")

    ax[2].plot(df.index, df['posture'], color='black', zorder=0)
    ax[2].plot(df.index, df['posture2'], color='red', zorder=1)
    ax[2].set_title("Algorithm")

    ylims = ax[2].get_ylim()
    for row in df_event.loc[df_event["Event"] == "Transition"].itertuples():
        ax[2].fill_between(x=[row.Start, row.Stop], y1=ylims[0], y2=ylims[1], color='grey', alpha=.5)

    ax[3].plot(pd.date_range(start=chest_ts[0], periods=len(gs), freq="1S"), gs, color='limegreen')
    ax[3].set_title("Reference")

    ax[4].plot(df.index, df['GaitMask'], color='purple')
    ax[4].set_title("Gait Mask")

    ax[-1].xaxis.set_major_formatter(xfmt)
    ax[-1].set_xlim(df_event.iloc[0]["Start"] + td(seconds=-120), df_event.iloc[-1]["Stop"] + td(seconds=120))

    plt.tight_layout()


# plot_results(lowpass_f=1.25)

vm = np.sqrt(np.square(np.array([chest_acc.signals[0], chest_acc.signals[1], chest_acc.signals[2]])).sum(axis=0))
vm = np.abs(vm)

# 5Hz lowpass
alpha_filt = filter_signal(data=vm[:200000], low_f=5, filter_order=4, sample_f=100, filter_type='lowpass')

# .25s rolling median
rm_alpha = [np.mean(alpha_filt[i:i+int(100/4)]) for i in range(len(alpha_filt))]
rm_alpha = np.abs([i-9.81 for i in rm_alpha])

c = pywt.cwt(data=alpha_filt, scales=[1, 64], wavelet='gaus1', sampling_period=1/100)
cwt_power = c[0][1]


def detect_peaks(wavelet_data, method='raw', sample_rate=100, min_seconds=5):

    if method == 'raw':
        d = wavelet_data
    if method == 'abs' or method == 'absolute':
        d = np.abs(wavelet_data)
    if method == 'le' or method == 'linear envelop' or method == 'linear envelope':
        d = filter_signal(data=np.abs(wavelet_data), low_f=.25, filter_type='lowpass', sample_f=sample_rate, filter_order=5)

    power_sd = np.std(d)
    peaks = peakutils.indexes(y=d, min_dist=int(min_seconds*sample_rate), thres_abs=True, thres=power_sd*2)

    return peaks, power_sd, d


def plot_s2s_results(signal, events_list=()):

    fig, ax = plt.subplots(4, sharex='col', figsize=(12, 8))

    ax[0].plot(df.index, df.posture2, color='black')
    ylims = ax[0].get_ylim()
    for row in df_event.loc[df_event["Event"] == "Transition"].itertuples():
        ax[0].fill_between(x=[row.Start, row.Stop], y1=ylims[0], y2=ylims[1], color='grey', alpha=.5)
    ax[0].set_title("Algorithm + Gait")

    x_filt = filter_signal(data=chest_acc.signals[0][:200000], high_f=.025, sample_f=100, filter_type='highpass', filter_order=3)
    y_filt = filter_signal(data=chest_acc.signals[1][:200000], high_f=.025, sample_f=100, filter_type='highpass', filter_order=3)
    z_filt = filter_signal(data=chest_acc.signals[2][:200000], high_f=.025, sample_f=100, filter_type='highpass', filter_order=3)

    ax[1].plot(chest_ts[:200000], x_filt, color='black')
    ax[1].plot(chest_ts[:200000], y_filt, color='red')
    ax[1].plot(chest_ts[:200000], z_filt, color='dodgerblue')
    ax[1].set_title(".1Hz Highpassed Chest Acceleration")

    ax[2].plot(chest_ts[:200000], signal, color='purple')
    ax[2].axhline(y=thresh, color='black', linestyle='dashed')
    ax[2].scatter(chest_ts[peaks], [signal[i]*1.2 for i in peaks], color='dodgerblue', marker='v',
                  label=f"{len(peaks)} peaks")
    ax[2].legend()
    ax[2].set_title("Wavelet data")
    ax[2].set_ylim(-.025, 1.5)

    ax[3].plot(pd.date_range(start=chest_ts[0], periods=len(gs), freq="1S"), gs, color='limegreen')
    ax[3].set_title("Reference data")

    ax[-1].xaxis.set_major_formatter(xfmt)

    for i in events_list:
        ax[2].axvline(x=pd.to_datetime(i), color='limegreen', linestyle='dashed')

    plt.tight_layout()


def create_peaks_df():

    ts = chest_ts[peaks]
    x_vals, y_vals, z_vals = [], [], []

    for peak in peaks:
        x = chest_acc.signals[0][peak-50:peak+50]
        x_peak = np.argmax(np.abs(x))
        x_vals.append(x[x_peak])

        y = chest_acc.signals[1][peak-50:peak+50]
        y_peak = np.argmax(np.abs(y))
        y_vals.append(y[y_peak])

        z = chest_acc.signals[2][peak-50:peak+50]
        z_peak = np.argmax(np.abs(z))
        z_vals.append(z[z_peak])

    df_out = pd.DataFrame({"Timestamp": ts, "xpeak": x_vals, "ypeak": y_vals, 'zpeak': z_vals})
    return df_out


peaks, thresh, processed_data = detect_peaks(wavelet_data=cwt_power, method='le')
df_peaks = create_peaks_df()
plot_s2s_results(processed_data,
                 ['2021-06-07 15:37:04', '2021-06-07 15:39:07', '2021-06-07 15:46:05', '2021-06-07 15:47:16',
                  '2021-06-07 15:49:00', '2021-06-07 15:51:17', '2021-06-07 15:58:05', '2021-06-07 15:59:04'])
