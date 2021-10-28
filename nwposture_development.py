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
log_file = "C:/Users/ksweber/Desktop/PosturePilot001_EventLog.xlsx"

df_event = pd.read_excel(log_file, usecols=["Event", "EventShort", "SuperShort", "Duration", "Type", "Start", "Stop"])
df_event["Event"] = df_event["Event"].fillna("Transition")
df_event["Duration"] = [(j-i).total_seconds() for i, j in zip(df_event["Start"], df_event["Stop"])]
df_event = df_event.fillna(method='bfill')
start_time = df_event.iloc[0]['Start']
stop_time = df_event.iloc[-1]['Stop']

chest_acc = nwdata.NWData()
chest_acc.import_edf(file_path=chest_file, quiet=False)
chest_acc_indexes = {"Acc_x": chest_acc.get_signal_index("Accelerometer x"),
                     "Acc_y": chest_acc.get_signal_index("Accelerometer y"),
                     "Acc_z": chest_acc.get_signal_index("Accelerometer z")}
chest_axes = {"X": "Up", "Y": "Left", "Z": "Anterior"}
chest_fs = chest_acc.signal_headers[chest_acc_indexes["Acc_x"]]['sample_rate']

chest_start = int((start_time - chest_acc.header['startdate']).total_seconds() * chest_fs)
chest_stop = int((df_event.iloc[-1]['Stop'] - chest_acc.header['startdate']).total_seconds() * chest_fs)
chest_ts = pd.date_range(start=start_time, periods=len(chest_acc.signals[0][chest_start:chest_stop]),
                         freq="{}ms".format(1000/chest_fs))
chest_acc.signals[chest_acc_indexes['Acc_x']] = chest_acc.signals[chest_acc_indexes['Acc_x']][chest_start:chest_stop]
chest_acc.signals[chest_acc_indexes['Acc_y']] = chest_acc.signals[chest_acc_indexes['Acc_y']][chest_start:chest_stop]
chest_acc.signals[chest_acc_indexes['Acc_z']] = chest_acc.signals[chest_acc_indexes['Acc_z']][chest_start:chest_stop]

la_acc = nwdata.NWData()
la_acc.import_edf(file_path=la_file, quiet=False)
la_acc_indexes = {"Acc_x": la_acc.get_signal_index("Accelerometer x"),
                  "Acc_y": la_acc.get_signal_index("Accelerometer y"),
                  "Acc_z": la_acc.get_signal_index("Accelerometer z")}
la_axes = {"X": "Up", "Y": "Anterior", "Z": "Right"}
la_fs = la_acc.signal_headers[la_acc_indexes["Acc_x"]]['sample_rate']
la_start = int((start_time - la_acc.header['startdate']).total_seconds() * la_fs)
la_stop = int((df_event.iloc[-1]['Stop'] - chest_acc.header['startdate']).total_seconds() * chest_fs)
la_ts = pd.date_range(start=start_time, periods=len(la_acc.signals[0][la_start:la_stop]),
                      freq="{}ms".format(1000/la_fs))
la_acc.signals[la_acc_indexes['Acc_x']] = la_acc.signals[la_acc_indexes['Acc_x']][la_start:la_stop]
la_acc.signals[la_acc_indexes['Acc_y']] = la_acc.signals[la_acc_indexes['Acc_y']][la_start:la_stop]
la_acc.signals[la_acc_indexes['Acc_z']] = la_acc.signals[la_acc_indexes['Acc_z']][la_start:la_stop]


""" ============================================= FACTOR IN GAIT ================================================== """

"""This is in place of the nwgait output --> will need reformatting"""
df_gait = df_event.loc[df_event["EventShort"] == "Walking"]

# Binary list of gait (1) or no gait (0) in 1-sec increments
gait_mask = np.zeros(int(len(chest_ts)/chest_fs))
for row in df_gait.itertuples():
    start = int((row.Start - start_time).total_seconds())
    stop = int((row.Stop - start_time).total_seconds())
    gait_mask[int(start):int(stop)] = 1

""" ========================================== POSTURE DETECTION ================================================== """

post = NWPosture(ankle_file_path=la_file, chest_file_path=chest_file)


def format_nwposture_output():

    def find_posture_changes(df):

        def get_index(data, start_index=0):

            data = list(data)
            current_value = data[start_index]

            for i in range(start_index, len(data) - 1):
                if data[i] != current_value:
                    return i
                if data[i] == current_value:
                    pass

        indexes = [0]

        for i in range(df.shape[0]):
            if i > indexes[-1]:
                index = get_index(df["posture2"], start_index=i)
                indexes.append(index)
            try:
                if i <= indexes[-1]:
                    pass
            except TypeError:
                if df.shape[0] not in indexes:
                    indexes.append(df.shape[0] - 1)

        indexes = [i for i in indexes if i is not None]

        transitions = np.zeros(df.shape[0])
        transitions[indexes] = 1

        df["Transition"] = transitions

        return indexes

    # Reformatting posture output -------------------------
    code_dict = {0: "Prone", 1: "Supine", 2: "Side", 3: "Sitting", 4: 'Sit/Stand', 5: 'Dynamic', 6: "Walking", 7: "Standing"}
    df = post.processing()
    df = df.reset_index()
    df.columns = ['Timestamp', "posture", "XY", "Y", "dyno", "vm", "dyno_vm", "ankle"]
    df = df.loc[(df['Timestamp'] >= start_time) & (df['Timestamp'] < stop_time)]
    df['GaitMask'] = gait_mask

    df['Timestamp'] = [i.round(freq='1S') for i in df['Timestamp']]

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

    # Replaces coded values with strings
    df['posture'] = [code_dict[i] for i in df['posture']]
    df['posture2'] = [code_dict[i] for i in df['posture2']]

    df['posture2'] = df['posture2'].replace(to_replace='Dynamic', method='bfill')

    df = df.loc[(df['Timestamp'] >= df_event.iloc[0]["Start"]) & (df['Timestamp'] < df_event.iloc[-1]["Stop"])]
    df = df.reset_index(drop=True)

    transitions_indexes = find_posture_changes(df)

    return df, transitions_indexes


df, transitions_indexes = format_nwposture_output()


# GOLD STANDARD REFERENCE DATA ---------------------
n_secs = int(np.ceil((chest_ts[-1] - start_time).total_seconds()))
# gs = ["Nothing" for i in range(int((df_event.iloc[0]["Start"]-chest_ts[0]).total_seconds()))]
gs = []
for row in df_event.itertuples():
    start = int((row.Start - start_time).total_seconds())
    end = int((row.Stop - start_time).total_seconds())
    for i in range(end-start):
        gs.append(row.SuperShort)


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

    ax[1].plot(la_ts, la_acc.signals[la_acc_indexes["Acc_x"]], color='black')
    ax[1].plot(la_ts, la_acc.signals[la_acc_indexes["Acc_y"]], color='red')
    ax[1].plot(la_ts, la_acc.signals[la_acc_indexes["Acc_z"]], color='dodgerblue')
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

""" --------------------------------- Data processing for sit-to-stand transitions ---------------------------------"""


def process_for_peak_detection():
    # Acceleration magnitudes
    vm = np.sqrt(np.square(np.array([chest_acc.signals[chest_acc_indexes["Acc_x"]],
                                     chest_acc.signals[chest_acc_indexes["Acc_y"]],
                                     chest_acc.signals[chest_acc_indexes["Acc_z"]]])).sum(axis=0))
    vm = np.abs(vm)

    # 5Hz lowpass
    alpha_filt = filter_signal(data=vm, low_f=3, filter_order=4, sample_f=chest_fs, filter_type='lowpass')

    # .25s rolling median
    rm_alpha = [np.mean(alpha_filt[i:i+int(chest_fs/4)]) for i in range(len(alpha_filt))]

    # Continuous wavelet transform; focus on <.5Hz band
    c = pywt.cwt(data=alpha_filt, scales=[1, 64], wavelet='gaus1', sampling_period=1/chest_fs)
    cwt_power = c[0][1]

    return vm, alpha_filt, rm_alpha, cwt_power


def detect_peaks(wavelet_data, method='raw', sample_rate=100, min_seconds=5):

    if method == 'raw':
        d = wavelet_data
    if method == 'abs' or method == 'absolute':
        d = np.abs(wavelet_data)
    if method == 'le' or method == 'linear envelop' or method == 'linear envelope':
        d = filter_signal(data=np.abs(wavelet_data), low_f=.25, filter_type='lowpass',
                          sample_f=sample_rate, filter_order=5)

    power_sd = np.std(d)
    peaks = peakutils.indexes(y=d, min_dist=int(min_seconds*sample_rate), thres_abs=True, thres=power_sd*1.5)

    return peaks, power_sd, d


def plot_s2s_results(signal, highpass_acc_data=False, events_list=()):

    fig, ax = plt.subplots(4, sharex='col', figsize=(12, 8))

    ax[0].plot(df.Timestamp, df.posture2, color='black', zorder=1)
    ylims = ax[0].get_ylim()
    for row in df_event.loc[df_event["Event"] == "Transition"].itertuples():
        ax[0].fill_between(x=[row.Start, row.Stop], y1=ylims[0], y2=ylims[1], color='grey', alpha=.5)
    for row in df.itertuples():
        if row.Transition:
            ax[0].axvline(x=row.Timestamp, color='orange', zorder=0)
    ax[0].set_title("Algorithm + Gait")

    if highpass_acc_data:
        x_filt = filter_signal(data=chest_acc.signals[0][:200000], high_f=.025, sample_f=chest_fs, filter_type='highpass', filter_order=3)
        y_filt = filter_signal(data=chest_acc.signals[1][:200000], high_f=.025, sample_f=chest_fs, filter_type='highpass', filter_order=3)
        z_filt = filter_signal(data=chest_acc.signals[2][:200000], high_f=.025, sample_f=chest_fs, filter_type='highpass', filter_order=3)

        ax[1].plot(chest_ts[:200000], x_filt, color='black')
        ax[1].plot(chest_ts[:200000], y_filt, color='red')
        ax[1].plot(chest_ts[:200000], z_filt, color='dodgerblue')

    if not highpass_acc_data:
        ax[1].plot(chest_ts[:200000], chest_acc.signals[chest_acc_indexes["Acc_x"]][:200000], color='black')
        ax[1].plot(chest_ts[:200000], chest_acc.signals[chest_acc_indexes["Acc_y"]][:200000], color='red')
        ax[1].plot(chest_ts[:200000], chest_acc.signals[chest_acc_indexes["Acc_z"]][:200000], color='dodgerblue')

    ax1_title = ".25Hz Highpassed Chest Acceleration" if highpass_acc_data else "Raw Chest Acceleration"
    ax[1].set_title(ax1_title)
    ax[1].legend(labels=[chest_axes["X"], chest_axes["Y"], chest_axes["Z"]])

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


def create_peaks_df(start_time):

    ts = chest_ts[peaks]
    x_vals, y_vals, z_vals = [], [], []

    for peak in peaks:
        x = filter_signal(data=chest_acc.signals[0][peak-int(chest_fs):peak+int(chest_fs)], sample_f=chest_fs,
                          high_f=.1, filter_type='highpass', filter_order=5)
        x_peak = np.argmax(np.abs(x))
        x_vals.append(x[x_peak])

        y = filter_signal(data=chest_acc.signals[1][peak-int(chest_fs):peak+int(chest_fs)], sample_f=chest_fs,
                          high_f=.1, filter_type='highpass', filter_order=5)
        y_peak = np.argmax(np.abs(y))
        y_vals.append(y[y_peak])

        z = filter_signal(data=chest_acc.signals[2][peak-int(chest_fs):peak+int(chest_fs)], sample_f=chest_fs,
                          high_f=.1, filter_type='highpass', filter_order=5)
        z_peak = np.argmax(np.abs(z))
        z_vals.append(z[z_peak])

    df_out = pd.DataFrame({"Timestamp": ts, "xpeak": x_vals, "ypeak": y_vals, 'zpeak': z_vals})
    df_out['Timestamp'] = [i.round('1S') for i in df_out['Timestamp']]
    df_out['Ind'] = [int((i - start_time).total_seconds()) for i in df_out.Timestamp]

    return df_out.reset_index(drop=True)


vm, alpha_filt, rm_alpha, cwt_power = process_for_peak_detection()
peaks, thresh, processed_data = detect_peaks(wavelet_data=cwt_power, method='le', sample_rate=chest_fs)
df_peaks = create_peaks_df(start_time=df.iloc[0]['Timestamp'])
# df_peaks['Ind'] = [int((i - df.iloc[0]['Timestamp']).total_seconds()) for i in df_peaks.Timestamp]
# df_peaks = df_peaks.loc[df_peaks['Ind'] >= 0]

plot_s2s_results(processed_data, highpass_acc_data=True,
                 events_list=['2021-06-07 15:37:04', '2021-06-07 15:39:07',
                              '2021-06-07 15:46:05', '2021-06-07 15:47:16',
                              '2021-06-07 15:49:00', '2021-06-07 15:51:17',
                              '2021-06-07 15:58:05', '2021-06-07 15:59:04'])

first_standing = df.loc[df['posture2'] == 'Standing'].index[0]

# List of the posture for each 'bout'
use_indexes = [i for i in transitions_indexes if i >= first_standing]  # peaks after first standing/gait period
unknown_indexes = [i for i in transitions_indexes if i < first_standing]
unknown_peaks = df_peaks.loc[(df_peaks["Timestamp"] >= df['Timestamp'].iloc[0]) &
                             (df_peaks["Timestamp"] < df['Timestamp'].iloc[first_standing])].reset_index(drop=True)


def v1():
    # Dictionary of event that starts at each transition index
    bout_dict = {}
    # Any transitions before first know standing/gait bout
    for i in [i for i in transitions_indexes if i < first_standing]:
        bout_dict[i] = "Unknown"
    # All other transitions
    for i in use_indexes:
        bout_dict[i] = df.iloc[i]['posture2']

    """ ------------------------------------ Figuring out unknowns at start of data ----------------------------------- """

    """if unknown_peaks.shape[0] % 2 == 1:
        bout_dict[0] = 'Sitting'
    
        for i, peak in enumerate(unknown_peaks["Timestamp"]):
            index = int((peak - df.iloc[0]['Timestamp']).total_seconds())
            bout_dict[index] = 'Standing' if i % 2 == 0 else "Sitting"
    
    bout_dict = dict(sorted(bout_dict.items()))
    """

    """ -------------------------- Algorithm once unknowns at beginning have been figured out ------------------------- """
    for start, stop in zip(use_indexes[:], use_indexes[1:]):
        start_ind = start-5 if start-5 >= 0 else start
        stop_ind = stop+5 if stop+5 <= len(df) else len(df)-1
        df_posture = df.iloc[start_ind:stop_ind]

        curr_event = df_posture.iloc[5]['posture2']

        try:
            next_event = df.iloc[stop+1]['posture2']
        except IndexError:
            next_event = None

        p = df_peaks.loc[(df_peaks["Timestamp"] >= df_posture['Timestamp'].iloc[0]) &
                         (df_peaks["Timestamp"] < df_posture['Timestamp'].iloc[-1])]
        n_events = p.shape[0]

        suffix = 's' if n_events != 1 else ""
        print(f"{df_posture.iloc[0]['Timestamp']}, {curr_event} -> {next_event}, {n_events} transition{suffix}")

        if n_events == 0:
            pass

        # Temporary: if only only transition found
        if n_events == 1:
            # If currently standing and next bout is sit/stand
            if curr_event == "Standing" and next_event == 'Sit/Stand':
                # If STS peak found < 5 seconds after end of gait bout
                if np.abs((p['Timestamp'].iloc[0] - df.iloc[stop]['Timestamp']).total_seconds()) < 5:
                    bout_dict[stop] = "Sitting"

    inds = [i for i in bout_dict.keys()]
    values = [i for i in bout_dict.values()]
    postures = []
    for a, b in zip(inds[:], inds[1:]):
        for i in range(b-a):
            postures.append(bout_dict[a])

    gs2 = gs[len(gs) - df.shape[0]:]
    fig, ax = plt.subplots(2, sharex='col')
    ax[0].plot(np.arange(len(gs2)), [i if i != "Walking" else "Standing" for i in gs2], color='green', label="GS")
    ax[0].plot(np.arange(df.shape[0]), df['posture2'], color='red', label="AlgV1")
    ax[0].legend()
    ax[1].plot(np.arange(len(postures)), postures, color='black', label='AlgV2.')
    ax[1].legend()
    for i in df_peaks.Timestamp:
        ax[1].axvline((i - df.iloc[0]['Timestamp']).total_seconds(), color='orange')
        ax[0].axvline((i - df.iloc[0]['Timestamp']).total_seconds(), color='orange')


df_index = pd.DataFrame({"Timestamp": [df.iloc[i]['Timestamp'] for i in use_indexes],
                         "Ind": use_indexes, "Posture": [df.iloc[i]['posture2'] for i in use_indexes],
                         'Type': ["Transition" for i in range(len(use_indexes))]})
df_index = df_index.append(pd.DataFrame({"Timestamp": [df.iloc[i]['Timestamp'] for i in df_peaks['Ind']],
                                         "Ind": df_peaks['Ind'],
                                         "Posture": [df.iloc[i]['posture2'] for i in df_peaks['Ind']],
                                         "Type": ["Peak" for i in range(df_peaks.shape[0])]}))
df_index = df_index.sort_values("Timestamp")
df_index = df_index.reset_index(drop=True)


def compare_algorithms():
    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8), gridspec_kw={"height_ratios": [.5, .5, 1]})
    ax[0].plot(df['posture2'], color='red', label='V1')
    ax[0].plot(np.arange(len(postures)), postures, color='dodgerblue', label='V2')
    ax[0].set_title("Original vs. New Algorithm")
    ax[0].legend()

    ax[1].set_title("Gait Mask")
    ax[1].plot(df["GaitMask"], color='purple')
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(labels=['No', 'Gait'])

    ax[2].plot(np.arange(len(postures)), postures, color='dodgerblue', label='V2')
    ax[2].plot(gs[len(gs) - df.shape[0]:], color='limegreen', label='GS')
    ax[2].legend()
    ax[2].set_title("New Algorithm vs. Gold Standard")

    for i in df_peaks.loc[df_peaks["Ind"] > 0]['Ind']:
        ax[1].axvline(x=i, color='orange')
        ax[2].axvline(x=i, color='orange')

    ax[2].fill_between(x=[0, 120], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[365, 427], y1=0, y2=8, alpha=.25, color='grey')
    ax[2].fill_between(x=[547, 602], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[606, 679], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[681, 780], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[1081, 1146], y1=0, y2=8, alpha=.25, color='grey')
    ax[2].fill_between(x=[1203, 1262], y1=0, y2=8, alpha=.25, color='grey')

    plt.tight_layout()


def v2():
    # Dictionary of event that starts at each transition index
    bout_dict = {0: 'unknown'}

    # All other transitions
    for i in use_indexes:
        bout_dict[i] = df.iloc[i]['posture2']

    for row in df_peaks.loc[df_peaks["Ind"] > first_standing].itertuples():
        # Windows +/- 3 seconds from potential STS peak
        window = list(df.iloc[row.Ind-3:row.Ind+3]["posture2"])
        print(f"{row.Timestamp}/{row.Ind}: {window}")

        # If going from Standing to Sit/Stand --> Standing to Sitting
        if "Standing" in window[:3] and "Sit/Stand" in window[-3:]:
            bout_dict[row.Ind] = "Sitting"

    #for peak in df_peaks["Ind"]:
    #    bout_dict[peak] = "PotentialSTS"

    bout_dict = dict(sorted(bout_dict.items()))

    # if sit/stand surrounded by any time of lying w/ no transitions --> sitting

    inds = [i for i in bout_dict.keys()]
    postures = []
    for a, b in zip(inds[:], inds[1:]):
        for i in range(b - a):
            postures.append(bout_dict[a])
        print(f"{a}-{b} = {bout_dict[a]}")

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8), gridspec_kw={"height_ratios": [.5, .5, 1]})
    ax[0].plot(df['posture2'], color='red', label='V1')
    ax[0].plot(np.arange(len(postures)), postures, color='dodgerblue', label='V2')
    ax[0].set_title("Original vs. New Algorithm")
    ax[0].legend()

    ax[1].set_title("Gait Mask")
    ax[1].plot(df["GaitMask"], color='purple')
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(labels=['No', 'Gait'])

    ax[2].plot(np.arange(len(postures)), postures, color='dodgerblue', label='V2')
    ax[2].plot(gs[len(gs) - df.shape[0]:], color='limegreen', label='GS')
    ax[2].legend()
    ax[2].set_title("New Algorithm vs. Gold Standard")

    for i in df_peaks.loc[df_peaks["Ind"] > 0]['Ind']:
        ax[1].axvline(x=i, color='orange')
        ax[2].axvline(x=i, color='orange')

    ax[2].fill_between(x=[0, 120], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[365, 427], y1=0, y2=8, alpha=.25, color='grey')
    ax[2].fill_between(x=[547, 602], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[606, 679], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[681, 780], y1=0, y2=8, alpha=.25, color='red')
    ax[2].fill_between(x=[1081, 1146], y1=0, y2=8, alpha=.25, color='grey')
    ax[2].fill_between(x=[1203, 1262], y1=0, y2=8, alpha=.25, color='grey')

    ax[2].set_xlabel("Seconds")

    plt.tight_layout()


df_use = df_index.loc[(df_index['Ind'] >= first_standing) & (df_index['Type'] == "Peak")].reset_index(drop=True)
win_size = 6
posture_list = np.array(df_use['Posture'])
posture_output = np.array(df['posture2'])

for row in range(df_use.shape[0]-1):
    curr_row = df_use.iloc[row]

    # Windows +/- 3 seconds from potential STS peak
    # df_window = df.iloc[curr_row['Ind'] - win_size:curr_row['Ind'] + win_size]
    df_window = posture_output[curr_row['Ind'] - win_size:curr_row['Ind'] + win_size]
    # window = list(df_window["posture2"])
    window = posture_output[curr_row['Ind']-win_size:curr_row['Ind'] + win_size]
    # print(f"{curr_row['Timestamp']}/{curr_row.Ind}: {window}")

    # FOLLOW LOGIC SPLITS DATA WINDOWS IN HALF AROUND THE POTENTIAL TRANSITION PEAK -----------

    # Standing + transition + sit/stand --> sitting
    if "Standing" in window[:win_size] and "Sit/Stand" in window[-win_size:]:
        posture_list[row] = "Sitting"
        posture_output[curr_row.Ind:df_use.iloc[row+1]['Ind']] = "Sitting"
        print("1. ", curr_row.Ind, df_use.iloc[row+1]['Ind'])

    # Sit/Stand + transition + "Standing" --> sitting
    if "Sit/Stand" in window[-win_size:] and "Standing" in window[:win_size]:
        posture_list[row] = "Sitting"
        posture_output[curr_row.Ind:df_use.iloc[row+1]['Ind']] = "Sitting"
        print("2. ", curr_row.Ind, df_use.iloc[row+1]['Ind'])

df_use["Posture"] = posture_list

fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': [1, .25]})
ax[0].plot(gs, color='limegreen', label="Ref.")
ax[1].plot(gait_mask, color='purple')
ax[0].plot(df['posture2'], color='red', label="V1")
for row in df_use.itertuples():
    if row.Index == 0:
        ax[0].axvline(x=row.Ind, color='orange', label='STS?', linestyle='dashed')
    if row.Index > 0:
        ax[0].axvline(x=row.Ind, color='orange', linestyle='dashed')

    ax[1].axvline(x=row.Ind, color='orange', linestyle='dashed')

ax[0].fill_between(x=[363, 424], y1=0, y2=8, color='grey', alpha=.2)
df_index = df_index.loc[df_index['Type'] == "Transition"].append(df_use).sort_values("Timestamp").reset_index(drop=True)

for i in range(df_index.shape[0]-1):
    try:
        if abs(df_index.iloc[i]['Ind'] - df_index.iloc[i+1]["Ind"]) <= 1:
            df_index = df_index.drop(index=i).reset_index(drop=True)

        if i >= 1:
            ax[0].plot([df_index.iloc[i]['Ind'], df_index.iloc[i+1]['Ind']],
                     [df_index.iloc[i]['Posture'], df_index.iloc[i]['Posture']],
                     color='dodgerblue')
        if i < 1:
            ax[0].plot([df_index.iloc[i]['Ind'], df_index.iloc[i + 1]['Ind']],
                       [df_index.iloc[i]['Posture'], df_index.iloc[i]['Posture']],
                       color='dodgerblue', label='V2')

    except IndexError:
        pass

ax[0].legend()
