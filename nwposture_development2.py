import pandas as pd
# from nwposture_dev.nwposture import NWPosture
from nwposture_dev.nwposture.NWPosture2 import NWPosture
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

"""
folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/001_Pilot/Converted/"
chest_file = folder + "001_chest_AX6_6014664_Accelerometer.edf"
la_file = folder + "001_left ankle_AX6_6014408_Accelerometer.edf"
log_file = "C:/Users/ksweber/Desktop/PosturePilot001_EventLog.xlsx"
df_nwposture2 = pd.read_excel("C:/Users/ksweber/Desktop/Pilot001_NWPosture2.xlsx")
"""

folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/002/Converted/"
chest_file = folder + "002_Axivity_Chest_Accelerometer.edf"
la_file = folder + "002_Axivity_LAnkle_Accelerometer.edf"
log_file = "C:/Users/ksweber/Desktop/PosturePilot002_EventLog.xlsx"
df_nwposture2 = pd.read_excel("C:/Users/ksweber/Desktop/Pilot002_NWPosture2.xlsx")



def import_goldstandard_df(filename):

    df_event = pd.read_excel(filename, usecols=["Event", "EventShort", "SuperShort",
                                                "Duration", "Type", "Start", "Stop"])
    df_event["Event"] = df_event["Event"].fillna("Transition")
    df_event["Duration"] = [(j-i).total_seconds() for i, j in zip(df_event["Start"], df_event["Stop"])]
    df_event = df_event.fillna(method='bfill')
    start_time = df_event.iloc[0]['Start']
    stop_time = df_event.iloc[-1]['Stop']

    return df_event, start_time, stop_time


df_event, start_time_gs, stop_time_gs = import_goldstandard_df(filename=log_file)

chest_acc = nwdata.NWData()
chest_acc.import_edf(file_path=chest_file, quiet=False)
chest_acc_indexes = {"Acc_x": chest_acc.get_signal_index("Accelerometer x"),
                     "Acc_y": chest_acc.get_signal_index("Accelerometer y"),
                     "Acc_z": chest_acc.get_signal_index("Accelerometer z")}
chest_axes = {"X": "Up", "Y": "Left", "Z": "Anterior"}
chest_fs = chest_acc.signal_headers[chest_acc_indexes["Acc_x"]]['sample_rate']

chest_start = int((start_time_gs - chest_acc.header['startdate']).total_seconds() * chest_fs)
chest_stop = int((df_event.iloc[-1]['Stop'] - chest_acc.header['startdate']).total_seconds() * chest_fs)
# chest_ts = pd.date_range(start=start_time, periods=len(chest_acc.signals[0][chest_start:chest_stop]), freq="{}ms".format(1000/chest_fs))
chest_ts = pd.date_range(start=chest_acc.header['startdate'], periods=len(chest_acc.signals[0]), freq="{}ms".format(1000/chest_fs))

la_acc = nwdata.NWData()
la_acc.import_edf(file_path=la_file, quiet=False)
la_acc_indexes = {"Acc_x": la_acc.get_signal_index("Accelerometer x"),
                  "Acc_y": la_acc.get_signal_index("Accelerometer y"),
                  "Acc_z": la_acc.get_signal_index("Accelerometer z")}
la_axes = {"X": "Up", "Y": "Anterior", "Z": "Right"}
la_fs = la_acc.signal_headers[la_acc_indexes["Acc_x"]]['sample_rate']
la_start = int((start_time_gs - la_acc.header['startdate']).total_seconds() * la_fs)
la_stop = int((df_event.iloc[-1]['Stop'] - chest_acc.header['startdate']).total_seconds() * chest_fs)
# la_ts = pd.date_range(start=start_time, periods=len(la_acc.signals[0][la_start:la_stop]), freq="{}ms".format(1000/la_fs))
la_ts = pd.date_range(start=la_acc.header['startdate'], periods=len(la_acc.signals[0]), freq="{}ms".format(1000/la_fs))

"""chest_acc.signals[chest_acc_indexes['Acc_x']] = chest_acc.signals[chest_acc_indexes['Acc_x']][chest_start:chest_stop]
chest_acc.signals[chest_acc_indexes['Acc_y']] = chest_acc.signals[chest_acc_indexes['Acc_y']][chest_start:chest_stop]
chest_acc.signals[chest_acc_indexes['Acc_z']] = chest_acc.signals[chest_acc_indexes['Acc_z']][chest_start:chest_stop]
la_acc.signals[la_acc_indexes['Acc_x']] = la_acc.signals[la_acc_indexes['Acc_x']][la_start:la_stop]
la_acc.signals[la_acc_indexes['Acc_y']] = la_acc.signals[la_acc_indexes['Acc_y']][la_start:la_stop]
la_acc.signals[la_acc_indexes['Acc_z']] = la_acc.signals[la_acc_indexes['Acc_z']][la_start:la_stop]"""


""" ================================================ FUNCTIONS =================================================== """


def create_gait_mask(df_gait, duration_seconds, start_stamp):
    """This is in place of the nwgait output --> will need reformatting"""

    # Binary list of gait (1) or no gait (0) in 1-sec increments
    gait_mask = np.zeros(duration_seconds)

    for row in df_gait.itertuples():
        start = int((row.Start - start_stamp).total_seconds())
        stop = int((row.Stop - start_stamp).total_seconds())
        gait_mask[int(start):int(stop)] = 1

    return gait_mask


def create_gs_list(df_event, start_time, stop_time):

    gs = []

    for i in range(int((df_event.iloc[0]['Start'] - start_time).total_seconds())):
        gs.append('other')

    for row in df_event.itertuples():
        start = int((row.Start - start_time).total_seconds())
        end = int((row.Stop - start_time).total_seconds())

        for i in range(end - start):
            gs.append(row.EventShort)

    for i in range(int((stop_time - df_event.iloc[-1]['Stop']).total_seconds())):
        gs.append('other')

    return gs


def format_nwposture_output(nwposture_obj, start_stamp, stop_stamp, gait_mask, df_event):

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
    code_dict = {0: "Prone", 1: "Supine", 2: "Side", 3: "Sitting", 4: 'Sit/Stand',
                 5: 'Dynamic', 6: "Walking", 7: "Standing"}

    df = nwposture_obj.processing()

    df = df.reset_index()
    df.columns = ['Timestamp', "posture", "XY", "Y", "dyno", "vm", "dyno_vm", "ankle"]
    df = df.loc[(df['Timestamp'] >= start_stamp) & (df['Timestamp'] < stop_stamp)]
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

    df = df.loc[(df['Timestamp'] >= start_stamp) & (df['Timestamp'] < stop_stamp)]
    df = df.reset_index(drop=True)

    # Returns indexes that correspond to each 1s interval when a posture change occurs
    transitions_indexes = find_posture_changes(df)

    df['GS'] = create_gs_list(df_event=df_event)

    return df, transitions_indexes


def find_posture_changes(df, colname='posture'):

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
            index = get_index(data=df[colname], start_index=i)
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


def format_nwposture_output2(df_nwposture, start_stamp, stop_stamp, df_event):

    df = df_nwposture.reset_index()
    df = df.loc[(df['timestamp'] >= start_stamp) & (df['timestamp'] < stop_stamp)]
    df = df.reset_index(drop=True)

    df['timestamp'] = [i.round(freq='1S') for i in df['timestamp']]

    # flags gait bouts at "Standing"
    df.loc[df["posture"] == 'gait', 'posture'] = 'stand'

    # Returns indexes that correspond to each 1s interval when a posture change occurs
    transitions_indexes = find_posture_changes(df=df, colname='posture')

    if "GS" not in df.columns:
        df['GS'] = create_gs_list(df_event=df_event, start_time=start_stamp, stop_time=stop_stamp)

    df["GS"] = df["GS"].replace({"SitRec": "sit", "SitRe": 'sit', 'Sitti': "sit",
                                 "Walking": "standing", "Walki": "stand",
                                 "LL": "leftside", "LR": "rightside",
                                 "Prone": "prone", "Supine": "supine", "Supin": "supine",
                                 "Sitting": 'sit', "Transition": 'other',
                                 'Standing': 'stand', 'Stand': 'stand'})

    return df, transitions_indexes


def process_for_peak_detection(chest_nwdata_obj, index_dict, sample_f):

    # Acceleration magnitudes
    vm = np.sqrt(np.square(np.array([chest_nwdata_obj.signals[index_dict["Acc_x"]],
                                     chest_nwdata_obj.signals[index_dict["Acc_y"]],
                                     chest_nwdata_obj.signals[index_dict["Acc_z"]]])).sum(axis=0))
    vm = np.abs(vm)

    # 5Hz lowpass
    alpha_filt = filter_signal(data=vm, low_f=3, filter_order=4, sample_f=sample_f, filter_type='lowpass')

    # .25s rolling median
    rm_alpha = [np.mean(alpha_filt[i:i+int(sample_f/4)]) for i in range(len(alpha_filt))]

    # Continuous wavelet transform; focus on <.5Hz band
    c = pywt.cwt(data=alpha_filt, scales=[1, 64], wavelet='gaus1', sampling_period=1/sample_f)
    cwt_power = c[0][1]

    return vm, alpha_filt, rm_alpha, cwt_power


def detect_peaks(wavelet_data, method='raw', sample_rate=100, min_seconds=1):

    if method == 'raw':
        d = wavelet_data
    if method == 'abs' or method == 'absolute':
        d = np.abs(wavelet_data)
    if method == 'le' or method == 'linear envelop' or method == 'linear envelope':
        d = filter_signal(data=np.abs(wavelet_data), low_f=.25, filter_type='lowpass',
                          sample_f=sample_rate, filter_order=5)
        d = np.abs(d)

    power_sd = np.std(d)
    peaks = peakutils.indexes(y=d, min_dist=int(min_seconds*sample_rate), thres_abs=True, thres=power_sd*1.5)

    return peaks, power_sd * 1.5, d


def create_peaks_df(start_time, stop_time, sts_peaks, chest_nwdata_obj, raw_timestamps, index_dict, sample_f):

    # ts = raw_timestamps[sts_peaks]
    x_vals, y_vals, z_vals = [], [], []

    new_peak_inds = []
    for peak in sts_peaks:
        x = filter_signal(data=chest_nwdata_obj.signals[index_dict['Acc_x']][peak-int(2*sample_f):peak+int(2*sample_f)],
                          sample_f=sample_f, high_f=.1, filter_type='highpass', filter_order=5)
        x_peak = np.argmax(np.abs(x))
        x_vals.append(x[x_peak])

        y = filter_signal(data=chest_nwdata_obj.signals[index_dict['Acc_y']][peak-int(2*sample_f):peak+int(2*sample_f)],
                          sample_f=sample_f, high_f=.1, filter_type='highpass', filter_order=5)
        y_peak = np.argmax(np.abs(y))
        y_vals.append(y[y_peak])

        z = filter_signal(data=chest_nwdata_obj.signals[index_dict['Acc_z']][peak-int(2*sample_f):peak+int(2*sample_f)],
                          sample_f=sample_f, high_f=.1, filter_type='highpass', filter_order=5)
        z_peak = np.argmax(np.abs(z))
        z_vals.append(z[z_peak])

        # Finds actual local maxima and creates new peaks
        window = processed_data[int(peak - 2*sample_f):int(peak + 2*sample_f)]
        peak_val = max(window)
        peak_ind = int(peak - 2*sample_f + np.argwhere(window == peak_val))
        new_peak_inds.append(peak_ind)

    ts = raw_timestamps[new_peak_inds]

    df_out = pd.DataFrame({"Timestamp": ts, "xpeak": x_vals, "ypeak": y_vals, 'zpeak': z_vals})
    df_out['Timestamp'] = [i.round('1S') for i in df_out['Timestamp']]
    df_out['Ind'] = [int((i - ts[0]).total_seconds()) for i in df_out.Timestamp]
    df_out = df_out.loc[(df_out["Timestamp"] >= start_time) & (df_out['Timestamp'] <= stop_time)]

    return df_out.reset_index(drop=True)


def format_posture_change_dfs(df_posture, df_peaks):

    # Index (1s epochs) of first standing (gait) bout: used as starting point for algorithm
    try:
        first_standing = df_posture.loc[df_posture['posture'] == 'stand'].index[0]
    except IndexError:
        first_standing = 0

    # indexes of transitions that occur at/after first standing
    use_indexes = [i for i in transitions_indexes if i >= first_standing]

    # First row in df1s for each posture --> bouts
    df_index = pd.DataFrame({"Timestamp": [df_posture.iloc[i]['timestamp'] for i in use_indexes],
                             "Ind": use_indexes,
                             "Posture": [df_posture.iloc[i]['posture'] for i in use_indexes],
                             'Type': ["Transition" for i in range(len(use_indexes))],
                             'xpeak': [None for i in range(len(use_indexes))],
                             'ypeak': [None for i in range(len(use_indexes))],
                             'zpeak': [None for i in range(len(use_indexes))]})

    # df for postures at each potential STS peak
    df_sts_peaks = pd.DataFrame({"Timestamp": [df_posture.iloc[i]['timestamp'] for i in df_peaks['Ind']],
                                 "Ind": df_peaks['Ind'],
                                 "Posture": [df_posture.iloc[i]['posture'] for i in df_peaks['Ind']],
                                 "Type": ["Peak" for i in range(df_peaks.shape[0])],
                                 'xpeak': df_peaks['xpeak'],
                                 'ypeak': df_peaks['ypeak'],
                                 'zpeak': df_peaks['zpeak']})

    # Combines two previous dataframes and formats indexes
    df_index = df_index.append(df_sts_peaks)
    df_index = df_index.sort_values("Timestamp")  # chronological order
    df_index = df_index.reset_index(drop=True)

    return df_index, first_standing


def pass1(win_size, df_peak, input):

    print("\nUsing known standing/gait periods to adjust sit/stand classifications...")

    input = np.array(input)
    output_postures = input.copy()

    # Loops through each potenetial STS peak and generates window around peak
    for row in range(df_peak.shape[0]):
        curr_row = df_peak.iloc[row]
        prev_row = df_peak.iloc[row-1] if row >= 1 else df_peak.iloc[0]
        prev_row_ind = prev_row.Ind if row >= 1 else 0

        # Windows +/- win_size seconds from potential STS peak
        window = list(input[curr_row['Ind'] - win_size:curr_row['Ind'] + win_size])

        # If current row is after first standing/gait bout:
            # If standing + transition = sit/stand --> sit/stand period becomes sitting
        if curr_row.Ind > df_peak.iloc[0]["Ind"]:
            if "stand" in window[:win_size] and "sitstand" in window[-win_size:]:
                output_postures[curr_row.Ind:df_peak.iloc[row+1]['Ind']] = "sit"

            # If current row is after first standing/gait bout:
                # If sit/stand + transition = standing --> sit/stand period becomes sitting
            if "sitstand" in window[:win_size] and "stand" in window[-win_size:]:
                output_postures[prev_row_ind:curr_row.Ind] = "sit"

            if "sit" in window[:win_size] and "sitstand" in window[-win_size:]:
                output_postures[curr_row.Ind - win_size:curr_row.Ind + win_size] = pd.Series(window).replace({"sitstand": "stand"})
                print(curr_row['Timestamp'])


    print("Complete.")

    return output_postures


def compare_algorithms(alg_colname):

    fig, ax = plt.subplots(3, sharex='col', gridspec_kw={"height_ratios": [1, .66, .34]}, figsize=(11, 7))
    ax0 = ax[0].twiny()  # secondary x-axis for 1s-indexes
    ax0.plot(df1s['posture'], linestyle="")
    ax[0].plot(df1s['timestamp'], df1s["posture"], color='red', label='V1', zorder=2)
    ax[0].plot(df1s['timestamp'], df1s[alg_colname], color='dodgerblue', label=alg_colname.capitalize(), zorder=2)
    ax[0].plot(df1s['timestamp'], df1s['GS'],
               color='limegreen', label='GS', zorder=0)
    ax[0].axvline(x=df1s.loc[df1s['posture'] == 'stand'].iloc[0]['timestamp'],
                  color='fuchsia', linestyle='dashed', label='1stConf.Stand')
    ax[0].fill_between(x=[df1s.iloc[0]['timestamp'], df1s.loc[df1s['posture'] == 'stand'].iloc[0]['timestamp']],
                       y1=0, y2=7, color='fuchsia', alpha=.1)
    ax[0].legend()
    ax[0].grid(axis='y')

    ax[1].plot(chest_ts, processed_data, color='gold', label='STS_data')
    ax[1].axhline(thresh, color='black', linestyle='dashed', label='PeakThresh')
    ax[1].set_yticks([])
    ax[1].set_ylim(-.1, 1)
    ax[1].legend()

    ax[2].plot(df1s['timestamp'], df1s["chest_gait_mask"], color='purple', zorder=1)
    ax[2].fill_between(x=df1s['timestamp'], y1=0, y2=df1s["chest_gait_mask"], color='purple', alpha=.25)
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels(['NoGait', 'Gait'])
    ax[-1].xaxis.set_major_formatter(xfmt)

    for row in df_index.loc[df_index["Type"] == 'Peak'].itertuples():
        try:
            peak_val = max(processed_data[int(row.Ind*chest_fs - 2*chest_fs):int(row.Ind*chest_fs + 2*chest_fs)])
            ax[0].axvline(x=chest_ts[int(row.Ind*chest_fs)], color='orange', linestyle='dashed', zorder=0)
            ax[1].scatter(x=row.Timestamp, y=peak_val, color='black', marker='o')
        except ValueError:
            pass

    ax[-1].set_xlim(df_event.iloc[0]['Start'], df_event.iloc[-1]['Stop'])
    plt.tight_layout()

    return fig


""" ============================================= CODE EXECUTION =================================================="""

# df_nwposture2 = df_nwposture2.loc[(df_nwposture2["timestamp"] >= start_time) & (df_nwposture2["timestamp"] <= stop_time)]
start_time = df_nwposture2.iloc[0]['timestamp']
stop_time = df_nwposture2.iloc[-1]['timestamp']

chest = {"Anterior": chest_acc.signals[2]*-1, "Up": chest_acc.signals[0], "Left": chest_acc.signals[1],
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}

ankle = {"Anterior": la_acc.signals[1], "Up": la_acc.signals[0], "Left": chest_acc.signals[2],
         "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}

# post = NWPosture(ankle_file_path=la_file, chest_file_path=chest_file)
post = NWPosture(chest_dict=chest, ankle_dict=ankle)

df1s, transitions_indexes = format_nwposture_output2(df_nwposture=df_nwposture2,
                                                     df_event=df_event,
                                                     start_stamp=start_time, stop_stamp=stop_time)

vm, alpha_filt, rm_alpha, cwt_power = process_for_peak_detection(chest_nwdata_obj=chest_acc,
                                                                 index_dict=chest_acc_indexes,
                                                                 sample_f=chest_fs)

peaks, thresh, processed_data = detect_peaks(wavelet_data=cwt_power, method='le', sample_rate=chest_fs, min_seconds=5)

df_peaks = create_peaks_df(start_time=start_time, stop_time=stop_time, sts_peaks=peaks,
                           chest_nwdata_obj=chest_acc, raw_timestamps=chest_ts,
                           index_dict=chest_acc_indexes, sample_f=chest_fs)

df_index, first_stand_index = format_posture_change_dfs(df_posture=df1s, df_peaks=df_peaks)

# Crops df_index to at/after first Standing and only include rows for potential STS peaks
df_use = df_index.loc[(df_index['Timestamp'] >= df1s.iloc[first_stand_index]['timestamp']) &
                      (df_index['Type'] == "Peak")].reset_index(drop=True)

df1s["v1"] = pass1(win_size=8, input=np.array(df1s['posture']), df_peak=df_use)
transitions_indexes2 = find_posture_changes(df=df1s, colname='v1')
df_index2 = df1s.iloc[transitions_indexes2][["timestamp", 'posture', 'Transition', 'GS', "v1"]].reset_index()

fig = compare_algorithms(alg_colname='v2')


# TODO
# Work backwards from first gait bout
# bouted DF
    # remove/edit certain events based on surrounding
        # e.g. sit -> sitstand -> sit with no transitions = all sitting

stand_stamp = df1s.iloc[first_stand_index]['timestamp']

prestand_index = df_index2.loc[df_index2['timestamp'] <=
                               stand_stamp].sort_values("timestamp", ascending=False).reset_index(drop=True)
prestand_index.columns = ['Ind', 'timestamp', 'posture', 'Transition', 'GS', 'v1']



def reclassify_pre_firststand(df, win_size, df_peaks):

    v2 = np.array(df1s['v1'])

    for i in range(prestand_index.shape[0] - 1):
        curr_row = prestand_index.iloc[i]

        curr_peak = df_peaks.loc[(df_peaks['Timestamp'] >= curr_row['timestamp'] + td(seconds=-win_size)) &
                                 (df_peaks['Timestamp'] <= curr_row['timestamp'] + td(seconds=win_size))]

        # If standing and no transition when standing starts --> flags previous peak to current as standing
        if v2[curr_row.Ind] == 'stand' and curr_peak.shape[0] == 0:
            prev_peak = df_peaks.loc[df_peaks['Timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
            # print("Stand: ", df1s.iloc[prev_peak]['timestamp'], curr_row.timestamp)
            v2[prev_peak:curr_row.Ind] = 'stand'

        if v2[curr_row.Ind] == 'sit' and curr_peak.shape[0] == 0:
            prev_peak = df_peaks.loc[df_peaks['Timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
            # print("Sit:", df1s.iloc[prev_peak]['timestamp'], curr_row.timestamp)
            v2[prev_peak:curr_row.Ind] = 'sit'

        if v2[curr_row.Ind] == 'sit' and curr_peak.shape[0] > 0:
            try:
                prev_peak = df_peaks.loc[df_peaks['Timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
                print("Sit:", prev_peak, curr_row.Ind)
                v2[prev_peak:curr_row.Ind] = 'stand'
            except IndexError:
                pass


df1s["v2"] = v2


test = reclassify_pre_firststand(df=prestand_index, win_size=8, df_peaks=df_peaks.loc[df_peaks['Timestamp'] <= stand_stamp])