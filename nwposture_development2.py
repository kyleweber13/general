import pandas as pd
from nwposture_dev.nwposture.NWPosture2 import NWPosture, plot_all
import nwdata
from nwpipeline import NWPipeline
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
from datetime import timedelta as td
from scipy import signal
from Filtering import filter_signal
import pywt
import peakutils
import os
import scipy
import sit2standpy as s2s
import pickle

lying = ['supine', 'prone', 'leftside', 'rightside']


""" ================================================ GS DATA IMPORT =============================================== """


def import_goldstandard_df(filename):

    df_event = pd.read_excel(filename, usecols=["Event", "EventShort", "SuperShort",
                                                "Duration", "Type", "Start", "Stop"])
    df_event["Event"] = df_event["Event"].fillna("Transition")
    df_event["Duration"] = [(j-i).total_seconds() for i, j in zip(df_event["Start"], df_event["Stop"])]
    df_event = df_event.fillna(method='bfill')
    start_time = df_event.iloc[0]['Start']
    stop_time = df_event.iloc[-1]['Stop']

    return df_event, start_time, stop_time


# GOLD STANDARD DATASET ---------------------
folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/001_Pilot/Converted/"
chest_file = folder + "001_chest_AX6_6014664_Accelerometer.edf"
ankle_file = folder + "001_left ankle_AX6_6014408_Accelerometer.edf"
log_file = "C:/Users/ksweber/Desktop/PosturePilot001_EventLog.xlsx"
df_gait = pd.read_excel("C:/Users/ksweber/Desktop/Posture001_Gait.xlsx")
"""
folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/002/Converted/"
chest_file = folder + "002_Axivity_Chest_Accelerometer.edf"
ankle_file = folder + "002_Axivity_LAnkle_Accelerometer.edf"
log_file = "C:/Users/ksweber/Desktop/PosturePilot002_EventLog.xlsx"
df_gait = pd.read_excel("C:/Users/ksweber/Desktop/Posture002_Gait.xlsx")
"""

df_event, start_time_gs, stop_time_gs = import_goldstandard_df(filename=log_file)
df_event.columns = ['Event', 'Type', 'start_timestamp', 'Duration', 'end_timestamp', 'EventShort', 'SuperShort']

print("\nImporting Bittium...")
chest_acc = nwdata.NWData()
chest_acc.import_edf(file_path=chest_file, quiet=False)
chest_acc_indexes = {"Acc_x": chest_acc.get_signal_index("Accelerometer x"),
                     "Acc_y": chest_acc.get_signal_index("Accelerometer y"),
                     "Acc_z": chest_acc.get_signal_index("Accelerometer z")}
chest_fs = chest_acc.signal_headers[chest_acc_indexes["Acc_x"]]['sample_rate']

chest_ts = pd.date_range(start=chest_acc.header['startdate'],
                         periods=len(chest_acc.signals[chest_acc_indexes['Acc_x']]),
                         freq="{}ms".format(1000/chest_fs))

print("\nImporting ankle Axivity...")
ank_acc = nwdata.NWData()
ank_acc.import_edf(file_path=ankle_file, quiet=False)
ank_acc_indexes = {"Acc_x": ank_acc.get_signal_index("Accelerometer x"),
                   "Acc_y": ank_acc.get_signal_index("Accelerometer y"),
                   "Acc_z": ank_acc.get_signal_index("Accelerometer z"),
                   "Temperature": ank_acc.get_signal_index("Temperature")}
ank_fs = ank_acc.signal_headers[ank_acc_indexes["Acc_x"]]['sample_rate']
ank_ts = pd.date_range(start=ank_acc.header['startdate'],
                       periods=len(ank_acc.signals[ank_acc_indexes['Acc_x']]), freq="{}ms".format(1000/ank_fs))
ank_temp_fs = ank_acc.signal_headers[ank_acc_indexes["Temperature"]]['sample_rate']
ank_temp_ts = pd.date_range(start=ank_acc.header['startdate'],
                            periods=len(ank_acc.signals[ank_acc_indexes['Temperature']]),
                            freq="{}ms".format(1000/ank_temp_fs))

# Axivity chest orientation
chest = {"Anterior": chest_acc.signals[chest_acc_indexes['Acc_z']]*-1,
         "Up": chest_acc.signals[chest_acc_indexes['Acc_x']],
         "Left": chest_acc.signals[chest_acc_indexes['Acc_y']],
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}

ankle = {"Anterior": ank_acc.signals[ank_acc_indexes['Acc_y']],
         "Up": ank_acc.signals[ank_acc_indexes['Acc_x']],
         "Left": ank_acc.signals[ank_acc_indexes['Acc_z']],
         "start_stamp": ank_acc.header['startdate'], "sample_rate": ank_fs}

# df_gait = df_event.loc[df_event['EventShort'] == "Walking"]
post = NWPosture(chest_dict=chest, ankle_dict=ankle,
                 study_code='OND09', subject_id='Test', coll_id='01')
post.crop_data()
post.df_gait = post.load_gait_data(df_gait)
post.df_gait = post.df_gait.loc[post.df_gait['start_timestamp'] > df_event['start_timestamp'].iloc[0]]
post.gait_mask = post.create_gait_mask()
df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=df_event)
first_walk_index = df1s.loc[df1s['chest_gait_mask'] == 1].iloc[0]['index']

# plot_all(df1s, df_peaks=df_peaks, show_v0=False, show_v1=False, show_v2=False, show_v3=True, collapse_lying=False)
# bouts.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture.csv", index=False)
# df1s.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture_Epoch1s.csv", index=False)


""" ==================================================== HANDDS =================================================== """

"""
subj = '0001'
# folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"
folder = "C:/Users/ksweber/Desktop/"

start_seconds = 0
n_seconds = 3600*12

ankle = pickle.load(open("C:/Users/ksweber/Desktop/OND09_0001_RAnkle.pickle", 'rb'))
chest = pickle.load(open("C:/Users/ksweber/Desktop/OND09_0001_Chest.pickle", 'rb'))

"""
"""
ankle_file = folder + f"OND09_{subj}_01_AXV6_LAnkle.edf"
if not os.path.exists(ankle_file):
    ankle_file = folder + f"OND09_{subj}_01_AXV6_RAnkle.edf"

chest_file = folder + f"OND09_{subj}_01_BF36_Chest.edf"

print("\nImporting Bittium...")
chest_acc = nwdata.NWData()
chest_acc.import_edf(file_path=chest_file, quiet=False)
chest_acc_indexes = {"Acc_x": chest_acc.get_signal_index("Accelerometer x"),
                     "Acc_y": chest_acc.get_signal_index("Accelerometer y"),
                     "Acc_z": chest_acc.get_signal_index("Accelerometer z"),
                     'Temperature': chest_acc.get_signal_index('Temperature')}
chest_fs = chest_acc.signal_headers[chest_acc_indexes["Acc_x"]]['sample_rate']
chest_temp_fs = chest_acc.signal_headers[chest_acc_indexes['Temperature']]['sample_rate']

chest_acc.header['startdate'] += td(seconds=2.75)

chest_ts = pd.date_range(start=chest_acc.header['startdate'],
                         periods=len(chest_acc.signals[chest_acc_indexes['Acc_x']]),
                         freq="{}ms".format(1000/chest_fs))
chest_temp_ts = pd.date_range(start=chest_acc.header['startdate'],
                              periods=len(chest_acc.signals[chest_acc_indexes['Temperature']]),
                              freq="{}ms".format(1000/chest_temp_fs))

print("\nImporting ankle Axivity...")
ank_acc = nwdata.NWData()
ank_acc.import_edf(file_path=ankle_file, quiet=False)
ank_acc_indexes = {"Acc_x": ank_acc.get_signal_index("Accelerometer x"),
                   "Acc_y": ank_acc.get_signal_index("Accelerometer y"),
                   "Acc_z": ank_acc.get_signal_index("Accelerometer z"),
                   "Temperature": ank_acc.get_signal_index('Temperature')}
ank_fs = ank_acc.signal_headers[ank_acc_indexes["Acc_x"]]['sample_rate']
ank_temp_fs = ank_acc.signal_headers[ank_acc_indexes["Temperature"]]['sample_rate']
ank_ts = pd.date_range(start=ank_acc.header['startdate'],
                       periods=len(ank_acc.signals[ank_acc_indexes['Acc_x']]), freq="{}ms".format(1000/ank_fs))
ank_temp_ts = pd.date_range(start=ank_acc.header['startdate'],
                            periods=len(ank_acc.signals[ank_acc_indexes['Temperature']]),
                            freq="{}ms".format(1000/ank_temp_fs))

# Bittium orientation + conversion from mg to G
chest = {"Anterior": chest_acc.signals[chest_acc_indexes['Acc_z']][int(start_seconds*chest_fs):int(n_seconds*chest_fs)],
         "Up": chest_acc.signals[chest_acc_indexes['Acc_x']][int(start_seconds*chest_fs):int(n_seconds*chest_fs)],
         "Left": chest_acc.signals[chest_acc_indexes['Acc_y']][int(start_seconds*chest_fs):int(n_seconds*chest_fs)],
         "start_stamp": chest_acc.header['startdate'] + td(seconds=start_seconds), "sample_rate": chest_fs}

start_seconds += 374900/ank_fs
# Actually lateral left OR medial right ankle
if 'LAnkle' in ankle_file:
    ankle = {"Anterior": ank_acc.signals[ank_acc_indexes['Acc_y']][int(start_seconds*ank_fs):int(n_seconds*ank_fs)],
             "Up": ank_acc.signals[ank_acc_indexes['Acc_x']][int(start_seconds*ank_fs):int(n_seconds*ank_fs)],
             "Left": ank_acc.signals[ank_acc_indexes['Acc_z']][int(start_seconds*ank_fs):int(n_seconds*ank_fs)],
             "start_stamp": ank_acc.header['startdate'] + td(seconds=start_seconds), "sample_rate": ank_fs}

# Actually lateral right OR medial left ankle
if 'RAnkle' in ankle_file:
    # Right ankle
    ankle = {"Anterior": ank_acc.signals[ank_acc_indexes['Acc_y']][int(start_seconds*ank_fs):int(n_seconds*ank_fs)]*-1,
             "Up": ank_acc.signals[ank_acc_indexes['Acc_x']][int(start_seconds*ank_fs):int(n_seconds*ank_fs)],
             "Left": ank_acc.signals[ank_acc_indexes['Acc_z']][int(start_seconds*ank_fs):int(n_seconds*ank_fs)]*-1,
             "start_stamp": ank_acc.header['startdate'] + td(seconds=start_seconds), "sample_rate": ank_fs}
"""
"""
df_gait = pd.read_csv(f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv")
df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])

post = NWPosture(chest_dict=chest, ankle_dict=ankle, study_code='OND09', subject_id=subj, coll_id='01')
post.crop_data()
post.df_gait = post.load_gait_data(df_gait)
post.gait_mask = post.create_gait_mask()
"""
# df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=None)

# plot_all(df1s, df_peaks=None, show_v0=False, show_v1=False, show_v2=True, show_v3=True, collapse_lying=False)
# bouts.to_csv(f"C:/Users/ksweber/Desktop/Posture/OND09_{subj}_PostureBouts.csv", index=False)
# df1s.to_csv(f"C:/Users/ksweber/Desktop/Posture/OND09_{subj}_Posture_Epoch1s.csv", index=False)
# df_peaks.to_csv(f"C:/Users/ksweber/Desktop/Posture/OND09_{subj}_Posture_STSPeaks.csv", index=False)


# READING FROM CSV -----------------
"""subj = '0001'

df1s = pd.read_csv(f"C:/users/ksweber/Desktop/Posture/Archived/OND09_{subj}_Posture_Epoch1s.csv")
df1s['timestamp'] = pd.to_datetime(df1s['timestamp'])
#bouts = pd.read_csv(f"C:/users/ksweber/Desktop/Posture/OND09_{subj}_PostureBouts.csv")
#bouts['start_timestamp'] = pd.to_datetime(bouts['start_timestamp'])
#bouts['end_timestamp'] = pd.to_datetime(bouts['end_timestamp'])
df_peaks = pd.read_csv(f"C:/users/ksweber/Desktop/Posture/Archived/OND09_{subj}_Posture_STSPeaks.csv")
df_peaks['timestamp'] = pd.to_datetime(df_peaks['timestamp'])

post = NWPosture(chest_dict={'sample_rate': 25}, ankle_dict={'sample_rate': 50}, gait_bouts=None, study_code='OND09', subject_id=subj, coll_id='01')
df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=None, input_dfpeaks=df_peaks, input_df1s=df1s)
# plot_all(df1s=df1s, df_peaks=df_peaks, show_v0=False, show_v1=False, show_v2=False, show_v3=True, collapse_lying=False)
"""


def convert_accel_units(to_g=False, to_mss=True):
    """Function to convert chest accelerometer data between G's and m/s/s.

        arguments:
        -to_g: boolean; divides accel values by 9.81 if max(abs(accel)) >= 17 (assumes m/s/s)
        -to_mss: boolean; multiplies accel values by 9.81 if max(abs(accel)) <= 17 (assumes G's)
    """

    converted_to_mss = False

    if to_mss:
        if max(np.abs(post.chest['Anterior'])) <= 17:
            print("\nConverting chest accelerometer data from G to m/s^2...")

            post.chest['Anterior'] *= 9.81
            post.chest['Up'] *= 9.81
            post.chest['Left'] *= 9.81
            converted_to_mss = True

        else:
            print("\nChest accelerometer data already in correct units. Doing nothing.")

    if to_g:
        if max(np.abs(post.chest['Anterior'])) >= 17:
            print("\nConverting chest accelerometer data from m/s^2 to g...")

            post.chest['Anterior'] /= 9.81
            post.chest['Up'] /= 9.81
            post.chest['Left'] /= 9.81
            converted_to_mss = True

        else:
            print("\nChest accelerometer data already in correct units. Doing nothing.")

    return converted_to_mss


def process_sts_transfers(anterior, vertical, left, sample_rate, quiet=True, show_plot=False):
    """REQUIRES ACCELEROMETER DATA IN M/S2 - NOT G"""

    # INITIAL PROCESSING ---------------------------------------------------------------------------------
    # Acceleration magnitudes
    if not quiet:
        print("-Calculating acceleration magnitudes...")
    a = np.sqrt(np.square(np.array([anterior, vertical, left])).sum(axis=0))
    a = np.abs(a)

    # 5Hz lowpass
    if not quiet:
        print("-Filtering data...")
    a_filtered = np.abs(filter_signal(data=a, low_f=.025, high_f=3, filter_order=4, sample_f=sample_rate, filter_type='bandpass'))

    # .25-sec rolling mean and SD
    if not quiet:
        print("-Calculating rolling mean and SD...")
    a_rm = [np.mean(a_filtered[i:i + int(sample_rate / 4)]) for i in range(len(a_filtered))]
    a_rsd = [np.std(a_filtered[i:i + int(sample_rate / 4)]) for i in range(len(a_filtered))]

    # Calculates jerk of rolling mean and rolling SD
    if not quiet:
        print("-Calculating jerk...")
    j_rm = np.abs([(j-i)/(1/sample_rate) for i, j in zip(a_rm[:], a_rm[1:])])
    j_rm = np.append(j_rm, j_rm[-1])
    j_rsd = np.abs([(j-i)/(1/sample_rate) for i, j in zip(a_rsd[:], a_rsd[1:])])
    j_rsd = np.append(j_rsd, j_rsd[-1])

    # Continuous wavelet transform; power in <.5Hz band
    if not quiet:
        print("-Running wavelet transform...")
    coefs, freqs = pywt.cwt(a_rm, np.arange(1, 65), 'gaus1', sampling_period=1 / sample_rate)
    f_mask = (freqs <= .5) & (freqs >= 0)
    cwt_power = np.sum(coefs[f_mask, :], axis=0)

    # Initial peak detection: minimum 1-sec apart
    if not quiet:
        print("-Detecting peaks...")
    pp = peakutils.indexes(y=cwt_power, min_dist=sample_rate, thres_abs=True, thres=np.std(cwt_power))
    print(f"     -Found {len(pp)} potential transitions.")

    # Binary stillness list if mean/SD values below thresholds
    if not quiet:
        print("-Finding periods of stillness...")
    stillness = np.zeros(len(a_rm))
    for i in range(len(a_rm)):
        if a_rm[i] < .15 and a_rsd[i] < .1 and j_rm[i] < 2.5 and j_rsd[i] < 3:
            stillness[i] = 1

    # Finds indexes for start/stop of stillness periods at least .3-sec long
    if not quiet:
        print("-Finding periods of stillness longer than 0.3 seconds...")

    print("     -Calculating still period durations...")
    curr_index = 0
    starts = []
    stops = []

    for i in range(0, len(stillness) - int(sample_rate*.3)-1):
        if i > curr_index:
            if stillness[i] == 1:
                window = stillness[i:i+int(sample_rate*.3)]

                # sum makes sure window is at least .3-sec long (all values = still)
                if sum(window) == int(sample_rate*.3):
                    for j in range(i + int(sample_rate*.3), len(stillness) - int(sample_rate * .3)):
                        if stillness[j] == 0:
                            starts.append(i)
                            stops.append(j)
                            curr_index = j
                            break

    print("     -Removing short periods...")

    for stop, start in zip(stops[:], starts[1:]):
        if (start - stop) < sample_rate * .3:
            stops.remove(stop)
            starts.remove(start)

    # Filtering to estimate gravity component of signal
    if not quiet:
        print("-More filtering...")
    g_est = np.array([filter_signal(data=anterior, low_f=.25, filter_order=4, sample_f=sample_rate, filter_type='lowpass'),
                      filter_signal(data=vertical, low_f=.25, filter_order=4, sample_f=sample_rate, filter_type='lowpass'),
                      filter_signal(data=left, low_f=.25, filter_order=4, sample_f=sample_rate, filter_type='lowpass')])
    g_est_vm = np.sqrt(np.square(g_est).sum(axis=0))

    # Estimates gravity vector for Up and Anterior axes
    a_vert = vertical / g_est_vm
    a_ant = anterior / g_est_vm

    # Integrates data to get velocity; needed for detrending
    if not quiet:
        print("-Integrating accelerometer data to estimate velocity...")
    v_vert_all = [0]
    v_ant_all = [0]
    for i in range(len(a_vert) - 1):
        v_vert_all.append((a_vert[i] * (1 / sample_rate) + a_vert[i]))
        v_ant_all.append((a_ant[i] * (1 / sample_rate) + a_ant[i]))
    v_vert_all = filter_signal(data=v_vert_all, sample_f=sample_rate, high_f=.02, filter_type='highpass', filter_order=3)
    v_ant_all = filter_signal(data=v_ant_all, sample_f=sample_rate, high_f=.02, filter_type='highpass', filter_order=3)

    def find_sittostand(possible_peaks, vertical_acc, anterior_acc, stillness_starts, stillnes_stops,
                        pad_pre=2.0, velo_thresh=.2, sample_rate=50):
        sit_to_stand = [0]
        stand_to_sit = []

        for peak in possible_peaks:

            # limits of windowing around peak: peak - 2 seconds to peak + 30 seconds
            min_index = peak - int(pad_pre * sample_rate)  # original value = 2 seconds
            max_index = peak + int(5 * sample_rate)  # original value = 30 seconds

            # finds end(s) of still period within 2 seconds of potential STS peak
            window_starts = [i for i in stillnes_stops if min_index <= i <= peak]

            # finds start(s) of active periods within 5 seconds of potential STS peak
            window_ends = [i for i in stillness_starts if peak < i <= max_index]

            # ONLY RUNS BLOCK OF CODE IF STILL PERIOD END WITHIN 3 SECONDS (SIT-TO-STAND REQUIRES STILLNESS AT START)
            # If more than one stillness period ends within pre_time seconds of peak...
            if len(window_starts) == 0:
                stand_to_sit.append(peak)

            if len(window_starts) > 0:
                # Takes end of still period closest to peak if multiple are found
                window_start = window_starts[-1]

                # Takes first start of still period if multiple found
                # Sets STS transition ending as static to true
                if len(window_ends) > 0:
                    window_end = window_ends[0]

                # If no still periods are found within 30 seconds of peak, changes window to peak + 5 seconds
                if len(window_ends) == 0:
                    window_end = peak + int(5 * sample_rate)

                integration_window_vert = vertical_acc[window_start:window_end]
                integration_window_ant = anterior_acc[window_start:window_end]

                # Integrate a_vert and a_ant to get vertical velocities in both axes
                v_vert = [0]
                v_ant = [0]
                for i in range(len(integration_window_vert) - 1):
                    ant_val = (integration_window_ant[i] * (1 / sample_rate) + v_ant[i])
                    vert_val = (integration_window_vert[i] * (1 / sample_rate) + v_vert[i])

                    v_vert.append(vert_val)
                    v_ant.append(ant_val)

                v_vert_detrend = v_vert_all[window_start:window_end]
                v_ant_detrend = v_ant_all[window_start:window_end]

                # Integrate velocity to get displacement
                d_vert = [0]
                d_ant = [0]
                for i in range(len(v_vert_detrend) - 1):
                    dv = v_vert_detrend[i] * (1 / sample_rate) + v_vert_detrend[i]
                    d_vert.append(dv)

                    da = v_ant_detrend[i] * (1 / sample_rate) + v_ant_detrend[i]
                    d_ant.append(da)

                d_vert_total = d_vert[-1] - d_vert[0]

                # Checks if velocities exceed threshold of v_thresh m/s
                # Anterior velocity needs to be negative; vertical is positive
                if max(v_vert_detrend) >= velo_thresh or min(v_ant_detrend) <= -velo_thresh:
                    start_ind = window_start

                    # Peak index if vertical speed higher than anterior speed
                    if max(np.abs(v_vert_detrend)) >= max(np.abs(v_ant_detrend)):
                        new_peak = np.argmax(np.abs(v_vert_detrend)) + window_start

                        # Finds duration of movement using zero-crossings
                        for i in range(len(v_vert_detrend) - 1):
                            if v_vert_detrend[i] > 0 >= v_vert_detrend[i + 1]:
                                stop_ind = i

                    # Peak index if anterior speed higher than vertical speed
                    if max(np.abs(v_vert_detrend)) < max(np.abs(v_ant_detrend)):
                        new_peak = np.argmax(np.abs(v_ant_detrend)) + window_start

                        # Finds duration of movement using zero-crossings
                        for i in range(len(v_ant_detrend) - 1):
                            if v_ant_detrend[i] > 0 >= v_ant_detrend[i + 1] or v_ant_detrend[i] <= 0 < v_ant_detrend[i + 1]:
                                stop_ind = i

                    # Movement duration in seconds
                    movement_len = (stop_ind - start_ind) / sample_rate

                    # counts as peak if less than 4.5 seconds long, vertical displacement >= .125m,
                    # and 400ms after previous peak
                    # if movement_len < 4.5 and max(d_vert) >= .125 and (new_peak - sit_to_stand[-1]) > sample_rate * .4:
                    if movement_len < 4.5 and abs(d_vert_total) >= .125 and (new_peak - sit_to_stand[-1]) > sample_rate * .4:
                        sit_to_stand.append(new_peak)

        # Ignores first index which was set to 0
        return sit_to_stand[1:], stand_to_sit

    def find_standtosit(possible_peaks, vertical_acc, anterior_acc, stillness_starts,
                        pad_post=2.0, velo_thresh=.2, sample_rate=50):
        stand_to_sit = [0]
        ignored_peaks = []

        for peak in possible_peaks:

            # window boundaries: 5 seconds before and pad_post seconds after peak
            window_start = peak - int(5 * sample_rate)
            max_index = peak + int(pad_post * sample_rate)

            # Finds still periods that start in pad_post seconds post-peak
            window_ends = [i for i in stillness_starts if peak < i <= max_index]

            # if not stillness periods start close enough to peak, ignores peak
            if len(window_ends) == 0:
                ignored_peaks.append(peak)

            # Takes end of still period closest to peak if multiple are found
            if len(window_ends) > 0:
                window_end = window_ends[0]

                integration_window_vert = vertical_acc[window_start:window_end]
                integration_window_ant = anterior_acc[window_start:window_end]

                # Integrate a_vert and a_ant to get vertical velocities in both axes
                v_vert = [0]
                v_ant = [0]
                for i in range(len(integration_window_vert) - 1):
                    ant_val = (integration_window_ant[i] * (1 / sample_rate) + v_ant[i])
                    vert_val = (integration_window_vert[i] * (1 / sample_rate) + v_vert[i])

                    v_vert.append(vert_val)
                    v_ant.append(ant_val)

                v_vert_detrend = v_vert_all[window_start:window_end]
                v_ant_detrend = v_ant_all[window_start:window_end]

                # Integrate velocity to get displacement
                d_vert = [0]
                d_ant = [0]
                for i in range(len(v_vert_detrend) - 1):
                    dv = v_vert_detrend[i] * (1 / sample_rate) + v_vert_detrend[i]
                    d_vert.append(dv)

                    da = v_ant_detrend[i] * (1 / sample_rate) + v_ant_detrend[i]
                    d_ant.append(da)

                # Checks if velocities exceed threshold of v_thresh m/s
                # Anterior velocity needs to be positive; vertical is negative
                if min(v_vert_detrend) <= -velo_thresh or max(v_ant_detrend) >= velo_thresh:

                    # Peak index if vertical speed higher than anterior speed
                    if max(np.abs(v_vert_detrend)) > max(np.abs(v_ant_detrend)):
                        new_peak = np.argmax(np.abs(v_vert_detrend)) + window_start

                    # Peak index if anterior speed higher than vertical speed
                    if max(np.abs(v_vert_detrend)) < max(np.abs(v_ant_detrend)):
                        new_peak = np.argmax(np.abs(v_ant_detrend)) + window_start

                    # counts as peak if more than 400ms after previous peak and .125m displaced
                    if max(d_vert) >= .125 and (new_peak - stand_to_sit[-1]) > sample_rate * .4:
                        stand_to_sit.append(new_peak)
                    else:
                        ignored_peaks.append(new_peak)

        return stand_to_sit, ignored_peaks

    def plot_results():
        fig, ax = plt.subplots(5, figsize=(12, 9), sharex='col', gridspec_kw={"height_ratios": [.67, .67, 1, .25, .5]})
        plt.suptitle(f"Found {len(pp)} potential peaks; sit-to-stand = {len(sit_to_stand)}, "
                     f"stand-to-sit = {len(stand_to_sit)}, ignored = {len(ignored_peaks)}")
        ax[0].plot(chest_ts, post.chest['Anterior'], color='black', label='A_ant.')
        ax[0].plot(chest_ts, post.chest['Up'], color='red', label='A_vert.')
        ax[0].plot(chest_ts, post.chest['Left'], color='dodgerblue', label='A_left')
        ax[0].set_ylabel("m/s/s")
        ax[0].legend()

        ax[1].plot(chest_ts, v_ant_all, color='black', label='v_ant.')
        ax[1].plot(chest_ts, v_vert_all, color='red', label='v_vert.')
        ax[1].fill_between(x=[chest_ts[0], chest_ts[-1]], y1=-.2, y2=.2, color='grey', alpha=.2)
        ax[1].axhline(y=0, color='grey')
        ax[1].set_ylabel("m/s")
        ax[1].legend()

        ax[2].plot(df1s.loc[df1s['timestamp'] < pd.to_datetime(chest_ts[-1])]['timestamp'],
                   df1s.loc[df1s['timestamp'] < pd.to_datetime(chest_ts[-1])]['GS'], color='green')
        ax[2].grid()
        for peak in pp:
            ax[0].axvline(chest_ts[peak], color='limegreen')
            ax[4].axvline(chest_ts[peak], color='limegreen')
        for peak in sit_to_stand:
            ax[1].axvline(chest_ts[peak], color='fuchsia')
        for peak in stand_to_sit:
            ax[1].axvline(chest_ts[peak], color='limegreen')
        for peak in ignored_peaks:
            ax[1].axvline(chest_ts[peak], color='grey')

        ax[3].plot(chest_ts, stillness, color='grey', label='Stillness')
        ax[3].legend()
        for start, stop in zip(starts, stops):
            ax[3].axvline(chest_ts[start], color='green')
            ax[3].axvline(chest_ts[stop], color='red')

        ax[4].plot(chest_ts, cwt_power, color='navy')
        plt.tight_layout()

    if not quiet:
        print("-Finding sit-to-stand transitions...")

    sit_to_stand, stand_to_sit = find_sittostand(possible_peaks=pp, sample_rate=sample_rate,
                                                 pad_pre=2.5, velo_thresh=.175,
                                                 stillness_starts=starts, stillnes_stops=stops,
                                                 anterior_acc=a_ant, vertical_acc=a_vert)
    if not quiet:
        print("-Finding stand-to-sit transitions...")
    stand_to_sit, ignored_peaks = find_standtosit(possible_peaks=stand_to_sit, sample_rate=sample_rate,
                                                  pad_post=3, velo_thresh=.175, stillness_starts=starts,
                                                  anterior_acc=a_ant, vertical_acc=a_vert)

    if show_plot:
        plot_results()

    if not quiet:
        print("-COMPLETE")

    return sit_to_stand, stand_to_sit, ignored_peaks


def remove_single_second_postures(postures):
    """Removes one-epoch (one second) long postures and re-classifies it as previous posture.

    returns:
    -array of new postures
    """

    postures = np.array(postures)
    curr_ind = 0
    for i in range(len(postures)):
        if i > curr_ind:
            curr_post = postures[i]

            for j in range(i, len(postures)-1):
                try:
                    if postures[j] != curr_post and postures[j+i] != postures[j]:
                        postures[j] = curr_post
                        curr_ind = j
                        break
                except IndexError:
                    j = len(postures)
                    break

    return postures


def format_sts_data():
    """Reformats and combines dataframes from process_sts_transfers() function.

    returns:
    -combined dataframe
    """
    temp_postures = list(df1s['posture'])
    ts = pd.date_range(start=post.chest['start_stamp'], periods=len(post.chest['Anterior']),
                       freq="{}ms".format(1000/post.chest['sample_rate']))
    df_sitstand = pd.DataFrame({"RawIndex": sit_to_stand,
                                "Seconds": [int(i/post.chest['sample_rate']) for i in sit_to_stand],
                                "Epoch": [temp_postures[int(i/post.chest['sample_rate'])-3:
                                                        int(i/post.chest['sample_rate'])+4] for i in sit_to_stand],
                                "Type": ["Sit-stand"]*len(sit_to_stand),
                                'Timestamp': ts[sit_to_stand]})

    df_standsit = pd.DataFrame({"RawIndex": stand_to_sit,
                                "Seconds": [int(i/post.chest['sample_rate']) for i in stand_to_sit],
                                "Epoch": [temp_postures[int(i/post.chest['sample_rate'])-3:
                                                        int(i/post.chest['sample_rate'])+4] for i in stand_to_sit],
                                "Type": ["Stand-sit"]*len(stand_to_sit),
                                'Timestamp': ts[stand_to_sit]})

    df_sts = df_sitstand.append(df_standsit).sort_values("RawIndex").reset_index(drop=True)

    return df_sts


def process_sit2standpy(data, start_stamp, sample_rate, stand_sit=False):

    print("\nRunning sit-stand/stand-sit transition detection using sis2standpy...")

    # for unix time
    start_time = pd.to_datetime("1970-01-01 00:00:00")

    chest_ts = pd.date_range(start=start_stamp, periods=len(data), freq="{}ms".format(1000 / sample_rate))
    ts = np.array([(i - start_time).total_seconds() for i in chest_ts])

    ths = {'stand displacement': 0.125, 'transition velocity': 0.3, 'accel moving avg': 0.15,
           'accel moving std': 0.1, 'jerk moving avg': 2.5, 'jerk moving std': 3}

    if not stand_sit:
        # All default values except power_peak_kwargs
        sts = s2s.Sit2Stand(method='stillness',
                            gravity=9.81, thresholds=ths, long_still=0.3, still_window=0.3,
                            duration_factor=10,  # 6
                            displacement_factor=0.6,
                            lmin_kwargs={'height': -9.5},
                            power_band=[0, 0.5],
                            window=False,
                            # power_peak_kwargs={'distance': 128},
                            power_peak_kwargs={'distance': post.ankle['sample_rate']},
                            power_stdev_height=True, gravity_cut=.25)

    if stand_sit:
        sts = s2s.Sit2Stand(method='displacement',
                            gravity=9.81, thresholds=ths, long_still=0, still_window=0,
                            duration_factor=10,
                            displacement_factor=0.4,  # .6
                            lmin_kwargs={'height': -9.5},
                            power_band=[0, 0.5],
                            window=False,
                            # power_peak_kwargs={'distance': 128},
                            power_peak_kwargs={'distance': post.ankle['sample_rate']},
                            power_stdev_height=True, gravity_cut=.25)

    if not stand_sit:
        print("-Running data for sit-stand detection...")
    if stand_sit:
        print("-Reversing data for stand-sit detection...")
        data = np.flip(data)

    SiSt = sts.apply(accel=data, time=ts, time_units='s')

    if not stand_sit:
        t = SiSt.keys()
        s = [int((pd.to_datetime(i) - start_stamp).total_seconds()) for i in SiSt.keys()]
    if stand_sit:
        t = [chest_ts[0] + td(seconds=(chest_ts[-1] - pd.to_datetime(i)).total_seconds()) for i in SiSt.keys()]
        s = [int((i - chest_ts[0]).total_seconds()) for i in t]

    df_s2spy = pd.DataFrame({"RawIndex": [int((pd.to_datetime(i) - start_stamp).total_seconds() *
                                              sample_rate) for i in SiSt.keys()],
                             "Seconds": s,
                             "Type": ['Sit-stand' if not stand_sit else 'Stand-sit']*len(SiSt),
                             'Timestamp': t})

    print("Complete. Found {} {} transitions.".format(df_s2spy.shape[0],
                                                      'sit-to-stand' if not stand_sit else 'stand-to-sit'))

    return df_s2spy


def remove_toosoon_transitions(df_transition, n_seconds=4):
    """Removes sit/stand transitions that occur too close in time to previous transition.
       Excludes 'newest' transition in that case.

    returns:
        -edited transition dataframe
    """

    # Boolean array for transitions to keep
    keep = np.array([True]*df_transition.shape[0])

    for i in range(df_transition.shape[0]-1):
        curr_time = df_transition.iloc[i]['Seconds']  # current transition time
        next_time = df_transition.iloc[i+1]['Seconds']  # next transition time

        if next_time - curr_time < n_seconds:
            keep[i+1] = False

    df_transition['Keep'] = keep
    df_out = df_transition.loc[df_transition['Keep']]
    df_out = df_out.reset_index(drop=True)

    return df_out


def remove_transitions_during_gait(gait_mask, df_transitions, pad_len=1):

    df = df_transitions.copy()
    gait_mask = np.array(gait_mask)

    idx = []
    for row in df.itertuples():
        epoch = gait_mask[row.Seconds - pad_len:row.Seconds + pad_len + 1]

        # Accepts transition if whole epoch (transitions +- pad_len) is not gait
        if row.Type == 'Stand-sit':
            if sum(epoch) < pad_len * 2:
                idx.append(row.Index)

        # Accepts transition if whole epoch (transitions +- pad_len) is not gait
        if row.Type == 'Sit-stand':
            if sum(epoch) == 0:
                idx.append(row.Index)

    return df.iloc[idx]


def fill_between_walks(postures, gait_mask, df_transitions, max_break=5):
    """Re-does gait_mask by 'filling' small breaks in walking with the 'walking' designation and reflags postures
       using this new gait mask for 'stand' designation.

    returns:
    -edited postured array
    """

    print(f"\nFilling in postures between gait bouts with no STS transitions as 'stand' (max gap = {max_break} seconds)...")

    postures = np.array(postures)
    gait_mask = np.array(gait_mask)

    n_affected = 0

    curr_ind = 0
    for i in range(1, len(gait_mask)):
        if i > curr_ind:
            # Start of gait bout
            if gait_mask[i] == 1:

                # End of current bout
                for j in range(i+1, len(gait_mask)):
                    if gait_mask[j] == 0:
                        gait_end = j

                        # Start of next bout
                        next_start = len(gait_mask)
                        for k in range(j+1, len(gait_mask)):
                            if gait_mask[k] == 1:
                                next_start = k
                                break
                        break

                if next_start - gait_end <= max_break:
                    # Transitions that occur 1 second before end of current bout and 1 second after start of next
                    t = df_transitions.loc[(df_transitions['Seconds'] >= gait_end - 1) &
                                           (df_transitions['Seconds'] <= next_start + 1)]
                    # Only includes stand-sit transitions
                    t = t.loc[t['Type'] == 'Stand-sit']

                    # if no stand-sits in window, postures become 'stand'
                    if t.shape[0] == 0:
                        print(gait_end, next_start)
                        postures[gait_end:next_start] = 'stand'
                        n_affected += 1

                curr_ind = j

    print(f"-Affected {n_affected} bouts.")

    return postures


def apply_logic(df_transitions, gait_mask, postures, first_walk_index, first_pass, quiet=False):
    """Reclassifies postures based on context relating unknown sitstand periods to lying or known standing periods."""

    curr_postures = []
    prev_postures = []
    next_postures = []
    next_diff_postures = []
    curr_sts = []
    next_sts = []

    gait_mask = np.array(gait_mask)
    postures = np.array(postures)

    # First pass of data runs it chronologically
    # Starts at first_walk_index
    if first_pass:
        print("\nRunning first pass of context logic (post-first gait bout)...")
        df_transitions = df_transitions.loc[df_transitions['Seconds'] >= first_walk_index]
        prev_post_end = 0

    # Second pass on data runs it reverse chronologically
    # Starts at first_walk_index and runs backwards
    if not first_pass:

        # Deals with posture that occurs right before first walk if no STS found right at start of walk ------
        last_sts = df_transitions.loc[(df_transitions['Seconds'] <= first_walk_index) &
                                      (df_transitions['Type'] == 'Sit-stand')].iloc[-1]

        prev_post = postures[first_walk_index - 1]  # posture before walk
        prev_post_ind = 0  # default if collection starts with prev_post
        for i in range(first_walk_index):
            if postures[first_walk_index - i] != prev_post:
                prev_post_ind = i
                break

        ind = max([last_sts['Seconds'], prev_post_ind])

        postures[ind:first_walk_index] = 'stand'

        print("\nRunning second pass of context logic to fix pre-first gait bout data...")
        # Reverses order of dataframe
        df_transitions = df_transitions.loc[df_transitions['Seconds'] <
                                            first_walk_index].sort_values('Seconds',
                                                                          ascending=False).reset_index(drop=True)
        prev_post_end = df_transitions.iloc[0]['Seconds']
        prev_post = postures[prev_post_end]

    for row in df_transitions.itertuples():

        curr_sts.append(row.Type)

        # Posture at time of STS transition
        curr_post = postures[row.Seconds]

        # index of next STS transition ----------------------------
        try:
            next_transition = df_transitions.loc[df_transitions['Seconds'] > row.Seconds].iloc[0]['Seconds']
            next_t_type = df_transitions.loc[df_transitions['Seconds'] > row.Seconds].iloc[0]['Type']
        except IndexError:
            next_transition = len(postures)
            next_t_type = 'Sit-stand'
        next_sts.append(f"{next_t_type} ({next_transition})")

        # Finds where current posture ends -----------------
        for i in range(row.Seconds, len(postures)):
            if postures[i] != curr_post and postures[i] != 'other':
                curr_post_ends = i
                curr_postures.append(f"{curr_post} ({row.Seconds} - {curr_post_ends})")
                break

        # Determines start/end indexes of next posture ---------------
        for i in range(row.Seconds, len(postures)):
            if postures[i] != curr_post and postures[i] != 'other':
                next_post = postures[i]
                next_post_start = i
                break

        for j in range(i, len(postures)):
            if postures[j] != next_post:
                next_post_end = j
                next_postures.append(f"{next_post} ({next_post_start} - {next_post_end})")
                break

        # Determines previous posture and indexes ------------
        for i in range(row.Seconds):
            if postures[row.Seconds - i] != curr_post:
                prev_post = postures[row.Seconds - i]
                prev_post_end = row.Seconds - i
                break

        prev_post_start = None
        for j in range(prev_post_end):
            if postures[prev_post_end - j] != prev_post:
                prev_post_start = prev_post_end - j
                prev_postures.append(f"{prev_post} ({prev_post_start} - {prev_post_end})")
                break
        if prev_post_start is None:
            prev_post_start = 0

        # Finds next posture that is not next_post or 'sitstand' -------
        for i in range(row.Seconds+1, len(postures)):
            if postures[i] != curr_post and postures[i] != 'sitstand':
                next_diff = postures[i]
                next_diff_post_start = i
                next_diff_postures.append(f"{next_diff} ({next_diff_post_start})")
                break

        # Index of next transition or posture end
        end_ind = min([next_transition, next_post_end])

        if not quiet:
            try:
                pp = prev_post_start
            except:
                pp = None
            print(
                f"Curr={curr_post}({row.Seconds}), prev={prev_post}({prev_post_start}-{prev_post_end}), "
                f"next={next_post}({next_post_start}-{next_post_end}), "
                f"next diff={next_diff}({next_diff_post_start}), STS={row.Type}, "
                f"nextSTS={next_t_type}({next_transition})")

        # Apply some logic to dis =================================================================================

        # If currently standing ------------------------------
        if curr_post == 'stand':
            # Stand-to-sit transitions -------
            if row.Type == 'Stand-sit':
                if next_post == 'sit' or next_post == 'sitstand':
                    # postures[row.Seconds:min([next_transition, next_diff_post_start])] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for
                         x in range(row.Seconds, min([next_transition, next_diff_post_start]))]
                    postures[row.Seconds:min([next_transition, next_diff_post_start])] = r

                if next_post == 'sitstand' and next_diff == 'sit':
                    # postures[row.Seconds:end_ind] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, end_ind)]
                    postures[row.Seconds:end_ind] = r

            # Sit-to-stand transitions ------
            if row.Type == 'Sit-stand':
                if prev_post == 'sitstand':
                    # postures[prev_post_start:row.Seconds] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_start, row.Seconds)]
                    postures[prev_post_start:row.Seconds] = r

        # If currently unsure if sitstand ----------------------
        if curr_post == 'sitstand':
            # Stand-to-sit transitions ------
            if row.Type == 'Stand-sit':
                # if next_post == 'sit':
                #    postures[prev_post_end:next_post_start] = 'stand'

                if next_post == 'sit':
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_start)]
                    postures[row.Seconds:next_post_start] = r
                    postures[prev_post_end:row.Seconds] = 'stand'

                if prev_post == 'sit':
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_end, row.Seconds)]
                    postures[prev_post_end:row.Seconds] = r

                    postures[row.Seconds:next_post_start] = 'stand'

                if prev_post == 'stand':
                    if next_post in ['other', 'sitstand']:
                        r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_start)]
                        postures[row.Seconds:next_post_start] = r

                if next_post == 'stand':
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_start)]
                    postures[row.Seconds:next_post_start] = r

            # Sit-to-stand transitions -------
            if row.Type == 'Sit-stand':
                if next_post == 'sit' and prev_post == 'sit':
                    postures[row.Seconds:curr_post_ends] = 'stand'

                if prev_post == 'sit':
                    # postures[prev_post_end:row.Seconds] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_end, row.Seconds)]
                    postures[prev_post_end:row.Seconds] = r

                    postures[row.Seconds:next_post_start] = 'stand'

        # If currently sitting ------------------------------
        if curr_post == 'sit':
            # Sit-to-stand transitions -------
            if row.Type == 'Sit-stand':
                if next_post == 'sitstand':
                    postures[row.Seconds:next_post_end] = 'stand'

            if row.Type == 'Stand-sit':
                if prev_post == 'stand' and prev_post_end - row.Seconds <= 3:
                    if next_post == 'sitstand':
                        # postures[row.Seconds:next_post_end] = 'sit'
                        r = ['sit' if gait_mask[x] == 0 else postures[x] for x in
                             range(row.Seconds, min([next_transition, next_post_end]))]
                        postures[row.Seconds:min([next_transition, next_post_end])] = r

        if curr_post in lying:
            if next_post == 'stand':
                postures[curr_post_ends:next_post_start] = 'sit' if row.Type == 'Stand-sit' else 'stand'
            if row.Type == 'Stand-sit' and next_post == 'sitstand':
                # postures[row.Seconds:next_post_end] = 'sit'
                r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_end)]
                postures[row.Seconds:next_post_end] = r

        if curr_post == 'other':
            if row.Type == 'Sit-stand' and next_post in ['sitstand', 'stand']:
                # postures[prev_post_end:row.Seconds] = 'sit'
                r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_start, row.Seconds)]
                postures[prev_post_end:row.Seconds] = r

                postures[row.Seconds:next_post_start] = 'stand'

                if next_post == 'sitstand' and next_post_start < next_transition:
                    postures[next_post_start:next_post_end] = 'stand'

    # Removes single-epoch events
    for i in range(len(postures)-2):
        if postures[i] != postures[i+1] and postures[i+1] != postures[i+2]:
            postures[i+1] = postures[i]

    l = min([len(df_transitions['Seconds']), len(prev_postures), len(curr_postures), len(next_postures),
            len(curr_sts), len(next_sts), len(next_diff_postures)])
    df_out = pd.DataFrame({"Seconds": df_transitions['Seconds'][:l],
                           "Previous": prev_postures[:l], "Current": curr_postures[:l], "Next": next_postures[:l],
                           "STS": curr_sts[:l], "Next STS": next_sts[:l], "NextDiff": next_diff_postures[:l]})
    return postures, df_out


def final_logic(postures, df_transitions, split_difference=False):
    """Another layer of logic."""

    postures = postures.copy()
    gait_mask = np.array(df1s['chest_gait_mask'])
    n_sitstand = len([i for i in postures if i == 'sitstand'])

    curr_ind = 0
    for i in range(len(postures)):
        if i > curr_ind:

            # Dealing with sitstand postures -------
            if postures[i] == 'sitstand':
                curr_post = 'sitstand'

                for j in range(i+1, len(postures)):
                    if postures[j] != 'sitstand':
                        # Crops df_transitions to span from curr_post to j
                        t = df_transitions.loc[(df_transitions['Seconds'] >= i) & (df_transitions['Seconds'] <= j)]

                        print(i, j, postures[j], list(t['Seconds']), list(t['Type']))
                        # Logic if transitions are found before posture change
                        start_idx = i

                        for row in t.itertuples():
                            # Deals with sit-stand transitions -------
                            if row.Type == 'Sit-stand':
                                # Flags i to transition as sitting if not known standing
                                for x in range(start_idx, row.Seconds+1):
                                    # Makes sure gait isn't flagged as sitting
                                    postures[x] = 'sit' if gait_mask[x] == 0 else postures[x]

                            # Deals with stand-sit transitions
                            if row.Type == 'Stand-sit':
                                # postures[row.Seconds:j] = 'sit'
                                postures[i:j] = 'sit'

                            start_idx = row.Seconds

                        # Deals with time period between final transition in window and j ---------
                        if t.shape[0] > 0:
                            postures[row.Seconds:j] = 'sit' if row.Type == 'Sit-stand' else 'stand'

                        # Logic if no transitions found -------------------------------------------
                        if t.shape[0] == 0:
                            prev_post = postures[i-1]
                            next_post = postures[j]

                            if curr_post == 'sitstand':
                                # postures are sit, flags all non-gait as sitting
                                if prev_post == 'sit' and next_post == 'sit':
                                    for x in range(i, j+1):
                                        postures[x] = 'sit' if gait_mask[x] == 0 else 'stand'

                                # postures are standing, all become standing
                                if prev_post == 'stand' and next_post == 'stand':
                                    postures[i:j] = 'stand'

                                # if previous is sitting and next is standing:
                                if prev_post == 'sit' and next_post == 'stand':
                                    # Splits sitstand event in half if split_difference
                                    if split_difference:
                                        postures[i:i+int((j-i)/2)] = 'sit'
                                        postures[i+int((j-i)/2):j] = 'stand'
                                    # if not split difference, sitstand becomes previous posture (sit)
                                    if not split_difference:
                                        postures[i:j] = 'sit'

                                # if previous is stand and next is sit:
                                if prev_post == 'stand' and next_post == 'sit':
                                    # Splits sitstand event in half
                                    if split_difference:
                                        postures[i:i+int((j-i)/2)] = 'stand'
                                        postures[i+int((j-i)/2):j] = 'sit'
                                    if not split_difference:
                                        postures[i:j] = 'stand'

                                # if sitstand transitions to lying with no STS detected
                                if prev_post == 'sit' and next_post in lying:
                                    postures[i:j] = 'sit'

                                # if sitstand transitions from lying to standing/sitting with no STS detected
                                if prev_post in lying and (next_post == 'sit' or next_post == 'stand'):
                                    postures[i:j] = prev_post

                                # if standing to lying with no STS transition detected
                                if prev_post == 'stand' and next_post in lying:
                                    postures[i:j] = 'stand'

                                # Flags 'other' as next posture if next_post is sit or stand
                                if prev_post == 'other' and next_post in ['sit', 'stand']:
                                    postures[i:j] = next_post

                            # If no transitions found and 'other' surrounded by same posture, coded as said posture
                            if curr_post == 'other':
                                if prev_post == next_post:
                                    postures[i:j] = next_post

                        curr_ind = j
                        break

    print(f"-Went from {n_sitstand} ({round(100*n_sitstand/len(postures), 2)}%) to "
          f"{len([i for i in postures if i == 'sitstand'])} 'sitstands' "
          f"({round(100*len([i for i in postures if i == 'sitstand'])/len(postures), 2)}%)")

    return postures


def fill_other(postures):
    """If 'other' posture remains and is surrounded by same posture, gets flagged as that other posture."""

    postures = postures.copy()

    curr_ind = 0
    for i in range(len(postures)):
        if i > curr_ind:

            if postures[i] == 'other':

                for j in range(i + 1, len(postures)):
                    if postures[j] != 'other':
                        next_post = postures[j]
                        curr_ind = j
                        break
                for k in range(i):
                    if postures[i-k] != 'other':
                        prev_post = postures[i-k]
                        break

                postures[i:j] = next_post

    return postures


def final_logic2(postures, df_transitions):

    postures = postures.copy()
    gait_mask = np.array(df1s['chest_gait_mask'])
    sts_indexes = np.array(df_transitions['Seconds'])
    sts_types = np.array(df_transitions['Type'])

    curr_ind = 0
    # Loops through all postures
    for i in range(1, len(postures)):
        if i > curr_ind:

            # If posture is sitstand --------------------------------------------
            if postures[i] == 'sitstand':
                curr_post = 'sitstand'
                prev_post = postures[i-1]

                # Finds next posture's start time --------
                for j in range(i+1, len(postures)):
                    if postures[j] != curr_post:
                        print(i, j, postures[i], postures[j])

                        # t = df_transitions.loc[(df_transitions['Seconds'] >= i) & (df_transitions['Seconds'] <= j)]
                        next_post = postures[j]
                        t = [x for x in sts_indexes if i <= x < j]

                        """start_ind = i
                        for transition in t:
                            t_index = np.argwhere(sts_indexes == t)[0][0]
                            t_type = sts_types[t_index]

                            if t_type == 'Sit-stand':
                                if prev_post == 'sit' and next_post in ['sitstand', 'stand']:
                                    postures[i:t_index] = "stand"

                            start_ind = t_index"""

                    curr_ind = j
                    break

    # return postures


# final_logic2(postures=df1s['v3'], df_transitions=df_sts2)


def check_accuracy(test_list, crop_start=0, version_name=''):

    df_crop_ind = 0
    for i in range(df1s.shape[0]):
        if df1s.iloc[-i]['GS'] != 'other':
            df_crop_ind = df1s.shape[0] - i
            break

    df_check = df1s.iloc[crop_start:df_crop_ind]
    alg = test_list[crop_start:df_crop_ind]

    values = []

    for gs, a in zip(df_check['GS'], alg):
        if gs != 'other':
            values.append(gs == a)

    print(f"\n============ {version_name} ACCURACY ===========")
    print(f"-Ignoring first {crop_start} seconds (pre-gait), and final {df1s.shape[0] - df_crop_ind} seconds (not useful)")
    print(f"-Accuracy = {round(100 * values.count(True) / len(values), 1)}% ({values.count(True)}/{len(values)})")


def plot_posture_comparison(show_transitions=True, show_v0=True, show_v1=True, show_v2=True, collapse_lying=True,
                            show_v3=True, show_v4=True, show_gs=True, first_walk_index=0, use_timestamps=True):

    fig, ax = plt.subplots(7, sharex='col', figsize=(10, 9),
                           gridspec_kw={"height_ratios": [1, .67, .67, .33, .2, .67, .67]})

    if show_transitions:
        plt.suptitle("Red = stand-to-sit; green = sit-to-stand")

    start_ts = df1s.iloc[0]['timestamp']
    lying = ['leftside', 'rightside', 'prone', 'supine']

    if show_gs:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['GS'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['GS']],
                   color='black', label="GS", zorder=1, linestyle='dashed')
    if show_v0:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['posture'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['posture']],
                   color='dodgerblue', label='Original', zorder=0)

    if show_v1:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v1'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v1']],
                   color='limegreen', label='V1', zorder=0)

    if show_v2:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v2'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v2']],
                   color='fuchsia', label='v2', zorder=0)
    if show_v3:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v3'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v3']],
                   color='purple', label='v3', zorder=0)
    if show_v4:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v4'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v4']],
                   color='orange', label='v4', zorder=0)

    ax[0].scatter(df1s.loc[df1s['v4'] == 'sitstand']['timestamp'] if use_timestamps else
                  df1s.loc[df1s['v4'] == 'sitstand'].index,
                  df1s.loc[df1s['v4'] == 'sitstand']['v4'], marker='o', color='red')

    ylim = ax[0].get_ylim()
    ax[0].fill_between(x=[0 if not use_timestamps else start_ts,
                          first_walk_index if not use_timestamps else start_ts + td(seconds=first_walk_index)],
                       y1=-1, y2=np.ceil(ylim[1]), color='grey', alpha=.35, label='Pre-gait')
    ax[0].set_ylim(ylim)

    ax[0].legend(loc='lower right')

    if show_transitions:
        for row in df_sts2.itertuples():
            ax[0].axvline(row.Seconds if not use_timestamps else start_ts + td(seconds=row.Seconds),
                          color='limegreen' if row.Type == 'Sit-stand' else 'red',
                          linestyle='dashed', lw=1.5)

    ax[0].grid()

    ax[1].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['chest_anterior'],
               color='black', label='Chest_Ant')
    ax[1].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['chest_up'],
               color='red', label='Chest_Up')
    ax[1].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['chest_left'],
               color='dodgerblue', label='Chest_Left')
    ax[1].set_yticks([0, 90, 180])
    ax[1].set_ylabel("Deg.")
    ax[1].grid()
    ax[1].legend(loc='lower right')

    ax[2].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['ankle_anterior'], color='black', label='ankle_anterior')
    ax[2].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['ankle_up'], color='red', label='ankle_up')
    ax[2].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['ankle_left'], color='dodgerblue', label='ankle_left')
    ax[2].set_yticks([0, 90, 180])
    ax[2].set_ylabel("Deg.")
    ax[2].grid()
    ax[2].legend(loc='lower right')

    try:
        ax[3].plot(chest_temp_ts if use_timestamps else np.arange(0, len(chest_acc.signals[chest_acc_indexes['Temperature']]))/chest_temp_fs,
                   chest_acc.signals[chest_acc_indexes['Temperature']], color='black', label='Chest')
        ax[3].plot(ank_temp_ts if use_timestamps else np.arange(0, len(ank_acc.signals[ank_acc_indexes['Temperature']]))/ank_temp_fs,
                   ank_acc.signals[ank_acc_indexes['Temperature']], color='red', label='Ankle')
        ax[3].legend(loc='lower right')
        ax[3].grid()
    except:
        pass

    ax[3].set_ylabel("Deg. C")

    ax[4].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]), df1s['chest_gait_mask'], color='orange', label='Gait')
    ax[4].legend(loc='lower right')

    x = post.chest['Anterior'] if not converted_to_mss else post.chest['Anterior']/9.81
    y = post.chest['Up'] if not converted_to_mss else post.chest['Up']/9.81
    z = post.chest['Left'] if not converted_to_mss else post.chest['Left']/9.81
    ts = pd.date_range(start=post.chest['start_stamp'], periods=len(x), freq="{}ms".format(1000/post.chest['sample_rate']))

    ax[5].plot(ts if use_timestamps else np.arange(len(post.chest['Anterior'])) / chest['sample_rate'],
               x, color='black', label='chest_anterior')
    ax[5].plot(ts if use_timestamps else np.arange(len(post.chest['Anterior']))/chest['sample_rate'],
               y, color='red', label='chest_up')
    ax[5].plot(ts if use_timestamps else np.arange(len(post.chest['Anterior']))/chest['sample_rate'],
               z, color='dodgerblue', label='chest_left')
    ax[5].legend(loc='lower right')
    ax[5].set_ylabel("G")
    ax[5].grid()

    ts = pd.date_range(start=post.ankle['start_stamp'], periods=len(post.ankle['Anterior']),
                       freq="{}ms".format(1000/post.ankle['sample_rate']))
    ax[6].plot(ts if use_timestamps else np.arange(len(post.ankle['Anterior']))/ankle['sample_rate'],
               post.ankle['Anterior'], color='black', label='ankle_anterior')
    ax[6].plot(ts if use_timestamps else np.arange(len(post.ankle['Up']))/ankle['sample_rate'],
               post.ankle['Up'], color='red', label='ankle_up')
    ax[6].plot(ts if use_timestamps else np.arange(len(post.ankle['Left']))/ankle['sample_rate'],
               post.ankle['Left'], color='dodgerblue', label='ankle_left')
    ax[6].legend(loc='lower right')
    ax[6].set_ylabel("G")
    ax[6].grid()

    if not use_timestamps:
        ax[-1].set_xlabel("Seconds")
        ax[-1].set_xlim(0, df1s.shape[0])
    if use_timestamps:
        ax[-1].xaxis.set_major_formatter(xfmt)
        ax[-1].set_xlim(df1s['timestamp'].iloc[0], df1s['timestamp'].iloc[-1])

    plt.tight_layout()
    plt.subplots_adjust(hspace=.15)

    return fig, ax


# df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=None)

converted_to_mss = convert_accel_units(to_mss=True, to_g=False)

"""
sit_to_stand, stand_to_sit, ignored_peaks = process_sts_transfers(anterior=post.chest['Anterior'],
                                                                  vertical=post.chest['Up'],
                                                                  left=post.chest['Left'],
                                                                  sample_rate=post.chest['sample_rate'],
                                                                  show_plot=False, quiet=False)
df_sts = format_sts_data()
"""

df1s['posture'] = remove_single_second_postures(df1s['posture'])

df_sitst = process_sit2standpy(data=np.array([post.chest['Anterior'], post.chest['Up'], post.chest['Left']]).transpose(),
                               sample_rate=post.chest['sample_rate'], start_stamp=post.chest['start_stamp'],
                               stand_sit=False)
df_stsit = process_sit2standpy(data=np.array([post.chest['Anterior'], post.chest['Up'], post.chest['Left']]).transpose(),
                               sample_rate=post.chest['sample_rate'], start_stamp=post.chest['start_stamp'],
                               stand_sit=True)

# df_sts = df_sts.loc[df_sts['Type'] == 'Stand-sit'].append(df_sitst)
df_sts = df_stsit.append(df_sitst)
df_sts = df_sts.sort_values("Seconds").reset_index(drop=True)
df_sts2 = remove_transitions_during_gait(gait_mask=df1s['chest_gait_mask'], df_transitions=df_sts, pad_len=1)
df_sts2 = remove_toosoon_transitions(df_transition=df_sts2, n_seconds=3)  # 4

df1s['v1'] = fill_between_walks(postures=df1s['posture'], gait_mask=df1s['chest_gait_mask'], df_transitions=df_sts2, max_break=10)

# first_walk_index = int((df_gait.loc[df_gait['start_timestamp'] >= post.ankle['start_stamp']].iloc[0]['start_timestamp'] - post.ankle['start_stamp']).total_seconds())

df1s['v2'], df_logic1 = apply_logic(df_transitions=df_sts2, gait_mask=df1s['chest_gait_mask'], first_walk_index=first_walk_index, postures=df1s['v1'], first_pass=True, quiet=False)
df1s['v3'], df_logic2 = apply_logic(df_transitions=df_sts2, gait_mask=df1s['chest_gait_mask'], first_walk_index=first_walk_index, postures=df1s['v2'].copy(), first_pass=False, quiet=False)
# df1s['v4'] = final_logic2(postures=df1s['v3'].copy(), split_sitstand_anomalies=False)
df1s['v4'] = final_logic(postures=df1s['v3'], df_transitions=df_sts2)
df1s['v4'] = fill_other(postures=df1s['v4'])

fig = plot_posture_comparison(show_transitions=True, show_v0=False, show_v2=False, show_v3=False, show_v4=True, show_gs=True, use_timestamps=False, collapse_lying=False)
# fig = plot_posture_comparison(show_transitions=True, show_v0=True, show_v2=False, show_v3=False, show_v4=True, show_gs=False, use_timestamps=False, collapse_lying=False)
# check_accuracy(test_list=df1s['v1'], version_name="V1", crop_start=first_walk_index)
# check_accuracy(test_list=df1s['v2'], version_name="V2", crop_start=df1s.loc[df1s['GS'] != 'other'].iloc[0]['index']+1)
# check_accuracy(test_list=df1s['v3'], version_name="V3", crop_start=df1s.loc[df1s['GS'] != 'other'].iloc[0]['index']+1)
# check_accuracy(test_list=df1s['v4'], version_name="V4", crop_start=df1s.loc[df1s['GS'] != 'other'].iloc[0]['index']+1)

# TODO
# Remove unnecessary columns in df1s when actually running code
# removed 'sit' classification from NWPosture2
# need to stop overwriting walking with flawed logic
# final final layer of logic to remove remaining sitstand if no STS and surrounded by same posture
    # see GS001 ~ 1000 seconds
    # final_logic() function
