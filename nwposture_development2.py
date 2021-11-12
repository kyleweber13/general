import pandas as pd
# from nwposture_dev.nwposture import NWPosture
from nwposture_dev.nwposture.NWPosture2 import NWPosture, plot_all
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
import os
import scipy

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
# folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/001_Pilot/Converted/"
# chest_file = folder + "001_chest_AX6_6014664_Accelerometer.edf"
# ankle_file = folder + "001_left ankle_AX6_6014408_Accelerometer.edf"
# log_file = "C:/Users/ksweber/Desktop/PosturePilot001_EventLog.xlsx"

folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/002/Converted/"
chest_file = folder + "002_Axivity_Chest_Accelerometer.edf"
ankle_file = folder + "002_Axivity_LAnkle_Accelerometer.edf"
log_file = "C:/Users/ksweber/Desktop/PosturePilot002_EventLog.xlsx"

df_event, start_time_gs, stop_time_gs = import_goldstandard_df(filename=log_file)

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
la_acc = nwdata.NWData()
la_acc.import_edf(file_path=ankle_file, quiet=False)
la_acc_indexes = {"Acc_x": la_acc.get_signal_index("Accelerometer x"),
                  "Acc_y": la_acc.get_signal_index("Accelerometer y"),
                  "Acc_z": la_acc.get_signal_index("Accelerometer z")}
la_fs = la_acc.signal_headers[la_acc_indexes["Acc_x"]]['sample_rate']
la_ts = pd.date_range(start=la_acc.header['startdate'],
                      periods=len(la_acc.signals[la_acc_indexes['Acc_x']]), freq="{}ms".format(1000/la_fs))

# Axivity chest orientation
chest = {"Anterior": chest_acc.signals[chest_acc_indexes['Acc_z']]*-1,
         "Up": chest_acc.signals[chest_acc_indexes['Acc_x']],
         "Left": chest_acc.signals[chest_acc_indexes['Acc_y']],
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}

ankle = {"Anterior": la_acc.signals[la_acc_indexes['Acc_y']],
         "Up": la_acc.signals[la_acc_indexes['Acc_x']],
         "Left": la_acc.signals[la_acc_indexes['Acc_z']],
         "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}

post = NWPosture(chest_dict=chest, ankle_dict=ankle,
                 gait_bouts=df_event.loc[df_event['EventShort'] == "Walking"],
                 study_code='OND09', subject_id='Test', coll_id='01')
post.crop_data()
post.df_gait = post.load_gait_data(df_event.loc[df_event['EventShort'] == "Walking"])
post.gait_mask = post.create_gait_mask()
df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=df_event)
# plot_all(df1s, df_peaks=df_peaks, show_v0=False, show_v1=False, show_v2=False, show_v3=True, collapse_lying=False)
# bouts.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture.csv", index=False)
# df1s.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture_Epoch1s.csv", index=False)


""" ==================================================== HANDDS =================================================== """

"""
subj = '0001'
# for subj in ['0001', '0007', '0008']:
folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"
ankle_file = folder + f"OND09_{subj}_01_AXV6_LAnkle.edf"
if not os.path.exists(ankle_file):
    ankle_file = folder + f"OND09_{subj}_01_AXV6_RAnkle.edf"

chest_file = folder + f"OND09_{subj}_01_BF36_Chest.edf"

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
la_acc = nwdata.NWData()
la_acc.import_edf(file_path=ankle_file, quiet=False)
la_acc_indexes = {"Acc_x": la_acc.get_signal_index("Accelerometer x"),
                  "Acc_y": la_acc.get_signal_index("Accelerometer y"),
                  "Acc_z": la_acc.get_signal_index("Accelerometer z")}
la_fs = la_acc.signal_headers[la_acc_indexes["Acc_x"]]['sample_rate']
la_ts = pd.date_range(start=la_acc.header['startdate'],
                      periods=len(la_acc.signals[la_acc_indexes['Acc_x']]), freq="{}ms".format(1000/la_fs))

# Bittium orientation
chest = {"Anterior": chest_acc.signals[chest_acc_indexes['Acc_z']]*.001,
         "Up": chest_acc.signals[chest_acc_indexes['Acc_x']]*.001,
         "Left": chest_acc.signals[chest_acc_indexes['Acc_y']]*.001,
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}

# Actually lateral left OR medial right ankle
if 'LAnkle' in ankle_file:
    ankle = {"Anterior": la_acc.signals[la_acc_indexes['Acc_y']],
             "Up": la_acc.signals[la_acc_indexes['Acc_x']],
             "Left": la_acc.signals[la_acc_indexes['Acc_z']],
             "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}

# Actually lateral right OR medial left ankle
if 'RAnkle' in ankle_file:
    # Right ankle
    ankle = {"Anterior": la_acc.signals[la_acc_indexes['Acc_y']]*-1,
             "Up": la_acc.signals[la_acc_indexes['Acc_x']],
             "Left": la_acc.signals[la_acc_indexes['Acc_z']]*-1,
             "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}

df_gait = pd.read_csv(f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv")
df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])

post = NWPosture(chest_dict=chest, ankle_dict=ankle, gait_bouts=df_gait, study_code='OND09', subject_id=subj, coll_id='01')

post.crop_data()
post.df_gait = post.load_gait_data(df_gait)
post.gait_mask = post.create_gait_mask()

# df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=None)

# plot_all(df1s, df_peaks=None, show_v0=False, show_v1=False, show_v2=True, show_v3=True, collapse_lying=False)
# bouts.to_csv(f"C:/users/ksweber/Desktop/Posture/OND09_{subj}_PostureBouts.csv", index=False)
# df1s.to_csv(f"C:/users/ksweber/Desktop/Posture/OND09_{subj}_Posture_Epoch1s.csv", index=False)
# df_peaks.to_csv(f"C:/users/ksweber/Desktop/Posture/OND09_{subj}_Posture_STSPeaks.csv", index=False)
"""

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


# Conversion to m/s
post.chest['Anterior'] *= 9.81
post.chest['Anterior'] = post.chest['Anterior'][:200000]
post.chest['Up'] *= 9.81
post.chest['Up'] = post.chest['Up'][:200000]
post.chest['Left'] *= 9.81
post.chest['Left'] = post.chest['Left'][:200000]

chest_ts = chest_ts[:200000]

# Acceleration magnitudes
a = np.sqrt(np.square(np.array([post.chest['Anterior'], post.chest['Left'], post.chest['Up']])).sum(axis=0))
a = np.abs(a)

# 5Hz lowpass
a_filtered = np.abs(filter_signal(data=a, low_f=.025, high_f=3, filter_order=4, sample_f=post.chest['sample_rate'], filter_type='bandpass'))

# .25s rolling mean and SD
a_rm = [np.mean(a_filtered[i:i + int(post.chest['sample_rate'] / 4)]) for i in range(len(a_filtered))]
a_rsd = [np.std(a_filtered[i:i + int(post.chest['sample_rate'] / 4)]) for i in range(len(a_filtered))]

# Calculates jerk for rolling mean and rolling SD
j_rm = np.abs([(j-i)/(1/post.chest['sample_rate']) for i, j in zip(a_rm[:], a_rm[1:])])
j_rm = np.append(j_rm, j_rm[-1])
j_rsd = np.abs([(j-i)/(1/post.chest['sample_rate']) for i, j in zip(a_rsd[:], a_rsd[1:])])
j_rsd = np.append(j_rsd, j_rsd[-1])

# Continuous wavelet transform; focus on <.5Hz band
coefs, freqs = pywt.cwt(a_rm, np.arange(1, 65), 'gaus1', sampling_period=1 / post.chest['sample_rate'])
f_mask = (freqs <= .5) & (freqs >= 0)
cwt_power = np.sum(coefs[f_mask, :], axis=0)

# Initial peak detection: minimum 1-sec apart
pp = peakutils.indexes(y=cwt_power, min_dist=post.chest['sample_rate'], thres_abs=True, thres=np.std(cwt_power))

# Binary stillness list if mean/SD values below thresholds
stillness = np.zeros(len(a_rm))
for i in range(len(a_rm)):
    if a_rm[i] < .15 and a_rsd[i] < .1 and j_rm[i] < 2.5 and j_rsd[i] < 3:
        stillness[i] = 1

# Finds indexes for start/stop of stillness periods at least .33-sec long
curr_index = 0
starts = []
stops = []
for i in range(len(stillness) - int(post.chest['sample_rate']*.25)-1):
    if i >= curr_index:
        window = stillness[i:i+int(post.chest['sample_rate']*.3)]

        # sum makes sure window is at least .3-sec long
        if sum(window) == int(post.chest['sample_rate']*.3):
            starts.append(i)

            for j in range(i+1, len(stillness) - int(post.chest['sample_rate']*.3)):
                if stillness[j] == 1:
                    pass
                if stillness[j] == 0:
                    stops.append(j)
                    curr_index = j
                    break

for stop, start in zip(stops[:], starts[1:]):
    if (start - stop) < post.chest['sample_rate']*.3:
        stops.remove(stop)
        starts.remove(start)

# Filtering to estimate gravity component of signal
g_est = np.array([filter_signal(data=post.chest['Anterior'], low_f=.25, filter_order=4, sample_f=post.chest['sample_rate'], filter_type='lowpass'),
                  filter_signal(data=post.chest['Left'], low_f=.25, filter_order=4, sample_f=post.chest['sample_rate'], filter_type='lowpass'),
                  filter_signal(data=post.chest['Up'], low_f=.25, filter_order=4, sample_f=post.chest['sample_rate'], filter_type='lowpass')])
g_est_vm = np.sqrt(np.square(g_est).sum(axis=0))

# Estimates gravity vector for Up and Anterior axes
a_vert = post.chest['Up'] / g_est_vm
a_ant = post.chest['Anterior'] / g_est_vm

# Integrates data to get velocity; needed for detrending
v_vert_all = [0]
v_ant_all = [0]
for i in range(len(a_vert) - 1):
    v_vert_all.append((a_vert[i] * (1 / post.chest['sample_rate']) + a_vert[i]))
    v_ant_all.append((a_ant[i] * (1 / post.chest['sample_rate']) + a_ant[i]))
v_vert_all = filter_signal(data=v_vert_all, sample_f=post.chest['sample_rate'], high_f=.02, filter_type='highpass', filter_order=3)
v_ant_all = filter_signal(data=v_ant_all, sample_f=post.chest['sample_rate'], high_f=.02, filter_type='highpass', filter_order=3)

# Slopes for detrending
# slope_vant_whole = (v_ant_all[-1] - v_ant_all[0])/len(v_ant_all)
# slope_vvert_whole = (v_vert_all[-1] - v_vert_all[0])/len(v_vert_all)

# Fixes peaks to max absolute value within 1 second either direction of initial peak
# pp2 = pp.copy()
# fast_peaks = []
# v_thresh = .2  # m/s


def screen1():
    for peak in pp2:
        ends_static = False  # boolean if window around peak ends still or not

        # limits of windowing around peak: peak - 2 seconds to peak + 30 seconds
        min_index = peak - int(3*post.chest['sample_rate'])  # original value = 2 seconds
        max_index = peak + int(30*post.chest['sample_rate'])

        # finds end(s) of still period within 2 seconds of potential STS peak
        integration_starts = [i for i in stops if min_index <= i <= peak]

        # finds start(s) of active periods within 30 seconds of potential STS peak
        integration_ends = [i for i in starts if peak < i <= max_index]


        # Removes peak if no still periods end within 2 seconds of peak
        # if len(integration_starts) == 0:
        #    pp2.remove(peak)

        # If more than one stillness period ends within 3 seconds of peak...
        if len(integration_starts) > 0:
            # Takes end of still period closest to peak if multiple are found
            integration_start = integration_starts[-1]
            print(peak, chest_ts[peak], integration_start)

            # Takes first start of still period if multiple found
            # Sets STS transition ending as static to true
            if len(integration_ends) > 0:
                integration_end = integration_ends[0]
                ends_static = True

            # If no still periods are found within 30 seconds of peak, changes window to peak + 5 seconds
            if len(integration_ends) == 0:
                integration_end = peak + int(5 * post.chest['sample_rate'])

            integration_window_vert = a_vert[integration_start:integration_end]
            integration_window_ant = a_ant[integration_start:integration_end]

            # Integrate a_vert and a_ant to get vertical velocities in both axes
            v_vert = [0]
            v_ant = [0]
            for i in range(len(integration_window_vert)-1):
                ant_val = (integration_window_ant[i] * (1/post.chest['sample_rate']) + v_ant[i])
                vert_val = (integration_window_vert[i] * (1/post.chest['sample_rate']) + v_vert[i])

                v_vert.append(vert_val)
                v_ant.append(ant_val)

            v_vert_detrend = []
            v_ant_detrend = []

            # If period ends static, detrends using slope within window
            if ends_static:
                slope_vert = (v_vert[-1] - v_vert[0]) / len(v_vert)
                slope_ant = (v_ant[-1] - v_ant[0]) / len(v_ant)

                for i, vi in enumerate(v_vert):
                    v_vert_detrend.append(vi - slope_vert * i)
                    v_ant_detrend.append(v_ant[i] - slope_ant * i)

            # If period does not end static, detrends using slope of all data
            if not ends_static:
                for i in range(len(v_vert)):
                    v_vert_detrend.append(v_vert[i] - slope_vvert_whole * (i + peak - integration_start))
                    v_ant_detrend.append(v_ant[i] - slope_vant_whole * (i + peak - integration_start))

            # Integrate velocity to get displacement
            d_vert = [0]
            d_ant = [0]
            for i in range(len(v_vert_detrend)-1):
                dv = v_vert_detrend[i]*(1/post.chest['sample_rate']) + d_vert[i]
                d_vert.append(dv)

                da = v_ant_detrend[i] * (1 / post.chest['sample_rate']) + d_ant[i]
                d_ant.append(da)

            # Checks if velocities exceed threshold of v_thresh m/s
            # Adds peaks from vertical axis to fast_peaks
            if max(np.abs(v_vert_detrend)) >= v_thresh or max(np.abs(v_ant_detrend)) >= v_thresh:
                # fast_peaks.append(np.argmax(integration_window_vert) + min_index)
                if max(integration_window_vert) > max(integration_window_ant):
                    sit_to_stand.append(np.argmax(integration_window_vert) + min_index)
                if max(integration_window_vert) < max(integration_window_ant):
                    sit_to_stand.append(np.argmax(integration_window_ant) + min_index)


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

        # ONLY RUNS BLOCK OF CODE IS STILL PERIOD END WITHIN 3 SECONDS (SIT-TO-STAND REQUIRES STILLNESS AT START) ----
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

            # Checks if velocities exceed threshold of v_thresh m/s
            # Anterior velocity needs to be negative; vertical is positive
            if max(v_vert_detrend) >= velo_thresh or min(v_ant_detrend) <= -velo_thresh:
                start_ind = window_start

                # Peak index if vertical speed higher than anterior speed
                if max(np.abs(v_vert_detrend)) > max(np.abs(v_ant_detrend)):
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
                        # if v_ant_detrend[i] <= 0 and v_ant_detrend[i + 1] > 0:
                        #    start_ind = i + 1
                        if v_ant_detrend[i] > 0 >= v_ant_detrend[i + 1]:
                            stop_ind = i

                # Movement duration in seconds
                movement_len = (stop_ind - start_ind) / post.chest['sample_rate']

                # counts as peak if less than 4.5 seconds long, vertical displacement >= .125m,
                # and 400ms after previous peak
                if movement_len < 4.5 and max(d_vert) >= .125 and (new_peak - sit_to_stand[-1]) > sample_rate * .4:
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
                print(max(d_vert), min(d_vert))
                if max(d_vert) >= .125 and (new_peak - stand_to_sit[-1]) > sample_rate * .4:
                    stand_to_sit.append(new_peak)
                else:
                    ignored_peaks.append(new_peak)

    return stand_to_sit, ignored_peaks


"""sit_to_stand, stand_to_sit = find_sittostand(possible_peaks=pp, sample_rate=post.chest['sample_rate'],
                                             pad_pre=2.5, velo_thresh=.2,
                                             stillness_starts=starts, stillnes_stops=stops,
                                             anterior_acc=a_ant, vertical_acc=a_vert)"""

stand_to_sit, ignored_peaks = find_standtosit(possible_peaks=stand_to_sit, sample_rate=post.chest['sample_rate'],
                                              pad_post=3, velo_thresh=.2, stillness_starts=starts,
                                              anterior_acc=a_ant, vertical_acc=a_vert)


def plot_results():
    fig, ax = plt.subplots(5, figsize=(12, 9), sharex='col', gridspec_kw={"height_ratios": [.67, .67, 1, .25, .5]})
    plt.suptitle(f"Found {len(pp)} potential peaks; sit-to-stand = {len(sit_to_stand)}, "
                 f"stand-to-sit = {len(stand_to_sit2)}, ignored = {len(ignored_peaks)}")
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

plt.close("all")
plot_results()

# TODO
# Add time threshold between consecutive peaks
