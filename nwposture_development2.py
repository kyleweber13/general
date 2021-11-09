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


"""
folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Posture/Gold standard dataset/Raw data/001_Pilot/Converted/"
chest_file = folder + "001_chest_AX6_6014664_Accelerometer.edf"
ankle_file = folder + "001_left ankle_AX6_6014408_Accelerometer.edf"
log_file = "C:/Users/ksweber/Desktop/PosturePilot001_EventLog.xlsx"
"""

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
df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=df_event)
plot_all(df1s, df_peaks=df_peaks, show_v0=False, show_v1=False, show_v2=False, show_v3=True, collapse_lying=False)
# bouts.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture.csv", index=False)
# df1s.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture_Epoch1s.csv", index=False)


""" ==================================================== HANDDS =================================================== """

subj = '0007'
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


""" ============================================= CODE EXECUTION =================================================="""

crop_sec = 12*3600
# crop_sec = (223300-1) / 100

# Bittium orientation
chest = {"Anterior": chest_acc.signals[chest_acc_indexes['Acc_z']][:int(crop_sec*chest_fs)]*.001,
         "Up": chest_acc.signals[chest_acc_indexes['Acc_x']][:int(crop_sec*chest_fs)]*-.001,
         "Left": chest_acc.signals[chest_acc_indexes['Acc_y']][:int(crop_sec*chest_fs)]*-.001,
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}

"""chest = {"Anterior": chest_acc.signals[chest_acc_indexes['Acc_z']]*-.001,
         "Up": chest_acc.signals[chest_acc_indexes['Acc_x']]*-.001,
         "Left": chest_acc.signals[chest_acc_indexes['Acc_y']]*-.001,
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}"""

# Left ankle
"""ankle = {"Anterior": la_acc.signals[la_acc_indexes['Acc_y']][:int(crop_sec*la_fs)],
         "Up": la_acc.signals[la_acc_indexes['Acc_x']][:int(crop_sec*la_fs)],
         "Left": la_acc.signals[la_acc_indexes['Acc_z']][:int(crop_sec*la_fs)]*-1,
         "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}"""

if 'LAnkle' in ankle_file:
    ankle = {"Anterior": la_acc.signals[la_acc_indexes['Acc_y']],
             "Up": la_acc.signals[la_acc_indexes['Acc_x']],
             "Left": la_acc.signals[la_acc_indexes['Acc_z']]*-1,
             "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}

if 'RAnkle' in ankle_file:
    # Right ankle
    ankle = {"Anterior": la_acc.signals[la_acc_indexes['Acc_y']]*-1,
             "Up": la_acc.signals[la_acc_indexes['Acc_x']],
             "Left": la_acc.signals[la_acc_indexes['Acc_z']],
             "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}

df_gait = pd.read_csv(f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv")
df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])

# df_gait = df_gait.loc[df_gait['end_timestamp'] <= chest_ts[int(crop_sec * chest_fs)]]
# df_gait = df_gait.loc[df_gait['end_timestamp'] <= chest_ts[-1]]

post = NWPosture(chest_dict=chest, ankle_dict=ankle,
                 gait_bouts=df_gait,
                 study_code='OND09', subject_id=subj, coll_id='01')
df1s, bouts, df_peaks = post.calculate_postures(goldstandard_df=None)
plot_all(df1s, df_peaks=df_peaks, show_v0=False, show_v1=False, show_v2=False, show_v3=True, collapse_lying=False)
# bouts.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture.csv", index=False)
# df1s.to_csv(f"C:/users/ksweber/Desktop/OND09_{subj}_Posture_Epoch1s.csv", index=False)


# FIRST HANDDS TO PROCESS: 0001, 0007, 0008
