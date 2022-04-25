import nwdata
import pandas as pd
import os
import numpy as np


def import_ax_data(filename, use_accel=True):
    """Imports and formats Axivity data from .cwa or .EDF file. Epoches as sum of vector magnitudes.

        arguments:
        -filename: pathway to .cwa/.EDF file
        -epoch_len: number of seconds over which IMU magnitude values are summed
        -use_accel: boolean --> if True, will load/use accelerometer data, gyroscope if False
    """

    x = nwdata.NWData()

    print(f"\nImporting {filename}...")

    # Uses correct function to load in file based on file type
    if filename.split(".")[-1] in ['CWA', 'cwa', 'Cwa']:
        x.import_axivity(file_path=filename)
    if filename.split(".")[-1] in ['EDF', 'edf', 'Edf']:
        x.import_edf(file_path=filename)

    # channel indexes for accelerometer/gyro data
    sig_idx = {'ax': x.get_signal_index('Accelerometer x'),
               'ay': x.get_signal_index('Accelerometer y'),
               'az': x.get_signal_index('Accelerometer z'),
               'gx': x.get_signal_index('Gyroscope x'),
               'gy': x.get_signal_index('Gyroscope y'),
               'gz': x.get_signal_index('Gyroscope z')}

    if use_accel:
        prefix = "a"
    if not use_accel:
        prefix = "g"

    fs = x.signal_headers[sig_idx[f'{prefix}x']]['sample_rate']

    print("-Loaded {} data ({}Hz)".format('accelerometer' if use_accel else "gyroscope", fs))

    print("-Generating timestamps...")
    timestamps = pd.date_range(start=x.header['start_datetime'], periods=len(x.signals[sig_idx[f'{prefix}x']]),
                               freq="{}ms".format(round(1000/fs, 6)))

    return x, fs, timestamps


def import_gait_bouts(gait_filepath, start_stamp, coll_dur):
    """Imports and formats gait bout tabular data and creates gait mask in 1-s increments.

       arguments:
       -gait_filepath: full pathway
       -start_stamp: timestamp when IMU collection begins
        -file_dur: collection duration in seconds

       returns:
       -df for gait bouts, gait mask array
    """

    if os.path.exists(gait_filepath):
        print("\nImporting gait data...")

        df_gait = pd.read_csv(gait_filepath)
        df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
        df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])

        gait_mask = np.zeros(int(np.ceil(coll_dur)))

        for row in df_gait.itertuples():
            start = int((row.start_timestamp - start_stamp).total_seconds())
            end = int((row.end_timestamp - start_stamp).total_seconds())
            gait_mask[start:end] = 1

    if not os.path.exists(gait_filepath):
        print("-Gait data not found --> flagging everything as not walking")
        df_gait = None
        gait_mask = None

    return df_gait, gait_mask


def import_sleep_bouts(sleep_filepath, coll_dur, start_stamp):
    """Imports and formats sleep bout tabular data and creates sleep mask in 1-s increments.

       arguments:
       -sleep_filepath: full pathway
       -start_stamp: timestamp when IMU collection begins
       -coll_dur: collection duration in seconds

       returns:
       -df for sleep bouts, sleep mask array
    """

    if os.path.exists(sleep_filepath):
        print("\nImporting sleep data...")

        df_sleep = pd.read_csv(sleep_filepath)
        df_sleep['start_timestamp'] = pd.to_datetime(df_sleep['start_time'])
        df_sleep['end_timestamp'] = pd.to_datetime(df_sleep['end_time'])

        sleep_mask = np.zeros(coll_dur)
        for row in df_sleep.itertuples():
            start = int((row.start_timestamp - start_stamp).total_seconds())
            end = int((row.end_timestamp - start_stamp).total_seconds())
            sleep_mask[start:end] = 1

    if not os.path.exists(sleep_filepath):
        print("-Sleep data not found.")
        df_sleep = None
        sleep_mask = None

    return df_sleep, sleep_mask


def import_nonwear_data(nw_filepath, start_stamp, coll_dur):

    nw_mask = np.zeros(coll_dur)
    df_nw = None

    if not os.path.exists(nw_filepath):
        print("-Nonwear file not found --> flagging everything as 'wear'")

    if os.path.exists(nw_filepath):
        print("\nImporting nonwear data...")
        start_stamp = pd.to_datetime(start_stamp)

        df_nw = pd.read_csv(nw_filepath)
        df_nw['start_timestamp'] = pd.to_datetime(df_nw['start_time'])
        df_nw['end_timestamp'] = pd.to_datetime(df_nw['end_time'])

        for row in df_nw.itertuples():
            start = int((row.start_timestamp - start_stamp).total_seconds())
            end = int((row.end_timestamp - start_stamp).total_seconds())
            nw_mask[start:end] = 1

    return df_nw, nw_mask


def create_df_mask(coll_dur, start_stamp, gait_filepath, sleep_filepath, nw_filepath):
    """Calls functions to import gait, sleep, and activity data. Combines masks into single DF.

       arguments:
       -sample_rate: ECG sample rate, Hz
       -max_i: length of ECG signal
       -start_stamp: ECG start timestamp
       -the rest are obvious (see individual import functions if not)

       returns individual DFs and unified gait mask DF
    """

    df_gait, gait_mask = import_gait_bouts(gait_filepath=gait_filepath, start_stamp=start_stamp, coll_dur=coll_dur)

    df_sleep, sleep_mask = import_sleep_bouts(sleep_filepath=sleep_filepath, coll_dur=coll_dur, start_stamp=start_stamp)

    df_nw, nw_mask = import_nonwear_data(nw_filepath=nw_filepath, start_stamp=start_stamp, coll_dur=coll_dur)

    df_mask = pd.DataFrame({"seconds": np.arange(1, coll_dur + 1),
                            'timestamp': pd.date_range(start=start_stamp, periods=coll_dur, freq='1S'),
                            'gait': gait_mask if gait_mask is not None else np.zeros(coll_dur),
                            'sleep': sleep_mask if sleep_mask is not None else np.zeros(coll_dur),
                            'nw': nw_mask if nw_mask is not None else np.zeros(coll_dur)})

    return df_mask, df_gait, df_sleep, df_nw
