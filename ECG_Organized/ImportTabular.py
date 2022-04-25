import pandas as pd
import numpy as np
import os
from datetime import timedelta


def import_cn_file(sample_rate, pathway, ecg_signal=None, use_arrs=None, start_time=None, quiet=True, timestamp_method='start',
                   ignore_events=('ST(ref)', "Max. HR", "Min. RR", "Afib Max. HR (total)", 'Min. RR', "Afib Min. HR (total)", "Max. RR", "Min. HR")):
    """Imports Cardiac Navigator arrhythmia event files. Formats timestamps. Removes undesired events.

        arguments:
        -pathway: full pathway to Cardiac Navigator events file
        -sample_rate: sample rate of ECG data, Hz
        -timestamp_method: 'start', 'middle', 'end' for what part of event Cardiac Navigator timestamp represents
        -use_arrs: list of arrhythmias to include. If you don't want to include all arrhythmias, set to None
        -ignore_events: list of useless events such as location of min/max HR, etc. Can leave as-is.
        -ecg_signal: required to calculate voltage range for each event

       returns separate DFs for sinus rhythms and arrhythmia events
    """

    if not quiet:
        print("\nImporting and formatting Cardiac Navigator data...")

    # events we will never care about
    # sinus gets saved as a second df

    # Import csc file
    df_all = pd.read_csv(pathway, delimiter=';')

    # datapoint index creation: ms --> index
    df_all['start_idx'] = [int(i / 1000 * sample_rate) for i in df_all['Msec']]
    df_all['end_idx'] = [row.start_idx + int(row.Length / 1000 * sample_rate) for row in df_all.itertuples()]

    if timestamp_method == 'start':
        pass

    if timestamp_method in ['middle', 'centre', 'center']:
        df_all['start_idx'] = [row.start_idx - int(row.Length/1000/2 * sample_rate) for row in df_all.itertuples()]
        df_all['end_idx'] = [row.start_idx + int(row.Length/1000 * sample_rate) for row in df_all.itertuples()]

    if timestamp_method == 'end':
        df_all['start_idx'] = [row.start_idx + int(row.Length/1000 * sample_rate) for row in df_all.itertuples()]
        df_all['end_idx'] = [row.start_idx + int(row.Length/1000 * sample_rate) for row in df_all.itertuples()]

    # Timestamp formatting ---------
    if start_time is not None:
        start_time = pd.to_datetime(start_time)

        if timestamp_method == 'start':
            df_all['start_timestamp'] = [start_time + timedelta(seconds=i/1000) for i in df_all['Msec']]
            df_all['end_timestamp'] = [start_time + timedelta(seconds=row.Msec/1000 + row.Length/1000) for row in df_all.itertuples()]

        if timestamp_method in ['middle', 'centre', 'center']:
            df_all['start_timestamp'] = [start_time + timedelta(seconds=row.Msec / 1000 - row.Length/1000/2) for row in df_all.itertuples()]
            df_all['end_timestamp'] = [row.start_timestamp + timedelta(seconds=row.Length / 1000) for row in df_all.itertuples()]

        if timestamp_method == 'end':
            df_all['start_timestamp'] = [start_time + timedelta(seconds=row.Msec / 1000 - row.Length / 1000) for row
                                         in df_all.itertuples()]
            df_all['end_timestamp'] = [row.start_timestamp + timedelta(seconds=row.Length / 1000) for row in df_all.itertuples()]

    if start_time is None:
        df_all['start_timestamp'] = [None] * df_all.shape[0]
        df_all['end_timestamp'] = [None] * df_all.shape[0]

    df_all = df_all[["start_idx", "end_idx", 'start_timestamp', 'end_timestamp', "Type"]]

    df_all['duration'] = [(row.end_idx - row.start_idx)/sample_rate for row in df_all.itertuples()]

    # combines multifocal with 'normal' events of same type
    df_all['Type'] = df_all['Type'].replace({"COUP(mf)": "COUP", "GEM(mf)": "GEM", "SALV(mf)": 'SALV', 'VT(mf)': "VT"})

    all_types = df_all['Type'].unique()

    if ecg_signal is not None:
        df_all['abs_volt'] = [max([np.abs(max(ecg_signal[row.start_idx:row.end_idx])),
                                   np.abs(min(ecg_signal[row.start_idx:row.end_idx]))]) for
                              row in df_all.itertuples()]

        # df_all['abs_volt_sd'] = [np.std(np.abs(ecg_signal[row.start_idx:row.end_idx])) for row in df_all.itertuples()]

    # separates sinus rhythm from arrhythmias
    df_sinus = df_all.loc[df_all['Type'] == 'Sinus']

    df_all = df_all.loc[(df_all['Type'] != "Sinus") & (~df_all['Type'].isin(ignore_events))]
    df_cn = df_all.loc[~df_all['Type'].isin(ignore_events)]

    if use_arrs is not None:
        df_cn['Include'] = [row.Type in use_arrs for row in df_cn.itertuples()]
    if use_arrs is None:
        df_cn['Include'] = [True] * df_cn.shape[0]

    df_cn['duration'] = [(row.end_idx - row.start_idx)/sample_rate for row in df_cn.itertuples()]

    remaining_types = df_cn['Type'].unique()
    removed_types = [i for i in all_types if i not in remaining_types]

    if not quiet:
        print(f"-Removed events: {removed_types}")
        print(f"-Remaining events: {remaining_types}")

    if use_arrs is not None:
        df_cn = df_cn.loc[df_cn['Type'].isin(use_arrs)]

    return df_all, df_sinus.reset_index(drop=True), df_cn.reset_index(drop=True)


def import_cn_beats_file(pathway, start_stamp, sample_rate):

        df_rr = pd.read_csv(pathway, delimiter=";")

        # column name formatting
        cols = [i for i in df_rr.columns][1:]
        cols.insert(0, "Msec")
        df_rr.columns = cols

        df_rr['idx'] = [int(i / 1000 * sample_rate) for i in df_rr['Msec']]

        if "Timestamp" not in df_rr.columns:
            df_rr["Timestamp"] = [start_stamp + timedelta(seconds=row.Msec / 1000) for row in df_rr.itertuples()]

        try:
            # Beat type value replacement (no short forms)
            type_dict = {'?': "Unknown", "N": "Normal", "V": "Ventricular", "S": "Premature", "A": "Aberrant"}
            df_rr["Type"] = [type_dict[i] for i in df_rr["Type"]]
        except KeyError:
            pass

        df_rr["HR"] = [60 / (r / 1000) for r in df_rr["RR"]]  # beat-to-beat HR

        df_rr = df_rr[['idx', "Timestamp", "RR", "HR", "Type", "Template", "PP", "QRS", "PQ",
                       "QRn", "QT", "ISO", "ST60", "ST80", "NOISE"]]

        return df_rr


def import_snr_bouts(filepath, sample_rate, snr_signal):
    """Imports Excel file of processed SNR bouts. Calculates min, average, and max voltage values during each event.
       Calculates what percent of file falls into each SNR category.

       -filepath: pathway to bouted Excel file
       -sample_rate: Hz
       -snr_signal: 'raw' SNR signal

       returns:
       -df_bouts: Excel file with more columns
       -quality_totals: df of percentage in each SNR category
    """

    print("\nImporting SNR bout data...")

    df_bouts = pd.read_excel(filepath)
    df_bouts['quality'] = df_bouts['quality'].replace({"Q1": 1, "Q2": 2, "Q3": 3})
    df_bouts['duration'] = [(row.end_idx - row.start_idx) / sample_rate for row in df_bouts.itertuples()]

    min_vals = []
    avg = []
    max_vals = []

    for row in df_bouts.itertuples():
        try:
            min_vals.append(min(snr_signal[row.start_idx:row.end_idx]))
            max_vals.append(max(snr_signal[row.start_idx:row.end_idx]))
            avg.append(np.mean(snr_signal[row.start_idx:row.end_idx]))
        except ValueError:
            min_vals.append(None)
            max_vals.append(None)
            avg.append(None)

    df_bouts['min_snr'] = min_vals
    df_bouts['avg_snr'] = avg
    df_bouts['max_snr'] = max_vals

    quality_totals = pd.DataFrame({"percent": df_bouts.groupby("quality")['duration'].sum() /
                                              df_bouts['duration'].sum() * 100})

    print("-Signal quality summary:")
    for row in quality_totals.itertuples():
        print(f"     -Q{row.Index}: {row.percent:.1f}% of data")

    return df_bouts, quality_totals


def import_gait_bouts(gait_folder, gait_file, start_stamp, sample_rate, max_i):
    """Imports and formats gait bout tabular data and creates gait mask in 1-s increments.

       arguments:
       -gait_folder: pathway to gait bout tabular data
       -gait_file: gait bout filename
       -start_stamp: timestamp when ECG collection begins
       -sample_rate: of ECG signal, Hz
       -max_i: length of ECG signal

       returns:
       -df for gait bouts, gait mask array
    """

    if os.path.exists(gait_folder + gait_file):
        print("\nImporting gait data...")

        df_gait = pd.read_csv(gait_folder + gait_file)
        df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
        df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])

        df_gait['start_idx'] = [int((row.start_timestamp - start_stamp).total_seconds() * sample_rate) for row in df_gait.itertuples()]
        df_gait['end_idx'] = [int((row.end_timestamp - start_stamp).total_seconds() * sample_rate) for row in df_gait.itertuples()]
        df_gait = df_gait.loc[(df_gait['start_idx'] >= 0) & (df_gait['end_idx'] <= max_i)]

        gait_mask = np.zeros(int(max_i / sample_rate))
        for row in df_gait.itertuples():
            start = int(row.start_idx / sample_rate)
            end = int(row.end_idx / sample_rate)
            gait_mask[start:end] = 1

    if not os.path.exists(gait_folder + gait_file):
        print("-Gait data not found --> flagging everything as not walking")
        df_gait = None
        gait_mask = []

    return df_gait, gait_mask


def import_sleep_bouts(sleep_folder, sleep_file, start_stamp, sample_rate, max_i):
    """Imports and formats sleep bout tabular data and creates sleep mask in 1-s increments.

       arguments:
       -sleep_folder: pathway to sleep bout tabular data
       -sleep_file: sleep bout filename
       -start_stamp: timestamp when ECG collection begins
       -sample_rate: of ECG signal, Hz
       -max_i: length of ECG signal

       returns:
       -df for sleep bouts, sleep mask array
    """

    if os.path.exists(sleep_folder + sleep_file):
        print("\nImporting sleep data...")

        df_sleep = pd.read_csv(sleep_folder + sleep_file)
        df_sleep['start_timestamp'] = pd.to_datetime(df_sleep['start_time'])
        df_sleep['end_timestamp'] = pd.to_datetime(df_sleep['end_time'])

        df_sleep['start_idx'] = [int((row.start_timestamp - start_stamp).total_seconds() * sample_rate)
                                 for row in df_sleep.itertuples()]
        df_sleep['end_idx'] = [int((row.end_timestamp - start_stamp).total_seconds() * sample_rate)
                               for row in df_sleep.itertuples()]
        df_sleep = df_sleep.loc[(df_sleep['start_idx'] >= 0) & (df_sleep['end_idx'] <= max_i)]

        sleep_mask = np.zeros(int(max_i / sample_rate))
        for row in df_sleep.itertuples():
            start = int(row.start_idx / sample_rate)
            end = int(row.end_idx / sample_rate)
            sleep_mask[start:end] = 1

    if not os.path.exists(sleep_folder + sleep_file):
        print("-Sleep data not found.")
        df_sleep = None
        sleep_mask = None

    return df_sleep, sleep_mask


def import_activity_counts(activity_folder, activity_file, start_stamp, sample_rate, max_i):
    """Imports and formats epoched wrist data and creates activity intensity mask in 1-s increments.

       arguments:
       -activity_folder: pathway to epoched wrist data
       -activity_file: epoched data filename
       -start_stamp: timestamp when ECG collection begins
       -sample_rate: of ECG signal, Hz
       -max_i: length of ECG signal

       returns:
       -df for epoched wrist data, intensity array
    """

    if os.path.exists(activity_folder + activity_file):
        print("\nImporting wrist activity data...")

        df_act = None
        intensity_mask = np.zeros(int(max_i / sample_rate))

    if not os.path.exists(activity_folder + activity_file):
        print("\nActivity counts file not found --> flagging everything as sedentary")
        df_act = None,
        intensity_mask = None

    if os.path.exists(activity_folder + activity_file):
        df_act = pd.read_csv(activity_folder + activity_file)
        df_act['start_timestamp'] = pd.to_datetime(df_act['start_time'])
        df_act['end_timestamp'] = pd.to_datetime(df_act['end_time'])

        df_act['start_idx'] = [int((row.start_timestamp - start_stamp).total_seconds() * sample_rate)
                               for row in df_act.itertuples()]
        df_act['end_idx'] = [int((row.end_timestamp - start_stamp).total_seconds() * sample_rate)
                             for row in df_act.itertuples()]
        df_act = df_act.loc[(df_act['start_idx'] >= 0) & (df_act['end_idx'] <= max_i)]

        int_dict = {'none': 0, 'sedentary': 0, 'light': 1, 'moderate': 2, 'vigorous': 2}

        for row in df_act.itertuples():
            start = int(row.start_idx / sample_rate)
            end = int(row.end_idx / sample_rate)
            intensity_mask[start:end] = int_dict[row.intensity]

    return df_act, intensity_mask


def import_nonwear_data(file, start_stamp, sample_rate, max_i, full_id=None):

    nw_mask = np.zeros(int(max_i / sample_rate))
    df_nw = None

    if not os.path.exists(file):
        print("-Nonwear file not found --> flagging everything as 'wear'")

    if os.path.exists(file):
        start_stamp = pd.to_datetime(start_stamp)

        df_nw = pd.read_excel(file)

        if full_id is not None:
            df_nw = df_nw.loc[df_nw['full_id'] == full_id]

        df_nw['start_timestamp'] = pd.to_datetime(df_nw['start_timestamp'])
        df_nw['end_timestamp'] = pd.to_datetime(df_nw['end_timestamp'])
        df_nw['subject_id'] = ["0" * (4 - len(str(row.subject_id))) + str(row.subject_id) for row in df_nw.itertuples()]

        for row in df_nw.itertuples():
            start = int((row.start_timestamp - start_stamp).total_seconds())
            end = int((row.end_timestamp - start_stamp).total_seconds())
            nw_mask[start:end] = 1

    return df_nw, nw_mask


def create_df_mask(sample_rate, max_i, start_stamp,
                   gait_folder, gait_file,
                   sleep_folder, sleep_file,
                   activity_folder, activity_file,
                   nw_folder, nw_file, full_id=None):
    """Calls functions to import gait, sleep, and activity data. Combines masks into single DF.

       arguments:
       -sample_rate: ECG sample rate, Hz
       -max_i: length of ECG signal
       -start_stamp: ECG start timestamp
       -the rest are obvious (see individual import functions if not)

       returns individual DFs and unified gait mask DF
    """

    df_gait, gait_mask = import_gait_bouts(gait_folder=gait_folder,
                                           gait_file=gait_file,
                                           sample_rate=sample_rate, max_i=max_i, start_stamp=start_stamp)
    df_sleep, sleep_mask = import_sleep_bouts(sleep_folder=sleep_folder,
                                              sleep_file=sleep_file,
                                              sample_rate=sample_rate, max_i=max_i, start_stamp=start_stamp)
    df_act, act_mask = import_activity_counts(activity_folder=activity_folder,
                                              activity_file=activity_file,
                                              sample_rate=sample_rate, max_i=max_i, start_stamp=start_stamp)

    max_len = int(np.floor(max_i / sample_rate))
    df_nw, nw_mask = import_nonwear_data(file=nw_folder + nw_file, start_stamp=start_stamp,
                                         sample_rate=sample_rate, max_i=max_i, full_id=full_id)

    df_mask = pd.DataFrame({"seconds": np.arange(max_len),
                            'timestamp': pd.date_range(start=start_stamp, periods=len(gait_mask), freq='1S'),
                            'gait': gait_mask if gait_mask is not None else np.zeros(max_len),
                            'sleep': sleep_mask if sleep_mask is not None else np.zeros(max_len),
                            'activity': act_mask if act_mask is not None else np.zeros(max_len),
                            'nw': nw_mask if nw_mask is not None else np.zeros(max_len)})

    return df_mask, df_gait, df_sleep, df_act, df_nw
    # return df_mask

