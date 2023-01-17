import nimbalwear
import pickle
import os
import Filtering
import pandas as pd
from ECG.Processing import *
import numpy as np
from datetime import timedelta


class ECG:

    def __init__(self, edf_folder, ecg_fname,
                 smital_edf_fname="", thresholds=(5, 20), bandpass=(.67, 40),
                 snr_hr_bout_filename="",
                 snr_fullanalysis_bout_filename="",
                 snr_all_bout_filename="",
                 nw_filename=""):

        self.thresholds = sorted(thresholds)

        if os.path.exists(edf_folder + ecg_fname):
            self.ecg = import_ecg_file(filepath=edf_folder + ecg_fname, low_f=bandpass[0], high_f=bandpass[1])

            self.fs = self.ecg.signal_headers[self.ecg.get_signal_index("ECG")]['sample_rate']
            self.signal = self.ecg.signals[self.ecg.get_signal_index("ECG")]

            try:
                self.start_stamp = self.ecg.header['start_datetime']
            except KeyError:
                self.start_stamp = self.ecg.header['startdate']

            self.ts = pd.date_range(self.start_stamp, periods=len(self.signal), freq="{}ms".format(1000/self.fs))

            try:
                self.temperature = self.ecg.signals[self.ecg.get_signal_index('Temperature')]
                self.temp_fs = self.ecg.signal_headers[self.ecg.get_signal_index('Temperature')]['sample_rate']
                self.temp_ts = pd.date_range(self.start_stamp,
                                             periods=len(self.ecg.signals[self.ecg.get_signal_index("Temperature")]),
                                             freq="{}ms".format(1000 / self.temp_fs))
            except TypeError:
                self.temperature = []
                self.temp_ts = []
                self.temp_fs = 1

        if not os.path.exists(edf_folder + ecg_fname):
            print("File does not exist.")

            self.ecg = None
            self.fs = 1
            self.signal = []
            self.filt = []
            self.start_stamp = None
            self.ts = []
            self.temperature = None
            self.temp_fs = 1

        if os.path.exists(smital_edf_fname):
            s = nimbalwear.Device()
            s.import_edf(smital_edf_fname)
            self.snr = s.signals[s.get_signal_index('snr')]

        if not os.path.exists(smital_edf_fname):
            self.snr = [None] * len(self.signal)

        # signal quality (SNR) data import -----------
        self.df_snr_hr = import_snr_bout_file(filepath=snr_hr_bout_filename)
        self.df_snr_q1 = import_snr_bout_file(filepath=snr_fullanalysis_bout_filename)
        self.df_snr_all = import_snr_bout_file(filepath=snr_all_bout_filename)
        self.df_snr_ignore = self.df_snr_hr.loc[self.df_snr_hr['quality_use'] > 1]

        self.df_nw = import_nw_file(filepath=nw_filename, start_timestamp=self.start_stamp, sample_rate=self.fs)

        try:
            self.df_nw = self.df_nw.loc[self.df_nw['event'] == 'nonwear'].reset_index(drop=True)
        except KeyError:
            pass


def import_ecg_file(low_f: float = 1.0, high_f: float = 25.0,
                    filepath="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_{}_01_BF36_Chest.edf"):
    """Imports Bittium Faros EDF file. Runs _-25Hz BP filter.

       argument:
       -subj: str for which subject's data to import
       -filepath_fmt: fullpathway with {} for subj to get passed in

       Returns ECG object, sample rate, and filtered data.
    """

    ecg = nimbalwear.Device()
    ecg.import_edf(file_path=filepath, quiet=False)
    ecg.fs = ecg.signal_headers[ecg.get_signal_index("ECG")]['sample_rate']
    try:
        ecg.acc_fs = ecg.signal_headers[ecg.get_signal_index("Accelerometer x")]['sample_rate']
    except TypeError:
        ecg.acc_fs = ecg.signal_headers[ecg.get_signal_index("Accelerometer_X")]['sample_rate']

    print(f"-Running {low_f}-{high_f}Hz bandpass filter...")
    ecg.filt = Filtering.filter_signal(data=ecg.signals[ecg.get_signal_index("ECG")], sample_f=ecg.fs,
                                       low_f=low_f, high_f=high_f, filter_order=5, filter_type='bandpass')

    return ecg


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
            df_all['start_time'] = [start_time + timedelta(seconds=i/1000) for i in df_all['Msec']]
            df_all['end_time'] = [start_time + timedelta(seconds=row.Msec/1000 + row.Length/1000) for row in df_all.itertuples()]

        if timestamp_method in ['middle', 'centre', 'center']:
            df_all['start_time'] = [start_time + timedelta(seconds=row.Msec / 1000 - row.Length/1000/2) for row in df_all.itertuples()]
            df_all['end_time'] = [row.start_timestamp + timedelta(seconds=row.Length / 1000) for row in df_all.itertuples()]

        if timestamp_method == 'end':
            df_all['start_time'] = [start_time + timedelta(seconds=row.Msec / 1000 - row.Length / 1000) for row
                                         in df_all.itertuples()]
            df_all['end_time'] = [row.start_timestamp + timedelta(seconds=row.Length / 1000) for row in df_all.itertuples()]

    if start_time is None:
        df_all['start_time'] = [None] * df_all.shape[0]
        df_all['end_time'] = [None] * df_all.shape[0]

    df_all = df_all[["start_idx", "end_idx", 'start_time', 'end_time', "Type"]]
    df_all.columns = ["start_idx", "end_idx", 'start_time', 'end_time', "arr_type"]

    df_all['duration'] = [(row.end_idx - row.start_idx)/sample_rate for row in df_all.itertuples()]

    # combines multifocal with 'normal' events of same type
    df_all['arr_type'] = df_all['arr_type'].replace({"COUP(mf)": "COUP", "GEM(mf)": "GEM",
                                                     "SALV(mf)": 'SALV', 'VT(mf)': "VT"})

    all_types = df_all['arr_type'].unique()

    if ecg_signal is not None:
        df_all['abs_volt'] = [max([np.abs(max(ecg_signal[row.start_idx:row.end_idx])),
                                   np.abs(min(ecg_signal[row.start_idx:row.end_idx]))]) for
                              row in df_all.itertuples()]

        # df_all['abs_volt_sd'] = [np.std(np.abs(ecg_signal[row.start_idx:row.end_idx])) for row in df_all.itertuples()]

    # separates sinus rhythm from arrhythmias
    df_sinus = df_all.loc[df_all['arr_type'] == 'Sinus']

    df_all = df_all.loc[(df_all['arr_type'] != "Sinus") & (~df_all['arr_type'].isin(ignore_events))]
    df_cn = df_all.loc[~df_all['arr_type'].isin(ignore_events)]

    if use_arrs is not None:
        df_cn['Include'] = [row.arr_type in use_arrs for row in df_cn.itertuples()]
    if use_arrs is None:
        df_cn['Include'] = [True] * df_cn.shape[0]

    df_cn['duration'] = [(row.end_idx - row.start_idx)/sample_rate for row in df_cn.itertuples()]

    remaining_types = df_cn['arr_type'].unique()
    removed_types = [i for i in all_types if i not in remaining_types]

    if not quiet:
        print(f"-Removed events: {removed_types}")
        print(f"-Remaining events: {remaining_types}")

    if use_arrs is not None:
        df_cn = df_cn.loc[df_cn['arr_type'].isin(use_arrs)]

    return df_all, df_sinus.reset_index(drop=True), df_cn.reset_index(drop=True)


def import_cn_beats_file(pathway, start_stamp, sample_rate):

        df_rr = pd.read_csv(pathway, delimiter=";")

        # column name formatting
        cols = [i for i in df_rr.columns][1:]
        cols.insert(0, "Msec")
        df_rr.columns = cols

        df_rr['idx'] = [int(i / 1000 * sample_rate) for i in df_rr['Msec']]

        if "timestamp" not in df_rr.columns:
            df_rr["timestamp"] = [start_stamp + timedelta(seconds=row.Msec / 1000) for row in df_rr.itertuples()]

        try:
            # Beat type value replacement (no short forms)
            type_dict = {'?': "Unknown", "N": "Normal", "V": "Ventricular", "S": "Premature", "A": "Aberrant"}
            df_rr["arr_type"] = [type_dict[i] for i in df_rr["arr_type"]]
        except KeyError:
            pass

        df_rr["HR"] = [60 / (r / 1000) for r in df_rr["RR"]]  # beat-to-beat HR

        df_rr = df_rr[['idx', "timestamp", "RR", "HR", "arr_type", "Template", "PP", "QRS", "PQ",
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

    df_bouts = pd.read_excel(filepath) if 'xlsx' in filepath else pd.read_csv(filepath)
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


def import_snr_bout_file(filepath: str):
    """ Imports signal-to-noise ratio (SNR) bout file from csv and formats column data appropriately.

        arguments:
        -filepath: pathway to SNR bout file

        returns:
        -dataframe
    """

    dtype_cols = {"study_code": str, 'subject_id': str, 'coll_id': str,
                  'start_idx': pd.Int64Dtype(), 'end_idx': pd.Int64Dtype(), 'bout_num': pd.Int64Dtype(),
                  'quality': str, 'avg_snr': float, 'quality_use': int}

    df = pd.DataFrame({'study_code': [], 'subject_id': [], 'coll_id': [], 'start_idx': [], 'end_idx': [],
                       'bout_num': [], 'quality': [], 'avg_snr': [], 'quality_use': []})

    if os.path.exists(filepath):
        try:
            date_cols = ['start_time', 'end_time']
            df = pd.read_csv(filepath, dtype=dtype_cols, parse_dates=date_cols)
            df['duration'] = [(row.end_time - row.start_time).total_seconds() for row in df.itertuples()]

        except (ValueError, AttributeError):
            date_cols = ['start_timestamp', 'end_timestamp']
            df = pd.read_csv(filepath, dtype=dtype_cols, parse_dates=date_cols)
            df['duration'] = [(row.end_timestamp - row.start_timestamp).total_seconds() for row in df.itertuples()]

        # replaces strings with numeric equivalents for signal qualities
        df['quality_use'] = df['quality'].replace({'ignore': 3, 'full': 1, 'HR': 1})

        # df = df.loc[(df['start_idx'] >= min_idx) & (df['end_idx'] <= max_idx if max_idx != -1 else df['end_idx'].max())]

    return df


def import_snr_edf(filepath):

    data = nimbalwear.Device()
    data.import_edf(filepath)

    data.fs = data.signal_headers[0]['sample_rate']
    data.start_time = data.header['start_datetime'] if 'start_datetime' in data.header.keys() else \
        data.header['start_timestamp']

    data.ts = pd.date_range(start=data.start_time, periods=len(data.signals[0]), freq=f"{1000/data.fs}ms")

    return data


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
        df_act['start_time'] = pd.to_datetime(df_act['start_time'])
        df_act['end_time'] = pd.to_datetime(df_act['end_time'])

        df_act['start_idx'] = [int((row.start_time - start_stamp).total_seconds() * sample_rate)
                               for row in df_act.itertuples()]
        df_act['end_idx'] = [int((row.end_time - start_stamp).total_seconds() * sample_rate)
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

        df_nw['start_time'] = pd.to_datetime(df_nw['start_timestamp'])
        df_nw['end_time'] = pd.to_datetime(df_nw['end_timestamp'])
        df_nw['subject_id'] = ["0" * (4 - len(str(row.subject_id))) + str(row.subject_id) for row in df_nw.itertuples()]

        df_nw = df_nw[[i for i in df_nw.columns if i not in ['start_timestamp', 'end_timestamp']]]
        for row in df_nw.itertuples():
            start = int((row.start_time - start_stamp).total_seconds())
            end = int((row.end_time - start_stamp).total_seconds())
            nw_mask[start:end] = 1

    return df_nw, nw_mask


def import_cn_beat_file(filename, start_time, sample_rate=250):

    print("\nImporting Cardiac Navigator beat data...")

    heart_beats = pd.read_csv(filename, sep=';')
    heart_beats.rename(columns={heart_beats.columns[0]: "msec"}, inplace=True)
    heart_beats['timestamp'] = [start_time + timedelta(seconds=row.msec / 1000) for row in heart_beats.itertuples()]
    heart_beats['idx'] = [int(i/1000*sample_rate) for i in heart_beats['msec']]
    heart_beats['rate'] = 60000 / heart_beats['RR']

    heart_beats = heart_beats[["timestamp", 'idx', 'rate', 'PP', 'QRS', 'PQ', 'QRn', 'QT', 'NOISE']]

    return heart_beats


def import_nw_file(filepath: str, sample_rate: float or int, start_timestamp: str, pad_mins: int or float = 0):
    """ Imports and formats nonwear bouts csv file.

        arguments:
        -filepath: pathway to csv file
        -sample_rate: of ECG signal, Hz
        -start_timestamp: of ECG signal
        -pad_mins: numbers of minutes added to start/end of each nonwear bout

        returns:
        -dataframe
    """

    if os.path.exists(filepath):
        start_timestamp = pd.to_datetime(start_timestamp)
        date_cols = ['start_time', 'end_time']
        df = pd.read_csv(filepath, parse_dates=date_cols)

        if pad_mins != 0:
            df['start_time'] = [i + timedelta(minutes=-pad_mins) for i in df['start_time']]
            df['end_time'] = [i + timedelta(minutes=-pad_mins) for i in df['end_time']]

        df['start_idx'] = [int((row.start_time - start_timestamp).total_seconds() * sample_rate) for row in df.itertuples()]
        df['end_idx'] = [int((row.end_time - start_timestamp).total_seconds() * sample_rate) for row in df.itertuples()]
        df['duration'] = [(j - i).total_seconds() for i, j in zip(df['start_time'], df['end_time'])]

    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=['start_time', 'end_time'])

    return df
