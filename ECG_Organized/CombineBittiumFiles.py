import os
import pandas as pd
from datetime import timedelta as td
import nwdata
import numpy as np
import pickle


def save_pickle(save_pathway, object):
    pickle_file = open(save_pathway, 'wb')
    pickle.dump(object, pickle_file)
    pickle_file.close()
    print(f"Saved to {save_pathway}")


def open_pickle(pickle_pathway):
    f = open(pickle_pathway, 'rb')
    data = pickle.load(f)

    return data


def check_start_times(files):
    """For each file in files, checks start/end timestamps and sample rates. Returns df sorted by start date"""

    print(f"\nChecking start times for {len(files)} files...")
    start_times = []
    end_times = []
    sample_rates = []
    acc_rates = []

    for file in files:
        print(f"-{file}")

        h = nwdata.EDF.EDFFile(file_path=file)
        h.read_header()

        start = h.header['startdate']
        end = start + td(seconds=h.header['duration'].total_seconds())

        start_times.append(start)
        end_times.append(end)

        for i in range(len(h.signal_headers)):
            if h.signal_headers[i]['label'] == 'ECG':
                sample_rates.append(h.signal_headers[i]['sample_rate'])
            if h.signal_headers[i]['label'] == 'Accelerometer_X':
                acc_rates.append(h.signal_headers[i]['sample_rate'])

    df = pd.DataFrame({"file": files, 'start': start_times, 'end': end_times,
                       'ecg_rate': sample_rates, 'acc_rate': acc_rates}).sort_values('start').reset_index(drop=True)

    df['duration_h'] = [(j-i).total_seconds() / 3600 for i, j in zip(df['start'], df['end'])]

    print("Complete.")

    return df


def calculate_sample_pad(df):
    """For each row in df, calculates how many samples will be needed to pad between end[i] and start[i+1]"""

    df = df.copy()

    n_ecg = []
    n_acc = []
    n_temp = []

    for i in range(df.shape[0] - 1):
        pad_ecg = int((df.iloc[i + 1]['start'] - df.iloc[i]['end']).total_seconds() * df.iloc[i]['ecg_rate'])
        pad_acc = int((df.iloc[i + 1]['start'] - df.iloc[i]['end']).total_seconds() * df.iloc[i]['acc_rate'])
        pad_temp = int((df.iloc[i + 1]['start'] - df.iloc[i]['end']).total_seconds())
        n_ecg.append(pad_ecg)
        n_acc.append(pad_acc)
        n_temp.append(pad_temp)

    n_ecg.append(None)
    n_acc.append(None)
    n_temp.append(None)

    df['ecg_pad'] = n_ecg
    df['acc_pad'] = n_acc
    df['temp_pad'] = n_temp

    return df


def create_signal(df):

    print(f"Combining files from {df.shape[0]} files...")

    obj_out = None
    for row in df.itertuples():
        print(f"-File {row.Index+1}/{df.shape[0]}")

        data = nwdata.NWData()
        data.import_edf(file_path=row.file)

        # combining new data segments with existing data -----------------------------
        if obj_out is not None:

            for signal in ["ECG", "Accelerometer_X", 'Accelerometer_Y', 'Accelerometer_Z', "DEV_Temperature"]:
                obj_out.signals[obj_out.get_signal_index(signal)] = np.append(obj_out.signals[obj_out.get_signal_index(signal)],
                                                                              data.signals[data.get_signal_index(signal)])

        # first iteration
        if obj_out is None:
            obj_out = data

        # pads zeros to fill between end of collection i and collection i+1 ------------------------------
        try:
            obj_out.signals[obj_out.get_signal_index('ECG')] = np.append(obj_out.signals[obj_out.get_signal_index('ECG')],
                                                                         np.zeros(int(row.ecg_pad)))

            obj_out.signals[obj_out.get_signal_index('Accelerometer_X')] = np.append(obj_out.signals[obj_out.get_signal_index('Accelerometer_X')],
                                                                                     np.zeros(int(row.acc_pad)))

            obj_out.signals[obj_out.get_signal_index('Accelerometer_Y')] = np.append(obj_out.signals[obj_out.get_signal_index('Accelerometer_Y')],
                                                                                     np.zeros(int(row.acc_pad)))

            obj_out.signals[obj_out.get_signal_index('Accelerometer_Z')] = np.append(obj_out.signals[obj_out.get_signal_index('Accelerometer_Z')],
                                                                                     np.zeros(int(row.acc_pad)))

            obj_out.signals[obj_out.get_signal_index('DEV_Temperature')] = np.append(obj_out.signals[obj_out.get_signal_index('DEV_Temperature')],
                                                                                     np.zeros(int(row.temp_pad)))
        except ValueError:
            pass

    return obj_out


"""
f = "W:/PD Case APR_6/"
files = [f + i for i in os.listdir(f)]
files = [i for i in files if "BF" in i]

df = check_start_times(files)
df = calculate_sample_pad(df)
data = create_signal(df)


from ECG_Organized.RunSmital import process_snr
data = open_pickle("C:/Users/ksweber/Desktop/OND90_9999_Combined.pickle")
snr = process_snr(out_dir="C:/Users/ksweber/Desktop/", edf_folder="C:/Users/ksweber/Desktop/", ecg_obj=data,
                  ecg_fname="OND90_9999_Combined.pickle", window_len=3600, overlap_secs=60, quiet=True"""