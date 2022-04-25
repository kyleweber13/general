import pandas as pd
from Filtering import filter_signal
import nwactivity
import numpy as np


def epoch_data(data_object, use_accel=True, epoch_len=5,
               use_activity_counts=True, use_sd=False,
               bandpass_epoching=True, remove_baseline=False,
               start=None, end=None):
    """Function to epoch data given specific inputs.

        arguments:
        -data_object: object returned from DataImport.import_ax_data()
        -use_accel: boolean for accelerometer or gyroscope data
        -epoch_len: integer, seconds
        -use_activity_counts: boolean to use activity counts (AVM)
        -use_sd: boolean to use acceleration/gyro SD instead of activity counts
        -bandpass_epoching: boolean to run a .1-20Hz BP filter on activity count data
        -remove_baseline: boolean to remove minimum activity count value from each other value
        -start/end: timestamp (str or timestamp) by which data is cropped.
            -Set to Nones for no cropping

        returns:
        -epoched dataframe
    """

    print("\nEpoching {} data into {}-second epochs...".format('accelerometer' if use_accel else 'gyroscope', epoch_len))

    # channel indexes for accelerometer/gyro data
    sig_idx = {'ax': data_object.get_signal_index('Accelerometer x'),
               'ay': data_object.get_signal_index('Accelerometer y'),
               'az': data_object.get_signal_index('Accelerometer z'),
               'gx': data_object.get_signal_index('Gyroscope x'),
               'gy': data_object.get_signal_index('Gyroscope y'),
               'gz': data_object.get_signal_index('Gyroscope z')}

    if use_accel:
        prefix = "a"
    if not use_accel:
        prefix = "g"

    fs = data_object.signal_headers[sig_idx[f'{prefix}x']]['sample_rate']

    # data cropping ---------
    print("\nData cropping:")
    if start is not None:
        n_start = int((start - data_object.header['startdate']).total_seconds() * fs)
    if start is None:
        n_start = 0
    print(f"-Data is being cropped by {n_start} samples at the start")

    if end is not None:
        n_end = int((end - data_object.header['startdate']).total_seconds() * fs)
    if end is None:
        n_end = -1

    print("-Data is being cropped by {} samples at the end".format(len(data_object.signals[0]) - n_end if
                                                                   n_end != -1 else 0))

    print()

    if use_activity_counts and not use_sd:
        if bandpass_epoching:
            print("-Bandpass filtering raw data (.1-20Hz)...")
            xf = filter_signal(data=data_object.signals[sig_idx[f'{prefix}x']][n_start:n_end],
                               sample_f=fs, filter_type='bandpass',
                               low_f=.1, high_f=20, filter_order=5)
            yf = filter_signal(data=data_object.signals[sig_idx[f'{prefix}y']][n_start:n_end],
                               sample_f=fs, filter_type='bandpass',
                               low_f=.1, high_f=20, filter_order=5)
            zf = filter_signal(data=data_object.signals[sig_idx[f'{prefix}z']][n_start:n_end],
                               sample_f=fs, filter_type='bandpass',
                               low_f=.1, high_f=20, filter_order=5)

            print("-Calculating AVM values...")
            df = nwactivity.calc_wrist_powell(x=xf, y=yf, z=zf, sample_rate=fs, epoch_length=1, quiet=True)

        if not bandpass_epoching:
            print("-No filtering...")
            print("-Calculating AVM values...")
            df = nwactivity.calc_wrist_powell(x=data_object.signals[sig_idx[f'{prefix}x']][n_start:n_end],
                                              y=data_object.signals[sig_idx[f'{prefix}y']][n_start:n_end],
                                              z=data_object.signals[sig_idx[f'{prefix}z']][n_start:n_end],
                                              sample_rate=fs, epoch_length=1, quiet=True)

        if remove_baseline:
            min_val = df['avm'].min()
            print(f"-Removing baseline value of {min_val} from epoched values...")
            df['value'] = [i - min_val for i in df['avm']]

            df = df.drop(columns='avm')

        if not remove_baseline:
            df['value'] = df['avm']
            df = df.drop(columns='avm')

        df['timestamp'] = pd.date_range(start=data_object.header['startdate'] if start is None else start,
                                        freq="{}S".format(epoch_len),
                                        periods=int(len(data_object.signals[sig_idx['{}x'.format(prefix)]][n_start:n_end])/fs/epoch_len))

        avm = np.array(df['value'])
        avm[avm < 0] = 0
        df['value'] = avm

    if use_sd and not use_activity_counts:
        print("-Calculating SD values...")
        vm = [np.sqrt(x ** 2 + y ** 2 + z ** 2) - 1 for x, y, z in zip(data_object.signals[sig_idx[f'{prefix}x']][n_start:n_end],
                                                                       data_object.signals[sig_idx[f'{prefix}y']][n_start:n_end],
                                                                       data_object.signals[sig_idx[f'{prefix}z']][n_start:n_end])]

        sd_vm = []

        for i in range(0, len(vm), int(fs * epoch_len)):
            sd_vm.append(np.std(vm[i:i + int(fs * epoch_len)]))

        df = pd.DataFrame({'value': sd_vm})

        df['timestamp'] = pd.date_range(start=data_object.header['startdate'] if start is None else None,
                                        freq="{}s".format(epoch_len), periods=df.shape[0])

    if not use_activity_counts and not use_sd:
        print("-Need to specifiy epoching method using 'use_activity_counts' or 'use_sd' arguments")
        return None

    print("\nComplete.")

    return df


def combine_epoched(df_dom, df_nondom, epoch_len):
    """Re-calculates epoching by averaging values over new timespan and combines AVM values from two dataframes.

        arguments:
        -df_dom: dominant wrist 1-sec epoch dataframe
        -df_nondom: non-dominant wrist 1-sec epoch dataframe
        -epoch_len: integer for new epoch length

        returns:
        -combined dataframe with new epoch length
    """

    og_epoch_len = int((df_dom.iloc[1]['timestamp'] - df_dom.iloc[0]['timestamp']).total_seconds())

    n_rows = int(epoch_len / og_epoch_len)

    print(f"\nRe-calculating epochs from {og_epoch_len} to {epoch_len}-second epochs...")

    ts = df_dom['timestamp'].iloc[::n_rows]

    avm_dom = np.array(df_dom['value'])
    avg_dom = [np.mean(avm_dom[i:i+n_rows]) for i in range(0, df_dom.shape[0], n_rows)]

    avm_nd = np.array(df_nondom['value'])
    avg_nd = [np.mean(avm_nd[i:i+n_rows]) for i in range(0, df_nondom.shape[0], n_rows)]

    df_out = pd.DataFrame({'timestamp': ts, 'dominant': avg_dom, 'nondom': avg_nd})

    return df_out
