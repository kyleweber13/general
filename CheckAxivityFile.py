from nwdata import nwfiles
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import pandas as pd
from Filtering import filter_signal
import numpy as np
import os


class CWAData:

    def __init__(self, filepath):
        self.filepath = filepath
        self.name = filepath.split("/")[-1]

        self.data_obj = self.read()

        self.data_obj.data['battery_filt'] = filter_signal(data=self.data_obj.data['battery'], filter_order=3,
                                                           filter_type='lowpass', low_f=.01,
                                                           sample_f=self.data_obj.header['packet_rate'])

        self.imu_ts = pd.date_range(start=self.data_obj.header["logging_start"],
                                    periods=len(self.data_obj.data['gx']),
                                    freq="{}ms".format(1000/self.data_obj.header['sample_rate']))

        self.util_ts = pd.date_range(start=self.data_obj.header["logging_start"],
                                     periods=len(self.data_obj.data['temperature']),
                                     freq="{}ms".format(1000/self.data_obj.header['packet_rate']))

    def read(self):
        cwa_obj = nwfiles.CWAFile(self.filepath)
        cwa_obj.read()

        return cwa_obj


def plot_data(CWAData_obj):

    ts = pd.date_range(start=CWAData_obj.data_obj.header["start_time"], periods=len(CWAData_obj.data_obj.data['gx']),
                       freq="{}ms".format(1000/CWAData_obj.data_obj.header['sample_rate']))

    fig, ax = plt.subplots(4, sharex='col', figsize=(14, 9))

    ax[0].plot(ts, CWAData_obj.data_obj.data['ax'], color='black')
    ax[0].plot(ts, CWAData_obj.data_obj.data['ay'], color='red')
    ax[0].plot(ts, CWAData_obj.data_obj.data['az'], color='dodgerblue')

    ax[0].set_title("Accelerometer")
    ax[0].set_ylabel("G")

    ax[1].plot(CWAData_obj.util_ts, CWAData_obj.data_obj.data['temperature'], color='darkorange')
    ax[1].set_title("Temperature")
    ax[1].set_ylabel("degrees C")

    ax[2].plot(CWAData_obj.util_ts, CWAData_obj.data_obj.data['battery'], color='navy', label='raw')
    ax[2].plot(CWAData_obj.util_ts, CWAData_obj.data_obj.data['battery_filt'], color='limegreen', label='.01Hz lowpass')
    ax[2].set_title("Battery")
    ax[2].set_ylabel("Volts")
    ax[2].legend()

    ax[3].plot(CWAData_obj.util_ts, CWAData_obj.data_obj.data['light'], color='gold', label='light')
    ax[3].legend()

    ax[-1].xaxis.set_major_formatter(xfmt)

    return fig


def calculate_voltage_drain(CWAData_obj, ax, timestamps, end_time=None):

    if end_time is None:
        duration = len(CWAData_obj.data_obj.data['gx']) / CWAData_obj.data_obj.header['sample_rate']
        end_ind = -1

    if end_time is not None:
        duration = (pd.to_datetime(end_time) - CWAData_obj.data_obj.header['start_time']).total_seconds()
        end_ind = int(duration * CWAData_obj.data_obj.header['packet_rate'])

    ts = timestamps[:end_ind]

    drain_rate = (max(CWAData_obj.data_obj.data['battery_filt'][:end_ind]) -
                  min(CWAData_obj.data_obj.data['battery_filt'][:end_ind])) / duration

    max_v = max(CWAData_obj.data_obj.data['battery'])
    ax.plot(ts[::100],
            [max_v - i / CWAData_obj.data_obj.header['packet_rate'] * drain_rate for
             i in range(0, len(CWAData_obj.data_obj.data['battery'][:end_ind]), 100)],
            linestyle='dashed', label=CWAData_obj.name)
    ax.legend()

    return drain_rate


def plot_multiple(CWAData_objs=()):

    fig, ax = plt.subplots(1, sharex='col', figsize=(14, 8))

    for obj in CWAData_objs:
        x = np.arange(0, len(obj.data_obj.data['battery_filt']))/obj.data_obj.header['packet_rate']/3600/24
        ax.plot(x, obj.data_obj.data['battery_filt'], label=obj.name)
    ax.legend()
    ax.set_ylabel("Voltage")
    ax.set_xlabel("Days")
    ax.set_xticks(np.arange(0, ax.get_xlim()[1]*1.1, 1))
    ax.set_xticks(np.arange(0, ax.get_xlim()[1]*1.1, .25), minor=True)
    ax.set_xlim(0, )

    return fig


def plot_all_data(wrist=None, ankle=None):
    n_subplots = 2 + int(wrist is not None) + int(ankle is not None)

    fig, ax = plt.subplots(n_subplots, sharex='col', figsize=(14, 10))
    plt.subplots_adjust(top=.925, bottom=.075, left=.05, right=.98)

    curr_subplot = 0
    if wrist is not None:
        ax[curr_subplot].plot(wrist.imu_ts, wrist.data_obj.data['ax'], color='black')
        ax[curr_subplot].plot(wrist.imu_ts, wrist.data_obj.data['ay'], color='red')
        ax[curr_subplot].plot(wrist.imu_ts, wrist.data_obj.data['az'], color='dodgerblue')
        ax[curr_subplot].set_title("Wrist")
        ax[curr_subplot].legend(loc='lower right')
        curr_subplot += 1

    if ankle is not None:
        ax[curr_subplot].plot(ankle.imu_ts, ankle.data_obj.data['ax'], color='black')
        ax[curr_subplot].plot(ankle.imu_ts, ankle.data_obj.data['ay'], color='red')
        ax[curr_subplot].plot(ankle.imu_ts, ankle.data_obj.data['az'], color='dodgerblue')
        ax[curr_subplot].set_title("Ankle")
        curr_subplot += 1

    if wrist is not None:
        ax[curr_subplot].plot(wrist.util_ts, wrist.data_obj.data['temperature'], color='red', label='Wrist')
        ax[curr_subplot+1].plot(wrist.util_ts, wrist.data_obj.data['battery_filt'], color='red', label='Wrist')

    if ankle is not None:
        ax[curr_subplot].plot(ankle.util_ts, ankle.data_obj.data['temperature'], color='black', label='Ankle')
        ax[curr_subplot+1].plot(ankle.util_ts, ankle.data_obj.data['battery_filt'], color='black', label='Ankle')

    ax[curr_subplot].legend()
    ax[curr_subplot].set_title("Temperature")

    ax[curr_subplot+1].legend()
    ax[curr_subplot+1].set_title("Battery")
    ax[curr_subplot + 1].set_ylabel("Volts")

    ax[-1].xaxis.set_major_formatter(xfmt)

    return fig


def print_summary(obj):

    file_size = round(os.path.getsize(fname) / 1000000, 2)
    dur = round((obj.imu_ts[-1] - obj.imu_ts[0]).total_seconds()/86400, 3)

    print(f"\n-File is {file_size}MB")
    print("-Logging from {} to {}".format(obj.data_obj.header['logging_start'], obj.data_obj.header['logging_end']))
    print(f"     -Recorded from {obj.imu_ts[0]} to {obj.imu_ts[-1]}")
    print(f"          -Recording is {dur} days")


fname = "O:/OBI/Personal Folders/Kyle Weber/6014987_0000000000.cwa"
file = CWAData(fname)
print_summary(file)
plot_all_data(file)
