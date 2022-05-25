from Filtering import filter_signal  # Kyle's script
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
import pandas as pd
import numpy as np
import peakutils
import nwdata
import random

""" ============================================== DATA IMPORT/PREPARATION ======================================== """

# Data import -------------------
ankle = nwdata.NWData()

ankle.import_edf(file_path='W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_0001_01_AXV6_RAnkle.edf')

# Pulling info from headers -----
start_stamp = ankle.header["startdate"]

index_dict = {"gyro x": ankle.get_signal_index('Gyroscope x'), "gyro y": ankle.get_signal_index('Gyroscope y'),
              "gyro z": ankle.get_signal_index('Gyroscope z'), "accel x": ankle.get_signal_index('Accelerometer x'),
              "accel y": ankle.get_signal_index('Accelerometer y'), "accel z": ankle.get_signal_index('Accelerometer z')}

sample_rate = int(ankle.signal_headers[0]['sample_rate'])

timestamps = pd.date_range(start=start_stamp, periods=len(ankle.signals[0]), freq=f"{1000/sample_rate}ms")

# data cropping by index
start_ind = random.randint(0, len(ankle.signals[0]) - int(1*3600*sample_rate))
# start_ind = 30416731  # good test section for OND09_0001
stop_ind = start_ind + int(1*3600*sample_rate)

""" ================================================ ACTUALLY DOING STUFF ========================================= """


def run_peak_detection(imu_signals, gyro_axis='z', accel_axis='y', start_index=0, stop_index=-1,
                       timestamps=None,
                       min_swing_dur_ms=250, max_swing_dur_ms=800, min_steptime_ms=250,
                       gyro_thresh=40, sample_rate=50, show_plot=True):
    """Function that runs step detection method adapted from Fraccaro, Coyle, & O'Sullivan (2014) on gyroscope data.

    arguments:
    -gyro_axis: str, "x"/"y"/"z" --> which gyroscope axis to use
    -accel_axis: str, "x", "y", "z" --> which accelerometer axis to use (plotting only)
    -start_index/stop_index: integer, for cropping imu_signals

    -min_swing_dur_ms/max_swing_dur_ms: float, min/max durations of swing phases; events outside range are ignored
    -min_steptime_ms: minimum time between consecutive steps in ms

    -gyro_thresh: minimum acceptable gyroscope threshold in deg/s
    -sample_rate: sampling frequency of imu_signals
    -show_plot: boolean --> shows plot of gyro and accel data with detected peaks

    returns:
    -df_peaks: dataframe of detected peaks with swing phase onset/onffset data indexes
    -figure: figure drawn if show_plot is True
    """

    fig = None
    ts = None

    gyro = imu_signals[index_dict[f"gyro {gyro_axis}"]][start_index:stop_index]
    accel = imu_signals[index_dict[f"accel {accel_axis}"]][start_index:stop_index]

    if timestamps is not None:
        ts = timestamps[start_index:stop_index]

    print(f"\nRunning gyroscope peak detection on {round(len(gyro)/sample_rate, 1)} seconds of data...")

    # Peak thresholding - minimum of 40 deg/s -------------------

    # Kyle's function that calls scipy.signal
    # Changed from 3Hz LP to .1-3Hz bandpass
    print("Running 5th order .1-3Hz bandpass filter...")
    gyro_f = filter_signal(data=gyro, filter_order=5, filter_type='bandpass', sample_f=sample_rate, low_f=.1, high_f=3)

    # "Derivative": difference of consecutive datapoints in deg/s as numpy array
    diff = np.array([(d2-d1) / (1/sample_rate) for d1, d2 in zip(gyro_f[:], gyro_f[1:])])

    print("\nDetecting peaks...")

    # Peak detection: minimum 40 deg/s and .25-sec between peaks
    peak_indexes1 = peakutils.indexes(y=diff, thres_abs=True, thres=gyro_thresh, min_dist=int(sample_rate/4))
    print(f"-Found {len(peak_indexes1)} peaks with initial check.")

    # list of values for each peak
    peak_vals = [gyro_f[i] for i in peak_indexes1]

    # Adaptive thresholding --------------

    # Sets threshold as 20% of value of highest 10 detected peaks, minimum .5-sec between peaks
    # sorted(peak_vals) --> sorts in ascending order
    # sorted(peak_vals)[-10:] --> crops to largest 10 values
    threshold = np.mean(sorted(peak_vals)[-10:]) * .2

    # Overrides threshold value to 40 if below 40
    if threshold < gyro_thresh:
        threshold = gyro_thresh

    # Peak detection again with new threshold criteria
    peak_indexes2 = peakutils.indexes(y=gyro_f, thres_abs=True, thres=threshold, min_dist=int(sample_rate/4))

    print(f"-Found {len(peak_indexes2)} peaks with adjusted threshold.")

    # swing phase duration (local minima): 250-800ms
    win_samples = int(sample_rate*.5)
    valid_peaks = []
    local_minima = []

    print("\nFinding onset/offset of swing phases...")

    for peak in peak_indexes2:
        window = np.array(gyro_f[peak-win_samples:peak+win_samples])

        try:
            min1 = peak -win_samples + np.argmin(window[:win_samples])
            min2 = np.argmin(window[win_samples:]) + peak
            local_minima.append([min1, min2])
            valid_peaks.append(peak)

        except ValueError:
            pass

    # Output dataframe: peaks and onset/offset of swing phases
    local_minima = np.array(local_minima).transpose()
    df_peaks = pd.DataFrame({"Peak": valid_peaks, "Start": local_minima[0], "End": local_minima[1]})
    df_peaks["Width"] = df_peaks["End"] - df_peaks["Start"]

    # Only includes swing phases between 250 and 800ms long
    len1 = df_peaks.shape[0]

    df_peaks = df_peaks.loc[(df_peaks['Width'] >= sample_rate*min_swing_dur_ms/1000) &
                            (df_peaks["Width"] <= sample_rate*max_swing_dur_ms/1000)]
    len2 = df_peaks.shape[0]
    print(f"\nRemoved {len1-len2} swing phases shorter than {min_swing_dur_ms}ms and "
          f"longer than {max_swing_dur_ms}ms.")

    # Minimum number of samples between consecutive steps
    min_samples = min_steptime_ms / (1000/sample_rate)
    step_time = [j-i for i, j in zip(df_peaks["Peak"].iloc[:], df_peaks["Peak"].iloc[1:])]
    step_time.append(None)
    df_peaks["StepTime"] = step_time

    df_peaks = df_peaks.loc[df_peaks["StepTime"] >= min_samples]
    removed_steptime = len2 - df_peaks.shape[0]
    suffix = "s" if removed_steptime != 1 else ""
    print(f"\nRemoved {removed_steptime} step{suffix} with less than {min_steptime_ms}ms between them.")

    df_peaks = df_peaks.reset_index(drop=True)

    # Converts data indexes to timestamps if timestamps argument is given
    if timestamps is not None:
        df_peaks['PeakStamp'] = [ts[i] for i in df_peaks["Peak"]]
        df_peaks['StartStamp'] = [ts[i] for i in df_peaks["Start"]]
        df_peaks['EndStamp'] = [ts[i] for i in df_peaks["End"]]
        df_peaks["WidthSec"] = [(j - i)/sample_rate for i, j in zip(df_peaks["Start"], df_peaks["End"])]

    if show_plot:
        fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))
        plt.suptitle(f"Gyroscope peak detection \n"
                     f"(swing duration = {min_swing_dur_ms}-{max_swing_dur_ms}ms; min. step time = {min_steptime_ms}ms)")

        ax[0].plot(ts if ts is not None else np.arange(0, len(gyro_f))/sample_rate, gyro_f, color='black', zorder=0)

        for row in df_peaks.itertuples():

            # Includes legend labels for first row
            if row.Index == 0:
                # Peaks
                ax[0].scatter(row.Peak/sample_rate if timestamps is None else row.PeakStamp,
                              gyro_f[row.Peak]*1.1,
                              color='dodgerblue', marker='v', s=25, label="Peak")

                # Local minimum: onset of swing
                ax[0].scatter(row.Start/sample_rate if timestamps is None else row.StartStamp,
                              gyro_f[int(row.Start)],
                              color='limegreen', marker='x', s=15, label="Swing_onset")

                # Local minima: offset of swing
                ax[0].scatter(row.End/sample_rate if timestamps is None else row.EndStamp,
                              gyro_f[int(row.End)],
                              color='red', marker='x', s=15, label="Swing_offset")

                ax[0].legend(loc='upper right')

            # Does not include legend labels for subsequent data
            if row.Index != 0:
                ax[0].scatter(row.Peak / sample_rate if timestamps is None else row.PeakStamp,
                              gyro_f[row.Peak] * 1.1,
                              color='dodgerblue', marker='v', s=25)

                ax[0].scatter(row.Start / sample_rate if timestamps is None else row.StartStamp,
                              gyro_f[int(row.Start)],
                              color='limegreen', marker='x', s=15)

                ax[0].scatter(row.End / sample_rate if timestamps is None else row.EndStamp,
                              gyro_f[int(row.End)],
                              color='red', marker='x', s=15)

        ax[0].set_ylabel("Deg/s")
        ax[0].set_title(f"Gyro {gyro_axis} (.1-3Hz BP)")

        ax[1].plot(ts if ts is not None else np.arange(0, len(accel))/sample_rate, accel, color='purple', zorder=0)
        ax[1].set_title(f"Accel {accel_axis}")
        ax[1].set_ylabel("G")

        if timestamps is not None:
            ax[1].xaxis.set_major_formatter(xfmt)
        if timestamps is None:
            ax[1].set_xlabel("Seconds")

        return df_peaks, fig


df_peaks, figure = run_peak_detection(imu_signals=ankle.signals, gyro_axis='z', accel_axis='y', gyro_thresh=40,
                                      start_index=0, stop_index=int(len(ankle.signals[0])/10),
                                      min_swing_dur_ms=250, max_swing_dur_ms=800, min_steptime_ms=250,
                                      timestamps=timestamps, sample_rate=sample_rate, show_plot=True)

