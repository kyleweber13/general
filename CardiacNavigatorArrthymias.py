import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import pyedflib
from datetime import timedelta as td
import os
from BittiumFarosNonwear import BittiumNonwear
import random
from Filtering import filter_signal
from fpdf import FPDF


""" ============================================== FUNCTION DEFINITIONS ========================================== """


def load_data(filepath_fmt, nonwear_file, gait_file, sleep_output_file):
    file = pyedflib.EdfReader(filepath_fmt)
    start_stamp = file.getStartdatetime()
    ecg = file.readSignal(0)
    file_dur = file.file_duration
    fs = int(file.getSampleFrequency(0))  # ECG sample rate
    ts = pd.date_range(start=start_stamp, periods=len(ecg), freq="{}ms".format(int(1000 / fs)))
    file.close()

    df_nw = pd.read_excel(nonwear_file)
    df_nw["Subj"] = [subj in i for i in df_nw["File"]]
    df_nw = df_nw.loc[df_nw["Subj"]]

    df_gait = pd.read_csv(gait_file)
    df_gait = df_gait[["start_timestamp", "end_timestamp"]]
    df_gait["start_timestamp"] = pd.to_datetime(df_gait["start_timestamp"])
    df_gait["end_timestamp"] = pd.to_datetime(df_gait["end_timestamp"])

    df_sleep_alg = pd.read_csv(sleep_output_file)
    df_sleep_alg["start_time"] = pd.to_datetime(df_sleep_alg["start_time"])
    df_sleep_alg["end_time"] = pd.to_datetime(df_sleep_alg["end_time"])

    f = filter_signal(data=ecg, sample_f=fs, filter_type='bandpass', filter_order=3, low_f=.67, high_f=30)

    return ecg, f, start_stamp, ts, fs, df_nw, df_gait, df_sleep_alg, file_dur


class ArrhythmiaProcessor:

    def __init__(self, raw_ecg=None, epoched_nonwear=None, epoched_signal_quality=None, card_nav_data=None,
                 card_nav_rr_data=None,
                 details={"start_time": None, "sample_rate": 250, "file_dur_sec": 1, "epoch_len": 15}):

        self.ecg = raw_ecg

        self.df_nonwear = epoched_nonwear
        self.df_qc_epochs = epoched_signal_quality

        self.df_card_nav = card_nav_data
        self.df_rr = card_nav_rr_data
        self.details = details
        self.df_valid_arr_qc, self.df_valid_arr_nw, self.df_desc_qc, self.df_desc_nw = None, None, None, None
        self.df_valid, self.df_desc = None, None

        if self.ecg is not None:
            self.timestamps = pd.date_range(start=self.details["start_time"], periods=len(self.ecg),
                                            freq="{}ms".format(int(1000/self.details["sample_rate"])))
        if self.ecg is None:
            self.timestamps = None

        self.xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")

        """METHODS"""
        self.df_card_nav, self.df_sums = self.format_card_nav_file()
        self.df_rr = self.import_rr_data(n_roll_beats_avg=10)
        # self.df_card_nav = self.find_longqt()
        self.df_qc_epochs = self.format_qc_data()
        self.df_nonwear = self.format_nonwear_data()

    def format_card_nav_file(self):

        if self.df_card_nav is None:
            print("Requires Cardiac Navigator event data. Try again.")
            return None, None

        if type(self.df_card_nav) is str:
            df = pd.read_csv(self.df_card_nav, delimiter=";")

            # ms timestamps -> actual timestamps
            df["Timestamp"] = [self.details["start_time"] + td(seconds=row.Msec / 1000) for row in df.itertuples()]

            df["Duration"] = [row.Length / 1000 for row in df.itertuples()]  # event duration, seconds

            # Indexes of raw data
            df["Start_Ind"] = [int(row.Msec / 1000 * self.details["sample_rate"]) for row in df.itertuples()]
            df["Stop_Ind"] = [int(row.Start_Ind + row.Duration * self.details["sample_rate"]) for row in df.itertuples()]
            df = df[["Timestamp", "Start_Ind", "Stop_Ind", "Duration", "Type", "Info"]]

            df["Type"] = [i if i != "AV2/II" else "AV2II" for i in df["Type"]]

        if type(self.df_card_nav) is pd.core.frame.DataFrame:
            df = self.df_card_nav

        # Sums time spent in each arrhythmia type, in seconds
        sums = []
        for t in df["Type"].unique():
            if t not in ["Min. RR", "Max. RR", "Min. HR", "Max. HR", "ST(ref)"]:
                d = df.loc[df["Type"] == t]
                sums.append([t, d["Duration"].sum()])
        df_sums = pd.DataFrame(sums)
        df_sums.columns = ["Type", "Duration"]
        df_sums["%"] = [100 * i / self.details["file_dur_sec"] for i in df_sums["Duration"]]

        return df, df_sums

    def import_rr_data(self, n_roll_beats_avg=30):

        if self.df_rr is None:
            print("Requires RR data. Try again.")
            return None

        if type(self.df_rr) is str:
            df_rr = pd.read_csv(self.df_rr, delimiter=";")

            # column name formatting
            cols = [i for i in df_rr.columns][1:]
            cols.insert(0, "Msec")
            df_rr.columns = cols

        if type(self.df_rr) is pd.core.frame.DataFrame:
            df_rr = self.df_rr

        if "Timestamp" not in df_rr.columns:
            df_rr["Timestamp"] = [self.details["start_time"] + td(seconds=row.Msec / 1000) for row in df_rr.itertuples()]

        try:
            # Beat type value replacement (no short forms)
            type_dict = {'?': "Unknown", "N": "Normal", "V": "Ventricular", "S": "Premature", "A": "Aberrant"}
            df_rr["Type"] = [type_dict[i] for i in df_rr["Type"]]
        except KeyError:
            pass

        df_rr["HR"] = [60 / (r / 1000) for r in df_rr["RR"]]  # beat-to-beat HR
        df_rr["RollHR"] = df_rr["HR"].rolling(window=n_roll_beats_avg, center=True).mean()  # rolling average HR

        df_rr = df_rr[["Timestamp", "Msec", "RR", "HR", "RollHR", "Type", "Template", "PP", "QRS", "PQ",
                       "QRn", "QT", "ISO", "ST60", "ST80", "NOISE"]]

        return df_rr

    def format_qc_data(self):

        if self.df_qc_epochs is None:
            print("Requires epoched signal qualilty data. Try again.")
            return None

        if type(self.df_qc_epochs) is pd.core.frame.DataFrame:
            df = self.df_qc_epochs

        if type(self.df_qc_epochs) is str:
            df = pd.read_csv(self.df_qc_epochs)

        if "Timestamp" not in df.columns:
            epoch_len = int((df.iloc[1]["Index"] - df.iloc[0]["Index"]) / self.details["sample_rate"])
            df["Timestamp"] = pd.date_range(start=self.details["start_time"],
                                            periods=df.shape[0], freq="{}S".format(epoch_len))

        """TEMPORARY"""
        df.columns = ["Index", "Validity", "Timestamp"]
        return df

    def format_nonwear_data(self):

        if self.df_nonwear is None:
            print("\nNo nonwear data given --> running nonwear algorithm...")
            nw = BittiumNonwear(signal=self.ecg,
                                run_filter=True,
                                # temperature=ecg.signals[5],
                                # tri_accel=np.array([ecg.signals[1], ecg.signals[2], ecg.signals[3]]),
                                start_time=start_stamp,
                                sample_rate=fs, temp_sample_rate=1, accel_sample_rate=25,
                                epoch_len=self.details["epoch_length"], rolling_mean=None,
                                min_nw_dur=3, min_nw_break=5,
                                power_percent_thresh=80, freq_thresh=55,
                                reference_nonwear=None, quiet=False)

            self.df_nonwear = nw.df_nw

        if type(self.df_nonwear) is str:
            df = pd.read_csv(self.df_nonwear)
            if "Unnamed: 0" in df.columns:
                df = df[[i for i in df.columns if i != "Unnamed: 0"]]

            df["bout_start"] = pd.to_datetime(df["bout_start"])
            df["bout_end"] = pd.to_datetime(df["bout_end"])

            return df

        if type(self.df_nonwear) is pd.core.frame.DataFrame:
            return self.df_nonwear

    def remove_bad_quality(self, df_events, df_qc, show_boxplot=False,
                          excl_types=("Sinus", "Min. RR", "Afib Max. HR (total)", "Min. HR", "Max. HR", "Afib Min. HR (total)", "Max. RR")):

        print("\nRunning ECG signal quality check data on arrhythmia events...")
        df = df_events.copy()

        print(f"-Omitting {excl_types}")

        epoch_len = int((df_qc.iloc[1]["Timestamp"] - df_qc.iloc[0]['Timestamp']).total_seconds())

        event_contains_invalid = []
        for row in df.itertuples():
            # Row index in df_qc that start of event falls into
            epoch_start_ind = int(np.floor(row.Start_Ind / (epoch_len * self.details["sample_rate"])))

            # Row index in df_qc that end of event falls into (inclusive)
            epoch_stop_ind = int(np.ceil(row.Stop_Ind / (epoch_len * self.details["sample_rate"]))) + 1

            qc = df_qc.iloc[epoch_start_ind:epoch_stop_ind]  # Cropped QC df

            # Whether or not event contains invalid data
            event_contains_invalid.append("Invalid" if "Invalid" in qc["Validity"].unique() else "Valid")

        df["Validity"] = event_contains_invalid
        df_valid_arr = df.loc[df["Validity"] == "Valid"]
        df_valid_arr = df_valid_arr.loc[~df_valid_arr["Type"].isin(excl_types)]

        if show_boxplot:
            fig, axes = plt.subplots(1, figsize=(10, 6))
            df_valid_arr.boxplot(column="Duration", by="Type", ax=axes)

            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                tick.label.set_rotation(45)
            axes.set_ylabel("Seconds")

        orig_len = df_events.shape[0]
        new_len = df_valid_arr.shape[0]

        print(f"\nRemoved {orig_len - new_len} ({round(100 * (orig_len - new_len) / orig_len, 1)}%) "
              f"events due to poor-quality data.")

        # Descriptive stats for each arrhythmia's duration
        desc = df_valid_arr.groupby("Type")["Duration"].describe()
        desc["Duration"] = [df_valid_arr.loc[df_valid_arr["Type"] == row.Index]["Duration"].sum() for
                            row in desc.itertuples()]

        print("\nResults summary:")
        for row in desc.itertuples():
            bout_name = "bouts" if row.count > 1 else "bout"
            print(f"-{row.Index}: {int(row.count)} {bout_name}, total duration = {round(row.Duration, 1)} seconds")

        df_out = df_valid_arr[["Timestamp", "Type", "Duration"]]

        try:
            df_out["MeanNoise"] = df_valid_arr["MeanNoise"]
        except KeyError:
            pass

        return df_out, desc

    def remove_nonwear(self, df_events, nw_bouts, show_boxplot=False, pad_nw_window=30,
                       excl_types=("Sinus", "Min. RR", "Afib Max. HR (total)", "Min. HR", "Max. HR", "Afib Min. HR (total)", "Max. RR")):

        print("\nRemoving arrhythmia events that occur during detected non-wear periods...")

        print(f"-Omitting {excl_types}")
        df = df_events.loc[~df_events["Type"].isin(excl_types)]

        # Handling NW bout data -----------------
        epoch_stamps = pd.date_range(start=self.details["start_time"],
                                     end=(self.details["start_time"] + td(seconds=self.details["file_dur_sec"])),
                                     freq="1S")

        epoch_nw = np.zeros(len(epoch_stamps), dtype=bool)
        for row in nw_bouts.itertuples():
            try:
                epoch_nw[int(row.start_dp / self.details["sample_rate"]):
                         int(row.end_dp / self.details["sample_rate"])] = True
            except AttributeError:
                start_ind = int(np.floor((row.Start - self.details["start_time"]).total_seconds()))
                end_ind = int(np.ceil((row.Stop - self.details["start_time"]).total_seconds()))
                epoch_nw[start_ind:end_ind] = True

        df_nw = pd.DataFrame({'Timestamp': epoch_stamps, "Nonwear": epoch_nw})

        # Finds events that occur during nonwear bouts ---------
        event_contains_nw = []
        for row in df.itertuples():
            # Cropped event dataframe to nonwear events
            nw = df_nw.loc[(df_nw["Timestamp"] >= row.Timestamp + td(seconds=-pad_nw_window)) &
                           (df_nw["Timestamp"] <= row.Timestamp + td(seconds=row.Duration + pad_nw_window))]

            # Whether or not event contains invalid data
            event_contains_nw.append(True in nw["Nonwear"].unique())

        df["Nonwear"] = event_contains_nw
        df_valid_arr = df.loc[~df["Nonwear"]]  # locates events that are not non-wear

        if show_boxplot:
            fig, axes = plt.subplots(1, figsize=(10, 6))
            df_valid_arr.boxplot(column="Duration", by="Type", ax=axes)

            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                tick.label.set_rotation(45)
            axes.set_ylabel("Seconds")

        orig_len = df_events.shape[0]
        new_len = df_valid_arr.shape[0]

        print(f"\nRemoved {orig_len - new_len} ({round(100 * (orig_len - new_len) / orig_len, 1)}%) "
              f"events due to non-wear periods.")

        # Descriptive stats for each arrhythmia's duration
        desc = df_valid_arr.groupby("Type")["Duration"].describe()
        desc["Duration"] = [df_valid_arr.loc[df_valid_arr["Type"] == row.Index]["Duration"].sum() for
                            row in desc.itertuples()]

        print("\nResults summary:")
        for row in desc.itertuples():
            bout_name = "bouts" if row.count > 1 else "bout"
            print(f"-{row.Index}: {int(row.count)} {bout_name}, total duration = {round(row.Duration, 1)} seconds")

        return df_valid_arr, desc

    def combine_valid_dfs(self, show_boxplot=False):
        """Combines nonwear and quality check dataframes and returns df of events in both dfs"""

        print("\nCombining quality check and nonwear check cardiac events...")

        df_valid = self.df_valid_arr_nw.loc[self.df_valid_arr_nw["Timestamp"].isin(self.df_valid_arr_qc["Timestamp"])].reset_index(drop=True)

        if show_boxplot:
            fig, axes = plt.subplots(1, figsize=(10, 6))
            df_valid.boxplot(column="Duration", by="Type", ax=axes)

            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                tick.label.set_rotation(45)
            axes.set_ylabel("Seconds")

        orig_len = max(self.df_valid_arr_nw.shape[0], self.df_valid_arr_qc.shape[0])
        new_len = df_valid.shape[0]

        print(f"\nRemoved {orig_len - new_len} ({round(100 * (orig_len - new_len) / orig_len, 1)}%) "
              f"events due to non-wear periods.")

        # Descriptive stats for each arrhythmia's duration
        desc = df_valid.groupby("Type")["Duration"].describe()
        desc["Duration"] = [df_valid.loc[df_valid["Type"] == row.Index]["Duration"].sum() for
                            row in desc.itertuples()]

        print("\nResults summary:")
        for row in desc.itertuples():
            bout_name = "bouts" if row.count > 1 else "bout"
            print(f"-{row.Index}: {int(row.count)} {bout_name}, total duration = {round(row.Duration, 1)} seconds")

        if show_boxplot:
            if show_boxplot:
                fig, axes = plt.subplots(1, figsize=(10, 6))
                df_valid.boxplot(column="Duration", by="Type", ax=axes)

                for tick in axes.xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)
                    tick.label.set_rotation(45)
                axes.set_ylabel("Seconds")

        df_out = df_valid[["Timestamp", "Type", "Duration"]]

        try:
            df_out["MeanNoise"] = df_valid["MeanNoise"]
        except KeyError:
            pass

        return df_out, desc

    def plot_arrhythmias(self, df, downsample=3, plot_noise=True,
                         types=("COUP(mf)", "SALV(mf)", "GEM(mf)", "COUP", "AF",  "PAC/SVE", "IVR", "SALV", "Arrest", "Block", "GEM", "AV2/II", "ST+")):

        if self.ecg is None or self.timestamps is None:
            print("\nNeed raw data to generate this plot. Try again.")

        color_dict = {"Arrest": 'red', 'AF': 'orange', "COUP(mf)": 'dodgerblue', "COUP": 'dodgerblue',
                      "SALV(mf)": 'green', "SALV": 'green', "Block": 'purple', "GEM(mf)": "pink", "AV1": 'skyblue',
                      "GEM": 'pink', "PAC/SVE": 'grey', 'IVR': 'limegreen', "ST-": "gold", "ST+": "fuchsia",
                      "VT": 'navy', 'VT(mf)': 'navy', 'Tachy': 'navy', "Brady": 'turquoise', 'AV2/II': 'teal'}

        print(f"\nPlotting raw data ({round(self.details['sample_rate']/downsample)}Hz) with overlaid events:")

        if type(types) is list or type(types) is tuple:
            types = list(types)
            for key in types:
                print(f"{key} = {color_dict[key]}")
        if type(types) is str:
            types = [types]
            for key in types:
                print(f"{key} = {color_dict[key]}")

        data = df.loc[df["Type"].isin(types)]

        if data.shape[0] == 0:
            print(f"\nNo arrhythmias of type(s) {types} found.")

        if "COUP(mf)" in types and "COUP" in types:
            types.remove("COUP(mf)")
        if "SALV(mf)" in types and "SALV" in types:
            types.remove("SALV(mf)")
        if "GEM(mf)" in types and "GEM" in types:
            types.remove("GEM(mf)")

        max_val = max(self.ecg)
        min_val = min(self.ecg)

        if not plot_noise:
            fig, axes = plt.subplots(1, figsize=(12, 8))
            axes.plot(self.timestamps[::downsample], self.ecg[::downsample], color='black')

            for row in data.itertuples():
                # Fill between region
                axes.fill_between(x=[row.Timestamp, row.Timestamp + td(seconds=row.Duration)],
                                  y1=min_val*1.1, y2=max_val*1.1, color=color_dict[row.Type], alpha=.35)
                # Triangle markers for visibility
                axes.scatter(row.Timestamp + td(seconds=row.Duration/2), max_val*1.1, marker="v",
                             color=color_dict[row.Type])

            plt.title(f"Showing {types} events")
            axes.xaxis.set_major_formatter(xfmt)

        if plot_noise:
            fig, axes = plt.subplots(2, sharex='col', figsize=(12, 8))
            axes[0].plot(self.timestamps[::downsample], self.ecg[::downsample], color='black')

            for row in data.itertuples():
                # Fill between region
                axes[0].fill_between(x=[row.Timestamp, row.Timestamp + td(seconds=row.Duration)],
                                     y1=min_val * 1.1, y2=max_val * 1.1, color=color_dict[row.Type], alpha=.35)
                # Triangle markers for visibility
                axes[0].scatter(row.Timestamp + td(seconds=row.Duration / 2), max_val * 1.1, marker="v",
                                color=color_dict[row.Type])

            axes[1].plot(self.df_card_nav['Timestamp'], self.df_card_nav["MeanNoise"], marker='o', linestyle="", color='red')
            axes[1].set_ylabel("Noise")


def flag_events_as_gait(start_time, file_dur_sec, df_gait, df_arrhyth):

    """ ========================================= FLAGGING AS GAIT/NOT GAIT ======================================= """
    epoch_stamps = pd.date_range(start=start_time,
                                 end=(start_time + td(seconds=file_dur_sec)),
                                 freq="1S")

    binary_gait = np.zeros(len(epoch_stamps), dtype=bool)

    try:
        for row in df_gait.itertuples():
            start_ind = int((row.start_timestamp - start_time).total_seconds())
            end_ind = int((row.end_timestamp - start_time).total_seconds())

            binary_gait[start_ind:end_ind] = True

        df_gait_epoch = pd.DataFrame({'Timestamp': epoch_stamps, "Gait": binary_gait})

        # Finds events that occur during nonwear bouts ---------
        event_contains_gait = []
        for row in df_arrhyth.itertuples():
            # Cropped event dataframe to nonwear events
            gait_window = df_gait_epoch.loc[(df_gait_epoch["Timestamp"] >= row.Timestamp) &
                                            (df_gait_epoch["Timestamp"] <= row.Timestamp + td(seconds=row.Duration))]

            # Whether or not event contains invalid data
            event_contains_gait.append(True in gait_window["Gait"].unique())

        df_arrhyth["Gait"] = event_contains_gait

    except AttributeError:
        df_gait_epoch = pd.DataFrame({'Timestamp': epoch_stamps,
                                      "Gait": ["Unknown" for i in range(len(epoch_stamps))]})


def flag_events_as_sleep(start_time, file_dur_sec, df_sleep, df_arrhyth):

    """ ========================================= FLAGGING AS GAIT/NOT GAIT ======================================= """
    epoch_stamps = pd.date_range(start=start_time,
                                 end=(start_time + td(seconds=file_dur_sec)),
                                 freq="1S")

    binary_sleep = np.zeros(len(epoch_stamps), dtype=bool)

    try:
        for row in df_sleep.itertuples():
            start_ind = int((row.start_time - start_time).total_seconds())
            end_ind = int((row.end_time - start_time).total_seconds())

            binary_sleep[start_ind:end_ind] = True

        df_sleep_epoch = pd.DataFrame({'Timestamp': epoch_stamps, "Sleep": binary_sleep})

        # Finds events that occur during nonwear bouts ---------
        event_contains_sleep = []
        for row in df_arrhyth.itertuples():
            # Cropped event dataframe to nonwear events
            sleep_window = df_sleep_epoch.loc[(df_sleep_epoch["Timestamp"] >= row.Timestamp) &
                                              (df_sleep_epoch["Timestamp"] <= row.Timestamp + td(seconds=row.Duration))]

            # Whether or not event contains invalid data
            event_contains_sleep.append(True in sleep_window["Sleep"].unique())

        df_arrhyth["Sleep"] = event_contains_sleep

    except AttributeError:
        df_sleep_epoch = pd.DataFrame({'Timestamp': epoch_stamps,
                                       "Sleep": ["Unknown" for i in range(len(epoch_stamps))]})


def average_event_noise_values(df_beats, df_arrhyth):

    print("\nCalculating mean noise values from Cardiac Navigator for each arrhythmia event...")

    mean_noise = []
    for row in df_arrhyth.itertuples():

        df_beat = df_beats.loc[(df_beats["Timestamp"] >= row.Timestamp) &
                               (df_beats["Timestamp"] <= row.Timestamp + td(seconds=row.Duration))]
        avg = df_beat["NOISE"].mean()
        mean_noise.append(avg)

    df_arrhyth["MeanNoise"] = mean_noise
    print("Complete.")


def calculate_epoch_noise(df_qc, df_beat, epoch_len=15):
    print("\nCalculating mean noise values from Cardiac Navigator for each {}-second epoch...")

    mean_noise = []
    for row in df_qc.itertuples():
        if row.Index % 250 == 0:
            print(100 * row.Index / df_qc.shape[0])

        epoch = df_qc[(df_beat["Timestamp"] >= row.Timestamp) &
                      (df_beat["Timestamp"] <= row.Timestamp + td(seconds=epoch_len))]
        avg = epoch["NOISE"].mean()
        mean_noise.append(avg)

    df_qc["MeanNoise"] = mean_noise

    print("Complete.")


def generate_random_valid_period(signal, df_qc, start_time, image_n=1, image_of=1, filter_data=True,
                                 window_len=10, sample_f=250, figsize=(10, 6)):

    df_orph_valid = df_qc.loc[df_qc["Validity"] == "Valid"]
    rand_index = random.choice([i for i in df_orph_valid["Index"]])

    window = signal[rand_index:rand_index + int(sample_f * window_len)]

    if filter_data:
        window = filter_signal(data=window, sample_f=sample_f, low_f=.33, high_f=40,
                               filter_type='bandpass', filter_order=3)

    ts = pd.date_range(start=start_time + td(seconds=rand_index/sample_f), periods=len(window),
                       freq="{}ms".format(1000/sample_f))

    fig, ax = plt.subplots(figsize=figsize)

    n_hours = round((ts[0] - start_time).total_seconds() / 3600, 1)
    ax.plot(np.arange(len(window)) / sample_f, window, color='black')
    ax.set_title(f"Sample Clean Data #{image_n}/{image_of}\n{ts[0]} ({n_hours} hours into collection)", color='green')
    ax.set_ylabel("uV")
    ax.set_xlabel("Seconds")

    return fig


def print_n_events(include_events=('Sinus', 'ST(ref)', 'PAC/SVE', 'AV1', 'COUP(mf)', 'COUP', 'GEM(mf)', 'GEM', 'SALV(mf)', 'SALV', 'Min. RR', 'SVT', 'VT', 'VT(mf)', 'ST-', 'Arrest', 'Block', 'Max. HR', 'Min. HR', 'AV2/II', 'AF', 'Max. RR', 'Afib Max. HR (total)', 'Afib Min. HR (total)')):

    if arr.df_card_nav is not None:
        raw = arr.df_card_nav.loc[arr.df_card_nav["Type"].isin(include_events)]
        print(f"\nRaw: {raw.shape[0]} events")

        if arr.df_valid_arr_nw is not None:
            nw_arr = arr.df_valid_arr_nw.loc[arr.df_valid_arr_nw["Type"].isin(include_events)]
            print(f"Nonwear: {nw_arr.shape[0]} events ({(round(100*nw_arr.shape[0]/raw.shape[0], 1))}% remain)")

        if arr.df_valid_arr_qc is not None:
            qc_arr = arr.df_valid_arr_qc.loc[arr.df_valid_arr_qc["Type"].isin(include_events)]
            print(f"Quality check: {qc_arr.shape[0]} events ({(round(100*qc_arr.shape[0]/raw.shape[0], 1))}% remain)")

        if arr.df_valid is not None:
            valid_arr = arr.df_valid.loc[arr.df_valid["Type"].isin(include_events)]
            print(f"QC and NW: {valid_arr.shape[0]} events ({(round(100*valid_arr.shape[0]/raw.shape[0], 1))}% remain)")

        try:
            r = df_report.loc[df_report["Type"].isin(include_events)]
            print(f"In report: {r.shape[0]} events ({(round(100*r.shape[0]/raw.shape[0], 1))}% remain)")
        except (AttributeError, NameError):
            pass


def gen_stripchart(signal, start_stamp, df_beat=None, df_event=None, line_duration_sec=10,  sample_f=250,
                   plot_width_inch=10.0, title=None):

    # Signal bias removal and conversion to mV
    e = [i/1000 for i in signal]
    bias = np.mean(e)
    e_zeroed = [i-bias for i in e]

    # Plotting set-up ------------------------------------------------------------------------------------------------

    # How many subplots required to show all data, given line_duration value and length of data
    n_plots = int(np.ceil(len(signal) / (line_duration_sec * sample_f)))

    # Timestamps for each row
    ts_row = pd.date_range(start=start_stamp, periods=n_plots+1, freq=f'{line_duration_sec}S')

    # Indexes in signal for each row
    row_inds = [i for i in range(0, int(len(signal)), int(sample_f * line_duration_sec))]

    # Voltage ranges
    voltage_range = [min(e_zeroed), max(e_zeroed)]

    if abs(voltage_range[1] - voltage_range[0]) < 1.25:
        voltage_range = [min(e_zeroed), min(e_zeroed) + 1.25]

    # how many 1mm boxes required to contain voltage range +- .25mV
    # x10 since 1 box = .1mV
    y_range = (voltage_range[1] + .25 - (voltage_range[0] - .25)) * 10  # n 1mm lines

    # Height of each subplot given plot_width and signal amplitude
    h = y_range * plot_width_inch / (25 * (line_duration_sec + 1))

    # Figure set-up
    # Dimensions ensures grids are squares
    fig, ax = plt.subplots(n_plots, sharex='col', figsize=(plot_width_inch, h * n_plots))
    plt.subplots_adjust(hspace=0, top=.925, bottom=.075, left=.075, right=.925)

    if title is not None:
        plt.suptitle(title)

    # Plotting if only a single subplot ------------------------------------------------------------------------------
    if n_plots == 1:

        df_e = None

        if df_event is not None:
            df_e = df_event.copy()
            df_e['EndTimestamp'] = [row.Timestamp + td(seconds=row.Duration) for row in df_e.itertuples()]
            df_e = df_e.loc[(df_e["Timestamp"] >= start_stamp) & (df_e["EndTimestamp"] <= ts_row[-1])]

        # Loops through subplots
        for i in range(n_plots):
            # Crops data: fills each row. Last row likely not full
            try:
                d = e_zeroed[row_inds[i]:row_inds[i + 1]]
            except IndexError:
                d = e_zeroed[row_inds[i]:-1]

            # ECG data
            ax.plot([j / sample_f for j in np.arange(len(d))], d, color='black')

            # Adds faint vertical line every 1-sec
            for sec in range(0, line_duration_sec + 2):
                ax.axvline(sec, color='red', zorder=0, alpha=.35)

            # Y and X axis grid
            ax.set_xticks(np.arange(0, line_duration_sec + 1 + .01, .2))
            ax.set_xticks(np.arange(0, line_duration_sec + 1 + .01, .04), minor=True)

            ax.set_yticks(np.arange(voltage_range[0] - .25, voltage_range[1] + .5, .5))
            ytick_vals = ax.get_yticks()
            ax.set_yticks(np.arange(voltage_range[0] - .25, voltage_range[1] + .25, .1), minor=True)

            ax.grid(color='red', which='minor', linewidth=.35, alpha=.35)
            ax.grid(color='red', which='major', linewidth=.65, alpha=.65)
            ax.set_xlim(0, line_duration_sec + .825)

            ax.set_ylim(voltage_range[0] - .25, voltage_range[1] + .25)
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='x', colors='white')

            # Calibration square wave for first subplot: aligns with major grid lines
            if i == 0:
                # .4-second long wave
                sw = np.array([ytick_vals[1] for i in range(int(sample_f * .4))])

                # Sets middle .2-second segment to 1mV height
                sw[int(len(sw) / 2 - int(sample_f / 10)):int(len(sw) / 2 + int(sample_f / 10))] = ytick_vals[3]

                ax.plot([(j + len(d)) / sample_f + .12 for j in range(len(sw))], sw,
                        linewidth=.75, color='black', label='1mV\n0.2 sec')

                ax.text(x=line_duration_sec + .06, y=ytick_vals[0] + (ytick_vals[1] - ytick_vals[0]) / 4,
                        s="0.2s, 1mV", fontsize=8)

            # Figure out shading by row
            row_start = ts_row[i]
            row_stop = ts_row[i + 1]

            # Flags beats -------------------------------
            if df_beat is not None:
                df_beat_crop = df_beat.loc[(df_beat["Timestamp"] >= row_start) &
                                           (df_beat["Timestamp"] <= row_stop)]

                for row in df_beat_crop.itertuples():
                    ax.scatter(x=(row.Timestamp - row_start).total_seconds(), y=voltage_range[1] + .08,
                               marker='v', color='limegreen', s=25)

            # Shades arrhythmia event ------------------
            if df_event is not None:
                for row in df_e.itertuples():
                    if row.Timestamp >= row_start:
                        # Fills event start to end
                        if row.EndTimestamp <= row_stop:
                            ax.fill_between(x=[(row.Timestamp - row_start).total_seconds(),
                                               (row.EndTimestamp - row_start).total_seconds()],
                                            y1=voltage_range[0] - .25, y2=voltage_range[1] + .25,
                                            color='grey', alpha=.35)
                        # Fills event start to end of row if proceeds onto next row
                        if row.EndTimestamp > row_stop:
                            ax.fill_between(x=[(row.Timestamp - row_start).total_seconds(), len(d) / sample_f],
                                            y1=voltage_range[0] - .25, y2=voltage_range[1] + .25,
                                            color='grey', alpha=.35)

    # Plotting if multiple subplots ----------------------------------------------------------------------------------
    if n_plots >= 2:
        df_e = None

        if df_event is not None:
            df_e = df_event.copy()
            df_e['EndTimestamp'] = [row.Timestamp + td(seconds=row.Duration) for row in df_e.itertuples()]
            df_e = df_e.loc[(df_e["Timestamp"] >= start_stamp) & (df_e["EndTimestamp"] <= ts_row[-1])]

        # Loops through subplots
        for i in range(n_plots):
            # Crops data: fills each row. Last row likely not full
            try:
                d = e_zeroed[row_inds[i]:row_inds[i+1]]
            except IndexError:
                d = e_zeroed[row_inds[i]:-1]

            # ECG data
            ax[i].plot([j / sample_f for j in np.arange(len(d))], d, color='black')

            # Adds faint vertical line every 1-sec
            for sec in range(0, line_duration_sec + 2):
                ax[i].axvline(sec, color='red', zorder=0, alpha=.35)

            # Y and X axis grid
            ax[i].set_xticks(np.arange(0, line_duration_sec + 1 + .01, .2))
            ax[i].set_xticks(np.arange(0, line_duration_sec + 1 + .01, .04), minor=True)

            ax[i].set_yticks(np.arange(voltage_range[0] - .25, voltage_range[1] + .5, .5))
            ytick_vals = ax[i].get_yticks()
            ax[i].set_yticks(np.arange(voltage_range[0] - .25, voltage_range[1] + .25, .1), minor=True)

            ax[i].grid(color='red', which='minor', linewidth=.35, alpha=.35)
            ax[i].grid(color='red', which='major', linewidth=.65, alpha=.65)
            ax[i].set_xlim(0, line_duration_sec + .625)

            ax[i].set_ylim(voltage_range[0] - .25, voltage_range[1] + .25)
            ax[i].tick_params(axis='y', colors='white')
            ax[i].tick_params(axis='x', colors='white')

            # Calibration square wave for first subplot: aligns with major grid lines
            if i == 0:
                # .4-second long wave
                sw = np.array([ytick_vals[1] for i in range(int(sample_f * .4))])

                # Sets middle .2-second segment to 1mV height
                sw[int(len(sw) / 2 - int(sample_f / 10)):int(len(sw) / 2 + int(sample_f / 10))] = ytick_vals[3]

                ax[0].plot([(j + len(d))/sample_f + .12 for j in range(len(sw))], sw,
                           linewidth=.75, color='black', label='1mV\n0.2 sec')

                ax[0].text(x=line_duration_sec + .06, y=ytick_vals[0] + (ytick_vals[1] - ytick_vals[0])/4,
                           s="0.2s, 1mV", fontsize=8)

            # Figure out shading by row
            row_start = ts_row[i]
            row_stop = ts_row[i+1]

            # Flags beats -------------------------------
            if df_beat is not None:
                df_beat_crop = df_beat.loc[(df_beat["Timestamp"] >= row_start) &
                                           (df_beat["Timestamp"] <= row_stop)]

                for row in df_beat_crop.itertuples():
                    ax[i].scatter(x=(row.Timestamp - row_start).total_seconds(), y=voltage_range[1] + .08,
                                  marker='v', color='limegreen', s=25)

            # Shades arrhythmia event ------------------
            if df_event is not None:
                for row in df_e.itertuples():
                    if row.Timestamp >= row_start:
                        # Fills event start to end
                        if row.EndTimestamp <= row_stop:
                            ax[i].fill_between(x=[(row.Timestamp - row_start).total_seconds(),
                                                  (row.EndTimestamp - row_start).total_seconds()],
                                               y1=voltage_range[0] - .25, y2=voltage_range[1] + .25,
                                               color='grey', alpha=.35)
                        # Fills event start to end of row if proceeds onto next row
                        if row.EndTimestamp > row_stop:
                            ax[i].fill_between(x=[(row.Timestamp - row_start).total_seconds(), len(d) / sample_f],
                                               y1=voltage_range[0] - .25, y2=voltage_range[1] + .25,
                                               color='grey', alpha=.35)

    return fig


def gen_sample_pdf2(signal, df_arrhythmia, df_arrhythmia_all, start_timestamp, beat_data,
                    df_qc=None, bad_data_removed=True, sample_f=250, save_pdf=True, img_dpi=150,
                   arrhythmias=("VT", "Arrest", "AF", "Brady", "Tachy", "Block", "ST+", "AV2/II"),
                   img_folder="C:/Users/ksweber/Desktop/TemporaryImages/",
                    show_n_seconds=30, seconds_per_line=10, min_pad_len=10,
                   include_sample_data=3, orientation='vertical', collect_dur_hours=0,
                   save_dir="C:/Users/ksweber/Desktop/", logo_link="C:/Users/ksweber/Pictures/HANDDS-logo.jpg",
                   include_all_events=True):

    names_dict = {"Sinus": "Sinus", "PAC/SVE": "Premature atrial contraction/supraventricular ectopic beat",
                  "AV1": "1º AV block",
                  "COUP(mf)": "Multifocal couplet", "COUP": "Couplet",
                  "GEM(mf)": "Multifocal geminy", "GEM": "Geminy",
                  "SALV": "Salvos", "SALV(mf)": "Multifocal Salvos",
                  "SVT": "Supraventricular tachycardiac", "VT": "Ventricular tachycardia", "Tachy": "Tachycardia",
                  "VT(mf)": "Multifocal ventricular tachycardia", "Brady": "Bradycardia",
                  "ST-": "ST depression", "ST+": "ST elevation",
                  "Arrest": "Arrest", "Block": "Block", "AV2II": "2º AV block/Mobitz II", "AF": "Atrial fibrillation",
                  "IVR": "Idioventricular rhythm"}
    img_list = []
    titles = []
    ts = pd.date_range(start=start_timestamp, periods=len(signal), freq="{}ms".format(1000 / sample_f))

    # only includes desired arrhythmias ---------------------------------------
    all_detected = df_arrhythmia_all['Type'].unique()  # all types in original file
    df_arrhythmia = df_arrhythmia.loc[df_arrhythmia["Type"].isin(arrhythmias)]  # only desired arrhythmias

    print("\nGenerating report of the following arrhythmias:")
    print(arrhythmias)

    # arrhythmias to exclude --------------------------------
    excl = [i for i in all_detected if i not in arrhythmias]

    # list of detected arrhythmias
    found_arrs = df_arrhythmia["Type"].unique()
    print("\nDetected arrhythmias in current file:")
    print(found_arrs)

    # Descriptive stats of durations for each arrhythmia type
    try:
        df_desc = df_arrhythmia.groupby("Type")["Duration"].describe()
    except ValueError:
        df_desc = None

    # generating data for all arrhythmia events --------------------------------
    print("\nGenerating data for all events...")
    for t in found_arrs:
        n_events = int(df_desc.loc[t]["count"])  # number of events for each condition
        print(f"-{t} ({n_events} events)")

        row_i = 0

        df_current_arr = df_arrhythmia.loc[df_arrhythmia["Type"] == t]

        for row in df_current_arr.itertuples():

            # creates string to explain gait/sleep context --------
            context_str = ["During"]
            if row.Gait:
                context_str.append("gait")
                if row.Sleep:
                    context_str.append("and sleep")
            if not row.Gait:
                if row.Sleep:
                    context_str.append("sleep")
            if not row.Gait and not row.Sleep:
                context_str = ["Awake, no ambulation"]

            context_join = " ".join(context_str)
            ts_formatted = str(row.Timestamp.round("1S").strftime("%A, %B %m, %Y at %H:%M:%S"))
            title = "{}: event #{}/{}; {} ({} seconds); {}".format(names_dict[t], str(int(row_i + 1)), n_events,
                                                                   ts_formatted,
                                                                   round(row.Duration, 1),
                                                                   context_join)

            # Data cropping
            pad = (show_n_seconds - row.Duration) / 2
            if pad < min_pad_len:
                pad = min_pad_len

            window_start = row.Timestamp + td(seconds=-pad)
            window_end = row.Timestamp + td(seconds=row.Duration + pad)
            start_ind = int((window_start - start_timestamp).total_seconds() * sample_f)
            stop_ind = int((window_end - start_timestamp).total_seconds() * sample_f)

            data = signal[start_ind:stop_ind]

            df_beat = beat_data.loc[(beat_data["Timestamp"] >= ts[start_ind]) &
                                    (beat_data["Timestamp"] <= ts[stop_ind])]

            plt.close("all")

            fig = gen_stripchart(signal=data, start_stamp=ts[start_ind],
                                 df_beat=df_beat, df_event=df_arrhythmia,
                                 sample_f=sample_f, line_duration_sec=seconds_per_line,
                                 plot_width_inch=15, title=None)

            title = title + f"; Plot width = {seconds_per_line} seconds"
            titles.append(title)

            if save_pdf:
                # Saves png file
                t_name = t if t != "PAC/SVE" else "PACSVE"
                plt.savefig(f"{img_folder}{t_name}_Index{row.Index}.png", dpi=img_dpi)
                img_list.append(f"{img_folder}{t_name}_Index{row.Index}.png")

                row_i += 1

    titles_dict = {}
    for n, t in zip(img_list, titles):
        titles_dict[n] = t

    plt.close("all")

    """ ============================================= PDF GENERATION ========================================= """
    if save_pdf:
        print(f"\nCombining {include_sample_data + df_arrhythmia.shape[0]} images into PDF...")
        # Creates pdf from all png files

        pdf = FPDF("L", 'mm', 'Letter')
        pdf.set_auto_page_break(auto=True)
        pdf.add_page()

        # TITLE PAGE -----------------------------------------------------------------------------------------------
        if os.path.exists(logo_link):
            pdf.image(name=logo_link, x=200, y=140, w=75, h=75)

        if df_qc is not None and bad_data_removed:
            p_valid = round(100 * df_qc["Validity"].value_counts().loc["Valid"] / df_qc.shape[0], 1)
        if df_qc is None or not bad_data_removed:
            p_valid = 100

        pdf.set_font("Arial", 'B', 12)

        # w=0 makes cell the width of the page
        pdf.cell(w=0, h=8, txt=f"Subject {subj}", ln=True, align="L")
        pdf.cell(w=0, h=8, txt="", ln=True, align="L")

        pdf.set_font("Arial", 'U', 12)
        pdf.cell(w=0, h=8, txt="Collection details", ln=True, align="L")

        pdf.set_font("Arial", '', 12)
        pdf.cell(w=0, h=8, txt=f"-Device orientation: {orientation}", ln=True, align="L")

        coll_days = int(np.floor(collect_dur_hours / 24))
        coll_hours = int(np.floor(collect_dur_hours - 24 * coll_days))
        coll_mins = int(collect_dur_hours * 60 - 24*60 * coll_days - 60 * coll_hours)
        coll_dur_str = "{} day{}, {} hour{}, {} minute{}".format(coll_days, "s" if coll_days != 1 else "",
                                                                 coll_hours, "s" if coll_hours != 1 else "",
                                                                 coll_mins, "s" if coll_mins != 1 else "")
        pdf.cell(w=0, h=8, txt=f"-Collection duration: {coll_dur_str}", ln=True, align="L")

        usable_dur = round(p_valid/100*collect_dur_hours, 1)
        coll_days = int(np.floor(usable_dur / 24))
        coll_hours = int(np.floor(usable_dur - 24 * coll_days))
        coll_mins = int(usable_dur * 60 - 24*60 * coll_days - 60 * coll_hours)
        coll_dur_str = "{} day{}, {} hour{}, {} minute{}".format(coll_days, "s" if coll_days != 1 else "",
                                                                 coll_hours, "s" if coll_hours != 1 else "",
                                                                 coll_mins, "s" if coll_mins != 1 else "")
        pdf.cell(w=0, h=8, txt=f"     -Used in analysis: {coll_dur_str}", ln=True, align="L")

        pdf.cell(w=0, h=8, txt=" ", ln=True, align="L")

        pdf.set_font("Arial", 'U', 12)
        pdf.cell(w=0, h=8, txt="Arrhythmia analysis", ln=True, align="L")

        pdf.set_font("Arial", '', 12)
        pdf.cell(w=0, h=8, txt=f"-Clinically significant arrhythmias:", ln=True, align="L")

        try:
            for row in df_desc.itertuples():
                event_label = "event" if row.count == 1 else "events"
                pdf.cell(w=0, h=8, txt=f"     -{names_dict[row.Index]}: {int(row.count)} {event_label}, "
                                       f"total duration = {round(row.mean*row.count, 1)} seconds",
                         ln=True, align="L")
        except AttributeError:
            pdf.cell(w=0, h=8, ln=True, align="L", txt="     -No clinically significant events found.")

        pdf.set_font("Arial", '', 12)

        if include_sample_data > 0 and df_qc is None:
            print("\nCannot generate random, clean data - no quality check dataframe given.")

        if include_sample_data > 0 and df_qc is not None:
            for i in range(1, include_sample_data+1):
                plt.close("all")

                df_orph_valid = df_qc.loc[df_qc["Validity"] == "Valid"]
                rand_index = random.choice([i for i in df_orph_valid["Index"]])

                window = signal[rand_index:rand_index + int(sample_f * 40)]

                df_beat = beat_data.loc[(beat_data["Timestamp"] >= ts[rand_index]) &
                                        (beat_data["Timestamp"] <= ts[rand_index + int(sample_f * 40)])]

                fig = gen_stripchart(signal=window, start_stamp=ts[rand_index],
                                     df_beat=df_beat, df_event=None,
                                     sample_f=sample_f, line_duration_sec=seconds_per_line,
                                     plot_width_inch=15, title=None)

                fig.savefig(f"{img_folder}SampleData{i}.png", dpi=img_dpi)
                img_list.insert(i-1, f"{img_folder}SampleData{i}.png")

                hours_into = round((ts[rand_index] - start_timestamp).total_seconds()/3600, 1)
                titles_dict[f"{img_folder}SampleData{i}.png"] = f"SampleData{i}: {hours_into} hours into collection; " \
                                                                f"Plot width = {show_n_seconds} seconds"

            plt.close("all")

        # DATA PAGES -------------------------------------------------------------------------------------------------
        pdf.set_font("Arial", '', 14)

        sample_n = 1
        for i, img in enumerate(img_list):
            pdf.add_page(orientation="L")

            if "SampleData" not in img:
                try:
                    pdf.cell(w=0, h=10, txt=titles_dict[img].split(";")[0], ln=True, align="C")
                    pdf.cell(w=0, h=10, txt=titles_dict[img].split(";")[1], ln=True, align="C")
                    pdf.cell(w=0, h=10, txt=titles_dict[img].split(";")[2], ln=True, align="C")
                    pdf.cell(w=0, h=10, txt=titles_dict[img].split(";")[3], ln=True, align="C")

                except KeyError:
                    pass
                pdf.image(img, x=-20, y=50, w=315)
            if "SampleData" in img:
                pdf.cell(w=0, h=10, txt=f"Sample Clean Data #{sample_n}: " +
                                        titles_dict[img].split(";")[0].split(":")[1], ln=True, align="C")
                pdf.cell(w=0, h=10, txt=titles_dict[img].split(";")[1], ln=True, align="C")
                pdf.image(img, x=-20, y=50, w=315)
                sample_n += 1

            os.remove(img)

        pdf.output("{}{}_{}.pdf".format(save_dir, subj, "Random" if not include_all_events else "Complete"))
        print(f"PDF created (saved to {save_dir}).")

    return df_arrhythmia


""" ================================================= RUNNING SCRIPT ============================================== """


subj = "OND09_0001"
"""
ecg, f, start_stamp, ts, fs, df_nw, df_gait, df_sleep_alg, file_dur = load_data(filepath_fmt=f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{subj}_01_BF36_Chest.edf",
                                                                                nonwear_file="C:/Users/ksweber/Desktop/OmegaSnap_Nonwear.xlsx",
                                                                                gait_file=f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/{subj}_01_GAIT_BOUTS.csv",
                                                                                sleep_output_file=f"W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/{subj}_01_SLEEP_BOUTS.csv"
                                                                                )

arr = ArrhythmiaProcessor(
                          raw_ecg=ecg,
                          epoched_nonwear=df_nw,
                          epoched_signal_quality=f"C:/Users/ksweber/Desktop/Processed ECG/OND09/Orphanidou/{subj}_Orphanidou.csv",
                          card_nav_data=f"C:/Users/ksweber/Desktop/Processed ECG/OND09/CardiacNavigator1.5/{subj}_Arrhythmias_CustomNotFinal_CardiacNavigator15.csv",
                          card_nav_rr_data=f"C:/Users/ksweber/Desktop/Processed ECG/OND09/CardiacNavigator1.5/{subj}_Default_RRDetails_CardiacNavigator15.csv",
                          details={"start_time": start_stamp, "sample_rate": fs,
                                   "file_dur_sec": file_dur, "epoch_length": 15})

# average_event_noise_values(df_beats=arr.df_rr, df_arrhyth=arr.df_card_nav)

arr.df_valid_arr_qc, arr.df_desc_qc = arr.remove_bad_quality(df_events=arr.df_card_nav, df_qc=arr.df_qc_epochs, show_boxplot=False)
arr.df_valid_arr_nw, arr.df_desc_nw = arr.remove_nonwear(df_events=arr.df_card_nav, nw_bouts=arr.df_nonwear, show_boxplot=False)
arr.df_valid, arr.df_desc = arr.combine_valid_dfs(show_boxplot=False)

flag_events_as_gait(start_time=start_stamp, file_dur_sec=file_dur, df_gait=df_gait, df_arrhyth=arr.df_valid_arr_nw)
flag_events_as_sleep(start_time=start_stamp, file_dur_sec=file_dur, df_sleep=df_sleep_alg, df_arrhyth=arr.df_valid_arr_nw)
# calculate_epoch_noise(df_qc=arr.df_qc_epochs, df_beat=arr.df_rr, epoch_len=15)

handds_arrs = ["VT", "Arrest", "AF", "Brady", "Tachy", "Block", "AV2II"]
other_arrs = ("COUP", "COUP(mf)", "PAC/SVE", "GEM", "VT", "Arrest", "AF", "Brady", "Tachy", "Block", "ST+")
# arr.plot_arrhythmias(arr, df=arr.df_valid_arr_nw, downsample=3, types=other_arrs, plot_noise=True)
# arr.df_valid.to_csv(f"C:/Users/ksweber/Desktop/{subj}_ValidArrhythmias_CustomNotFinal_CardiacNavigator15.csv", index=False)
d = arr.df_valid_arr_nw.loc[arr.df_valid_arr_nw["Type"].isin(handds_arrs)]
print(d.value_counts("Type"))"""


df_report = gen_sample_pdf2(df_arrhythmia=arr.df_valid_arr_nw, df_arrhythmia_all=arr.df_card_nav,
                            signal=f, beat_data=arr.df_rr,
                            sample_f=fs, start_timestamp=start_stamp,
                            show_n_seconds=30, seconds_per_line=10, min_pad_len=10,
                            include_all_events=True, save_pdf=True, img_dpi=125,
                            arrhythmias=handds_arrs, include_sample_data=2,
                            df_qc=arr.df_qc_epochs, bad_data_removed=False,
                           orientation='vertical', collect_dur_hours=file_dur/3600)

# print_n_events(handds_arrs)

"""
raw = arr.df_card_nav.loc[arr.df_card_nav["Type"].isin(handds_arrs)]
print(f"Raw: {raw.shape[0]} events total")

valid_nw = arr.df_valid_arr_nw.loc[arr.df_valid_arr_nw["Type"].isin(handds_arrs)]
print(f"Non-wear removed: {valid_nw.shape[0]} events remain")

valid_qc = arr.df_valid_arr_qc.loc[arr.df_valid_arr_qc["Type"].isin(handds_arrs)]
print(f"Bad quality removed: {valid_qc.shape[0]} events remain")
"""

""" ======================================================= IGNORE ================================================ """

# TODO
# Put script in nwutils.Scripts
    # Uninstall nwutils package -> ensure importing from 'local' version/folder
# Rerun Orphanidou for 0008 (lower voltage threshold)
# Check how many raw arrhythmias in arrhythmia list --> manageable


def plot_segment(start_ind=None, win_len_sec=15, sample_f=250, show_noise=False, plot_pq=False, plot_qt=False,
                 plot_qrs=False, plot_adj_qrs=False, plot_raw_qrs=True):

    plt.close('all')
    win_len = int(win_len_sec*sample_f)

    # If start_ind is None, generate random section of data
    if start_ind is None:
        start_ind = random.choice(np.arange(len(ecg) - win_len))

    # .5-30Hz bandpass filter
    f = filter_signal(data=ecg[start_ind:start_ind + win_len], sample_f=sample_f,
                      filter_type='bandpass', filter_order=3, low_f=.5, high_f=30)

    # Crops dataframe to given section
    rr = arr.df_rr.loc[(arr.df_rr["Timestamp"] >= start_stamp + td(seconds=int(start_ind / sample_f)))]
    rr = rr.loc[rr['Timestamp'] <= start_stamp + td(seconds=int((start_ind+win_len)/sample_f))]

    # Raw data timestamps
    ts = pd.date_range(start=start_stamp + td(seconds=start_ind/sample_f),
                       periods=win_len, freq="{}ms".format(int(1000/sample_f)))

    # Attempts to find timestamp of QRS peak
    # Timestamp for row approximately end of QRS complex or T-wave peak?
    rr["QRS_stamp"] = [row.Timestamp + td(seconds=-(row.QRS - row.QRn)/1000) for row in rr.itertuples()]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    plt.suptitle(f"Start index = {start_ind}")

    # Raw and filtered data
    ax.plot(ts, ecg[start_ind:start_ind+win_len], color='red', zorder=1, label="Raw")
    ax.plot(ts, f, color='black', zorder=1, label="Filt")

    if show_noise:
        # Noise value given by Cardiac Navigator for each beat
        ax2 = ax.twinx()
        ax2.plot(rr["Timestamp"], rr["NOISE"], color='dodgerblue')
        ax2.set_ylabel("Noise", color='dodgerblue')

    ax.legend(loc='upper left')
    ax.set_title("Grey line = Msec timestamp; black line = 'adjusted' timestamp;\n"
                 "PQ = fuchsia; QR(up) = orange; QRS = green; QT = blue")

    true_qrs_stamp = []
    for row in rr.itertuples():

        ind = int((row.QRS_stamp - start_stamp).total_seconds() * fs)  # QRS_stamp index
        window = np.array(f[ind - int(fs / 5):ind + int(fs / 5)])  # 200ms window around QRS_stamp

        if len(window) == 0:
            true_qrs_stamp.append(row.QRS_stamp)

        # Finds timestamp of highest peak
        if len(window) > 0:
            max_ind = np.argwhere(window == max(window))[0][0]  # Index of tallest peak
            true_qrs_stamp.append(row.QRS_stamp + td(seconds=-.25 + max_ind / fs))

    rr["TrueQRS"] = true_qrs_stamp

    for row in rr.itertuples():

        if row.TrueQRS >= ts[0]:
            if plot_adj_qrs:
                ax.axvline(row.TrueQRS, linestyle='dashed', color='black')
            if plot_raw_qrs:
                ax.axvline(ts[0] + td(seconds=(row.Msec - rr.iloc[0]["Msec"])/1000), linestyle='dotted', color='grey')

            if plot_qrs:
            # QRS
                ax.fill_between(x=[row.TrueQRS + td(seconds=-row.QRS / 1000 / 2),
                                   row.TrueQRS + td(seconds=row.QRS / 1000 / 2)],
                                y1=0, y2=1000, color='green', alpha=.5)

            peak_ind = int((row.TrueQRS - (start_stamp + td(seconds=start_ind/sample_f))).total_seconds() * sample_f)

            # QRn
            # ax.fill_between(x=[row.TrueQRS + td(seconds=-row.QRn/1000), row.TrueQRS], y1=1000, y2=2000, color='orange', alpha=.5)

            if plot_pq:
                # PQ
                ax.fill_between(x=[row.TrueQRS + td(seconds=-row.QRS/1000/2 - row.PQ/1000),
                                   row.TrueQRS + td(seconds=-row.QRS/1000/2)],
                                y1=0, y2=1000, color='fuchsia', alpha=.5)

            if plot_qt:
                # QT
                ax.fill_between(x=[row.TrueQRS + td(seconds=-row.QRS/1000/2),
                                   row.TrueQRS + td(seconds=row.QT/1000)], y1=-1000, y2=0, color='dodgerblue', alpha=.5)

    return rr, start_ind, fig


# rr, index, fig = plot_segment(start_ind=None, win_len_sec=5, sample_f=fs, show_noise=False, plot_pq=False, plot_qt=False, plot_adj_qrs=False, plot_raw_qrs=True)

# fig, tld = gen_stripchart(signal=window, start_stamp=ts[rand_index], sample_f=250, line_duration_sec=8, plot_width_inch=17.5)


def crop_long_periods():
    pass


# TODO
# Figure out how to show overlaps on long plots
"""window_len = 30
min_pad = 10

test_event = arr.df_valid_arr_nw.loc[arr.df_valid_arr_nw["Duration"] > 60].sort_values("Duration")
test_row = test_event.iloc[0]

fig, ax = plt.subplots(3, sharex='col')
if window_len <= test_row["Duration"] <= window_len * 4:

    # Start segment
    start_ind = (test_row["Start_Ind"] - int(window_len/2 * fs))
    end_ind = (test_row["Start_Ind"] + int(window_len/2 * fs))
    # ax[0].plot(ts[start_ind:end_ind], f[start_ind: end_ind])
    gen_stripchart(signal=f[start_ind:end_ind], start_stamp=ts[start_ind], df_beat=None,
                   df_event=test_event, line_duration_sec=10, sample_f=250,
                   plot_width_inch=10.0, title=None)

    # Middle segment
    middle_ind = (test_row["Stop_Ind"] - test_row["Start_Ind"]) / 2
    middle_start_ind = int(middle_ind + test_row["Start_Ind"] - int(window_len/2*fs))
    middle_end_ind = int(middle_ind + test_row["Start_Ind"] + int(window_len/2*fs))
    # ax[1].plot(ts[middle_start_ind:middle_end_ind], f[middle_start_ind: middle_end_ind])
    gen_stripchart(signal=f[middle_start_ind:middle_end_ind], start_stamp=ts[middle_start_ind], df_beat=None,
                   df_event=test_event, line_duration_sec=10, sample_f=250,
                   plot_width_inch=10.0, title=None)

    # End segment
    end_start_ind = test_row["Stop_Ind"] - int(window_len/2*fs)
    end_end_ind = test_row["Stop_Ind"] + int(window_len/2*fs)
    # ax[2].plot(ts[end_start_ind:end_end_ind], f[end_start_ind: end_end_ind])

    gen_stripchart(signal=f[end_start_ind:end_end_ind], start_stamp=ts[end_start_ind], df_beat=None,
                   df_event=test_event, line_duration_sec=10, sample_f=250,
                   plot_width_inch=10.0, title=None)

    # Flags other sections on first section
    if middle_start_ind < end_ind:
        ax[0].axvline(ts[middle_start_ind], color='dodgerblue')
    if end_start_ind < end_ind:
        ax[0].axvline(ts[end_start_ind], color='red')

    # Flags other sections on middle data
    if end_ind > middle_start_ind:
        ax[1].axvline(ts[end_ind], color='orange')
    if end_start_ind < middle_end_ind:
        ax[1].axvline(ts[end_start_ind], color='red')

    if end_ind > end_start_ind:
        ax[2].avhline(ts[end_ind], color='dodgerblue')
    if middle_end_ind > end_start_ind:
        ax[2].axvline(ts[middle_end_ind], color='orange')
"""