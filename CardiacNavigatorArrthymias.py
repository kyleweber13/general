import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import pyedflib
from datetime import timedelta as td
import os
import Filtering
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

    f = filter_signal(data=ecg, sample_f=fs, filter_type='bandpass', filter_order=3, low_f=.5, high_f=30)

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

        self.df_valid_arr_qc, self.df_desc_qc = self.remove_bad_quality(df_events=self.df_card_nav,
                                                                        df_qc=self.df_qc_epochs,
                                                                        show_boxplot=False)
        self.df_valid_arr_nw, self.df_desc_nw = self.remove_nonwear(df_events=self.df_card_nav,
                                                                    nw_bouts=self.df_nonwear,
                                                                    show_boxplot=False)

        if self.df_valid_arr_nw is not None and self.df_valid_arr_qc is not None:
            self.df_valid, self.df_desc = self.combine_valid_dfs(show_boxplot=False)

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

    def find_longqt(self):

        print("\nFinding long QT beats...")
        long_qt = arr.df_rr.loc[arr.df_rr["QT"] >= 500]
        long_qt["Start_Ind"] = [int(i / 1000 * arr.details["sample_rate"]) for i in long_qt["Msec"]]

        stamps = [i for i in long_qt["Timestamp"]]
        start_inds = [i for i in long_qt["Start_Ind"]]
        stop_inds = [int(start + (t2 - t1).total_seconds() * arr.details["sample_rate"]) for start, t1, t2 in
                     zip(start_inds, stamps[:], stamps[1:])]
        stop_inds.append(start_inds[-1])
        long_qt["Stop_Ind"] = stop_inds
        long_qt["Type"] = ["LongQT" for i in range(long_qt.shape[0])]
        long_qt["Info"] = ["AUTOMATIC" for i in range(long_qt.shape[0])]
        long_qt["Duration"] = [(stop - start) / arr.details["sample_rate"] for
                               start, stop in zip(start_inds, stop_inds)]

        long_qt = long_qt[["Timestamp", "Start_Ind", "Stop_Ind", "Duration", "Type", "Info"]]

        df = self.df_card_nav.append(long_qt)
        df = df.sort_values("Timestamp")

        return df

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

            # Row index in df_qc that end of event falls into (includsive)
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

        return df_valid_arr[["Timestamp", "Type", "Duration"]], desc

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

        return df_valid[["Timestamp", "Type", "Duration"]], desc

    def plot_arrhythmias(self, df, downsample=3,
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


def generate_random_valid_period(signal, df_qc, start_time, image_n=1, image_of=1,
                                 window_len=10, sample_f=250, figsize=(10, 6)):

    df_orph_valid = df_qc.loc[df_qc["Validity"] == "Valid"]
    rand_index = random.choice([i for i in df_orph_valid["Index"]])

    window = signal[rand_index:rand_index + int(sample_f * window_len)]
    ts = pd.date_range(start=start_time + td(seconds=rand_index/sample_f), periods=len(window),
                       freq="{}ms".format(1000/sample_f))

    fig, ax = plt.subplots(figsize=figsize)

    n_hours = round((ts[0] - start_time).total_seconds() / 3600, 1)
    ax.plot(np.arange(len(window)) / sample_f, window, color='green')
    ax.set_title(f"{subj}: Sample Clean Data {image_n}/{image_of}\n{ts[0]} ({n_hours} hours into collection)", color='green')
    ax.set_ylabel("uV")
    ax.set_xlabel("Seconds")

    return fig


def gen_sample_pdf(signal, df_arrhythmia, start_timestamp, df_qc, beat_data, sample_f=250, save_pdf=True,
                   arrhythmias=("VT", "Arrest", "AF", "Brady", "Tachy", "Block", "ST+", "AV2/II"),
                   img_folder="C:/Users/ksweber/Desktop/TemporaryImages/", show_n_seconds=15,
                   include_sample_data=3, orientation='vertical', collect_dur_hours=0,
                   save_dir="C:/Users/ksweber/Desktop/", logo_link="C:/Users/ksweber/Pictures/HANDDS-logo.jpg",
                   pad_short_events=True, include_all_events=False):

    # only includes desired arrhythmias
    all_detected = arr.df_valid['Type'].unique()  # all types in original file
    df_arrhythmia = df_arrhythmia.loc[df_arrhythmia["Type"].isin(arrhythmias)]  # only desired arrhythmias
    print("\nGenerating report of the following arrhtyhmias:")
    print(arrhythmias)

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

    img_list = []

    # Loops each arrhythmia and generates a plot of a random event
    if not include_all_events:
        print("\nRandomly selecting one event from each arrhythmia type...")

    if include_all_events:
        print("\nGenerating data for all events...")

    if len(found_arrs) > 0:
        for t in found_arrs:
            print(f"-{t}")
            n_events = int(df_desc.loc[t]["count"])
            df_inds = [i for i in df_arrhythmia.loc[df_arrhythmia["Type"] == t].index]

            if not include_all_events:
                # randomly picks one event if more than one event detected
                if n_events > 1:
                    ind = random.choice(range(len(df_inds)-1))
                    row = df_inds[ind]

                if n_events == 1:
                    ind = 0
                    row = df_inds[0]

            if include_all_events:
                row = df_inds

            d = df_arrhythmia.loc[row]

            row_i = 0
            for row in d.itertuples():

                # Cropping data
                if row.Duration < show_n_seconds:
                    if pad_short_events:
                        pad = show_n_seconds - row.Duration
                    if not pad_short_events:
                        pad = 2

                    start_ind = int((row.Timestamp - start_timestamp).total_seconds() * sample_f - pad/2 * sample_f)
                    stop_ind = int((row.Timestamp + td(seconds=row.Duration) - start_timestamp).total_seconds() *
                                   sample_f + pad/2 * sample_f)

                if row.Duration >= show_n_seconds:
                    midpoint_ind = int((row.Timestamp + td(seconds=row.Duration/2) - start_timestamp).total_seconds() * sample_f)
                    start_ind = int((midpoint_ind - show_n_seconds / 2 * sample_f))
                    stop_ind = int(midpoint_ind + show_n_seconds / 2 * sample_f)

                data = signal[start_ind:stop_ind]

                df_beat = beat_data.loc[(beat_data["Timestamp"] >= ts[start_ind]) &
                                        (beat_data["Timestamp"] <= ts[stop_ind])]
                max_volt = max(data) * 1.1

                plt.close("all")
                fig, ax = plt.subplots(1, figsize=(7.5, 5.5))
                plt.subplots_adjust(top=0.875, bottom=0.125, left=0.075, right=0.975, hspace=0.2, wspace=0.2)

                ax.plot(ts[start_ind:stop_ind], data, color='black')
                ax.scatter(df_beat["Timestamp"], [max_volt for i in range(df_beat.shape[0])],
                           color='green', s=15, marker='v', label='Detected beats')
                ax.fill_between(x=[row.Timestamp, row.Timestamp + td(seconds=row.Duration)], y1=min(data), y2=max(data),
                                color='grey', alpha=.35)
                ax.legend(loc='lower left')
                show_lab = "full event" if row.Duration < show_n_seconds else "cropped"
                ax.set_title("{} Event {}/{}: start = {}, {} seconds ({})\n"
                             "Context: gait = {}, sleep = {}".format(t, str(int(row_i+1)), n_events,
                                                                     str(row.Timestamp.round("1S")),
                                                                     round(row.Duration, 1), show_lab,
                                                                     row.Gait, row.Sleep))
                ax.xaxis.set_major_formatter(xfmt)
                plt.xticks(rotation=45, fontsize=8)

                if row.Duration >= show_n_seconds:
                    ax.set_xlim(ts[start_ind], ts[stop_ind])

                if save_pdf:
                    # Saves png file
                    t_name = t if t != "PAC/SVE" else "PACSVE"
                    plt.savefig(f"{img_folder}{t_name}_Index{row.Index}.png", dpi=100)
                    img_list.append(f"{img_folder}{t_name}_Index{row.Index}.png")

                row_i += 1

    plt.close("all")

    if save_pdf:
        print(f"\nCombining {include_sample_data + df_arrhythmia.shape[0]} images into PDF...")
        # Creates pdf from all png files
        pdf = FPDF(orientation="L")  # "L" for landscape
        pdf.add_page()
        pdf.set_font("Arial", size=20)

        if os.path.exists(logo_link):
            pdf.image(name=logo_link, x=100, y=115, w=90, h=90)
        pdf.set_font("Arial", size=12)

        p_valid = round(100 * df_qc["Validity"].value_counts().loc["Valid"] / df_qc.shape[0], 1)

        pdf.cell(200, 10, txt=f"Subject {subj}", ln=1, align="L")
        pdf.cell(200, 10, txt=" ", ln=2, align="L")

        pdf.cell(200, 10, txt="Collection details", ln=3, align="L")
        pdf.cell(200, 10, txt=f"-Device orientation: {orientation}", ln=4, align="L")
        pdf.cell(200, 10, txt=f"-Collection duration: {round(collect_dur_hours, 1)} hours", ln=5, align="L")
        pdf.cell(200, 10, txt=f"-Usable data: {p_valid}% valid (Orphanidou algorithm); "
                              f"{round(p_valid/100*collect_dur_hours, 1)} hours of data", ln=6, align="L")

        pdf.cell(200, 10, txt=" ", ln=7, align="L")

        pdf.cell(200, 10, txt="Arrhythmia analysis", ln=8, align="L")
        pdf.cell(200, 10, txt=f"-Arrhythmias to include: {arrhythmias}", ln=9, align="L")
        pdf.cell(200, 10, txt=f"-Excluding detected arrhythmias: {excl}", ln=10, align="L")
        pdf.cell(200, 10, txt=f"-Found {df_arrhythmia.shape[0]} events", ln=11, align="L")

        if include_sample_data > 0:
            for i in range(1, include_sample_data+1):
                plt.close("all")
                fig = generate_random_valid_period(signal, window_len=10, sample_f=fs, figsize=(7.5, 5.5), df_qc=df_qc,
                                                   image_n=i, image_of=include_sample_data, start_time=start_stamp)
                fig.savefig(f"{img_folder}SampleData{i}.png", dpi=100)
                img_list.insert(i-1, f"{img_folder}SampleData{i}.png")
            plt.close("all")

        for img in img_list:
            pdf.image(img)
            os.remove(img)
        pdf.output("{}{}_{}.pdf".format(save_dir, subj, "Random" if not include_all_events else "Complete"))
        print(f"PDF created (saved to {save_dir}).")

    return df_arrhythmia


def print_n_events():

    if arr.df_card_nav is not None:
        raw = arr.df_card_nav.loc[arr.df_card_nav["Type"] != "Sinus"]
        print(f"\nRaw: {raw.shape[0]} events")

        if arr.df_valid_arr_nw is not None:
            nw_arr = arr.df_valid_arr_nw.loc[arr.df_valid_arr_nw["Type"] != "Sinus"]
            print(f"Nonwear: {nw_arr.shape[0]} events ({(round(100*nw_arr.shape[0]/raw.shape[0], 1))}% remain)")

        if arr.df_valid_arr_qc is not None:
            qc_arr = arr.df_valid_arr_qc.loc[arr.df_valid_arr_qc["Type"] != "Sinus"]
            print(f"Quality check: {qc_arr.shape[0]} events ({(round(100*qc_arr.shape[0]/raw.shape[0], 1))}% remain)")

        if arr.df_valid is not None:
            valid_arr = arr.df_valid.loc[arr.df_valid["Type"] != "Sinus"]
            print(f"QC and NW: {valid_arr.shape[0]} events ({(round(100*valid_arr.shape[0]/raw.shape[0], 1))}% remain)")

        if df_report is not None:
            print(f"In report: {df_report.shape[0]} events ({(round(100*df_report.shape[0]/raw.shape[0], 1))}% remain)")


""" ================================================= RUNNING SCRIPT ============================================== """

subj = "OND09_0001"

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

flag_events_as_gait(start_time=start_stamp, file_dur_sec=file_dur, df_gait=df_gait, df_arrhyth=arr.df_valid)
flag_events_as_sleep(start_time=start_stamp, file_dur_sec=file_dur, df_sleep=df_sleep_alg, df_arrhyth=arr.df_valid)

handds_arrs = ["VT", "Arrest", "AF", "Brady", "Tachy", "Block", "ST+", "AV2/II"]
other_arrs = ("COUP", "COUP(mf)", "PAC/SVE", "GEM", "VT", "Arrest", "AF", "Brady", "Tachy", "Block", "ST+")
# arr.plot_arrhythmias(df=arr.df_valid_arr_nw, downsample=5, types=other_arrs)
# arr.df_valid.to_csv(f"C:/Users/ksweber/Desktop/{subj}_ValidArrhythmias_CustomNotFinal_CardiacNavigator15.csv", index=False)

df_report = gen_sample_pdf(df_arrhythmia=arr.df_valid, signal=ecg, beat_data=arr.df_rr,
                           sample_f=fs, start_timestamp=start_stamp,
                           show_n_seconds=10, pad_short_events=False, include_all_events=True, save_pdf=True,
                           arrhythmias=other_arrs, include_sample_data=3, df_qc=arr.df_qc_epochs,
                           orientation='vertical', collect_dur_hours=file_dur/3600)

# print_n_events()

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
# Work on long QT stuff --> don't include every damn beat
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
    ax.set_title("Grey line = Msec timestamp; black line = 'adjusted' timestamp;\nPQ = fuchsia; QR(up) = orange; QRS = green; QT = blue")

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


def find_long_qt_periods(min_beats=10, min_duration=30):
    inds = [i for i in long_qt.index]
    diffs = [j-i for i, j in zip(inds[:], inds[1:])]
    diffs.append(0)

    long_qt["RowDiff"] = diffs

    indexes = []
    cur_index = 0
    for i in range(len(diffs)):
        if i > cur_index:
            if [i for i in set(diffs[i:i+min_beats])] == [1]:
                start = i

                for j in range(i+1, len(diffs)):
                    if diffs[j] == 1:
                        pass

                    if diffs[j] > 1:
                        end = j
                        cur_index = j
                        indexes.append([start, end])
                        break

    longqt_final = pd.DataFrame(columns=long_qt.columns)
    for i, period in enumerate(indexes):
        print(period)
        start = long_qt.iloc[period[0]]
        stop = long_qt.iloc[period[1]]

        df_out = pd.DataFrame({"Timestamp": [start["Timestamp"]],
                               "Start_Ind": [stop['Start_Ind']],
                               "Stop_Ind": [stop['Stop_Ind']],
                               "Duration": [(stop['Stop_Ind'] - start['Start_Ind'])/fs],
                               "Type": ["LongQT"],
                               "Info": ["AUTOMATIC"]})
        longqt_final = longqt_final.append(df_out)

    longqt_final = longqt_final.loc[longqt_final["Duration"] >= min_duration]

    return longqt_final

