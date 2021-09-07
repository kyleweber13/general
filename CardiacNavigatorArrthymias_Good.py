import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyedflib
from datetime import timedelta as td
import os
import Filtering
from OrphanidouQC import run_orphanidou
from BittiumFarosNonwear import BittiumNonwear
import random
from fpdf import FPDF

os.chdir("O:/OBI/Personal Folders/Kyle Weber/Cardiac Navigator Investigation/")


class ArrhythmiaProcessor:

    def __init__(self, raw_ecg=None, epoched_nonwear=None, epoched_signal_quality=None, card_nav_data=None,
                 card_nav_rr_data=None,
                 details={"start_time": None, "sample_rate": 250, "file_dur_sec": 1}):

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

        df_rr = df_rr[["Timestamp", "RR", "HR", "RollHR", "Type", "Template", "PP", "QRS", "PQ",
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
                          excl_types=("Sinus", "Brady", "Min. RR", "Tachy", "Afib Max. HR (total)", "Min. HR", "Max. HR", "Afib Min. HR (total)", "Max. RR")):

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

    def remove_nonwear(self, df_events, nw_bouts, show_boxplot=False,
                       excl_types=("Sinus", "Brady", "Min. RR", "Tachy", "Afib Max. HR (total)", "Min. HR", "Max. HR", "Afib Min. HR (total)", "Max. RR")):

        print("\nRemoving arrhythmia events that occur during detected non-wear periods...")

        print(f"-Omitting {excl_types}")
        df = df_events.loc[~df_events["Type"].isin(excl_types)]

        # Handling NW bout data -----------------
        epoch_stamps = pd.date_range(start=self.details["start_time"],
                                     end=(self.details["start_time"] + td(seconds=self.details["file_dur_sec"])),
                                     freq=f"{self.details['epoch_length']}S")

        epoch_nw = np.zeros(len(epoch_stamps), dtype=bool)
        for row in nw_bouts.itertuples():
            epoch_nw[int(row.start_dp/self.details["sample_rate"]/self.details['epoch_length']):
                     int(row.end_dp/self.details["sample_rate"]/self.details['epoch_length'])] = True

        df_nw = pd.DataFrame({'Timestamp': epoch_stamps, "Nonwear": epoch_nw})

        # Finds events that occur during nonwear bouts ---------
        event_contains_nw = []
        for row in df.itertuples():
            # Cropped event dataframe to nonwear events
            nw = df_nw.loc[(df_nw["Timestamp"] >= row.Timestamp) &
                           (df_nw["Timestamp"] <= row.Timestamp + td(seconds=row.Duration))]

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

        # return df_valid_arr[["Timestamp", "Type", "Duration"]], desc
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

    def plot_arrhythmias(self, downsample=3,
                         types=("COUP(mf)", "SALV(mf)", "GEM(mf)", "COUP", "AF",  "PAC/SVE", "IVR", "SALV", "Arrest", "Block", "GEM")):

        if self.ecg is None or self.timestamps is None:
            print("\nNeed raw data to generate this plot. Try again.")

        color_dict = {"Arrest": 'red', 'AF': 'orange', "COUP(mf)": 'dodgerblue', "COUP": 'dodgerblue',
                      "SALV(mf)": 'green', "SALV": 'green', "Block": 'purple', "GEM(mf)": "pink",
                      "GEM": 'pink', "PAC/SVE": 'grey', 'IVR': 'limegreen', "ST-": "gold", "ST+": "fuchsia"}

        print(f"\nPlotting raw data ({round(self.details['sample_rate']/downsample)}Hz) with overlaid events:")

        if type(types) is list or type(types) is tuple:
            types = list(types)
            for key in types:
                print(f"{key} = {color_dict[key]}")
        if type(types) is str:
            types = [types]
            for key in types:
                print(f"{key} = {color_dict[key]}")

        data = self.df_valid.loc[self.df_valid["Type"].isin(types)]

        if data.shape[0] == 0:
            print(f"\nNo arrhythmias of type(s) {types} found.")
            return None

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


subj = "008"
arr = ArrhythmiaProcessor(
                          raw_ecg=ecg,
                          epoched_nonwear=f"C:/Users/ksweber/Desktop/{subj}_Final_NW.csv",
                          epoched_signal_quality=f"C:/Users/ksweber/Desktop/{subj}_QC.csv",
                          card_nav_data=f'C:/Users/ksweber/Desktop/{subj}_Events_TestOut.csv',
                          # card_nav_rr_data=f"C:/Users/ksweber/Desktop/{subj}_CC_TestOut.csv",
                          details={"start_time": pd.to_datetime("2021-07-08 12:50:23"),
                                   "sample_rate": 250, "file_dur_sec": 802394, "epoch_length": 15})

# arr.plot_arrhythmias(downsample=3, types=("AF", "Arrest"))
