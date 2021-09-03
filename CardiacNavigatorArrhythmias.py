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
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


def import_arrhythmia_events(file, start_time, sample_f, file_dur):

    df = pd.read_csv(file, delimiter=";")

    df["Timestamp"] = [start_time + td(seconds=row.Msec/1000) for row in df.itertuples()]  # ms -> timestamps
    df["Duration"] = [row.Length/1000 for row in df.itertuples()]  # event duration, seconds
    df["Start_Ind"] = [int(row.Msec/1000 * sample_f) for row in df.itertuples()]  # raw data
    df["Stop_Ind"] = [int(row.Start_Ind + row.Duration * sample_f) for row in df.itertuples()]  # raw data
    df = df[["Timestamp", "Start_Ind", "Stop_Ind", "Duration", "Type", "Info"]]

    # Sums time spent in each arrhythmia type, in seconds
    sums = []
    for t in df["Type"].unique():
        if t not in ["Min. RR", "Max. RR", "Min. HR", "Max. HR", "ST(ref)"]:
            d = df.loc[df["Type"] == t]
            sums.append([t, d["Duration"].sum()])
    df_sums = pd.DataFrame(sums)
    df_sums.columns = ["Type", "Duration"]
    df_sums["%"] = [100*i/file_dur for i in df_sums["Duration"]]

    return df, df_sums


def import_rr_data(file, start_time, n_roll_beats_avg=30):

    df_rr = pd.read_csv(file, delimiter=";")

    # column name formatting
    cols = [i for i in df_rr.columns][1:]
    cols.insert(0, "Msec")
    df_rr.columns = cols

    df_rr["Timestamp"] = [start_time + td(seconds=row.Msec/1000) for row in df_rr.itertuples()]  # ms -> timestamp

    # Beat type value replacement (no short form)
    type_dict = {'?': "Unknown", "N": "Normal", "V": "Ventricular", "S": "Premature", "A": "Aberrant"}
    df_rr["Type"] = [type_dict[i] for i in df_rr["Type"]]

    df_rr["HR"] = [60/(r/1000) for r in df_rr["RR"]]  # beat-to-beat HR
    df_rr["RollHR"] = df_rr["HR"].rolling(window=n_roll_beats_avg, center=True).mean()  # rolling average HR

    df_rr = df_rr[["Timestamp", "RR", "HR", "RollHR", "Type", "Template", "PP", "QRS", "PQ",
                   "QRn", "QT", "ISO", "ST60", "ST80", "NOISE"]]

    return df_rr


def remove_orphanidou(events_df, df_orphanidou, epoch_len, sample_f, show_boxplot=False,
                      excl_types=("Sinus", "Brady", "Min. RR", "Tachy", "Afib Max. HR (total)", "Min. HR", "Max. HR", "Afib Min. HR (total)", "Max. RR")):

    print("\nRunning Orphanidou et al. (2015) quality check data on arrhythmia events...")
    df = events_df.copy()

    print(f"-Omitting {excl_types}")

    event_contains_invalid = []
    for row in df.itertuples():
        # Row index in df_qc that start of event falls into
        epoch_start_ind = int(np.floor(row.Start_Ind / (epoch_len * sample_f)))

        # Row index in df_qc that end of event falls into (includsive)
        epoch_stop_ind = int(np.ceil(row.Stop_Ind / (epoch_len * sample_f))) + 1

        qc = df_orphanidou.iloc[epoch_start_ind:epoch_stop_ind]  # Cropped QC df

        # Whether or not event contains invalid data
        event_contains_invalid.append("Invalid" if "Invalid" in qc["Orphanidou"].unique() else "Valid")

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

    orig_len = events_df.shape[0]
    new_len = df_valid_arr.shape[0]

    print(f"\nRemoved {orig_len - new_len} ({round(100*(orig_len - new_len)/orig_len, 1)}%) "
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


def remove_nonwear(events_df, df_nw, show_boxplot=False,
                       excl_types=("Sinus", "Brady", "Min. RR", "Tachy", "Afib Max. HR (total)", "Min. HR", "Max. HR", "Afib Min. HR (total)", "Max. RR")):

    print("\nRemoving arrhythmia events that occur during detected non-wear periods...")

    print(f"-Omitting {excl_types}")
    df = events_df.loc[~events_df["Type"].isin(excl_types)]

    event_contains_nw = []
    for row in df.itertuples():
        # Cropped nonwear dataframe to event
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

    orig_len = events_df.shape[0]
    new_len = df_valid_arr.shape[0]

    print(f"\nRemoved {orig_len - new_len} ({round(100*(orig_len - new_len)/orig_len, 1)}%) "
          f"events due to non-wear periods.")

    # Descriptive stats for each arrhythmia's duration
    desc = df_valid_arr.groupby("Type")["Duration"].describe()
    desc["Duration"] = [df_valid_arr.loc[df_valid_arr["Type"] == row.Index]["Duration"].sum() for
                        row in desc.itertuples()]

    print("\nResults summary:")
    for row in desc.itertuples():
        bout_name = "bouts" if row.count > 1 else "bout"
        print(f"-{row.Index}: {int(row.count)} {bout_name}, total duration = {round(row.Duration, 1)} seconds")

    return df_valid_arr[["Timestamp", "Type", "Duration"]], desc


def combine_valid_dfs(df_nw, df_qc, show_boxplot=False):
    """Combines nonwear and quality check dataframes and returns df of events in both dfs"""
    df_valid = df_nw.loc[df_nw["Timestamp"].isin(df_qc["Timestamp"])].reset_index(drop=True)

    if show_boxplot:
        fig, axes = plt.subplots(1, figsize=(10, 6))
        df_valid.boxplot(column="Duration", by="Type", ax=axes)

        for tick in axes.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
            tick.label.set_rotation(45)
        axes.set_ylabel("Seconds")

    orig_len = max(df_nw.shape[0], df_qc.shape[0])
    new_len = df_valid.shape[0]

    print(f"\nRemoved {orig_len - new_len} ({round(100*(orig_len - new_len)/orig_len, 1)}%) "
          f"events due to non-wear periods.")

    # Descriptive stats for each arrhythmia's duration
    desc = df_valid.groupby("Type")["Duration"].describe()
    desc["Duration"] = [df_valid.loc[df_valid["Type"] == row.Index]["Duration"].sum() for
                        row in desc.itertuples()]

    print("\nResults summary:")
    for row in desc.itertuples():
        bout_name = "bouts" if row.count > 1 else "bout"
        print(f"-{row.Index}: {int(row.count)} {bout_name}, total duration = {round(row.Duration, 1)} seconds")

    return df_valid[["Timestamp", "Type", "Duration"]], desc


def plot_arrhythmias(df_event, signal, timestamps, downsample=3,
                     types=("COUP(mf)", "SALV(mf)", "GEM(mf)", "COUP", "AF",  "PAC/SVE", "IVR", "SALV", "Arrest", "Block", "GEM")):

    color_dict = {"Arrest": 'red', 'AF': 'orange', "COUP(mf)": 'dodgerblue', "COUP": 'dodgerblue',
                  "SALV(mf)": 'green', "SALV": 'green', "Block": 'purple', "GEM(mf)": "pink",
                  "GEM": 'pink', "PAC/SVE": 'grey', 'IVR': 'limegreen'}

    if type(types) is list or type(types) is tuple:
        types = list(types)
        for key in types:
            print(f"{key} = {color_dict[key]}")
    if type(types) is str:
        types = [types]
        for key in types:
            print(f"{key} = {color_dict[key]}")

    data = df_event.loc[df_event["Type"].isin(types)]

    if data.shape[0] == 0:
        print(f"\nNo arrhythmias of type(s) {types} found.")
        return None

    if "COUP(mf)" in types and "COUP" in types:
        types.remove("COUP(mf)")
    if "SALV(mf)" in types and "SALV" in types:
        types.remove("SALV(mf)")
    if "GEM(mf)" in types and "GEM" in types:
        types.remove("GEM(mf)")

    max_val = max(signal)
    min_val = min(signal)

    fig, axes = plt.subplots(1, figsize=(12, 8))
    axes.plot(timestamps[::downsample], signal[::downsample], color='black')

    for row in data.itertuples():
        # Fill between region
        axes.fill_between(x=[row.Timestamp, row.Timestamp + td(seconds=row.Duration)],
                          y1=min_val*1.1, y2=max_val*1.1, color=color_dict[row.Type], alpha=.35)
        # Triangle markers for visibility
        axes.scatter(row.Timestamp + td(seconds=row.Duration/2), max_val*1.1, marker="v", color=color_dict[row.Type])

    plt.title(types)


""" ================================================ DATA IMPORT ================================================= """

subj = "008"

file = pyedflib.EdfReader(f"O:/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Data Files/OmegaSnap/{subj}_OmegaSnap.EDF")
start_stamp = file.getStartdatetime()
ecg = file.readSignal(0)
file_dur = file.file_duration
fs = int(file.getSampleFrequency(0))  # ECG sample rate
acc_fs = int(file.getSampleFrequency(1))  # Accelerometer sample rate
filt = Filtering.filter_signal(data=ecg, sample_f=fs, filter_type='bandpass', low_f=.67, high_f=30, filter_order=3)
ts = pd.date_range(start=start_stamp, periods=len(ecg), freq="{}ms".format(1000/fs))
epoch_len = 15
file.close()

""" ================================================== NON-WEAR =================================================== """

nw = BittiumNonwear(signal=Filtering.filter_signal(data=ecg, sample_f=fs, filter_type='highpass',
                                                   high_f=.25, filter_order=3),
                   # tri_accel=np.array([ecg.signals[1], ecg.signals[2], ecg.signals[3]]),
                    start_time=start_stamp, power_percent_thresh=80, freq_thresh=55,
                    sample_rate=fs, temp_sample_rate=1, accel_sample_rate=acc_fs,
                    epoch_len=epoch_len, min_nw_dur=3, min_nw_break=5,
                    rolling_mean=False, reference_nonwear=None)

# nw.df_epochs = nw.process_epochs(use_orphanidou=True)
nw.df_epochs = pd.read_csv(f"C:/Users/ksweber/Desktop/{subj}_nw_epochs.csv")
nw.df_nw, nw.nw_timeseries, nw.nw_epochs = nw.run_power_nonwear()

# nw.df_nw.to_excel(f"C:/Users/ksweber/Desktop/{subj}_FinalNW.xlsx", index=False)
# plt.savefig(f"{subj}_NW_Output.tiff", dpi=125)


""" =============================================== ORPHANIDOU PROCESSING ========================================= """

# df_qc, peaks_data = run_orphanidou(signal=filt, sample_rate=fs, epoch_len=epoch_len, volt_thresh=100)
df_qc = pd.read_csv(f"C:/Users/ksweber/Desktop/{subj}_QC.csv")
df_qc["Timestamp"] = pd.date_range(start=start_stamp, periods=df_qc.shape[0], freq="{}S".format(epoch_len))

""" =========================================== CARDIAC NAVIGATOR PROCESSING ====================================== """

# Data for each arrhythmia event and durations
df_events, df_sums = import_arrhythmia_events(file=f'C:/Users/ksweber/Desktop/{subj}_Events_TestOut.csv',
                                              start_time=start_stamp, sample_f=fs, file_dur=file_dur)

# Data for each cardiac cycle
# df_rr = import_rr_data(file=f"C:/Users/ksweber/Desktop/{subj}_CC_TestOut.csv", start_time=start_stamp, n_roll_beats_avg=30)

""" ================================================ REMOVING BAD DATA ============================================ """

# Removes events with invalid data according to Orphanidou algorithm
df_valid_qc, desc_arrhyth_qc = remove_orphanidou(events_df=df_events, df_orphanidou=df_qc,
                                                 epoch_len=epoch_len, sample_f=fs, show_boxplot=True)

# Removes events with non-wear detected
df_valid_nw, desc_arrhyth_nw = remove_nonwear(events_df=df_events, df_nw=nw.nw_epochs, show_boxplot=True,
                                              excl_types=("Sinus", "Brady", "Min. RR", "Tachy",
                                                          "Afib Max. HR (total)", "Min. HR", "Max. HR",
                                                          "Afib Min. HR (total)", "Max. RR"))

df_valid, df_desc = combine_valid_dfs(df_nw=df_valid_nw, df_qc=df_valid_qc)


def gen_sample_pdf(df_arrhythmia, img_folder="C:/Users/ksweber/Desktop/Images/", show_n_seconds=15,
                   save_dir="C:/Users/ksweber/Desktop/", logo_link="C:/Users/ksweber/Desktop/HANDDS-logo.jpg",
                   pad_short_events=True):

    # list of detected arrhythmias
    found_arrs = df_arrhythmia["Type"].unique()

    img_list = []

    # Loops each arrhythmia and generates a plot of a random event
    for t in found_arrs:
        n_events = int(df_desc.loc[t]["count"])
        df_inds = df_valid.loc[df_valid["Type"] == t].index

        # randomly picks one event if more than one event detected
        if n_events > 1:
            ind = random.choice(range(len(df_inds)))
            row = df_inds[ind]

        if n_events == 1:
            ind = 0
            row = df_inds[0]

        d = df_valid.iloc[row]

        # Cropping data
        if d.Duration < show_n_seconds:
            if pad_short_events:
                pad = show_n_seconds - d.Duration
            if not pad_short_events:
                pad = 2
            start_ind = int((d.Timestamp - start_stamp).total_seconds() * fs - pad/2 * fs)
            stop_ind = int((d.Timestamp + td(seconds=d.Duration) - start_stamp).total_seconds() * fs + pad/2 * fs)

        if d.Duration >= show_n_seconds:
            midpoint_ind = int((d.Timestamp + td(seconds=d.Duration/2) - start_stamp).total_seconds() * fs)
            start_ind = int((midpoint_ind - show_n_seconds / 2 * fs))
            stop_ind = int(midpoint_ind + show_n_seconds / 2 * fs)

        data = filt[start_ind:stop_ind]

        plt.close("all")
        fig, ax = plt.subplots(1, figsize=(7.5, 5.5))
        plt.subplots_adjust(top=0.9, bottom=0.125, left=0.075, right=0.975, hspace=0.2, wspace=0.2)

        ax.plot(ts[start_ind:stop_ind], data, color='black')
        ax.fill_between(x=[d.Timestamp, d.Timestamp + td(seconds=d.Duration)], y1=min(data), y2=max(data),
                        color='grey', alpha=.35)
        show_lab = "full event" if d.Duration < show_n_seconds else "cropped"
        ax.set_title("{}: start = {}, duration = {} "
                     "seconds ({})\nEvent {}/{}, df row index {}".format(t, str(d.Timestamp.round("500ms"))[:-3],
                                                                         round(d.Duration, 1), show_lab,
                                                                         str(int(ind+1)), n_events, d.name))
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        if d.Duration >= show_n_seconds:
            ax.set_xlim(ts[start_ind], ts[stop_ind])

        # Saves png file
        t_name = t if t != "PAC/SVE" else "PACSVE"
        plt.savefig(f"{img_folder}{t_name}.png", dpi=100)
        img_list.append(f"{img_folder}{t_name}.png")

    plt.close("all")

    # Creates pdf from all png files
    pdf = FPDF(orientation="L")  # "L" for landscape
    pdf.add_page()
    pdf.set_font("Arial", size=20)

    if os.path.exists(logo_link):
        pdf.image(name=logo_link, x=90, y=50, w=125)
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Subject {subj}:", ln=1, align="L")
    pdf.cell(200, 10, txt="Arrhythmia analysis", ln=2, align="L")
    pdf.cell(200, 10, txt=f"-Found {df_arrhythmia.shape[0]} events", ln=3, align="L")

    for img in img_list:
        pdf.image(img)
        os.remove(img)
    pdf.output(f"{save_dir}{subj}.pdf")
    print(f"\nPDF created (saved to {save_dir}).")


gen_sample_pdf(df_arrhythmia=df_valid, show_n_seconds=10, pad_short_events=False)

# plot_arrhythmias(signal=filt, downsample=5, timestamps=ts, types=("GEM", "GEM(mf)", "IVR"), df_event=df_valid)

# Plotting ------------------------------------------------------------------------------------------------------------
# df_use = df_rr.loc[(df_rr["Timestamp"] <= start_stamp + td(seconds=3600)) & (df_rr["Timestamp"] >= start_stamp + td(seconds=3585))]


# TODO

def plot_waves():
    print("NOT READY FOR USE")
    fig, axes = plt.subplots(1, sharex='col', figsize=(12, 8))
    # axes.plot(ts[:int(fs*3600)], filt[:int(fs*3600)], color='black')
    axes.plot(np.arange(1000*3585, 3600*1000, int(1000/fs)), filt[int(fs*59.75*60):int(fs*3600)], color='black')
    axes.set_ylabel("Voltage")
    plt.title("008: Minutes 59.75-60")

    for row in df_use.itertuples():
        if row.Index != df_use.index[-1]:
            # T wave?
            plt.axvline(x=row.Msec, color='limegreen')
            # QT segment
            axes.fill_between(x=[row.Msec - row.QT/2, row.Msec+row.QT/2], y1=-1000, y2=-333, color='dodgerblue', alpha=.35)
            # PR segment?
            axes.fill_between(x=[row.Msec - row.QT/2 - row.PQ, row.Msec - row.QT/2], y1=-333, y2=333, color='orange', alpha=.35)
            # QRS segment?
            axes.fill_between(x=[row.Msec - row.QT/2, row.Msec - row.QT/2 + row.QRS], y1=333, y2=1000, color='purple', alpha=.35)
        if row.Index == df_use.index[-1]:
            # T wave?
            plt.axvline(x=row.Msec, color='limegreen', label="T onset?")
            # QT segment
            axes.fill_between(x=[row.Msec - row.QT/2, row.Msec+row.QT/2], y1=-1000, y2=-333, color='dodgerblue', alpha=.35, label="QT?")
            # PR segment?
            axes.fill_between(x=[row.Msec - row.QT/2 - row.PQ, row.Msec - row.QT/2], y1=-333, y2=333, color='orange', alpha=.35, label="PR?")
            # QRS segment?
            axes.fill_between(x=[row.Msec - row.QT/2, row.Msec - row.QT/2 + row.QRS], y1=333, y2=1000, color='purple', alpha=.35, label="QRS?")

    axes.legend()
