import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pyedflib
xfmt = mdates.DateFormatter("%a\n%b-%d")
xfmt_raw = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
import numpy as np
from datetime import timedelta
import nwdata
import scipy.stats

subj = '0008'
tabular_file = "O:/OBI/ONDRI@Home/Participant Summary Data - Feedback/HANDDS Feedback Forms/Summary Dataframes/OND09 Summary Dataframes.xlsx"

gait_file = f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv"
steps_file = f"W:/NiMBaLWEAR/OND09/analytics/gait/steps/OND09_{subj}_01_GAIT_STEPS.csv"

activity_log_file = f"O:/OBI/HANDDS-ONT/Logistic Planning for Launch/File and Folder Structure/HANDDS_ActivityLogTemplate.xlsx"

sleep_output_file = f"W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/OND09_{subj}_01_SLEEP_BOUTS.csv"
sleep_window_file = f"W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/OND09_{subj}_01_SPTW.csv"

epoch_folder = "W:/NiMBaLWEAR/OND09/analytics/activity/epoch/"
edf_folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"

ankle_file = f"{edf_folder}OND09_{subj}_01_AXV6_RAnkle.edf"
if not os.path.exists(ankle_file):
    ankle_file = f"{edf_folder}OND09_{subj}_01_AXV6_LAnkle.edf"

wrist_file = f"{edf_folder}OND09_{subj}_01_AXV6_RWrist.edf"
if not os.path.exists(wrist_file):
    wrist_file = f"{edf_folder}OND09_{subj}_01_AXV6_LWrist.edf"

cutpoints = {"Dominant": [51*1000/30/15, 68*1000/30/15, 142*1000/30/15],
             "Non-dominant": [47*1000/30/15, 64*1000/30/15, 157*1000/30/15]}


def import_data():

    print("\nImporting and formatting summary dataframes...")

    # EDF file details ------------------------------------
    edfs = os.listdir(edf_folder)
    edfs = [i for i in edfs if subj in i and "Wrist" in i]

    start_time = pyedflib.EdfReader(edf_folder + edfs[0]).getStartdatetime()
    file_dur = pyedflib.EdfReader(edf_folder + edfs[0]).file_duration

    # Epoched wrist data -------------------------------------------------------
    df_epoch = pd.read_csv(epoch_folder + f"OND09_{subj}_01_EPOCH_ACTIVITY.csv")
    epoch_len = int(file_dur/df_epoch.shape[0])
    df_epoch["Timestamp"] = pd.date_range(start=start_time, freq=f"{epoch_len}S", periods=df_epoch.shape[0])
    df_epoch = df_epoch[["Timestamp", "avm"]]
    epoch_len = int((df_epoch["Timestamp"].iloc[1] - df_epoch["Timestamp"].iloc[0]).total_seconds())

    # Summary dataframe formatting -----------------------------------------------------------
    df_tab = pd.read_excel(tabular_file, sheet_name=f"{subj} Summary Dataframes", header=None)
    df_tab = df_tab.dropna(how='all')

    # Sedentary data
    sed_ind = df_tab.loc[df_tab[0] == "Sedentary"].index[0]

    stop_inds = []
    for i in range(sed_ind+1, df_tab.shape[0]):
        if type(df_tab.iloc[i][0]) is str:
            stop_ind = i
            stop_inds.append(stop_ind)
    stop_inds.append(df_tab.shape[0])

    df_sed = df_tab.iloc[sed_ind+1:stop_inds[0]].iloc[:, 1:3]
    df_sed.columns = ["Date", "Duration"]

    # Activity volume data
    df_act = df_tab.iloc[stop_inds[0]+1:stop_inds[1]].iloc[:, 1:4]
    df_act.columns = ["Date", "Light", "MVPA"]
    df_act.sort_values("Date")
    df_act = df_act.reset_index(drop=True)

    durs = []
    for i in df_sed["Duration"]:
        try:
            h = float(i.split("h")[0]) * 60
            try:
                m = float(i.split("h")[1][:-1])
            except ValueError:
                m = 0
            durs.append(h + m)
        except AttributeError:
            durs.append(None)
    df_act["Sed"] = durs

    df_act["Date"] = pd.to_datetime(df_act["Date"])
    df_act["Date"] = [i.date() for i in df_act["Date"]]

    # Walking data
    df_walk = df_tab.iloc[stop_inds[1]+1:stop_inds[2]].iloc[:, 1:6]
    df_walk.columns = ["Date", "LongestBout", "StepsLongest", "Bouts>3mins", "TotalSteps"]
    df_walk["Date"] = pd.to_datetime(df_walk["Date"])

    # Sleep data (totals)
    df_sleep = df_tab.iloc[stop_inds[2]+1:stop_inds[3]].iloc[1:, 1:7]
    df_sleep.columns = ["Date", "TST", "BedTime", "TimeOutBed", "NightlySleepTime", "NumWalks"]
    df_sleep["Date"] = pd.to_datetime(df_sleep["Date"])

    # Sleep data (algorithm output)
    df_sleep_alg = pd.read_csv(sleep_output_file)
    df_sleep_alg["start_time"] = pd.to_datetime(df_sleep_alg["start_time"])
    df_sleep_alg["end_time"] = pd.to_datetime(df_sleep_alg["end_time"])

    # Sleep windows
    df_sptw = pd.read_csv(sleep_window_file)
    df_sptw["start_time"] = pd.to_datetime(df_sptw["start_time"])
    df_sptw["end_time"] = pd.to_datetime(df_sptw["end_time"])

    # Activity descriptive data by day
    df_epoch["Day"] = [row.Timestamp.date() for row in df_epoch.itertuples()]
    avm_desc = df_epoch.groupby("Day")["avm"].describe()
    df_act["mean_avm"] = avm_desc['mean'].reset_index(drop=True)
    df_act["std_avm"] = avm_desc['std'].reset_index(drop=True)
    df_act["Date"] = pd.to_datetime(df_act["Date"])

    # Activity log (subjective)
    df_act_log = pd.read_excel(activity_log_file)
    df_act_log = df_act_log.loc[df_act_log["subject_id"] == int(subj)].reset_index(drop=True)
    df_act_log = df_act_log.fillna("")
    df_act_log.columns = ['coll_id', 'study_code', 'subject_id', 'activity', 'start_time',
                          'duration', 'Unnamed: 6', 'Notes.', 'Unnamed: 8']

    # Gait bout data
    df_gait = pd.read_csv(gait_file)
    df_gait = df_gait[["start_timestamp", "end_timestamp", "number_steps"]]

    print("\nImporting ankle data...")
    ankle = nwdata.NWData()
    ankle.import_edf(file_path=ankle_file, quiet=False)

    print("\nImporting wrist data...")
    wrist = nwdata.NWData()
    wrist.import_edf(file_path=wrist_file, quiet=False)
    print("Data imported.")

    return df_epoch, epoch_len, df_act, df_walk.reset_index(drop=True), df_sleep.reset_index(drop=True), \
           df_sleep_alg, df_gait, df_act_log, ankle, wrist


def summary_plot():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex='all')
    plt.suptitle(f"OND09_{subj}")
    plt.subplots_adjust(left=.05, right=.975, wspace=.15)
    plt.rcParams['axes.labelsize'] = '8'

    axes[0][0].bar(df_act["Date"], df_act["mean_avm"], edgecolor='black', color='grey', label="Active", alpha=.75)
    axes[0][0].set_ylim(0, )
    axes[0][0].set_title("AVM by Day")

    axes[0][1].bar(df_daily["Date"], df_daily["Steps"]/1000, zorder=1,
                   edgecolor='black', color='dodgerblue', label='Steps', alpha=.75)
    axes[0][1].axhline(y=7, color='green', linestyle='dashed', zorder=0)
    axes[0][1].set_title("Step Counts (x1000)")

    axes[0][2].set_title("Sleep Hours")
    # axes[0][2].bar(df_sleep["Date"], df_sleep["TST"], color='navy', edgecolor='black', alpha=.75, zorder=1)
    axes[0][2].fill_between(x=[df_daily.min()["Date"]+timedelta(hours=-12), df_daily.max()["Date"]+timedelta(hours=12)],
                            y1=7, y2=9, color='green', alpha=.25)
    axes[0][2].set_ylim(0, 10)

    axes[1][2].bar(df_daily["Date"], df_daily["Sed"], edgecolor='black', color='silver', label='Sed', alpha=.75)
    axes[1][2].set_ylim(0, 1510)
    axes[1][2].set_yticks(np.arange(0, 1501, 250))
    axes[1][2].axhline(y=1440, color='black', linestyle='dashed')
    axes[1][2].set_title("Sedentary Minutes")

    axes[1][0].bar(df_daily["Date"], df_daily["Light"], edgecolor='black', color='forestgreen', label="Light", alpha=.75)
    axes[1][0].set_title("Light Minutes")

    axes[1][1].bar(df_daily["Date"], df_daily["Mod"] + df_daily["Vig"], edgecolor='black', color='orange', label="MVPA", alpha=.75)
    axes[1][1].set_title("MVPA Minutes")

    axes[1][0].xaxis.set_major_formatter(xfmt)
    axes[1][1].xaxis.set_major_formatter(xfmt)
    axes[1][2].xaxis.set_major_formatter(xfmt)
    axes[1][0].tick_params(axis='x', labelsize=8)
    axes[1][1].tick_params(axis='x', labelsize=8)
    axes[1][2].tick_params(axis='x', labelsize=8)


def plot_raw(ds_ratio=1, dominant=True, ankle_gyro=True, incl_activity_volume=False, incl_step_count=False,
             shade_gait_bouts=False, show_activity_log=False, shade_sleep_windows=False, mark_steps=False):

    ankle_ts = pd.date_range(start=ankle.header["startdate"], periods=len(ankle.signals[0]),
                             freq="{}ms".format(1000/ankle.signal_headers[0]["sample_rate"]))

    wrist_ts = pd.date_range(start=wrist.header["startdate"], periods=len(wrist.signals[0]),
                             freq="{}ms".format(1000 / wrist.signal_headers[0]["sample_rate"]))

    fig, ax = plt.subplots(3, sharex='col', figsize=(14, 8))
    plt.suptitle(f"OND09_{subj}")
    ax[0].plot(df_epoch["Timestamp"], df_epoch["avm"], color='black')
    ax[0].set_title(f"Wrist AVM ({epoch_len}-s epochs)")
    ax[0].set_ylabel("AVM")

    if dominant:
        cp = cutpoints["Dominant"]
    if not dominant:
        cp = cutpoints['Non-dominant']

    if incl_activity_volume:
        ax_vol = ax[0].twinx()

        for row in df_daily.itertuples():
            ax_vol.fill_between(x=[row.Date, row.Date+timedelta(days=.999)], y1=0, y2=row.Light,
                                color='green', alpha=.35)
            ax_vol.fill_between(x=[row.Date, row.Date+timedelta(days=.999)], y1=row.Light, y2=row.Light + row.Mod,
                                color='orange', alpha=.35)
            ax_vol.fill_between(x=[row.Date, row.Date+timedelta(days=.999)],
                                y1=row.Light + row.Mod, y2=row.Light + row.Mod + row.Vig,
                                color='red', alpha=.35)
        ax_vol.set_ylabel("Activity Minutes", color='green')

    c = ['green', 'orange', 'red']
    for i, val in enumerate(cp):
        ax[0].axhline(y=val, color=c[i], linestyle='dotted')

    ax[1].set_title("Wrist Accelerometer ({}Hz)".format(int(wrist.signal_headers[wrist.get_signal_index("Accelerometer x")]["sample_rate"]/ds_ratio)))
    ax[1].plot(wrist_ts[::ds_ratio], wrist.signals[wrist.get_signal_index("Accelerometer x")][::ds_ratio], color='black')
    ax[1].plot(wrist_ts[::ds_ratio], wrist.signals[wrist.get_signal_index("Accelerometer y")][::ds_ratio], color='red')
    ax[1].plot(wrist_ts[::ds_ratio], wrist.signals[wrist.get_signal_index("Accelerometer z")][::ds_ratio], color='dodgerblue')
    ax[1].set_ylabel("G")

    if shade_sleep_windows:
        ylims = ax[1].get_ylim()
        for row in df_sleep_alg.itertuples():
            ax[1].fill_between(x=[row.start_time, row.end_time], y1=ylims[0]*1.1, y2=ylims[1]*1.1,
                               color='navy', alpha=.35)

    if not ankle_gyro:
        ax[2].set_title("Ankle Accelerometer ({}Hz)".format(int(ankle.signal_headers[ankle.get_signal_index("Accelerometer x")]["sample_rate"]/ds_ratio)))
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Accelerometer x")][::ds_ratio], color='black')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Accelerometer y")][::ds_ratio], color='red')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Accelerometer z")][::ds_ratio], color='dodgerblue')
        ax[2].set_ylabel("G")

        if mark_steps:
            max_val = max([max(ankle.signals[ankle.get_signal_index("Accelerometer x")]),
                           max(ankle.signals[ankle.get_signal_index("Accelerometer y")]),
                           max(ankle.signals[ankle.get_signal_index("Accelerometer z")])])

            ax[2].scatter(df_steps["step_time"], [max_val*1.1 for i in range(df_steps.shape[0])], marker="v", color='green', s=5)

    if ankle_gyro:
        ax[2].set_title("Ankle Gyroscope ({}Hz)".format(int(ankle.signal_headers[ankle.get_signal_index("Gyroscope x")]["sample_rate"]/ds_ratio)))
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Gyroscope x")][::ds_ratio], color='black')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Gyroscope y")][::ds_ratio], color='red')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Gyroscope z")][::ds_ratio], color='dodgerblue')
        ax[2].set_ylabel("deg/S")

        if mark_steps:
            max_val = max([max(ankle.signals[ankle.get_signal_index("Gyroscope x")]),
                           max(ankle.signals[ankle.get_signal_index("Gyroscope y")]),
                           max(ankle.signals[ankle.get_signal_index("Gyroscope z")])])

            ax[2].scatter(df_steps["step_time"], [max_val*1.1 for i in range(df_steps.shape[0])], marker="v", color='green', s=5)

    if incl_step_count:
        ax_step = ax[2].twinx()

        for row in df_daily.itertuples():
            ax_step.fill_between(x=[row.Date, row.Date+timedelta(days=.999)], y1=0, y2=row.Steps,
                                 color='grey', alpha=.35)
        ax_step.set_ylabel("Step Count", color='grey')

    if shade_gait_bouts:
        ylim = ax[2].get_ylim()

        for row in df_gait.itertuples():
            if row.Index < df_gait.index[-1]:
                ax[2].fill_between(x=[row.start_timestamp, row.end_timestamp], y1=ylim[0]*1.1, y2=ylim[1]*1.1,
                                   color='gold', alpha=.35)
            if row.Index == df_gait.index[-1]:
                ax[2].fill_between(x=[row.start_timestamp, row.end_timestamp], y1=ylim[0]*1.1, y2=ylim[1]*1.1,
                                   color='gold', alpha=.35, label='GaitBouts')
                ax[2].legend(loc='upper right')

    if show_activity_log:
        print("\nActivity log key:")
        for row in df_act_log.itertuples():
            try:
                ax[0].fill_between(x=[row.start_time, row.start_time + timedelta(minutes=row.duration)],
                                   y1=0, y2=max(df_epoch['avm']*1.1), color='purple', alpha=.35)
                ax[0].text(x=row.start_time + timedelta(minutes=row.duration/3),
                           y=df_epoch['avm'].max()*1.05 if row.Index % 2 == 0 else df_epoch['avm'].max()*1.1, s=row.Index)
            except TypeError:
                pass
            print(f"-#{row.Index}| {row.start_time} - {row.activity}")

    ax[-1].xaxis.set_major_formatter(xfmt_raw)

    plt.tight_layout()


def import_data2():
    # df_epoch, epoch_len, df_act, df_walk, df_sleep, df_sleep_alg, df_gait, df_act_log, ankle, wrist = None, None, None, None, None, None, None, None, None, None

    print("\nImporting and formatting summary dataframes...")

    # EDF file details ------------------------------------
    edfs = os.listdir(edf_folder)
    edfs = [i for i in edfs if subj in i and "Wrist" in i]

    # List of days in collection period
    dates = []

    start_time = pyedflib.EdfReader(edf_folder + edfs[0]).getStartdatetime()
    file_dur = pyedflib.EdfReader(edf_folder + edfs[0]).file_duration

    # Epoched wrist data -------------------------------------------------------
    df_epoch = pd.read_csv(epoch_folder + f"OND09_{subj}_01_EPOCH_ACTIVITY.csv")
    epoch_len = int(file_dur / df_epoch.shape[0])
    df_epoch["Timestamp"] = pd.date_range(start=start_time, freq=f"{epoch_len}S", periods=df_epoch.shape[0])
    df_epoch = df_epoch[["Timestamp", "avm"]]

    for day in set([i.date() for i in df_epoch['Timestamp']]):
        dates.append(day)

    # Sleep data (algorithm output)
    df_sleep_alg = pd.read_csv(sleep_output_file)
    df_sleep_alg["start_time"] = pd.to_datetime(df_sleep_alg["start_time"])
    df_sleep_alg["end_time"] = pd.to_datetime(df_sleep_alg["end_time"])

    # Sleep windows
    df_sptw = pd.read_csv(sleep_window_file)
    df_sptw["start_time"] = pd.to_datetime(df_sptw["start_time"])
    df_sptw["end_time"] = pd.to_datetime(df_sptw["end_time"])
    df_sptw['relative_date'] = pd.to_datetime(df_sptw['relative_date'])
    df_sptw['relative_date'] = [i.relative_date.date() for i in df_sptw.itertuples()]

    for day in set(df_sptw["relative_date"]):
        if day not in dates:
            dates.append(day)

    # Activity descriptive data by day
    df_epoch["Day"] = [row.Timestamp.date() for row in df_epoch.itertuples()]
    avm_desc = df_epoch.groupby("Day")["avm"].describe()
    avm_sum = df_epoch.groupby("Day")['avm'].sum()

    df_act = pd.DataFrame({"Date": [i for i in set([row.Timestamp.date() for row in df_epoch.itertuples()])],
                           "svm": [i*epoch_len for i in avm_sum],
                           "mean_avm": avm_desc['mean'].reset_index(drop=True),
                           "std_avm": avm_desc['std'].reset_index(drop=True)})

    # Activity log (subjective)
    df_act_log = pd.read_excel(activity_log_file)
    df_act_log = df_act_log.loc[df_act_log["subject_id"] == int(subj)].reset_index(drop=True)
    df_act_log = df_act_log.fillna("")
    df_act_log.columns = ['coll_id', 'study_code', 'subject_id', 'activity', 'start_time',
                          'duration', 'Unnamed: 6', 'Notes.', 'Unnamed: 8']

    stamps = []
    for row in df_act_log.itertuples():
        if not type(row.start_time) is str:
            stamps.append(row.start_time)
        if type(row.start_time) is str:
            stamps.append("NOT LEGIBLE")

    df_act_log["start_time"] = stamps

    # Gait bout data
    df_gait = pd.read_csv(gait_file)
    df_gait = df_gait[["start_timestamp", "end_timestamp", "number_steps"]]
    df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
    df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])
    df_gait['duration'] = [(j-i).total_seconds() for i, j in zip(df_gait['start_timestamp'], df_gait['end_timestamp'])]
    df_gait['cadence'] = 60*df_gait["number_steps"]/df_gait['duration']

    df_steps = pd.read_csv(steps_file)
    df_steps["step_time"] = pd.to_datetime(df_steps["step_time"])

    print("\nImporting ankle data...")
    ankle = nwdata.NWData()
    ankle.import_edf(file_path=ankle_file, quiet=False)

    print("\nImporting wrist data...")
    wrist = nwdata.NWData()
    wrist.import_edf(file_path=wrist_file, quiet=False)
    print("Data imported.")

    return df_epoch, epoch_len, df_act, df_sleep_alg, df_gait, df_steps, df_act_log, ankle, wrist


def calculate_daily_activity(side="Non-dominant"):

    days = sorted([i for i in set([i.date() for i in df_epoch['Timestamp']])])
    days = pd.date_range(start=days[0], end=days[-1] + timedelta(days=1), freq='1D')

    daily_vals = []
    for day1, day2 in zip(days[:], days[1:]):
        df_day = df_epoch.loc[df_epoch["Day"] == day1]

        sed = df_day.loc[df_day['avm'] < cutpoints[side][0]]
        sed_mins = sed.shape[0] * epoch_len / 60

        light = df_day.loc[(df_day['avm'] >= cutpoints[side][0]) & (df_day['avm'] < cutpoints[side][1])]
        light_mins = light.shape[0] * epoch_len / 60

        mod = df_day.loc[(df_day['avm'] >= cutpoints[side][1]) & (df_day['avm'] < cutpoints[side][2])]
        mod_mins = mod.shape[0] * epoch_len / 60

        vig = df_day.loc[df_day['avm'] >= cutpoints[side][2]]
        vig_mins = vig.shape[0] * epoch_len / 60

        df_gait_day = df_gait.loc[(day1 <= df_gait['start_timestamp']) & (df_gait['start_timestamp'] < day2)]
        n_steps = df_gait_day["number_steps"].sum()
        walk_mins = df_gait_day["duration"].sum()/60

        daily_vals.append([day1, sed_mins, light_mins, mod_mins, vig_mins, n_steps, walk_mins])

    df_daily = pd.DataFrame(daily_vals, columns=["Date", "Sed", "Light", "Mod", "Vig", "Steps", "MinutesWalking"])
    df_daily["Active"] = df_daily["Light"] + df_daily["Mod"] + df_daily["Vig"]
    df_daily["MVPA"] = df_daily["Mod"] + df_daily["Vig"]
    df_daily['avm'] = df_act['mean_avm']

    return df_daily


def generate_scatter(df, x, y):

    reg = scipy.stats.linregress(x=df[x], y=df[y])

    m = reg.slope
    b = reg.intercept
    r = reg.rvalue
    p = reg.pvalue

    x_vals = np.linspace(df[x].min(), df[x].max(), 100)
    y_vals = [i*m + b for i in x_vals]

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.scatter(df[x], df[y], color='black')
    ax.plot(x_vals, y_vals, color='red', label=f"r={round(r, 3)}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()


df_epoch, epoch_len, df_act, df_sleep_alg, df_gait, df_steps, df_act_log, ankle, wrist = import_data2()
df_daily = calculate_daily_activity()

# df_epoch, epoch_len, df_act, df_walk, df_sleep, df_sleep_alg, df_gait, df_act_log, ankle, wrist = import_data()
# summary_plot()
plot_raw(ds_ratio=5, dominant=True, incl_activity_volume=False, incl_step_count=True, show_activity_log=True, ankle_gyro=True, shade_gait_bouts=True, shade_sleep_windows=False, mark_steps=False)
# generate_scatter(df_daily, "Steps", "Active")

"""
intensity = []
side = 'Non-dominant'
for row in df_epoch.itertuples():

    if row.avm < cutpoints[side][0]:
        intensity.append("sedentary")

    if cutpoints[side][0] <= row.avm < cutpoints[side][1]:
        intensity.append("light")

    if cutpoints[side][1] <= row.avm < cutpoints[side][2]:
        intensity.append('moderate')

    if cutpoints[side][2] <= row.avm:
        intensity.append('vigorous')

df_epoch["Kyle"] = intensity
df_notmatched = df_epoch.loc[df_epoch["intensity"] != df_epoch['Kyle']]"""