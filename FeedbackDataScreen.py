import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pyedflib
xfmt = mdates.DateFormatter("%a\n%b-%d")
xfmt_raw = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")

from datetime import timedelta
import nwdata

subj = '0008'
tabular_file = "O:/OBI/ONDRI@Home/Participant Summary Data - Feedback/HANDDS Feedback Forms/Summary Dataframes/OND09 Summary Dataframes.xlsx"

gait_file = f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv"

activity_log_file = f"O:/OBI/HANDDS-ONT/Logistic Planning for Launch/File and Folder Structure/HANDDS_ActivityLogTemplate.xlsx"

epoch_folder = "W:/NiMBaLWEAR/OND09/analytics/activity/epoch/"
edf_folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"

ankle_file = f"{edf_folder}OND09_{subj}_01_AXV6_RAnkle.edf"
if not os.path.exists(ankle_file):
    ankle_file = f"{edf_folder}OND09_{subj}_01_AXV6_LAnkle.edf"

wrist_file = f"{edf_folder}OND09_{subj}_01_AXV6_RWrist.edf"
if not os.path.exists(wrist_file):
    wrist_file = f"{edf_folder}OND09_{subj}_01_AXV6_LWrist.edf"

cutpoints = {"Dominant": [51*1000/450, 68*1000/450, 142*1000/450],
             "Non-dominant": [47*1000/450, 64*1000/450, 157*1000/450]}


def import_data():

    print("\nImporting and formatting summary dataframes...")

    edfs = os.listdir(edf_folder)
    edfs = [i for i in edfs if subj in i and "Wrist" in i]

    start_time = pyedflib.EdfReader(edf_folder + edfs[0]).getStartdatetime()
    file_dur = pyedflib.EdfReader(edf_folder + edfs[0]).file_duration

    df_epoch = pd.read_csv(epoch_folder + f"OND09_{subj}_01_EPOCH_ACTIVITY.csv")
    epoch_len = int(file_dur/df_epoch.shape[0])
    df_epoch["Timestamp"] = pd.date_range(start=start_time, freq=f"{epoch_len}S", periods=df_epoch.shape[0])
    df_epoch = df_epoch[["Timestamp", "avm"]]
    epoch_len = int((df_epoch["Timestamp"].iloc[1] - df_epoch["Timestamp"].iloc[0]).total_seconds())

    df_tab = pd.read_excel(tabular_file, sheet_name=f"{subj} Summary Dataframes", header=None)
    df_tab = df_tab.dropna(how='all')

    sed_ind = df_tab.loc[df_tab[0] == "Sedentary"].index[0]

    stop_inds = []
    for i in range(sed_ind+1, df_tab.shape[0]):
        if type(df_tab.iloc[i][0]) is str:
            stop_ind = i
            stop_inds.append(stop_ind)
    stop_inds.append(df_tab.shape[0])

    df_sed = df_tab.iloc[sed_ind+1:stop_inds[0]].iloc[:, 1:3]
    df_sed.columns = ["Date", "Duration"]

    df_act = df_tab.iloc[stop_inds[0]+1:stop_inds[1]].iloc[:, 1:4]
    df_act.columns = ["Date", "Light", "MVPA"]
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

    df_walk = df_tab.iloc[stop_inds[1]+1:stop_inds[2]].iloc[:, 1:6]
    df_walk.columns = ["Date", "LongestBout", "StepsLongest", "Bouts>3mins", "TotalSteps"]
    df_walk["Date"] = pd.to_datetime(df_walk["Date"])

    df_sleep = df_tab.iloc[stop_inds[2]+1:stop_inds[3]].iloc[1:, 1:7]
    df_sleep.columns = ["Date", "TST", "BedTime", "TimeOutBed", "NightlySleepTime", "NumWalks"]
    df_sleep["Date"] = pd.to_datetime(df_sleep["Date"])

    df_epoch["Day"] = [row.Timestamp.date() for row in df_epoch.itertuples()]
    avm_desc = df_epoch.groupby("Day")["avm"].describe()
    df_act["mean_avm"] = avm_desc['mean'].reset_index(drop=True)
    df_act["std_avm"] = avm_desc['std'].reset_index(drop=True)
    df_act["Date"] = pd.to_datetime(df_act["Date"])

    df_act_log = pd.read_excel(activity_log_file)
    df_act_log = df_act_log.loc[df_act_log["subject_id"] == int(subj)].reset_index(drop=True)
    df_act_log = df_act_log.fillna("")
    df_act_log.columns = ['coll_id', 'study_code', 'subject_id', 'activity', 'start_time',
                          'duration', 'Unnamed: 6', 'Notes.', 'Unnamed: 8']

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
           df_gait, df_act_log, ankle, wrist


def summary_plot():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex='all')
    plt.suptitle(f"OND09_{subj}")
    plt.subplots_adjust(left=.05, right=.975, wspace=.15)
    plt.rcParams['axes.labelsize'] = '8'

    axes[0][0].bar(df_act["Date"], df_act["mean_avm"], edgecolor='black', color='grey', label="Active",
                   yerr=df_act["std_avm"], capsize=3, alpha=.75)
    axes[0][0].set_ylim(0, )
    axes[0][0].set_title("AVM by Day")

    axes[0][1].bar(df_walk["Date"], df_walk["TotalSteps"]/1000, zorder=1,
                   edgecolor='black', color='dodgerblue', label='Steps', alpha=.75)
    axes[0][1].axhline(y=7, color='green', linestyle='dashed', zorder=0)
    axes[0][1].set_title("Step Counts (x1000)")

    axes[0][2].set_title("Sleep Hours")
    axes[0][2].bar(df_sleep["Date"], df_sleep["TST"], color='navy', edgecolor='black', alpha=.75, zorder=1)
    axes[0][2].fill_between(x=[df_walk.min()["Date"]+timedelta(hours=-12), df_walk.max()["Date"]+timedelta(hours=12)],
                            y1=7, y2=9, color='green', alpha=.25)

    axes[1][2].bar(df_act["Date"], df_act["Sed"], edgecolor='black', color='red', label='Sed', alpha=.75)
    axes[1][2].set_title("Sedentary Minutes")

    axes[1][0].bar(df_act["Date"], df_act["Light"], edgecolor='black', color='forestgreen', label="Light", alpha=.75)
    axes[1][0].set_title("Light Minutes")

    axes[1][1].bar(df_act["Date"], df_act["MVPA"], edgecolor='black', color='orange', label="MVPA", alpha=.75)
    axes[1][1].set_title("MVPA Minutes")

    axes[1][0].xaxis.set_major_formatter(xfmt)
    axes[1][1].xaxis.set_major_formatter(xfmt)
    axes[1][2].xaxis.set_major_formatter(xfmt)
    axes[1][0].tick_params(axis='x', labelsize=8)
    axes[1][1].tick_params(axis='x', labelsize=8)
    axes[1][2].tick_params(axis='x', labelsize=8)


def plot_raw(ds_ratio=1, dominant=True, ankle_gyro=True, incl_activity_volume=False, incl_step_count=False,
             shade_gait_bouts=False, show_activity_log=False):

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

        for row in df_act.itertuples():
            ax_vol.fill_between(x=[row.Date, row.Date+timedelta(days=.999)], y1=0, y2=row.Light,
                                color='green', alpha=.35)
            ax_vol.fill_between(x=[row.Date, row.Date+timedelta(days=.999)], y1=row.Light, y2=row.Light + row.MVPA,
                                color='orange', alpha=.35)
        ax_vol.set_ylabel("Activity Minutes", color='green')

    c = ['green', 'orange', 'red']
    for i, val in enumerate(cp):
        ax[0].axhline(y=val, color=c[i], linestyle='dotted')

    ax[1].set_title("Wrist Accelerometer ({}Hz)".format(int(wrist.signal_headers[wrist.get_signal_index("Accelerometer x")]["sample_rate"]/ds_ratio)))
    ax[1].plot(wrist_ts[::ds_ratio], wrist.signals[wrist.get_signal_index("Accelerometer x")][::ds_ratio], color='black')
    ax[1].plot(wrist_ts[::ds_ratio], wrist.signals[wrist.get_signal_index("Accelerometer y")][::ds_ratio], color='red')
    ax[1].plot(wrist_ts[::ds_ratio], wrist.signals[wrist.get_signal_index("Accelerometer z")][::ds_ratio], color='dodgerblue')
    ax[1].set_ylabel("G")

    if not ankle_gyro:
        ax[2].set_title("Ankle Accelerometer ({}Hz)".format(int(ankle.signal_headers[ankle.get_signal_index("Accelerometer x")]["sample_rate"]/ds_ratio)))
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Accelerometer x")][::ds_ratio], color='black')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Accelerometer y")][::ds_ratio], color='red')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Accelerometer z")][::ds_ratio], color='dodgerblue')
        ax[2].set_ylabel("G")

    if ankle_gyro:
        ax[2].set_title("Ankle Gyroscope ({}Hz)".format(int(ankle.signal_headers[ankle.get_signal_index("Gyroscope x")]["sample_rate"]/ds_ratio)))
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Gyroscope x")][::ds_ratio], color='black')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Gyroscope y")][::ds_ratio], color='red')
        ax[2].plot(ankle_ts[::ds_ratio], ankle.signals[ankle.get_signal_index("Gyroscope z")][::ds_ratio], color='dodgerblue')
        ax[2].set_ylabel("deg/S")

    if incl_step_count:
        ax_step = ax[2].twinx()

        for row in df_walk.itertuples():
            ax_step.fill_between(x=[row.Date, row.Date+timedelta(days=.999)], y1=0, y2=row.TotalSteps,
                                 color='grey', alpha=.35)
        ax_step.set_ylabel("Step Count", color='grey')

    if shade_gait_bouts:
        ylim = ax[2].get_ylim()

        for row in df_gait.itertuples():
            ax[2].fill_between(x=[row.start_timestamp, row.end_timestamp], y1=ylim[0]*1.1, y2=ylim[1]*1.1,
                               color='gold', alpha=.35)

    if show_activity_log:
        for row in df_act_log.itertuples():
            ax[0].fill_between(x=[row.start_time, row.start_time + timedelta(minutes=row.duration)],
                               y1=0, y2=max(df_epoch['avm']*1.1), color='purple', alpha=.35)
            ax[0].text(x=row.start_time + timedelta(minutes=row.duration/3), y=df_epoch['avm'].max()*1.05, s=row.Index)

    ax[-1].xaxis.set_major_formatter(xfmt_raw)

    plt.tight_layout()


df_epoch, epoch_len, df_act, df_walk, df_sleep, df_gait, df_act_log, ankle, wrist = import_data()
# summary_plot()
plot_raw(ds_ratio=2, dominant=True, incl_activity_volume=False, incl_step_count=False, show_activity_log=True,
         ankle_gyro=True, shade_gait_bouts=True)

# TODO
# Pull in gait bouts, sleep windows, etc.
