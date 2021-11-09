import nwactivity
import nwdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")

"""--------------------------------------------------- Data import -------------------------------------------------"""


def import_data(accel_filepath, cp_epochlen, gait_filepath, posture_filepath):

    d = nwdata.NWData()
    d.import_edf(file_path=accel_filepath)

    df_powell = nwactivity.calc_wrist_powell(x=d.signals[0], y=d.signals[1], z=d.signals[2],
                                             sample_rate=d.signal_headers[0]["sample_rate"], epoch_length=cp_epochlen)

    # Timestamps
    start_time = d.header["startdate"]
    timestamps = pd.date_range(start=start_time, periods=df_powell.shape[0], freq=f"{cp_epochlen}S")
    df_powell["Timestamp"] = timestamps

    # Gait data ------------------------------------------------
    df_gait = pd.read_csv(gait_filepath) if "csv" in gait_filepath else pd.read_excel(gait_filepath, engine='openpyxl')

    # Bout durations - LIKELY ALREADY IN THE FILE
    df_gait = df_gait[["start_timestamp", "end_timestamp", "bout_length_sec"]]

    df_gait["start_timestamp"] = pd.to_datetime(df_gait["start_timestamp"])
    df_gait["start_timestamp"] = [i.round("1S") for i in df_gait["start_timestamp"]]

    df_gait["end_timestamp"] = pd.to_datetime(df_gait["end_timestamp"])
    df_gait["end_timestamp"] = [i.round("1S") for i in df_gait["end_timestamp"]]

    # Posture data --------------------------------------------
    df_post = pd.read_csv(posture_filepath) if "csv" in posture_filepath else \
        pd.read_excel(posture_filepath, engine='openpyxl')

    df_post["start_timestamp"] = pd.to_datetime(df_post["start_timestamp"])
    df_post["end_timestamp"] = pd.to_datetime(df_post["end_timestamp"])

    return df_powell, df_gait, df_post, start_time


"""
df_wrist, df_gait, df_posture, start_time = import_data(accel_filepath="/Volumes/Kyle's External HD/OND06 NDWrist Data/Accelerometer/OND06_SBH_3413_GNAC_ACCELEROMETER_LWrist.edf",
                                                        gait_filepath="/Users/kyleweber/Desktop/Posture_GS/gait_LAnkleRAnkle_OND06_3413.csv",
                                                        posture_filepath="/Users/kyleweber/Desktop/Posture_GS/posture_bouts_OND06_3413.xlsx",
                                                        cp_epochlen=15)
"""


"""--------------------------------------------------- Actual code ------------------------------------------------"""


def import_tabular_data(accel_filepath, gait_filepath, posture_filepath, wrist_epoch_length=15):
    df_posture = pd.read_csv(posture_filepath)
    df_posture = df_posture[["timestamp", 'v3' if 'v3' in df_posture.columns else 'posture']]
    df_posture.columns = ['timestamp', 'posture']
    df_posture['timestamp'] = pd.to_datetime(df_posture['timestamp'])

    stop = df_posture.iloc[-1]['timestamp']
    print("REMOVE 'STOP' FUNCTIONALITY")

    df_wrist = pd.read_csv(accel_filepath)
    df_wrist = df_wrist.iloc[:int((stop - df_posture.iloc[0]['timestamp']).total_seconds() / wrist_epoch_length)]

    df_gait = pd.read_csv(gait_filepath)
    df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
    df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])
    df_gait = df_gait.loc[df_gait['end_timestamp'] <= stop]

    return df_posture, df_wrist, df_gait


def combine_data(start_stamp, df_wrist, df_gait, df_posture, epoch_len=15,
                 lying_is_sedentary=False, plot_data=False):
    """More advanced physical activity intensity estimation that combines the outputs from
       cut-point-derived intensity with gait and posture data.
       Gait periods are set as moderate intensity. Periods of standing are set as light intensity. For each 1-sec
       period, the highest intensity is used.
        :argument
        -df_cp: dataframe containing time series data of epoched cut-point-derived intensity
                -columns: "Timestamp", "intensity"
        -df_gait: dataframe containing gait bout start/stop times
                -columns: "start_timestamp", "end_timestamp"
        -df_posture: dataframe containing bout start/stop times for each posture
                -columns: "bout_start", "bout_end", "posture"
        -sedentary_lying: boolean of whether to treat all supine/prone/lying on side as sedentary
                -if False, highest intensity will be taken
                -if True, lying periods will all be sedentary regardless of other domains' intensity measure
        :returns
        -df_ts: dataframe of time series intensity data in 1-second epochs
        -df_bouts: dataframe with start/stop timestamps for each intensity bout
    """

    print("\nRecalculating activity intensity using gait and posture data...")

    # Data preparation -----------------------------------------------------------------------------------------------

    epoch_stamps = list(pd.date_range(start=df_posture.iloc[0]['timestamp'],
                                      end=df_posture.iloc[-1]['timestamp'], freq='1S'))
    posture_epoch_len = (df_posture.iloc[1]['timestamp'] - df_posture.iloc[0]['timestamp']).total_seconds()
    if posture_epoch_len != 1:
        print("\nPosture data was not generated using 1-second epochs --> chaos will now ensue")

    # Dataframe cropping to ensure same length data
    # Crop based on posture/wrist timestamps (ignore gait bout timestamps)

    # Powell data to 1-sec epochs ---------------

    # Converts str categories from output to integer categories
    intensity_dict = {"sedentary": 0, "light": 1, "moderate": 2, "vigorous": 3}

    # Cutpoints' intensity ---------
    cp = []
    for i in df_wrist["intensity"]:
        # Repeats epoch value epoch_len number of times (generates 1-sec epochs)
        for j in range(epoch_len):
            cp.append(intensity_dict[i])

    # maintains data output length
    if len(cp) < df_posture.shape[0]:
        df_posture = df_posture.iloc[:-(df_posture.shape[0] - len(cp))]

    # Posture/gait data ----------------------------

    # Gait data to 1-s epochs ------------
    gait = np.zeros(len(cp))

    # Assigns moderate intensity (2) to all gait bouts
    for row in df_gait.itertuples():
        start = int((row.start_timestamp - start_stamp).total_seconds())
        stop = int((row.end_timestamp - start_stamp).total_seconds())
        gait[start:stop] = 2

    # Posture data to 1-s epochs ---------
    # Assigns light intensity to standing bouts ---------------------------------
    # Index corresponds to number of seconds since start of collection since it's 1-second data
    standing = np.array(df_posture['posture'].replace({"sit": 0, 'stand': 1, 'gait': 0, 'other': 0, 'sitstand': 0,
                                                       'supine': 0, 'prone': 0, 'lyingleft': 0, 'lyingright': 0}))

    # Flags regions where posture is lying down in any posture ------------------------------------
    lying = np.array(df_posture['posture'].replace({"sit": 0, 'stand': 0, 'gait': 0, 'other': 0, 'sitstand': 0,
                                                    'supine': 1, 'prone': 1, 'lyingleft': 1, 'lyingright': 1}))

    # finalizing data -----------
    epoch_stamps = epoch_stamps[:len(cp)]
    df_out = pd.DataFrame({"Timestamp": epoch_stamps, "Wrist": cp, "Gait": gait,
                           "Stand": standing, "Lying": lying})

    # All sedentary epochs will be sedentary regardless of wrist/gait intensity ------------------------------------

    print("===== To fix: is all lying sedentary regardless of wrist movement? =====")

    final_intensity = []
    for row in df_out.itertuples():

        if row.Lying == 1 and lying_is_sedentary:
            final_intensity.append(0)
        if row.Lying == 0 or not lying_is_sedentary:
            final_intensity.append(max(row.Wrist, row.Gait, row.Stand))

    df_out["Final"] = final_intensity

    # bout flagging -------------
    indexes = []
    intensity = []
    for i in range(len(final_intensity) - 1):
        if final_intensity[i] != final_intensity[i+1]:
            indexes.append(i+1)
            intensity.append(int(final_intensity[i+1]))
    indexes.append(len(final_intensity)-1)
    intensity.append(0)

    starts = [epoch_stamps[i] for i in indexes]
    ends = [i for i in starts[1:]]

    if len(ends) < len(starts):
        ends.append(epoch_stamps[-1])

    df_bout = pd.DataFrame({"bout_num": np.arange(1, len(starts) + 1), "start_timestamp": starts,
                            "end_timestamp": ends, "intensity": intensity})
    df_bout['duration'] = [(a-b).total_seconds() for a, b in zip(df_bout["end_timestamp"], df_bout["start_timestamp"])]

    if plot_data:
        print("\nGenerating plot...")

        fig, axes = plt.subplots(5, sharex='col', figsize=(12, 9),
                                 gridspec_kw={"height_ratios": [1, .33, .335, .33, 1]})

        # Wrist cutpoints intensity
        axes[0].fill_between(df_out["Timestamp"], y1=0, y2=df_out["Wrist"], where=(df_out["Wrist"] == 1),
                             color='green', alpha=.5)
        axes[0].fill_between(df_out["Timestamp"], y1=0, y2=df_out["Wrist"], where=(df_out["Wrist"] == 2),
                             color='orange', alpha=.5)
        axes[0].fill_between(df_out["Timestamp"], y1=0, y2=df_out["Wrist"], where=(df_out["Wrist"] == 3),
                             color='red', alpha=.5)
        axes[0].grid()

        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].set_title("Cut-Points Intensity")
        axes[0].set_ylabel("Intensity")

        # Gait bouts
        axes[1].fill_between(df_out["Timestamp"], y1=0, y2=df_out['Gait']/2, color='grey', alpha=.5)
        axes[1].set_yticks([0, 1])
        axes[1].set_title("Gait")

        # Standing classification
        axes[2].fill_between(df_out["Timestamp"], y1=0, y2=df_out['Stand'], color='purple', alpha=.5)
        axes[2].set_title("Stand")
        axes[2].set_yticks([0, 1])

        # Lying classification
        axes[3].fill_between(df_out["Timestamp"], y1=0, y2=df_out['Lying'], color='dodgerblue', alpha=.5)
        axes[3].set_title("Lying")
        axes[3].set_yticks([0, 1])

        # Final activity intensity
        axes[4].fill_between(df_out["Timestamp"], y1=0, y2=df_out["Final"], where=(df_out["Final"] == 1),
                             color='green', alpha=.5)
        axes[4].fill_between(df_out["Timestamp"], y1=0, y2=df_out["Final"], where=(df_out["Final"] == 2),
                             color='orange', alpha=.5)
        axes[4].fill_between(df_out["Timestamp"], y1=0, y2=df_out["Final"], where=(df_out["Final"] == 3),
                             color='red', alpha=.5)
        axes[4].set_yticks([0, 1, 2, 3])
        axes[4].set_title("Final Intensity")
        axes[4].set_ylabel("Intensity")
        axes[4].grid()

        axes[4].set_xlim(df_posture.iloc[0]['timestamp'] + timedelta(seconds=-300),
                         df_posture.iloc[-1]['timestamp'] + timedelta(seconds=+300))
        axes[-1].xaxis.set_major_formatter(xfmt)

        plt.tight_layout()

    print("Complete.")

    return df_out, df_bout


df_posture, df_wrist, df_gait = import_tabular_data(accel_filepath = "W:/NiMBaLWEAR/OND09/analytics/activity/epoch/OND09_0020_01_EPOCH_ACTIVITY.csv",
                                                    gait_filepath = "W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_0020_01_GAIT_BOUTS.csv",
                                                    posture_filepath = "C:/users/ksweber/Desktop/OND09_0020_PostureTest.csv")
df_ts, df_bout = combine_data(start_stamp=pd.to_datetime("2021-09-27 15:12:25"),
                              df_wrist=df_wrist, df_gait=df_gait, df_posture=df_posture,
                              lying_is_sedentary=True, plot_data=True)


# TODO
# make sure timestamps in bouts are not flagging epochs as multiple things
    # Kit: should bout_end and next bout_start be 1-sec apart or same timestamp?
# Time resolution? Round to nearest second?
# Minimum bout durations?
# Should all lying be sedentary regardless of wrist intensity?
# Wrist dataframe has no timestamps --> assuming start time is same as wrist EDF
