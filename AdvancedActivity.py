import nwactivity
import nwdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")


def import_tabular_data(accel_filepath, gait_filepath, posture_filepath, wrist_epoch_length=15):
    """Imports outputs from NWGait, NWPosture, and NWActivity. Returns as separate DFs.

        argument:
        -wrist_epoch_length: epoch length in seconds used to derive wrist activity data

       Currently crops data to end when posture data ends. This will throw out data after the Bittium dies.
    """

    """POSTURE"""
    df = pd.read_csv(posture_filepath)
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])

    # Reconstructs bouted posture dataframe into 1-sec epoched
    stamps = pd.date_range(start=df.iloc[0]['start_timestamp'], end=df.iloc[-1]['end_timestamp'], freq='1S')

    post_dict = {"sit": 0, 'supine': 1, 'stand': 2, 'other': 3, 'prone': 4, 'sitstand': 5, 'leftside': 6,
                 'rightside': 7}
    keys = list(post_dict.keys())
    posture = np.array([None] * (len(stamps) - 1))
    for row in df.itertuples():
        start = int((row.start_timestamp - stamps[0]).total_seconds())
        end = int((row.end_timestamp - stamps[0]).total_seconds())
        posture[start:end] = post_dict[row.posture]
    posture = np.append(posture, posture[-1])

    df_posture = pd.DataFrame({"timestamp": stamps, 'posture': [keys[i] for i in posture]})

    stop = df_posture.iloc[-1]['timestamp']
    print("REMOVE 'STOP' FUNCTIONALITY")

    """WRIST"""
    df_wrist = pd.read_csv(accel_filepath)
    df_wrist = df_wrist.iloc[:int((stop - df_posture.iloc[0]['timestamp']).total_seconds() / wrist_epoch_length)]

    """GAIT"""
    df_gait = pd.read_csv(gait_filepath)
    df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
    df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])
    df_gait = df_gait.loc[df_gait['end_timestamp'] <= stop]

    return df_posture, df_wrist, df_gait


def combine_data(start_stamp, df_wrist, df_gait, df_posture, wrist_epoch_len=15,
                 lying_is_sedentary=True, plot_data=False):
    """More advanced physical activity intensity estimation that combines the outputs from
       cut-point-derived intensity with gait and posture data.
       Gait periods are set as moderate intensity. Periods of standing are set as light intensity. For each 1-sec
       period, the highest intensity is used.
        :argument
        -start_stamp: timestamp data begins, used to generate timestamps for wrist data
        -df_wrist: dataframe containing time series data of epoched cut-point-derived intensity
                  -columns: "Timestamp", "intensity"
        -df_gait: dataframe containing gait bout start/stop times
                -columns: "start_timestamp", "end_timestamp"
        -df_posture: dataframe containing 1-second epoched posture data
                -columns: 'timestamp', 'posture'/'v3'
        -lying_is_sedentary: boolean of whether to treat all supine/prone/lying on side as sedentary
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
        for j in range(wrist_epoch_len):
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
                                                       'supine': 0, 'prone': 0, 'leftside': 0, 'rightside': 0}))

    # Flags regions where posture is lying down in any posture ------------------------------------
    lying = np.array(df_posture['posture'].replace({"sit": 0, 'stand': 0, 'gait': 0, 'other': 0, 'sitstand': 0,
                                                    'supine': 1, 'prone': 1, 'leftside': 1, 'rightside': 1}))

    # finalizing data -----------
    epoch_stamps = epoch_stamps[:len(cp)]
    df_out = pd.DataFrame({"Timestamp": epoch_stamps, "Wrist": cp, "Gait": gait,
                           "Stand": standing, "Lying": lying})

    # All sedentary epochs will be sedentary regardless of wrist/gait intensity ------------------------------------

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


"""
# ----------------------------------------------------- Sample run ----------------------------------------------------
subj = '0008'
df_posture, df_wrist, df_gait = import_tabular_data(accel_filepath=f"W:/NiMBaLWEAR/OND09/analytics/activity/epoch/OND09_{subj}_01_EPOCH_ACTIVITY.csv",
                                                    gait_filepath=f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv",
                                                    posture_filepath=f"C:/users/ksweber/Desktop/OND09_{subj}_Posture2.csv")
df_ts, df_bout = combine_data(start_stamp=pd.to_datetime(df_posture.iloc[0]['timestamp']),
                              df_wrist=df_wrist, df_gait=df_gait, df_posture=df_posture,
                              lying_is_sedentary=True, plot_data=True)
"""

# TODO
# make sure timestamps in bouts are not flagging epochs as multiple things
    # Kit: should bout_end and next bout_start be 1-sec apart or same timestamp?
