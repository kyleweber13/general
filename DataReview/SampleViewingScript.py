# python packages
import matplotlib.pyplot as plt  # plotting
from matplotlib import dates as mdates  # timestamp formatting on graphs
import pandas as pd  # dataframe

# our stuff
import nimbalwear  # pipeline
from nimbalwear.activity import activity_wrist_avm  # activity counts


def import_imu(filepath):

    # initializes empty class instance
    data = nimbalwear.Device()

    # imports EDF file
    data.import_edf(file_path=filepath, quiet=False)

    # dictionary key for start time
    start_key = 'start_datetime' if 'start_datetime' in data.header.keys() else 'start_time'

    # raw timestamps for signal[0]
    data.ts = pd.date_range(start=data.header[start_key], periods=len(data.signals[0]),
                            freq="{}ms".format(1000 / data.signal_headers[0]["sample_rate"]))

    # creates timestamps for temperature if there's a channel named 'Temperature'
    try:
        data.temp_ts = pd.date_range(start=data.header[start_key],
                                     periods=len(data.signals[data.get_signal_index('Temperature')]),
                                     freq="{}ms".format(1000 / data.signal_headers[data.get_signal_index('Temperature')]["sample_rate"]))
    except KeyError:
        pass

    return data


def import_tabular(filepath):

    print(f"\nImporting {filepath}...")

    # Reads csv or xlsx file
    df = pd.read_csv(filepath) if 'csv' in filepath else pd.read_excel(filepath)

    # converts timestamps from str to timestamp objects for all columns with 'time' in their name
    for col in df.columns:
        if 'time' in col:
            print(f"-Converting column {col} to timestamps")
            df[col] = pd.to_datetime(df[col])

    print("Complete.")

    return df


""" PLAY AROUND WITH THIS. THIS IS A GENERIC FORMAT WITH SOME FUNCTIONALITY SHOWN """


def plot_data(downsample_ratio=1):
    """ downsample_ratio helps manage data volume getting plotted by plotting every nth datapoint """

    # Creates plotting window and 'ax' object with is an array for each subplot
    fig, ax = plt.subplots(nrows=4, ncols=1,  # subjplot layout
                           sharex='col',  # columns share x-axis data so you can zoom/scroll together
                           figsize=(10, 6)  # figure size in inches
                           )

    # line plot on first subplot ---------------------------------------------------------
    # gets accel-x channel and plots with timestamps in black
    ax[0].plot(imu.ts[::downsample_ratio], imu.signals[imu.get_signal_index('Accelerometer x')][::downsample_ratio],
               color='black',  # can also use hex/HTML codes for precise colors. Yes, colors. Silly Americans.
               label='accel x'  # label for legend
               )

    # sets y-axis label
    ax[0].set_ylabel("G")

    # legend in fixed location (slow to render if not set)
    ax[0].legend(loc='upper right')

    # turns on grid
    ax[0].grid()

    # line plot on second subplot ---------------------------------------------------------
    # gets accel-y channel and plots with timestamps in black
    ax[1].plot(imu.ts[::downsample_ratio], imu.signals[imu.get_signal_index('Accelerometer y')][::downsample_ratio],
               color='red',
               label='accel y'  # label for legend
               )

    # sets y-axis label
    ax[1].set_ylabel("G")

    # legend in fixed location (slow to render if not set)
    ax[1].legend(loc='upper right')

    # turns on grid
    ax[1].grid()

    # line plot on third subplot ---------------------------------------------------------
    # gets accel-z channel and plots with timestamps in black
    ax[2].plot(imu.ts[::downsample_ratio], imu.signals[imu.get_signal_index('Accelerometer z')][::downsample_ratio],
               color='dodgerblue',
               label='accel z'  # label for legend
               )

    # sets y-axis label
    ax[2].set_ylabel("G")

    # legend in fixed location (slow to render if not set)
    ax[2].legend(loc='upper right')

    # turns on grid
    ax[2].grid()

    """ Bar graph for activity counts ----------------------------------------------- """

    # Figures out epoch length from time difference between first start and end times; needed for bar width
    epoch_len = (df_epoch.iloc[0]['end_time'] - df_epoch.iloc[0]['start_time']).total_seconds()

    ax[3].bar(df_epoch['start_time'], df_epoch['avm'],
              color='grey', edgecolor='black',  # grey bars with black outline
              alpha=.5,  # 50% transparency
              align='edge',  # aligns bar to left of timestamp instead of centering around start_time
              width=epoch_len/86400  # bar width for timestamps is in fraction of 1 day (86400 seconds) for some reason
              )

    """ SHADING REGIONS OF GRAPH FROM TABULAR DATA ----------------------------- """
    # wonderful feature called axvspan

    # loops subplots so we don't have to type this out for all subplots
    for subplot in range(len(ax)):

        # loops through each row in dataframe
        # columns are now accessed as row.<column name> --> never put spaces in column names because this doesn't work then
        for row in df_gait.itertuples():
            ax[subplot].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp,  # x-axis limits (actual units; time) for shading
                                ymin=0, ymax=1,  # y-axis limits (normalized to range) for shading
                                color='gold', alpha=.5  # gold shading with 50% transparency
                                )

    """ timestamp formatting for x-axis ---------------------------------------- """
    # timestamp formatting. See https://docs.python.org/3/library/datetime.html (table near bottom)
    # YYYY-MM-DD <newline> HH:MM:SS
    xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")

    # only call this for last ([-1]) axis when sharex feature used
    ax[-1].xaxis.set_major_formatter(xfmt)

    # expands graph to fill plotting window
    plt.tight_layout()


# running stuff -----------------------------------------------------------------------------------
# this statement prevents the script from being run if it's imported by another script ------------
if __name__ == 'main':

    # ID in format <study_code>_<sitecode><subject_id>, e.g., OND09_007 or OND09_SBH007
    full_id = 'OND09_0003'

    # imports single device data
    imu = import_imu(filepath=f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{full_id}_01_AXV6_RWrist.edf")

    # epochs imu data
    epoch_data = activity_wrist_avm(x=imu.signals[imu.get_signal_index('Accelerometer x')],  # x-axis accel
                                    y=imu.signals[imu.get_signal_index('Accelerometer y')],  # y-axis accel
                                    z=imu.signals[imu.get_signal_index('Accelerometer z')],  # z-axis accel
                                    sample_rate=imu.signal_headers[imu.get_signal_index('Accelerometer x')]['sample_rate'],  # accel sample rate
                                    start_datetime=imu.ts[0],  # first timestamp,
                                    epoch_length=15  # epoch length in seconds
                                    )
    df_epoch = epoch_data[0]

    # imports tabular data
    df_gait = import_tabular(filepath=f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/{full_id}_01_GAIT_BOUTS.csv")

    # event durations. zip function iterates through two lists at same time
    df_gait['duration'] = [(end - start).total_seconds() for start, end in zip(df_gait['start_timestamp'], df_gait['end_timestamp'])]

    # cropped based on criteria ---------------
    # creates a boolean mask for what rows meet the condition and then only keeps those 'True' row indexes
    df_gait_long = df_gait.loc[df_gait['duration'] >= 180]

    # multiple conditions formatting: each condition in round brackets with & operator between
    df_gait_medium = df_gait.loc[(df_gait['duration'] >= 60) & (df_gait['duration'] < 180)]

    # plotting -------------------------------------------
    # play around with plot_data() function
    plot_data(downsample_ratio=2)
