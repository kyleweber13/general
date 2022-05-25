import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


def plot_device(data_object, raw_timestamps, df_epoch, ax1=None, ax2=None, label=""):

    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))

    ax1.set_title(f"{label}: Raw")
    ax1.plot(raw_timestamps, data_object.signals[data_object.get_signal_index("Accelerometer x")], color='black')
    ax1.plot(raw_timestamps, data_object.signals[data_object.get_signal_index("Accelerometer y")], color='red')
    ax1.plot(raw_timestamps, data_object.signals[data_object.get_signal_index("Accelerometer z")], color='dodgerblue')
    ax1.set_ylabel("G")
    ax1.grid()

    ax2.plot(df_epoch['timestamp'], df_epoch['value'], color='black')
    ax2.set_title(f"{label}: Epoched")

    ax2.xaxis.set_major_formatter(xfmt)

    plt.tight_layout()


def plot_epoched_context(df_epoch):

    fig, ax = plt.subplots(3, sharex='col', figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1, .33]})

    epoch_len = int((df_epoch.iloc[1]['timestamp'] - df_epoch.iloc[0]['timestamp']).total_seconds())

    max_y = max([df_epoch['dominant'].max(), df_epoch['nondom'].max()]) * 1.05
    ax[0].plot(df_epoch['timestamp'], df_epoch['dominant'], color='red')
    ax[0].set_ylim(-5, max_y)
    ax[0].set_title("Dominant")

    ax[1].plot(df_epoch['timestamp'], df_epoch['nondom'], color='dodgerblue')
    ax[1].set_ylim(-5, max_y)
    ax[1].set_title("Non-Dominant")

    ax[2].plot(df_epoch['timestamp'], [row.gait/4 for row in df_epoch.itertuples()], color='green')
    ax[2].plot(df_epoch['timestamp'], [.33 + row.sleep/4 for row in df_epoch.itertuples()], color='navy')
    ax[2].plot(df_epoch['timestamp'], [.66 + row.nonwear/4 for row in df_epoch.itertuples()], color='grey')

    ax[2].set_yticks([0, .33, .66])
    ax[2].set_ylim(-.16, 1)
    ax[2].grid()
    ax[2].set_yticklabels(['gait', 'sleep', 'nw'])

    ax[2].xaxis.set_major_formatter(xfmt)

    plt.tight_layout()
