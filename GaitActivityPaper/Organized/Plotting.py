import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")


def plot_bouts(ankle_obj, df_wrist=None, cutpoints=(62.5, 92.5), ankle_axis='x',
               df_long_bouts=None, df_all_bouts=None, df_steps=None, bout_steps_only=False):

    fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))

    ax[0].plot(ankle_obj.ts, ankle_obj.signals[ankle_obj.get_signal_index(f"{ankle_axis}")], color='black', zorder=0)

    long_fill_val = [0, 1] if df_all_bouts is None else [.5, 1]
    all_fill_val = [0, 1] if df_long_bouts is None else [0, .5]

    if df_long_bouts is not None:

        for row in df_long_bouts.itertuples():
            ax[0].axvspan(xmin=ankle_obj.ts[int(row.start)], xmax=ankle_obj.ts[int(row.end)],
                          ymin=long_fill_val[0], ymax=long_fill_val[1], color='gold', alpha=.35)

            if bout_steps_only and df_steps is not None:
                df_steps_bout = df_steps.loc[(df_steps['step_time'] >= ankle_obj.ts[row.start]) &
                                             (df_steps['step_time'] < ankle_obj.ts[row.end])]

                ax[0].scatter(ankle_obj.ts[df_steps_bout['step_index']],
                              [5 if 'Accelerometer' in ankle_axis else 250]*df_steps_bout.shape[0],
                              color='limegreen', s=25, marker='v', zorder=1)

    if df_steps is not None and not bout_steps_only:
        ax[0].scatter(ankle_obj.ts[df_steps['step_index']],
                      [5 if 'Accelerometer' in ankle_axis else 250]*df_steps.shape[0],
                      color='limegreen', s=25, marker='v', zorder=1)

    if df_all_bouts is not None:
        for row in df_all_bouts.itertuples():
            try:
                ax[0].axvspan(xmin=ankle_obj.ts[int(row.start)], xmax=ankle_obj.ts[int(row.end)],
                              ymin=all_fill_val[0], ymax=all_fill_val[1], color='dodgerblue', alpha=.35)
            except (KeyError, AttributeError):
                ax[0].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp,
                              ymin=all_fill_val[0], ymax=all_fill_val[1], color='dodgerblue', alpha=.35)

    ax[0].set_title("Raw Ankle {} Data with steps marked "
                    "({} bouts)\n ({} {})".format(ankle_axis, 'all' if not bout_steps_only else 'long',
                                                  "all bouts in blue, " if df_all_bouts is not None else "",
                                                  "long bouts in yellow" if df_long_bouts is not None else ""))

    if df_wrist is not None:
        epoch_len = int((df_wrist.iloc[1]['start_time'] - df_wrist.iloc[0]['start_time']).total_seconds())
        ax[1].set_title(f"Wrist AVM ({epoch_len}-sec epochs)")
        ax[1].plot(df_wrist['start_time'], df_wrist['avm'], color='black')
        ax[1].axhline(y=0, color='grey', linestyle='dotted')
        ax[1].axhline(y=cutpoints[0], color='limegreen', linestyle='dotted')
        ax[1].axhline(y=cutpoints[1], color='orange', linestyle='dotted')

        ax[1].set_ylim(0, )

        if df_long_bouts is not None:
            for row in df_long_bouts.itertuples():
                ax[1].axvspan(xmin=ankle_obj.ts[int(row.start)], xmax=ankle_obj.ts[int(row.end)],
                              ymin=0, ymax=1, color='gold', alpha=.35)

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig
