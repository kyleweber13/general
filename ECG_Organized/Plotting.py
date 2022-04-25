import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S.%f")


def plot_data(df, fs, ecg_data, snr_data, incl_context=True, incl_arrs=None, t=None, ds_ratio=1,
              gait_mask=None, intensity_mask=None, sleep_mask=None, nw_mask=None, q1_thresh=25):
    """Plots a bunch of data.

       arguments:
       -df: arrhythmia df
       -fs: sample rate of ECG/SNR signal
       -snr_data: SNR array
       -incl_context: boolean --> if True, adds subplots for gait/sleep/activity
       -incl_arrs: list of which arrhythmias to include
       -t: can pass in timestamps for x-axis. If None, will show time as seconds
       -ds_ratio: downsample ratio for ECG and SNR signals
       -gait/intensity/nw/sleep_mask: mask data from create_df_mask()
       -q1_thresh: threshold for highest quality data. Gets drawn on graph. That's it.

       returns figure
    """

    if t is None:
        t = np.arange(len(ecg_data)) / fs

    arrs = {'COUP': 'pink', 'Tachy': 'limegreen', 'GEM': "blue", 'SALV': 'dodgerblue', 'PAC/SVE': "dodgerblue",
            'VT': 'gold', 'ST-': 'grey', "ST+": 'grey', 'AV1': 'lightgrey', 'AV2/I': 'lightgrey', 'IVR': 'purple',
            'AV2II': "black", 'AV2/II': 'black', 'SVT': 'orange', 'AF': 'brown', 'Arrest': 'red', 'Block': 'darkorange'}

    if not incl_context:
        fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))
    if incl_context:
        fig, ax = plt.subplots(7, sharex='col', figsize=(12, 8),
                               gridspec_kw={"height_ratios": [1, 1, 1, .33, .33, .33, .67]})

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    ax[0].plot(t[::ds_ratio], ecg_data[::ds_ratio], color='black', zorder=1, label='ECG')

    if incl_arrs is None:
        incl_arrs = df['Type'].unique()

    for arrhythmia in incl_arrs:
        df2 = df.loc[df['Type'] == arrhythmia].reset_index(drop=True)

        if df2.shape[0] > 0:

            for row in df2.itertuples():
                if row.Index == 0:
                    ax[0].plot([t[row.start_idx], t[row.end_idx]], [0, 0], color=arrs[row.Type],
                               lw=6, zorder=0, label=arrhythmia, alpha=.75)
                if row.Index > 0:
                    ax[0].plot([t[row.start_idx], t[row.end_idx]], [0, 0], color=arrs[row.Type],
                               lw=6, zorder=0, alpha=.75)

                ax[0].axvline(t[row.start_idx], color=arrs[row.Type], linestyle='dashed', zorder=0)
                ax[0].axvline(t[row.end_idx], color=arrs[row.Type], linestyle='dashed', zorder=0)

                # box with min, max, and median values
                try:
                    ax[2].plot([t[row.start_idx], t[row.end_idx]], [row.p0, row.p0], color='red', zorder=2, lw=3)
                    ax[2].plot([t[row.start_idx], t[row.end_idx]], [row.p50, row.p50], color='dodgerblue', zorder=2, lw=3)
                    ax[2].plot([t[row.start_idx], t[row.end_idx]], [row.p100, row.p100], color='limegreen', zorder=2, lw=3)

                    ax[2].plot([t[row.start_idx], t[row.start_idx]], [row.p0, row.p100], color='black', zorder=2)
                    ax[2].plot([t[row.end_idx], t[row.end_idx]], [row.p0, row.p100], color='black', zorder=2)
                except AttributeError:
                    pass

    ax[0].legend(loc='lower right')

    ax[1].plot(t[:min([len(snr_data), len(t)]):ds_ratio], snr_data[::ds_ratio], color='dodgerblue', zorder=1)
    ax[1].set_ylabel("dB")
    ax[1].axhline(q1_thresh, color='limegreen', linestyle='dashed', zorder=0, label=f'Q1 ({q1_thresh}dB)')
    ax[1].axhline(5, color='red', linestyle='dashed', zorder=0, label='Q2 (5dB)')
    ax[1].legend(loc='lower right')

    ax[2].set_ylabel("dB")
    ax[2].axhline(5, color='red', linestyle='dashed', zorder=0)
    ax[2].axhline(q1_thresh, color='limegreen', linestyle='dashed', zorder=0)

    if incl_context:
        ax[3].plot(t[::int(fs)], gait_mask, label='Gait', color='black')
        ax[3].legend(loc='lower right')

        ax[4].plot(t[::int(fs)], sleep_mask, label='Sleep', color='black')
        ax[4].legend(loc='lower right')

        ax[5].plot(t[::int(fs)], nw_mask, label='NW', color='black')
        ax[5].legend(loc='lower right')

        ax[6].plot(t[::int(fs)], intensity_mask, color='black')
        ax[6].set_yticks([0, 1, 2])
        ax[6].set_yticklabels(['sed', 'light', 'mod'])
        ax[6].grid()

    if not type(t[0]) in [int, float, np.float64]:
        ax[-1].xaxis.set_major_formatter(xfmt)

    plt.tight_layout()

    return fig


def flag_all_df_cn_criteria(df, sleep_thresh=50, max_active=100, max_nw=0,
                            snr_thresh=20, snr_var='min_snr',
                            use_arrs=("Tachy", "Brady", "Arrest", "AF", "VT", "SVT", "ST+")):

    min_durs = {"VT": 30, "SVT": 30, "AF": 30, "Arrest": 10, 'Pause': 10, 'Brady': 60}

    df = df.copy()

    df['incl_arr'] = df['Type'].isin(use_arrs)

    if snr_var not in df.columns:
        print(f"'{snr_var}' not a valid column in dataframe. Options:")
        for col in df.columns:
            print(f"-{col}")

        return df

    len_crit = []

    df['sleep_crit'] = df['sleep%'] <= sleep_thresh
    df['active_crit'] = df['active%'] <= max_active
    df['nw_crit'] = df['nw%'] <= max_nw
    df['snr_crit'] = df[snr_var] >= snr_thresh

    for row in df.itertuples():
        # if event has minimum duration ------
        if row.Type in min_durs.keys():
            if row.duration >= min_durs[row.Type]:
                len_crit.append(True)
            if row.duration < min_durs[row.Type]:
                len_crit.append(False)

        if row.Type not in min_durs.keys():
            len_crit.append(True)

    df['length_crit'] = len_crit

    df['final_screen'] = [row.incl_arr and row.length_crit and row.sleep_crit and row.active_crit \
                          and row.nw_crit and row.snr_crit for row in df.itertuples()]

    df_summary = pd.DataFrame(columns=["type", 'incl_arr', 'n_events', 'avg_snr', 'min_snr', 'max_snr',
                                       'sleep_crit%', 'active_crit%', 'nw_crit%', 'length_crit%', 'snr_crit%',
                                       'final_screen%'])

    df_summary['type'] = df['Type'].unique()
    df_summary['incl_arr'] = [i in use_arrs for i in df_summary['type']]
    df_summary['n_events'] = [df.loc[df['Type'] == event_type].shape[0] for event_type in df_summary['type']]
    df_summary['min_snr'] = [df.loc[df['Type'] == event_type]['p0'].min() for event_type in df_summary['type']]
    df_summary['max_snr'] = [df.loc[df['Type'] == event_type]['p100'].max() for event_type in df_summary['type']]

    avg = []
    n_events = []
    sleep = []
    act = []
    nw = []
    snr = []
    length = []
    final = []
    for event_type in df_summary['type']:
        d = df.loc[df['Type'] == event_type]
        s = d['duration'] * d['avg_snr']
        a = s.sum() / d['duration'].sum()
        avg.append(a)
        n_events.append(d.shape[0])
        sleep.append(d.loc[d['sleep_crit']].shape[0] / d.shape[0] * 100)
        act.append(d.loc[d['active_crit']].shape[0] / d.shape[0] * 100)
        nw.append(d.loc[d['nw_crit']].shape[0] / d.shape[0] * 100)
        snr.append(d.loc[d['snr_crit']].shape[0] / d.shape[0] * 100)
        length.append(d.loc[d['length_crit']].shape[0] / d.shape[0] * 100)
        final.append(d.loc[d['final_screen']].shape[0] / d.shape[0] * 100)

    df_summary['avg_snr'] = avg
    df_summary['n_events'] = n_events
    df_summary['sleep_crit%'] = sleep
    df_summary['active_crit%'] = act
    df_summary['nw_crit%'] = nw
    df_summary['snr_crit%'] = snr
    df_summary['length_crit%'] = length
    df_summary['final_screen%'] = final

    return df, df_summary


def plot_smital_bouts(ecg_signal, sample_rate, snr_signal, df_snr_bouts, ds_ratio=2):

    fig, ax = plt.subplots(3, sharex='col', figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1, .33]})
    ax[0].plot(np.arange(0, int(len(ecg_signal)))[::ds_ratio]/sample_rate, ecg_signal[::ds_ratio], color='black')

    c = {1: 'limegreen', 2: 'orange', 3: 'red'}
    for row in df_snr_bouts.itertuples():
        ax[2].plot([row.start_idx / sample_rate, row.end_idx / sample_rate], [row.quality, row.quality], color=c[row.quality], lw=3)

    ax[2].set_yticks([1, 2, 3])
    ax[2].grid()

    ax[1].plot(np.arange(0, int(len(snr_signal)))[::ds_ratio]/sample_rate, snr_signal[::ds_ratio],
               color='dodgerblue', label='SNR', zorder=0)
    ax[1].axhline(y=5.1, color='orange', label='Q2 thresh.')
    ax[1].axhline(y=4.9, color='red')
    ax[1].axhline(y=24.9, color='orange')
    ax[1].axhline(y=25.1, color='limegreen', label='Q1 thresh.')
    ax[1].set_ylabel("dB")
    ax[1].legend(loc='lower right')
    plt.tight_layout()


def plot_for_nw_detection(ecg_obj, df_nw=None):

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 9))

    ax[0].set_title("Raw Voltage")
    ax[0].plot(ecg_obj.ts[::2], ecg_obj.signal[::2], color='red')

    ax[1].set_title("Temperature")
    ax[1].plot(ecg_obj.ts[::int(ecg_obj.fs / ecg_obj.ecg.signal_headers[ecg_obj.ecg.get_signal_index('Temperature')]['sample_rate'])],
               ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Temperature')], color='orange')
    val = ax[1].get_ylim()
    val = (val[1] - val[0]) / 2 + val[0]

    if df_nw is not None:
        for row in df_nw.itertuples():
            ax[1].plot([row.start_timestamp, row.end_timestamp], [val, val], color='black', lw=3)
    ax[1].grid()

    ax[2].set_title("Raw Acceleration")
    ax[2].plot(ecg_obj.ts[::int(2 * ecg_obj.fs / ecg_obj.ecg.signal_headers[ecg_obj.ecg.get_signal_index('Accelerometer x')]['sample_rate'])],
               ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Accelerometer x')][::2], color='black')
    ax[2].plot(ecg_obj.ts[::int(2 * ecg_obj.fs / ecg_obj.ecg.signal_headers[ecg_obj.ecg.get_signal_index('Accelerometer y')]['sample_rate'])],
               ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Accelerometer y')][::2], color='red')
    ax[2].plot(ecg_obj.ts[::int(2 * ecg_obj.fs / ecg_obj.ecg.signal_headers[ecg_obj.ecg.get_signal_index('Accelerometer z')]['sample_rate'])],
               ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Accelerometer z')][::2], color='dodgerblue')
    ax[2].grid()

    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))

    return fig
