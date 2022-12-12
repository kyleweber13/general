import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import numpy as np
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


def plot_corrected_peaks(ecg_signal, timestamps, og_peaks, corr_peaks, ds_ratio):

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(timestamps[::ds_ratio], ecg_signal[::ds_ratio], color='black', zorder=0)

    ax.scatter(timestamps[og_peaks], ecg_signal[og_peaks]*1.05,
               s=30, color='red', marker='x', label='Original', zorder=1)

    ax.scatter(timestamps[corr_peaks], ecg_signal[corr_peaks],
               s=30, color='limegreen', marker='v', label='Corrected', zorder=1)

    ax.legend(loc='lower right')
    ax.set_ylabel("Voltage")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.\n%f"))
    ax.grid()

    return fig


def plot_hr_and_peaks(df_peaks, peaks_colname, df_epoch, timestamps, used_signal, snr_signal, snr_roll,
                      ecg_obj, raw_signal=(), ds_ratio=3,
                      show_rr=None, show_jumping=True, thresholds=(5, 20), plot_peaks=True,
                      start_idx=0, end_idx=-1):

    fig, ax = plt.subplots(6, figsize=(12, 8), sharex='col', gridspec_kw={"height_ratios": [1, 1, .5, 1, .5, 1]})

    if show_rr is not None:
        ax[0].plot(df_peaks['timestamp'], df_peaks[show_rr], color='black', label=show_rr)
    if show_jumping:
        ax[0].plot(df_epoch['timestamp'], df_epoch['hr'], color='dodgerblue', label='WindowEpoch')

    ax[0].legend()
    ax[0].set_ylabel("HR")
    ax[0].set_ylim(0, 250)
    ax[0].grid()

    if used_signal is not None:
        ax[1].plot(timestamps[start_idx:end_idx:ds_ratio], used_signal[start_idx:end_idx:ds_ratio],
                   color='black', zorder=1, label='ECG')

        if plot_peaks:
            df_peaks = df_peaks.loc[(df_peaks[peaks_colname] >= start_idx) & (df_peaks[peaks_colname] < end_idx)]

            q1 = df_peaks.loc[df_peaks['quality'] == 1]
            q2 = df_peaks.loc[df_peaks['quality'] == 2]
            q3 = df_peaks.loc[df_peaks['quality'] == 3]
            ax[1].scatter(q1['timestamp'], used_signal[q1[peaks_colname]],
                          color='limegreen', zorder=4, marker='v', label='Q1')
            ax[1].scatter(q2['timestamp'], used_signal[q2[peaks_colname]],
                          color='orange', zorder=3, marker='o', label='Q2')
            ax[1].scatter(q3['timestamp'], used_signal[q3[peaks_colname]],
                          color='red', zorder=2, marker='x', label='Q3')

    if raw_signal is not None:
        ax[1].plot(timestamps[start_idx:end_idx:ds_ratio], raw_signal[start_idx:end_idx:ds_ratio],
                   color='red', zorder=0, label='ECG')

    ax[1].set_ylabel("uV")
    ax[1].legend(loc='lower left')

    # df_epoch_use = df_epoch.loc[df_epoch['n_invalid'] > 0]
    # ax[2].scatter(df_epoch_use['timestamp'], df_epoch_use['n_invalid'], s=5, color='red', label='n_invalid beats')
    ax[2].scatter(df_peaks['timestamp'], df_peaks['r'], color='limegreen', label='QRS_corr')
    ax[2].set_ylabel("r")
    ax[2].legend(loc='upper right')

    if snr_signal is not None:
        ax[3].plot(timestamps[start_idx:end_idx:ds_ratio], snr_signal[start_idx:end_idx:ds_ratio],
                   color='dodgerblue',  label='SNR', zorder=0)
    if snr_roll is not None:
        ax[3].plot(timestamps[start_idx:end_idx:ds_ratio], snr_roll[start_idx:end_idx:ds_ratio],
                   color='red',  label='SNR_roll', zorder=1)

    ax[3].axhline(thresholds[0], color='red', linestyle='dotted')
    ax[3].axhline(thresholds[1], color='limegreen', linestyle='dotted')
    ax[3].legend(loc='upper right')
    ax[3].set_ylabel("dB")

    if ecg_obj is not None:

        try:
            c = {1: 'limegreen', 2: 'orange', 3: 'red'}
            if end_idx == -1:
                snr_end = ecg_obj.df_snr.iloc[-1]['end_idx']
            else:
                snr_end = end_idx
            for row in ecg_obj.df_snr.loc[(ecg_obj.df_snr['start_idx'] >= start_idx) &
                                          (ecg_obj.df_snr['end_idx'] <= snr_end)].itertuples():
                ax[4].plot([row.start_timestamp, row.end_timestamp], [row.quality, row.quality], color=c[row.quality])
        except AttributeError:
            pass

        ax[4].set_yticks([1, 2, 3])
        ax[4].set_yticklabels(['Q1', 'Q2', 'Q3'])

        acc_fs = int(ecg_obj.ecg.signal_headers[ecg_obj.ecg.get_signal_index('Accelerometer x')]['sample_rate'])
        fs = int(ecg_obj.ecg.signal_headers[ecg_obj.ecg.get_signal_index('ECG')]['sample_rate'])

        t_acc = timestamps[::int(fs/acc_fs)]
        r = int(fs/acc_fs)

        ax[5].plot(t_acc[int(start_idx/r):int(end_idx/r):ds_ratio],
                   ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Accelerometer x')][int(start_idx/r):int(end_idx/r):ds_ratio],
                   color='black', label='X')
        ax[5].plot(t_acc[int(start_idx/r):int(end_idx/r):ds_ratio],
                   ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Accelerometer y')][int(start_idx/r):int(end_idx/r):ds_ratio],
                   color='red', label='Y')
        ax[5].plot(t_acc[int(start_idx/r):int(end_idx/r):ds_ratio],
                   ecg_obj.ecg.signals[ecg_obj.ecg.get_signal_index('Accelerometer z')][int(start_idx/r):int(end_idx/r):ds_ratio],
                   color='dodgerblue', label='Z')

    ax[5].set_ylabel("G")
    ax[5].legend(loc='lower right')
    ax[-1].xaxis.set_major_formatter(xfmt)

    plt.tight_layout()
    plt.subplots_adjust(hspace=.05)

    return fig


def compare_hr(data_lists, overlay=False):

    if not overlay:
        fig, ax = plt.subplots(len(data_lists), sharex='col', sharey='col', figsize=(10, 8))

        for i, data in enumerate(data_lists):
            ax[i].plot(data[0]['timestamp'], data[0][data[1]], label=data[2], color='black')

            ax[i].legend(loc='lower right')
            ax[i].set_ylabel("HR")

        ax[-1].xaxis.set_major_formatter(xfmt)

    if overlay:
        fig, ax = plt.subplots(1, sharex='col', sharey='col', figsize=(10, 8))

        for i, data in enumerate(data_lists):
            ax.plot(data[0]['timestamp'], data[0][data[1]], label=data[2])

        ax.legend(loc='lower right')
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_ylabel("HR")

    plt.tight_layout()


def overlay_template(qrs_temp, df, ecg_signal, peaks_colname='peak_idx', timestamps=None, ds_ratio=3):

    # win_size = int(len(qrs_temp)/2)

    pre_idx = np.argmax(qrs_temp)
    post_idx = len(qrs_temp) - pre_idx

    start_i = 0
    end_i = len(ecg_signal)

    if 'r' in df.columns:
        thresh = round(df.loc[~df['valid']]['r'].max(), 2)

        fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))
        ax[0].plot(timestamps[start_i:end_i:ds_ratio], ecg_signal[start_i:end_i:ds_ratio], color='black', lw=2)

        for peak in df[peaks_colname]:
            ecg_window = ecg_signal[peak - pre_idx:peak + post_idx]
            ecg_range = max(ecg_window) - min(ecg_window)

            ax[0].plot(timestamps[peak - pre_idx:peak + post_idx], [i * ecg_range for i in qrs_temp], color='red')

        valid = df.loc[df['valid']]
        invalid = df.loc[~df['valid']]

        ax[1].scatter(valid['timestamp'], valid['r'], color='limegreen', marker='o')
        ax[1].scatter(invalid['timestamp'], invalid['r'], color='red', marker='x')

        ax[1].axhline(y=thresh, color='red', linestyle='dashed')
        ax[1].set_ylabel("r")
        ax[1].grid()

        ax[-1].xaxis.set_major_formatter(xfmt)

    if 'r' not in df.columns:
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.plot(timestamps[start_i:end_i:ds_ratio], ecg_signal[start_i:end_i:ds_ratio], color='black', lw=2)

        for peak in df[peaks_colname]:
            ecg_window = ecg_signal[peak - pre_idx:peak + post_idx]
            ecg_range = max(ecg_window) - min(ecg_window)

            ax.plot(timestamps[peak - pre_idx:peak + post_idx], [i * ecg_range for i in qrs_temp], color='red')

        ax.xaxis.set_major_formatter(xfmt)

    plt.tight_layout()

    return fig


def plot_beat_snr_windowing(signal, timestamps, snr_signal, df_peaks, peaks_colname,
                            window_size=.33, sample_rate=250, ds_ratio=3):

    df_peaks = df_peaks.reset_index(drop=True)

    fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))

    ax[0].plot(timestamps[df_peaks.iloc[0]['idx_corr']:df_peaks.iloc[-1]['idx_corr']:ds_ratio],
               signal[df_peaks.iloc[0]['idx_corr']:df_peaks.iloc[-1]['idx_corr']:ds_ratio], color='black')

    ax[0].scatter(timestamps[list(df_peaks[peaks_colname])], signal[list(df_peaks[peaks_colname])], color='limegreen', marker='v', s=15)

    ax[1].plot(timestamps[df_peaks.iloc[0]['idx_corr']:df_peaks.iloc[-1]['idx_corr']:ds_ratio],
               snr_signal[df_peaks.iloc[0]['idx_corr']:df_peaks.iloc[-1]['idx_corr']:ds_ratio], color='dodgerblue')

    win_samples = int(window_size * sample_rate)

    for row in df_peaks.itertuples():
        ax[0].axvspan(xmin=timestamps[row.idx - win_samples], xmax=timestamps[row.idx + win_samples],
                      ymin=0, ymax=1, color='dimgrey' if row.Index % 2 == 0 else 'lightgrey', alpha=.5)
        ax[1].axvspan(xmin=timestamps[row.idx - win_samples], xmax=timestamps[row.idx + win_samples],
                      ymin=0, ymax=1, color='dimgrey' if row.Index % 2 == 0 else 'lightgrey', alpha=.5)
        m = np.mean(snr_signal[row.idx - win_samples:row.idx + win_samples])
        ax[1].plot([timestamps[row.idx - win_samples], timestamps[row.idx + win_samples]],
                   [m, m], lw=2, color='black')

    ax[1].legend(labels=['snr', 'avg_snr'])
    ax[1].set_ylabel("SNR")
    ax[0].set_ylabel("Voltage")
    ax[0].set_title(f"SNR Averaging in {window_size}-sec beat windows")
