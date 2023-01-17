import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from datetime import timedelta
import numpy as np
import pandas as pd
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


def plot_corrected_peaks(ecg_signal: np.array or tuple or list, timestamps: np.array or tuple or list,
                         original_peaks: np.array or tuple or list, corr_peaks: np.array or tuple or list,
                         ds_ratio: int = 1):
    """Plots ECG timeseries data with two sets of marked peaks.

        arguments:
        -ecg_signal: timeseries ECG signal upon which peak detection was run
        -timestamps: timestamps of ecg_signal
        -original_peaks: array-like of first set of peaks
        -corr_peaks: array-like of second set of peaks
        -ds_ratio: down sample ratio, int (plots every nth datapoint)

        returns:
        -figure

    """

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(timestamps[::ds_ratio], ecg_signal[::ds_ratio], color='black', zorder=0)

    ax.scatter(timestamps[original_peaks], ecg_signal[original_peaks]*1.05,
               s=30, color='red', marker='x', label='Original', zorder=1)

    ax.scatter(timestamps[corr_peaks], ecg_signal[corr_peaks],
               s=30, color='limegreen', marker='v', label='Corrected', zorder=1)

    ax.legend(loc='lower right')
    ax.set_ylabel("Voltage")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.\n%f"))
    ax.grid()

    return fig


def compare_hr(data_dict: dict, data_keys: list or tuple = (), overlay=False):
    """Plots heart rate(s) from specified data either overlaid or on individual subplots.

         arguments:
         -data_dict: dictionary output from run_algorithm()
         -data_keys: keys in data_dict for which data to plot
         -overlay: boolean. If True, all data on one subplot. If False, each HR dataset given its own subplot.

         returns:
         -figure
     """

    if not overlay:
        fig, ax = plt.subplots(len(data_keys), sharex='col', sharey='col', figsize=(10, 8))

        for i, key in enumerate(data_keys):
            try:
                ax[i].plot(data_dict[key]['start_time'], data_dict[key]['hr'], label=key, color='black')
            except KeyError:
                pass

            ax[i].legend(loc='lower right')
            ax[i].set_ylabel("HR")

        ax[-1].xaxis.set_major_formatter(xfmt)

    if overlay:
        fig, ax = plt.subplots(1, sharex='col', sharey='col', figsize=(10, 8))

        for i, key in enumerate(data_keys):
            ax.plot(data_dict[key]['start_time'], data_dict[key]['hr'], label=key)

        ax.legend(loc='lower right')
        ax.set_ylabel("HR")
        ax.xaxis.set_major_formatter(xfmt)

    plt.tight_layout()


def overlay_template(ecg_signal: np.array or list or tuple, timestamps: np.array or list or tuple,
                     qrs_temp: tuple or list or np.array,
                     df: pd.DataFrame = pd.DataFrame(), peaks_colname: str = 'peak_idx',  ds_ratio: int = 3):
    """ Overlays average QRS template on top of ECG signal centered on detected peaks.

        arguments:
        -qrs_temp: QRS template which is overlaid on peaks. It's scaled to each peak's voltage range.
        -ecg_signal: timeseries ECG signal
        -timestamps: timestamps of ecg_signal
        -ds_ratio: down sample ratio, int. Plots every nth ecg_signal datapoint
    """

    # Normalizing QRS to have a range of 1uV for scaling
    qrs_min = min(qrs_temp)
    qrs_max = max(qrs_temp)
    qrs_norm = [(i - qrs_min) / (qrs_max - qrs_min) for i in qrs_temp]
    mean_val = np.mean(qrs_norm)
    qrs_norm = [i - mean_val for i in qrs_norm]

    pre_idx = np.argmax(qrs_temp)
    post_idx = len(qrs_temp) - pre_idx

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(timestamps[::ds_ratio], ecg_signal[::ds_ratio], color='black', lw=3)

    for peak in df[peaks_colname]:
        ecg_window = ecg_signal[peak - pre_idx:peak + post_idx]
        ecg_range = max(ecg_window) - min(ecg_window)
        ecg_mean = np.mean(ecg_window)

        ax.plot(timestamps[peak - pre_idx:peak + post_idx], [i * ecg_range + ecg_mean for i in qrs_norm], color='red')

    ax.xaxis.set_major_formatter(xfmt)

    plt.tight_layout()

    return fig


def plot_results(data_dict: dict, ecg_signal: np.array or list or tuple, ecg_timestamps: np.array or list or tuple,
                 subj: str = "", peak_cols: list or tuple = ('original', 'valid'),
                 hr_cols: list or tuple = ('original', 'valid', 'epoch'),
                 orphanidou_bouts: pd.DataFrame = None, smital_quality: pd.DataFrame = None,
                 smital_raw: np.array or list or tuple = None,
                 df_nw: pd.DataFrame = None, ds_ratio: int = 3):
    """ Plots output of run_algorithm() function using specified data for heart rate and signal quality indices.

        arguments:
        -data_dict: dictionary output from run_algorithm() containing many dataframes
        -ecg_signal: timeseries ECG signal
        -ecg_timestamps: timestamps for ecg_signal
        -subj: str for participant ID
        -peak_cols: list of keys in data_dict that will be used to plot detected peaks on ecg_signal
        -hr_cols: list of keys in data_dict that will be used to plot heart rate
        -orphanidou_bouts: dataframe of Orphanidou signal quality bouts to plot. Recommend only invalid bouts
        -smital_quality: dataframe of Smital quality bouts to plot.
        -smital_raw: timeseries Smital SNR data

        returns:
        -figure

    """

    # dictionaries for plotting specifications for all available data
    color_dict = {"original": 'red', 'quality_screen': 'orange', 'low_quality': 'orange', 'nonwear_screen': 'grey',
                  'nonwear': 'grey', 'processed': 'mediumorchid',
                  'valid': 'dodgerblue', 'invalid': 'dodgerblue', 'epoch': 'black',
                  'orph_valid': 'limegreen', 'orph_invalid': 'limegreen', 'orph_epochs': 'pink'}
    marker_dict = {"original": 'v', 'quality_screen': 'v', 'low_quality': 'x', 'nonwear_screen': 'v',
                   'nonwear': 'x', 'processed': 'v',
                   'valid': 'v', 'invalid': 'x', 'orph_valid': 'v', 'orph_invalid': 'x',
                   'orph_epochs': None, 'epoch': None}
    pairs_dict = {'valid': 'invalid', 'invalid': 'valid',
                  'quality_screen': 'low_quality', 'low_quality': 'quality_screen',
                  'nonwear': 'nonwear_screen', 'nonwear_screen': 'nonwear',
                  'orph_valid': 'orph_invalid', 'orph_invalid': 'orph_valid'}
    plotted = []

    # subplot formatting based on what data is specified
    n_subplots = 2
    heights = [1, .66]

    if smital_quality is not None:
        n_subplots += 1
        heights.append(.33)
    if orphanidou_bouts is not None:
        n_subplots += 1
        heights.append(.33 if smital_raw is None else .5)
    if smital_raw is not None and smital_quality is None:
        n_subplots += 1
        heights.append(.5)

    fig, ax = plt.subplots(n_subplots, sharex='col', figsize=(12, 8), gridspec_kw={"height_ratios": heights})

    plt.suptitle(subj)

    ax[0].plot(ecg_timestamps[:len(ecg_signal):ds_ratio], ecg_signal[::ds_ratio], color='black')

    if df_nw is not None:
        for row in df_nw.itertuples():
            if row.Index == 0:
                ax[0].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1,
                              color='grey', alpha=.2, label='nonwear')
            if row.Index > 0:
                ax[0].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1,
                              color='grey', alpha=.2)

    offset = 0
    for col in peak_cols:

        has_pair = col in pairs_dict
        try:
            pair_plotted = pairs_dict[col] in plotted or col in plotted
        except KeyError:
            pair_plotted = False

        if has_pair and not pair_plotted:
            offset += 200

        if not has_pair:
            offset += 200

        plotted.append(col)

        ax[0].scatter(data_dict[col]['start_time'], ecg_signal[data_dict[col]['idx']] + offset,
                      color=color_dict[col], marker=marker_dict[col],
                      label=f"{col.capitalize()} (n={data_dict[col].shape[0]})")

    ax[0].legend(loc='lower right')

    for col in hr_cols:
        if col not in ['epoch', 'orphanidou']:
            ax[1].plot(data_dict[col]['start_time'], data_dict[col]['hr'], color=color_dict[col], label=col.capitalize())

        if col in ['epoch', 'orphanidou']:
            epoch_len = int((data_dict[col].iloc[1]['start_time'] -
                             data_dict[col].iloc[0]['start_time']).total_seconds())
            df_use = data_dict[col].dropna()
            ax[1].errorbar(df_use['start_time'] + timedelta(seconds=epoch_len/2),
                           df_use['hr'], label=f'{col.capitalize()} ({epoch_len}s)',
                           marker=None, ecolor=color_dict[col], fmt='none',
                           capsize=4, xerr=timedelta(seconds=epoch_len/2)
                           )

    ax[1].grid()
    ax[1].legend(loc='lower right')
    ax[1].set_ylabel("HR")

    if orphanidou_bouts is not None:
        ax[2].set_ylabel("Orphanidou\nInvalid")

        for row in orphanidou_bouts.itertuples():
            ax[2].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, alpha=.1,
                          color='red' if not row.valid_period else 'limegreen')
        ax[2].set_yticks([])

    if smital_quality is not None:
        use_ax = ax[2] if orphanidou_bouts is None else ax[3]

        c = {"3": 'red', '2': 'dodgerblue', '1': 'limegreen', '0': 'grey'}
        for row in smital_quality.itertuples():
            use_ax.axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, alpha=.1,
                           color=c[str(row.quality_use)])

        use_ax.set_ylabel("Smital\nQuality")

        if smital_raw is None:
            use_ax.set_yticks([])

    if smital_raw is not None:
        if smital_quality is not None:
            pass
        if smital_quality is None:
            use_ax = ax[-1]

        use_ax.plot(ecg_timestamps[:len(smital_raw):25], smital_raw[::25], color='black')
        use_ax.grid()

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig
