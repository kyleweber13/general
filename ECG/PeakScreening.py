import pandas as pd
import numpy as np
import scipy.stats

""" ===== CHECKED ===== """


def correct_peak_locations(df_peaks: pd.DataFrame,
                           peaks_colname: str,
                           ecg_signal: np.array or list or tuple,
                           timestamps: np.array or list or tuple,
                           sample_rate: float or int,
                           window_size: float or int = 0.15,
                           use_abs_peaks: bool = False,
                           use_correlation: bool = True,
                           quiet: bool = True):
    """Adjusts peak indexes to correspond to highest amplitude value in window surrounding each beat.

        arguments:
        -df_peaks: dataframe containing peaks, timestamps, etc.
        -peaks_colname: column name in df_peaks corresponding to peak indexes
        -ecg_signal: timeseries ECG signal
        -timestamps: of ecg_signal
        -sample_rate: sample rate of ecg_signal, Hz
        -window_size: number of seconds included in window on either side of each peak that is checked for
                      more appropriate peak location
        -use_abs_peaks: if True, uses absolute values. If False, only looks at positive values for possible peaks.
        -use_correlation: if True, only changes peak location if new peak is more highly-correlated with the previous
                          peak than the original peak
        -quiet: whether to print processing progress to console

        returns:
        -copy of df_peaks with new 'idx_corr' column (correct indexes for peaks)
    """

    if not quiet:
        print(f"\nAdjusting peak locations using a window size of +- {window_size} seconds and "
              f"{'positive' if not use_abs_peaks else 'largest amplitude'} peaks...")

    df_out = df_peaks.copy()

    new_peaks = []
    win_size = int(window_size * sample_rate)

    peaks = list(df_peaks[peaks_colname])

    # Loops through peaks
    for peak_idx, peak in enumerate(peaks):

        # segment of data: peak +- window_size duration
        window = ecg_signal[int(peak - win_size):int(peak + win_size)]

        # index of largest value/absolute value in the window
        if use_abs_peaks:
            p = np.argmax(abs(window))
        if not use_abs_peaks:
            p = np.argmax(window)

        new_peak = p + peak - win_size

        # correlates new potential peak and current peak with previous peak
        # output which peak is more highly correlated with previous peak
        if use_correlation:
            new_window = ecg_signal[int(new_peak - win_size):int(new_peak + win_size)]
            if peak_idx >= 1:
                prev_window = ecg_signal[int(peaks[peak_idx - 1] - win_size):int(peaks[peak_idx - 1] + win_size)]

                try:
                    r_new = scipy.stats.pearsonr(new_window, prev_window)[0]
                    r_old = scipy.stats.pearsonr(window, prev_window)[0]

                    # keeps peak if no improvement
                    if r_old >= r_new:
                        new_peak = peak
                    # uses new peak if improved correlation
                    if r_new > r_old:
                        new_peak = p + peak - win_size
                except ValueError:
                    pass

        # converts window's index to whole collection's index
        new_peaks.append(new_peak)

    # difference in seconds between peak and corrected peak
    diff = [(i - j) / sample_rate for i, j in zip(new_peaks, peaks)]

    df_out['idx_corr'] = new_peaks
    df_out['idx_diff'] = diff
    df_out['start_time'] = timestamps[new_peaks]
    df_out['corrected'] = ~(df_out['idx'] == df_out['idx_corr'])

    # drops duplicates in case two beats get corrected to the same location (unlikely)
    df_out.drop_duplicates(subset='idx_corr', keep='first', inplace=True)
    df_out = df_out.reset_index(drop=True)

    return df_out


def remove_peaks_during_bouts(df_peaks: pd.DataFrame(),
                              idx_colname: str = 'idx',
                              stage_name: str = "",
                              dfs_events_to_remove: list or tuple = (),
                              quiet: bool = True):
    """Removes peaks in df_peaks that occur in any df in df_events_to_remove.

        arguments:
        -df_peaks: pd.DataFrame containing ECG peak information.
        -idx_colname: column name in df_peaks used to match events in df(s) contained in dfs_events_to_remove
        -stage_name: optional string used as to identify which processing stage peaks were removed as
        -dfs_events_to_remove: list/tuple of pd.DataFrames. Dfs are event dataframes (start and end times) which
                               requires columns start_idx and end_idx, which correspond to the same indexes of the ECG
                               data used for df_peaks. All peaks in each event in every dataframe are removed
        -quiet: boolean to print to console

        returns:
        -df_peaks_out: rows in df_peaks that were not contained in any events
        -df_peaks_rem: rows in df_peaks that were contained in any event
    """

    df_peaks_out = df_peaks.copy()

    df_peaks_rem = pd.DataFrame(columns=df_peaks_out.columns)

    if type(dfs_events_to_remove) is not list and type(dfs_events_to_remove) is not tuple:
        dfs_events_to_remove = [dfs_events_to_remove]

    if not quiet:
        print(f"\nRemoving peaks using events from {len(dfs_events_to_remove)} "
              f"dataframe{'s' if len(dfs_events_to_remove) != 1 else ''} ({stage_name})...")

    n = df_peaks.shape[0]  # number of input peaks

    # loops through dataframe(s)
    for df in dfs_events_to_remove:
        # loops through rows in dataframe
        for row in df.itertuples():
            df_peaks_out = df_peaks_out.loc[(df_peaks_out[idx_colname] < row.start_idx) |
                                            (df_peaks_out[idx_colname] > row.end_idx)]

            df_rem = df_peaks.loc[(df_peaks[idx_colname] >= row.start_idx) & (df_peaks[idx_colname] <= row.end_idx)]
            df_peaks_rem = pd.concat([df_peaks_rem, df_rem])

    df_peaks_out.reset_index(drop=True, inplace=True)

    df_peaks_rem.drop_duplicates(inplace=True)
    df_peaks_rem.reset_index(drop=True, inplace=True)
    df_peaks_rem[idx_colname] = df_peaks_rem[idx_colname].astype(int)
    df_peaks_rem['stage'] = [stage_name] * df_peaks_rem.shape[0]

    n2 = df_peaks_out.shape[0]  # output peaks

    if not quiet:
        print(f"-Removed {n - n2}/{n} peaks ({(n-n2)/n*100:.1f}%)")

    return df_peaks_out, df_peaks_rem


""" ===== NOT CHECKED ===== """

