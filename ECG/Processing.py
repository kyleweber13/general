import pandas as pd
import nwecg.ecg_quality as ecg_quality
from datetime import timedelta as td
import numpy as np
import bottleneck
from peakutils import indexes
import matplotlib.pyplot as plt
from ECG.ImportFiles import *
from scipy.stats import pearsonr


def create_df_peaks(timestamps, peaks):

    df_out = pd.DataFrame({'start_time': timestamps[peaks], 'idx': peaks, 'valid': [True] * len(peaks)})

    return df_out


def create_snr_bouts(snr_signal, sample_rate, start_stamp, thresholds=(5, 20), shortest_time=30, quiet=True):
    """Creates SNR bouts based on given SNR thresholds and duration requirements. Formats into DF.

        arguments:
        -snr_signal: output from import_snr_pickle() (time series data)
        -sample_rate: Hz, of ECG/SNR signals
        -start_stamp: timestamp of start of ECG collection
        -thresholds: list/tuple of length 2 for SNR values that differentiate Q3/Q2 and Q2/Q1
            -Thresholds in ascending order
        -shortest_time: shortest duration of a 'bout'
        -quiet: whether to print processing progress to console

        returns:
        -df containing SNR bouts
    """

    if not quiet:
        print(f"\n-Bouting SNR with thresholds of {thresholds} dB and minimum event durations of {shortest_time}...")

    snr_bouts = ecg_quality._annotate_SNR(rolling_snr=snr_signal, signal_len=len(snr_signal), thresholds=thresholds,
                                          sample_rate=sample_rate, shortest_time=shortest_time,
                                          Nonwear_mask=np.array([False]*len(snr_signal)),
                                          apply_rules=True)

    df_snr = pd.DataFrame(snr_bouts.get_annotations())
    df_snr['duration'] = [(row.end_idx - row.start_idx) / sample_rate for row in df_snr.itertuples()]
    df_snr['start_timestamp'] = [start_stamp + td(seconds=row.start_idx / sample_rate) for row in df_snr.itertuples()]
    df_snr['end_timestamp'] = [start_stamp + td(seconds=row.end_idx / sample_rate) for row in df_snr.itertuples()]

    df_snr['quality'] = [row.quality.value for row in df_snr.itertuples()]
    df_snr['quality'] = df_snr['quality'].replace({"Q1": 1, 'Q2': 2, 'Q3': 3})

    df_snr['avg_snr'] = [np.mean(snr_signal[row.start_idx:row.end_idx]) for row in df_snr.itertuples()]

    return df_snr


def create_df_mask(sample_rate, max_i, start_stamp,
                   gait_folder, gait_file,
                   sleep_folder, sleep_file,
                   activity_folder, activity_file,
                   nw_folder, nw_file, full_id=None):
    """Calls functions to import gait, sleep, and activity data. Combines masks into single DF.

       arguments:
       -sample_rate: ECG sample rate, Hz
       -max_i: length of ECG signal
       -start_stamp: ECG start timestamp
       -the rest are obvious (see individual import functions if not)

       returns individual DFs and unified gait mask DF
    """

    df_gait, gait_mask = import_gait_bouts(gait_folder=gait_folder,
                                           gait_file=gait_file,
                                           sample_rate=sample_rate, max_i=max_i, start_stamp=start_stamp)
    df_sleep, sleep_mask = import_sleep_bouts(sleep_folder=sleep_folder,
                                              sleep_file=sleep_file,
                                              sample_rate=sample_rate, max_i=max_i, start_stamp=start_stamp)
    df_act, act_mask = import_activity_counts(activity_folder=activity_folder,
                                              activity_file=activity_file,
                                              sample_rate=sample_rate, max_i=max_i, start_stamp=start_stamp)

    max_len = int(np.floor(max_i / sample_rate))
    df_nw, nw_mask = import_nonwear_data(file=nw_folder + nw_file, start_stamp=start_stamp,
                                         sample_rate=sample_rate, max_i=max_i, full_id=full_id)

    df_mask = pd.DataFrame({"seconds": np.arange(max_len),
                            'timestamp': pd.date_range(start=start_stamp, periods=len(gait_mask), freq='1S'),
                            'gait': gait_mask if gait_mask is not None else np.zeros(max_len),
                            'sleep': sleep_mask if sleep_mask is not None else np.zeros(max_len),
                            'activity': act_mask if act_mask is not None else np.zeros(max_len),
                            'nw': nw_mask if nw_mask is not None else np.zeros(max_len)})

    return df_mask, df_gait, df_sleep, df_act, df_nw


def calculate_beat_snr(snr_data, sample_rate, df_peaks, peak_colname='idx_corr',
                       thresholds=(5, 20), window_width=.25, quiet=True):
    """Function that removes peaks whose surrounding window_width-window averages below given SNR threshold.
        Classifies as Q1, Q2, or Q3 data based on thresholds.

        arguments:
        -snr_data: time series SNR signal
        -sample_rate: Hz, of snr_data
        -df_peaks: df containing peak indexes
        -thresholds: snr thresholds
        -window_width: window size around each peak to test
        -drop_invalid: boolean, will remove rows of invalid beats if True
        -quiet: whether to print processing progress to console

        returns:
        -snr values for each beat
    """

    if not quiet:
        print("\nCalculating mean SNR values for each beat +- {} second{}...".format(window_width, "s" if window_width != 1 else ""))

    df_peaks = df_peaks.copy()

    win_size = int(sample_rate*window_width)

    snr_vals = [np.mean(snr_data[i-win_size:i+win_size]) for i in df_peaks[peak_colname]]

    df_peaks['snr'] = snr_vals

    q = []
    thresholds = sorted(thresholds)

    for val in snr_vals:

        if val < thresholds[0]:
            q.append(3)
        if thresholds[0] <= val < thresholds[1]:
            q.append(2)
        if val >= thresholds[1]:
            q.append(1)
        if np.isnan(val):
            q.append(None)

    df_peaks['quality'] = q

    return df_peaks


def get_zncc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Get zero-normalized cross-correlation (ZNCC) of the given vectors.
    Adapted from: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    To improve performance, the formula given by Wikipedia is rearranged as follows::
        1 / (n * std(x) * std(y)) * (sum[x(i) * y(i)] - n * mean(x) * mean(y))
    """

    # TODO: also return whether peak meets statistical significance criteria.
    # TODO: return array of lags.
    # Ensure x is the longer signal.
    x, y = sorted((x, y), key=len, reverse=True)

    # Calculate rolling mean and standard deviation. Discard first few NaN values.
    x_mean = bottleneck.move_mean(x, len(y))[len(y) - 1:]
    x_std = bottleneck.move_std(x, len(y))[len(y) - 1:]

    # Avoid division by zero and numerical errors caused by zero or very small standard deviations.
    x_std_reciprocal = np.reciprocal(x_std, where=np.abs(x_std) > 0.0000001)

    y_mean = np.mean(y)
    y_std_reciprocal = 1 / np.std(y)

    n = len(y)

    # Calculate correlation and normalize.
    correlation = np.correlate(x, y, mode="valid")
    return (1 / n) * x_std_reciprocal * y_std_reciprocal * (correlation - n * x_mean * y_mean)


def filter_rr(df_peaks, sample_rate, timestamps, peaks_colname='idx', threshold=30,
              max_iters=20, plot_hr=False, quiet=True):
    """Applies 'RR-interval filtering' from Wilkund et al. (2008) to remove potentially artificial R peaks.

        '...the series of all RR intervals related to normal-to-normal (N-N) interbeat intervals was filtered using
         a recursive procedure, similar to the approach suggested by Wictherele et al. After each step, N-N intervals
         were removed if they differed more than 30% from the mean of the preceding and following value. The filtering
         was repeated on the remaining series of N-N intervals until no values were removed. The maximal number of
         iterations was set to twenty.'

        arguments:
        -peaks: peak indexes
        -sample_rate: of ecg signal, correspdonding to peaks
        -threshold: percentage threshold of maximal difference (e.g., 30% = 30)
        -max_iters: maximum number of iterations to run
        -timestamps: corresponding to peak indexes
        -plot_hr: boolean, plots HR after each iteration
        -quiet: whether to print processing progress to console

        returns:
        -dataframe of remaining beats + timestamps
    """

    if not quiet:
        print(f"\nRunning Wilkund et al. (2008) RR-filtering technique using a "
              f"threshold of {threshold}% and <={max_iters} iterations...")

    peaks = list(df_peaks[peaks_colname])

    n_input = len(peaks)

    if plot_hr:
        fig, ax = plt.subplots(int(np.ceil(max_iters/5)+1), figsize=(12, 8), sharex='col', sharey='col')

    for i in range(max_iters):

        if not quiet:
            print(f"-Iteration {i+1}")

        # recalculates RR intervals with remaining peaks (set at end of each iteration after 1st)
        rr = [(j - i) / sample_rate for i, j in zip(peaks[:], peaks[1:])]

        df_rr = pd.DataFrame({'timestamp': timestamps[peaks[:-1]], 'idx': peaks[:-1], 'rr': rr})

        n1 = len(rr)  # number of peaks at start of iteration

        diffs = [None]
        for prev_beat, curr_beat, next_beat in zip(rr[:], rr[1:], rr[2:]):

            m = np.mean([prev_beat, next_beat])  # mean of previous and next RR intervals

            # uses average of prev/next beats if average RR is < 5 seconds
            # accounts for removed beats so current beat is not removed when it shouldn't
            if m <= 5:
                diff = 100 * (np.abs(m - curr_beat)) / m  # %diff from surrounding peaks' RR intervals

            if m > 5:
                diff = 100 * (np.abs(prev_beat - curr_beat) / prev_beat)

            diffs.append(diff)

        diffs.append(None)
        df_rr['idx'] = peaks[:-1]
        df_rr['diff'] = diffs
        df_rr['diff%'] = diffs
        df_rr['hr'] = 60 / df_rr['rr']  # HR from RR interval

        df_rr = df_rr.loc[df_rr['diff'] <= threshold].reset_index(drop=True)
        n2 = df_rr.shape[0]

        if not quiet:
            print(f"     -Went from {n1} to {n2} beats ({n1-n2} removed)")

        if n1-n2 == 0:
            if not quiet:
                print(f"     -LOOP BROKEN ON ITERATION {i+1} --> no peaks removed")
                print(f"          -Total of {n_input - n2} peaks removed ({round((n_input - n2) / n_input, 2)})%")
            break

        if plot_hr:
            subplot_i = int(np.floor(i / int(np.ceil(max_iters/5))))
            df_plot = df_rr.loc[(df_rr['rr'] >= .25) & (df_rr['rr'] <= 2)]
            ax[subplot_i].plot(df_plot['timestamp'], df_plot['hr'], lw=1, label=f"Iter{i+1}")

        peaks = list(df_rr['idx'])

    if plot_hr:
        for i in range(int(np.ceil(max_iters / 5) + 1)):
            ax[i].legend(loc='lower right')
            ax[i].set_ylabel("HR")
            ax[i].grid()

        plt.tight_layout()

    df_out = df_peaks.loc[df_peaks[peaks_colname].isin(df_rr['idx'])].iloc[:-1]
    df_out['rr'] = [(j - i) / sample_rate for i, j in zip(peaks[:], peaks[1:])]

    return df_out.reset_index(drop=True)


def find_snr_bouts(df_snr, min_snr=18, min_duration=60, n_bouts=5, min_total_minutes=20, quiet=True):
    """Finds bouts in df_snr that meet minimum SNR and minimum duration criteria.
    Returns n_bouts with highest SNR values.

    arguments:
    -df_snr: SNR dataframe in ecg object
    -min_snr: minimum average SNR for a period to be included
    -min_duration: minimum duration in seconds for period to be included
    -n_bouts: minimum number of bouts to include
    -min_total_minutes: minimum total duration of bouts to include in minutes
    -quiet: whether to print processing progress to console

    Set n_bouts=-1 to return all bouts
    """

    if not quiet:
        print(f"\nLocation best (at least) {n_bouts} SNR bouts with a minimum SNR of {min_snr} dB, bout "
              f"duration of {min_duration} seconds, and totalling > {min_total_minutes} minutes of data...")

    df_out = df_snr.loc[(df_snr['avg_snr'] >= min_snr) & (df_snr['duration'] >= min_duration)].reset_index(drop=True)

    df_out = df_out.sort_values('avg_snr', ascending=False).reset_index(drop=True)

    df_out['total_minutes'] = df_out['duration'].cumsum() / 60

    if df_out.shape[0] <= n_bouts:
        df_out2 = df_snr.loc[df_snr['duration'] > min_duration].sort_values("avg_snr", ascending=False).iloc[:n_bouts]
        print(f"-Criteria not satisfied. Returning best {n_bouts} bouts of data.")

    if df_out.shape[0] > n_bouts:
        if df_out.iloc[n_bouts]['total_minutes'] >= min_total_minutes:
            df_out2 = df_out.iloc[:n_bouts]

        try:
            if df_out.iloc[n_bouts]['total_minutes'] < min_total_minutes:
                df_out2 = df_out.iloc[:df_out.loc[df_out['total_minutes'] >= min_total_minutes].index[0]]
        except IndexError:
            pass

    return df_out2


def remove_low_quality_signal(ecg_signal, df_snr=None,df_nw=None, min_duration=15, min_quality=2, min_snr=5, quiet=True):

    if type(ecg_signal) is list:
        signal = np.array(ecg_signal.copy())

    if type(ecg_signal) is np.ndarray:
        signal = np.asarray(np.copy(ecg_signal))

    if df_snr is not None:
        if not quiet:
            print(f"-Removing ECG data during regions with quality lower than "
                  f"Q{min_quality} and/or SNR < {min_snr} and/or bouts < {min_duration} seconds long...")
        df_snr = df_snr.copy()
        df_snr = df_snr.loc[df_snr['end_idx'] < len(signal)].reset_index(drop=True)

        for row in df_snr.itertuples():

            if row.quality > min_quality or row.duration < min_duration or row.avg_snr < min_snr:
                if row.start_idx != 0:
                    signal[row.start_idx-1:row.end_idx+1] = 0
                else:
                    signal[row.start_idx:row.end_idx+1] = 0

    if df_nw is not None:
        if not quiet:
            print("-Flagging non-wear periods...")
        df_nw = df_nw.copy()
        df_nw = df_nw.loc[df_nw['end_idx'] < len(signal)].reset_index(drop=True)

        for row in df_nw.itertuples():
            if row.start_idx != 0:
                signal[row.start_idx - 1:row.end_idx + 1] = 0
            else:
                signal[row.start_idx:row.end_idx + 1] = 0

    return signal


def remove_peaks_during_bouts(df_peaks, stage_name="", dfs_events_to_remove=(), quiet=True):
    """Removes peaks in df_peaks that occur in any df in df_events_to_remove"""

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
            df_peaks_out = df_peaks_out.loc[(df_peaks_out['idx'] < row.start_idx) | (df_peaks_out['idx'] > row.end_idx)]

            df_rem = df_peaks.loc[(df_peaks['idx'] >= row.start_idx) & (df_peaks['idx'] <= row.end_idx)]
            df_peaks_rem = pd.concat([df_peaks_rem, df_rem])

    df_peaks_out.reset_index(drop=True, inplace=True)

    df_peaks_rem.drop_duplicates(inplace=True)
    df_peaks_rem.reset_index(drop=True, inplace=True)
    df_peaks_rem['idx'] = df_peaks_rem['idx'].astype(int)
    df_peaks_rem['stage'] = [stage_name] * df_peaks_rem.shape[0]

    n2 = df_peaks_out.shape[0]  # output peaks

    if not quiet:
        print(f"-Removed {n - n2}/{n} peaks ({(n-n2)/n*100:.1f}%)")

    return df_peaks_out, df_peaks_rem


def window_beat(ecg_signal, sample_rate, window_size, idx=None, stamp=None, timestamps=None):

    ws = int(sample_rate * window_size)

    if stamp is not None and timestamps is not None:
        idx = int((pd.to_datetime(stamp) - timestamps[0]).total_seconds() * sample_rate)
    if idx is not None:
        idx = idx

    w = ecg_signal[idx - ws:idx + ws]

    return w


def crop_template(template: np.array or list, sample_rate: int or float, window_size: int or float):
    # ensures template length matches specified window size
    max_i = np.argmax(abs(template))
    pad_i = int(sample_rate * window_size)
    template = template[int(max_i - pad_i) if int(max_i - pad_i) >= 0 else 0:int(max_i + pad_i)]

    return template


def find_first_highly_correlated_beat(ecg_signal: list or np.array,
                                      peaks: list or np.array,
                                      template: list or np.array,
                                      correl_thresh: float = .7,
                                      n_consec: int = 5):
    """Looks for first peak that matches correlation to template criteria.
       Requires a streak of n_consec consecutive peaks that meet correlation threshold.

        arguments:
        -ecg_signal: timeseries ECG signal
        -sample_rate: of ecg_signal, Hz
        -peaks: array-like of peak indexes corresponding to ecg_signal
        -template: array-like of average QRS template that windows are each beat are correlated with
        -correl_thresh: Pearson correlation threshold required for "valid" beat
        -n_consec: number of consecutive beats above correlation threshold

        returns:
        -index of first beat in the sequence of n_consecutive beats that exceed correlation threshold
    """

    window_samples = int(len(template)/2)

    all_r = []
    for idx, peak in enumerate(peaks):
        window = ecg_signal[peak - window_samples:peak + window_samples]
        r = pearsonr(window, template)[0]
        all_r.append(r >= correl_thresh)

        if len(all_r) >= n_consec:
            if sum(all_r[-n_consec:]) == n_consec:
                return idx - n_consec


def window_signal(ecg_signal: list or np.array, sample_rate: int or float,
                  centre_idx: int, window_size: int or float = .2):
    """ Function that returns a window of data given indexes.

        parameters:
        -ecg_signal: array/list of ECG signal to crop
        -sample_rate: of ecg_signal, Hz
        -centre_idx: index in ecg_signal that represents the centre of the returned window
        -window_size: number of seconds that make up the window on both sides of centre_idx

        returns:
        -cropped section of ecg_signal
    """

    window_samples = int(sample_rate * window_size)
    window = ecg_signal[centre_idx - window_samples:centre_idx + window_samples]

    return window


def correct_premature_beat(ecg_signal: list or np.array,

                           sample_rate: int or float,
                           peak_idx: int, next_peak_idx: int,
                           qrs_template: list or np.array,
                           search_window: int or float = .5,
                           correl_window: int or float = .125,
                           volt_thresh: int or float = 200):
    """Function designed to correct prematurely detected beats.
       Scans window specified in search_window *after* the given peak until the next peak,
       runs a simple peak detection, correlates each window with qrs_template.
       Of the beats with an amplitude above volt_thresh, returns index of the beat with the highest correlation value.


    """

    w = window_signal(ecg_signal=ecg_signal, sample_rate=sample_rate,
                      centre_idx=peak_idx,
                      window_size=search_window)

    w = w[int(len(w)/2):]
    win_size = int(correl_window * sample_rate)

    # simple threshold-based peak detection (not ECG-specific)
    # simple_peaks = indexes(y=w, thres=volt_thresh, thres_abs=True)
    simple_peaks = indexes(y=abs(w), thres=volt_thresh, thres_abs=True)
    simple_peaks += peak_idx
    simple_peaks = [i for i in simple_peaks if peak_idx < i < next_peak_idx]

    r_vals = []
    for idx, peak in enumerate(simple_peaks):
        # if ecg_signal[peak] >= volt_thresh:
        if abs(ecg_signal[peak]) >= volt_thresh:
            window = ecg_signal[peak - win_size:peak + win_size]
            r = pearsonr(qrs_template, window)[0]
            r_vals.append(r)

    try:
        max_r = np.argmax(r_vals)
        max_r_val = max(r_vals)
        max_r_peak = simple_peaks[max_r]

    # if no alternative peaks, returns original peak
    except ValueError:
        max_r_val = None
        max_r_peak = peak_idx

    return max_r_peak, max_r_val, simple_peaks


def row_by_timestamp(df: pd.DataFrame, timestamp: str or pd.Timestamp, n_rows: int = 1):
    """ Search function to return n_rows from dataframe after given timestamp.

        arguments:
        -df: dataframe with column 'start_time'
        -timestamp: timestamp used to crop df. Converted to pd.datetime object
        -n_rows: number of dataframe rows after timestamp to return

        returns:
        -cropped dataframe
    """

    d = df.loc[df['start_time'] >= pd.to_datetime(timestamp)].iloc[:n_rows]

    return d


def crop_df_snr(df, start_idx=0, end_idx=None, quiet=True):

    if not quiet:
        print(f"-Cropping Smital SNR dataframe to indexes {start_idx}-{end_idx if end_idx is not None else -1}")

    df = df.loc[df['start_idx'] >= start_idx]

    if end_idx is not None:
        df = df.loc[df['end_idx'] <= end_idx]

    return df
