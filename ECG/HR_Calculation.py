import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta


def epoch_hr(df_beats, avg_period, start_time, end_time, centre=False):

    # window timestamps
    ts = list(pd.date_range(start=start_time + timedelta(seconds=-avg_period/2 if centre else 0),
                            end=end_time, freq=f"{avg_period}S"))

    if avg_period is not None:
        print(f"-Calculating HR in {avg_period}-second windows...")

        r = []

        for t1, t2 in zip(ts[:], ts[1:]):
            d = df_beats.loc[(df_beats['timestamp'] >= t1) & (df_beats['timestamp'] < t2)]
            n_beats = d.shape[0]

            try:
                if d.shape[0] >= avg_period / 2:
                    t = (d.iloc[-1]['timestamp'] - d.iloc[0]['timestamp']).total_seconds()
                    hr = (n_beats - 1)/t * 60

                    r.append(hr)

                if d.shape[0] < avg_period / 2:
                    r.append(None)

            except IndexError:
                r.append(None)

        roll_hr = pd.DataFrame({"timestamp": ts[:len(r)], 'rate': r})

    if avg_period is None:
        roll_hr = pd.DataFrame({"timestamp": ts, 'rate': [None] * len(ts)})

    return roll_hr


def filter_rr(df_peaks, sample_rate, timestamps, peaks_colname='idx', threshold=30, max_iters=20, plot_hr=False):
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

        returns:
        -dataframe of remaining beats + timestamps
    """

    print(f"\nRunning Wilkund et al. (2008) RR-filtering technique using a "
          f"threshold of {threshold}% and <={max_iters} iterations...")

    peaks = list(df_peaks[peaks_colname])

    n_input = len(peaks)

    if plot_hr:
        fig, ax = plt.subplots(int(np.ceil(max_iters/5)+1), figsize=(12, 8), sharex='col', sharey='col')

    for i in range(max_iters):

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

        print(f"     -Went from {n1} to {n2} beats ({n1-n2} removed)")

        if n1-n2 == 0:
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


def calculate_inst_hr(sample_rate, df_peaks, peak_colname='idx', min_quality=3, max_break=3, quiet=True):
    """Calculates beat-to-beat HR from given peaks (RR intervals)

        arguments:
        -df_peaks: df with timestamps, indexes, and a validity check column
        -min_quality: worst quality data to include. Integer of 1, 2, or 3 (3 is lowest quality)
        -sample_rate: Hz, of ecg signal
        -max_break: number of seconds consecutive beats can be and still calculate a HR. If RR interval is above this
            value, a None will be given instead of a very low HR

        returns:
        -heart rates

    """

    if not quiet:
        print(f"\nCalculating beat-to-beat HR using beats in Smital category <={min_quality} "
              f"and maximum gap between beats of {max_break} seconds...")

    inst_hr = []
    peaks = list(df_peaks[peak_colname])

    if 'quality' in df_peaks.columns:
        q = list(df_peaks['quality'])
    if 'quality' not in df_peaks.columns:
        q = [1] * df_peaks.shape[0]
        if not quiet:
            print("-No quality data provided. All beats will be included.")

    for i in range(df_peaks.shape[0] - 1):
        p1 = peaks[i]
        p2 = peaks[i+1]

        dt = (p2 - p1) / sample_rate

        if dt <= max_break and q[i] <= min_quality and q[i+1] <= min_quality:
            inst_hr.append(round(60 / dt, 2))

        else:
            inst_hr.append(None)

    inst_hr.append(None)

    return inst_hr


def calculate_epoch_hr_sliding(df_peaks, sample_rate, epoch_len=15):
    """Calculates average heart rate over beat + epoch_len seconds interval for each beat.

        arguments:
        -df_peaks: dataframe with peaks
        -sample_rate: Hz, of ecg signal
        -epoch_len: number of seconds over which HR is averaged

        returns:
        -list of heart rates
    """

    print(f"\nCalculating HR in {epoch_len}-second intervals...")

    peaks = list(df_peaks['idx'])
    idx = np.array(df_peaks['idx'])
    idx_window = int(sample_rate*epoch_len)

    start_i = idx[0] + int(epoch_len * sample_rate / 2)
    start_peak_i = np.argwhere(idx >= start_i)[0]

    hrs = [None] * start_peak_i[0]

    for i, peak in enumerate(idx[start_peak_i[0]:]):
        peak_window = []

        # Stops loop if window exceeds final peak
        if peak + idx_window >= idx[-1]:
            break

        # Loops through peaks
        for j in range(i, len(idx)):
            if idx[j] <= peak + idx_window:
                peak_window.append(idx[j])
            if idx[j] > peak + idx_window:
                break

        # requires at least half as many beats as duration of window (i.e., HR > 30bpm)
        if len(peak_window) >= epoch_len/2:
            delta_t = (peak_window[-1] - peak_window[0])/sample_rate
            n_beats = len(peak_window)
            hrs.append(60 * (n_beats - 1)/delta_t)

        if len(peak_window) < epoch_len/2:
            hrs.append(None)

    for i in range(len(idx) - len(hrs)):
        hrs.append(None)

    return hrs


def calculate_epoch_hr_jumping(df_peaks, sample_rate, ecg_signal, ecg_timestamps, min_quality=2, epoch_len=15):
    """Epochs HR in jumping epoch_len windows.

        arguments:
        -df_peaks: dataframe with peaks
        -ecg_signal: array
        -ecg_timestamps: timestamps for ecg_signal
        -sample_rate: Hz, of ecg_signal
        -epoch_len: in seconds
        -min_quality: minimum Smital quality category for inclusion of beats

        returns:
        df

    """

    epoch_idx = np.arange(0, len(ecg_signal), int(sample_rate * epoch_len))
    min_beats = epoch_len / 2

    epoch_valid_beats = []
    epoch_invalid_beats = []
    epoch_hr = []

    epoch_i = 0
    curr_epoch = (epoch_idx[epoch_i], epoch_idx[epoch_i + 1])
    curr_valid = []
    curr_invalid = []
    n_beats = []
    durations = []
    n_invalid = []

    # df_peaks = df_peaks.loc[df_peaks['snr'] <= min_quality]

    for row in df_peaks.itertuples():

        # if beat falls within current epoching window
        if curr_epoch[0] <= row.idx < curr_epoch[1]:
            try:
                if row.quality <= min_quality:
                    curr_valid.append(row.idx)
                if row.quality > min_quality:
                    curr_invalid.append(row.idx)
            except AttributeError:
                curr_valid.append(row.idx)

        if row.idx >= curr_epoch[1]:

            if len(curr_valid) >= min_beats:
                total_t = (curr_valid[-1] - curr_valid[0]) / sample_rate

                hr = (len(curr_valid) - 1) / total_t * 60

                epoch_hr.append(round(hr, 2))

                n_beats.append(len(curr_valid))
                n_invalid.append(len(curr_invalid))
                durations.append(total_t)

            if len(curr_valid) < min_beats:
                epoch_hr.append(None)
                n_beats.append(len(curr_valid))
                n_invalid.append(len(curr_invalid))
                durations.append(None)

            epoch_valid_beats += curr_valid
            epoch_invalid_beats += curr_invalid

            curr_valid = []
            curr_invalid = []

            epoch_i += 1
            try:
                curr_epoch = (epoch_idx[epoch_i], epoch_idx[epoch_i + 1])
            except IndexError:
                break

    df = pd.DataFrame({"timestamp": ecg_timestamps[epoch_idx][:len(epoch_hr)],
                       'beats': n_beats, 'n_invalid': n_invalid,
                       'duration': durations, 'hr': epoch_hr})

    return df


def calculate_delta_hr(hr, absolute_hr=True):

    # delta HR. Values are beat and change from previous beat
    if absolute_hr:
        ch = [j - i for i, j in zip(hr[:], hr[1:])]
    if not absolute_hr:
        ch = [(j - i) * 100 / i for i, j in zip(hr[:], hr[1:])]

    ch.insert(0, None)  # padding

    return ch


def jumping_epoch_hr(sample_rate, peaks, timestamps, epoch_len=15):
    print(f"\nEpoching data into {epoch_len}-second windows...")

    epoch_samples = int(sample_rate * epoch_len)

    p = np.array(peaks)
    epoch_idxs = range(0, len(timestamps), epoch_samples)
    hrs = []
    beats = []

    for epoch_idx in tqdm(epoch_idxs):
        epoch_p = sorted(np.where((p >= epoch_idx) & (p < epoch_idx + epoch_samples))[0])
        epoch_p = p[epoch_p]
        n_beats = len(epoch_p)
        beats.append(n_beats)

        try:
            dt = (epoch_p[-1] - epoch_p[0]) / sample_rate
            hr = (n_beats - 1) / dt * 60
            hrs.append(round(hr, 1))
        except (IndexError, ValueError):
            hrs.append(None)

    secs = epoch_len - 1 / sample_rate
    df_epoch = pd.DataFrame({"start_time": timestamps[epoch_idxs], "idx": epoch_idxs, 'n_beats': beats, 'hr': hrs})
    df_epoch.insert(loc=1, column='end_time', value=df_epoch['start_time'] + timedelta(seconds=secs))

    return df_epoch

