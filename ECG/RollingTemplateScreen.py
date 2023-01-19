from tqdm import tqdm
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta as td
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")


def format_qrs_template(qrs_template, window_size, sample_rate, peak_align):

    # makes sure qrs_template is correct length given peak_align, sample_rate, and window_size
    max_idx = np.argmax(qrs_template)  # index of highest peak in qrs_template, for alignment
    req_max_idx = window_size * sample_rate * peak_align  # index of peak to align with peak_align
    qrs_use = qrs_template.copy()

    # pads qrs_template pre/post peak to ensure max peak alignment
    for i in range(int(req_max_idx - max_idx)):
        qrs_use = np.insert(qrs_use, 0, qrs_use[0])
    for i in range(int(sample_rate * window_size) - len(qrs_use)):
        qrs_use = np.insert(qrs_use, -1, qrs_use[-1])

    return qrs_use


def correct_peak_locations(ecg_signal, pre_idx, post_idx, peaks, sample_rate, max_hr=220):

    print(f"\nCorrecting peaks locations...")
    min_samples = int(60/max_hr * sample_rate)

    corrected_peaks = []
    for peak_i, peak in enumerate(peaks):
        if peak - pre_idx >= 0:
            window = ecg_signal[peak - pre_idx:peak + post_idx]
        else:
            window = ecg_signal[peak:peak + post_idx]
            for i in range(int(sample_rate * window_size) - len(window)):
                window.insert(0, 0)

        if peak_i == 0 or peak - corrected_peaks[-1] >= min_samples:
            true_peak_idx = np.argmax(window) + peak - pre_idx
            offset = peak - true_peak_idx

            corrected_peaks.append(peak - offset)

    return corrected_peaks


def calculate_valid_rr(df, validity_colname='valid', sample_rate=250):
    # calculates RR-based HR for consecutive valid beats
    rr = []
    for p1, p2, v1, v2 in zip(df['idx'].iloc[:], df['idx'].iloc[1:], df[validity_colname].iloc[:], df[validity_colname].iloc[1:]):
        t = (p2 - p1) / sample_rate
        rr.append(60 / t)
    rr.append(None)

    return rr


def epoch_avg_rr(df, sample_rate, epoch_len=30):

    print(f"\nEpoching HR using average RR intervals over {epoch_len}-second epochs...")

    # indexes for epoching windows
    epoch_idx = np.arange(df['idx'].min(), df['idx'].max() + 1, int(sample_rate * epoch_len))

    # epochs HR using average RR interval from valid beats in epoch windows
    avg_rr = []
    n_beats = []
    for i in tqdm(range(len(epoch_idx) - 1)):
        i1 = epoch_idx[i]
        i2 = epoch_idx[i+1]

        d = df.loc[(df['idx'] >= i1) & (df['idx'] <= i2)]
        d = d.reset_index(drop=True)
        n_beats.append(d.shape[0])

        if d.shape[0] >= 1:
            avg_rr.append(np.mean(d['rr']))
        if d.shape[0] == 0:
            avg_rr.append(None)

    df_epoch = pd.DataFrame({"timestamp": pd.date_range(start=df.iloc[0]['timestamp'] + td(seconds=epoch_len/2),
                                                        freq=f'{epoch_len}S', periods=len(avg_rr)),
                             'n_beats': n_beats, 'rr_avg': avg_rr})

    return df_epoch


def create_initial_template(ecg_signal, peaks, template, remove_mean=True, n_beats_template=6):

    print(f"\nGenerating template from start of file using {n_beats_template} beats...")

    window_beats = []
    beat_snr = []

    for start_beat_idx, peak in enumerate(peaks):

        # stops loop once enough beats (n_beats_template) have been found
        if len(window_beats) >= n_beats_template:
            break

        if len(window_beats) < n_beats_template:

            window = ecg_signal[peak - pre_idx:peak + post_idx]

            if remove_mean:
                mean_val = np.mean(window)
                window = [i - mean_val for i in window]

            r = scipy.stats.pearsonr(window, template)

            if r[0] >= corr_thresh:
                window_beats.append(window)

            beat_snr_val = np.mean(snr[peak - pre_idx:peak + post_idx])
            beat_snr.append(beat_snr_val)

    template_out = np.mean(np.array(window_beats, dtype=object), axis=0)

    return start_beat_idx, template_out, window_beats


def create_initial_template_consecutive(ecg_signal, peaks, template, remove_mean=True, n_beats_template=6):

    print(f"\nGenerating template from start of file using {n_beats_template} beats...")

    beat_snr = []
    beat_idx = -1

    for start_beat_idx, peak in enumerate(peaks):
        window_beats = []

        if start_beat_idx > beat_idx:
            window = ecg_signal[peak - pre_idx:peak + post_idx]

            if remove_mean:
                mean_val = np.mean(window)
                window = [i - mean_val for i in window]

            beat_snr_val = np.mean(snr[peak - pre_idx:peak + post_idx])
            beat_snr.append(beat_snr_val)

            r = scipy.stats.pearsonr(window, template)

            # if beat correlated enough, looks for n_beats_template subsequent highly-correlated beats
            if r[0] >= corr_thresh:

                for beat_idx in range(start_beat_idx + 1, len(peaks)):
                    if len(window_beats) < n_beats_template:
                        window2 = ecg_signal[peaks[beat_idx] - pre_idx:peaks[beat_idx] + post_idx]

                        if remove_mean:
                            mean_val = np.mean(window2)
                            window2 = [i - mean_val for i in window2]

                        r2 = scipy.stats.pearsonr(window2, template)

                        if r2[0] >= corr_thresh:
                            window_beats.append(window2)

                        if r2[0] < corr_thresh:
                            break

                    if len(window_beats) >= n_beats_template:
                        break

        if len(window_beats) >= n_beats_template:
            template_out = np.mean(np.array(window_beats, dtype=object), axis=0)

            return beat_idx, template_out, window_beats


def rolling_template(peaks, ecg_signal, start_peak_idx, window_beats, timestamps, corr_thresh=.7):

    r_values = [None] * start_peak_idx
    beat_snr = [None] * start_peak_idx
    window_peaks = [None] * start_peak_idx
    avg_height = [None] * start_peak_idx
    indiv_height = [None] * start_peak_idx
    peak_diff_percent = [None] * start_peak_idx
    peak_diff_abs = [None] * start_peak_idx

    last_beat_valid = False

    window_dict = {}

    for i, peak in enumerate(tqdm(peaks[start_peak_idx:])):

        # removes first beat and adds next required beat IF last tested beat was valid
        if last_beat_valid:
            window_beats = window_beats[1:]
            window_peaks = window_peaks[1:]

        peak_heights = [ecg_signal[j] for j in window_peaks if j is not None]

        window = ecg_signal[peak - pre_idx:peak + post_idx]
        peak_height = ecg_signal[peak]

        mean_peak = np.mean(peak_heights)
        # peak_diff = (mean_peak - peak_height) * 100 / ((mean_peak + peak_height) / 2)
        peak_diff = (peak_height - mean_peak) * 100 / mean_peak
        peak_diff_percent.append(np.abs(peak_diff))
        avg_height.append(mean_peak)
        indiv_height.append(peak_height)
        try:
            peak_diff_abs.append(np.abs(peak_height - avg_height))
        except TypeError:
            peak_diff_abs.append(None)

        mean_val = np.mean(window)
        beat = [i - mean_val for i in window]

        # average beat for beats in window_beats
        template = np.mean(np.array(window_beats, dtype=object), axis=0)

        window_dict[peak] = template

        # correlation between beat and rolling template
        try:
            r = scipy.stats.pearsonr(beat, template)[0]
            r_values.append(r)

            if r >= corr_thresh:
                last_beat_valid = True
                window_beats.append(beat)
                window_peaks.append(peak)

            if r < corr_thresh:
                last_beat_valid = False

        except ValueError:
            r_values.append(None)
            last_beat_valid = False

        beat_snr_val = np.mean(snr[peak - pre_idx:peak + post_idx])
        beat_snr.append(beat_snr_val)

    df = pd.DataFrame({'timestamp': timestamps[peaks], 'idx': peaks, 'r_roll': r_values, 'snr': beat_snr,
                       'avg_peak': avg_height, 'peak_height': indiv_height, 'peak_change': peak_diff_percent,
                       'peak_change_abs': peak_diff_abs})
    df['valid_r'] = df['r_roll'] >= corr_thresh
    df['valid_snr'] = df['snr'] >= snr_thresh
    df['valid'] = df['valid_r'] & df['valid_snr']

    return df, window_dict


def plot_data(ds_ratio=2, show_template=False):
    max_i = df_test['idx'].max()

    fig, ax = plt.subplots(6, sharex='col', figsize=(12, 8))

    # ecg
    ax[0].plot(ecg.ts[:max_i:ds_ratio], ecg.filt[:max_i:ds_ratio], color='black', zorder=0)

    # invalid beats from SNR/template
    ax[0].scatter(ecg.ts[df_test_invalid['idx']], [ecg.filt[i] + 100 for i in df_test_invalid['idx']],
                  color='fuchsia', s=20, zorder=1, marker='v', label=f'invalid (n={df_test_invalid.shape[0]})')

    # invalid beats from RR outside range
    ax[0].scatter(ecg.ts[df_test_invalid_hr['idx']], [ecg.filt[i] + 50 for i in df_test_invalid_hr['idx']],
                  color='red', s=20, zorder=1, marker='x', label=f'HR rej. (n={df_test_invalid_hr.shape[0]})')

    # valid beats
    ax[0].scatter(ecg.ts[df_test_valid_hr['idx']], ecg.filt[df_test_valid_hr['idx']],
                  color='limegreen', s=20, zorder=1, marker='o', label=f'valid (n={df_test_valid_hr.shape[0]})')

    # overlays rolling template onto each beat
    if show_template:
        for key in templates.keys():
            template = templates[key]
            t = ecg.ts[key - pre_idx:key + post_idx]

            ax[0].plot(t[::2], template[::2], color='red')

    ax[0].legend(loc='upper right')
    ax[0].set_ylabel("V")

    # beat-to-beat HR for valid beats
    ax[1].scatter(df_test_valid_hr['timestamp'], df_test_valid_hr['rr'], color='black', s=10, label='B2B')

    # beat-to-beat HR for invalid beats
    ax[1].scatter(df_test_invalid_hr['timestamp'], df_test_invalid_hr['rr'], color='red', s=10, label='Inv. B2B')

    # epoched HR
    epoch_len = int((df_epoch['timestamp'].iloc[1] - df_epoch.iloc[0]['timestamp']).total_seconds())
    ax[1].plot(df_epoch['timestamp'], df_epoch['rr_avg'], color='dodgerblue', label=f'{epoch_len}-sec avg')
    ax[1].set_ylabel("HR")
    ax[1].legend(loc='upper right')
    ax[1].grid()

    # timeseries SNR
    ax[2].plot(ecg.ts[:max_i:25], ecg.snr[:max_i:25], color='dodgerblue')

    # beats' SNR
    ax[2].scatter(df_test['timestamp'], df_test['snr'], color='black')
    ax[2].axhspan(xmin=0, xmax=1, ymin=ax[2].get_ylim()[0], ymax=snr_thresh, color='red', alpha=.25)
    ax[2].set_ylabel("SNR")

    # Template correlation values
    ax[3].scatter(df_test['timestamp'], df_test['r_roll'], color='green', s=3)
    ax[3].axhspan(xmin=0, xmax=1, ymin=ax[3].get_ylim()[0], ymax=corr_thresh, color='red', alpha=.25)
    ax[3].set_ylabel("SNR")
    ax[3].set_ylabel("Rolling Correlation")
    ax[3].grid()

    # percent change in peak height
    ax[4].scatter(df_test['timestamp'], df_test['peak_change'], color='black')
    ax[4].grid()
    ax[4].set_ylabel("Peak height\n(% change)")

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig


def calculate_invalidpeak_loc_difference(df):
    """ For each invalid peak (SNR/template correlation), calculates expected peak location based on midpoint of
        previous and next peaks if both are valid.

        Outputs list of number of samples the actual peak and expected location differ by
    """

    peak_diff = []

    for row in df.iloc[:-1].itertuples():

        if row.valid:
            peak_diff.append(None)

        # only checks invalid beats (snr/template)
        if not row.valid:
            peak_prev = df.iloc[row.Index - 1]
            peak_next = df.iloc[row.Index + 1]

            # boolean for previous + next peaks valid
            surround_valid = peak_prev['valid'] and peak_next['valid']

            if surround_valid:
                # midpoint index of previous and next peaks
                mid_idx = (peak_next['idx'] - peak_prev['idx']) / 2 + peak_prev['idx']
                diff_idx = row.idx - mid_idx
                peak_diff.append(abs(diff_idx))

            if not surround_valid:
                peak_diff.append(None)

    # keeps list length consistent
    peak_diff.append(None)

    return peak_diff


def rolling_template2(peaks, ecg_signal, start_peak_idx, window_beats, timestamps, corr_thresh=.7):

    r_values = [None] * start_peak_idx
    beat_snr = [None] * start_peak_idx
    window_peaks = [None] * start_peak_idx
    avg_height = [None] * start_peak_idx
    indiv_height = [None] * start_peak_idx
    peak_diff_percent = [None] * start_peak_idx
    peak_diff_abs = [None] * start_peak_idx

    last_beat_valid = False

    window_dict = {}

    for i, peak in enumerate(tqdm(peaks[start_peak_idx:])):

        # removes first beat and adds next required beat IF last tested beat was valid
        if last_beat_valid:
            window_beats = window_beats[1:]
            window_peaks = window_peaks[1:]

        peak_heights = [ecg_signal[j] for j in window_peaks if j is not None]

        window = ecg_signal[peak - pre_idx:peak + post_idx]
        peak_height = ecg_signal[peak]

        mean_peak = np.mean(peak_heights)
        peak_diff = (peak_height - mean_peak) * 100 / mean_peak
        peak_diff_percent.append(np.abs(peak_diff))
        avg_height.append(mean_peak)
        indiv_height.append(peak_height)

        try:
            peak_diff_abs.append(np.abs(peak_height - avg_height))
        except TypeError:
            peak_diff_abs.append(None)

        mean_val = np.mean(window)
        beat = [i - mean_val for i in window]

        # average beat for beats in window_beats
        template = np.mean(np.array(window_beats, dtype=object), axis=0)

        window_dict[peak] = template

        # correlation between beat and rolling template
        try:
            r = scipy.stats.pearsonr(beat, template)[0]
            r_values.append(r)

            if r >= corr_thresh:
                last_beat_valid = True
                window_beats.append(beat)
                window_peaks.append(peak)

            if r < corr_thresh:
                last_beat_valid = False

        except ValueError:
            r_values.append(None)
            last_beat_valid = False

        beat_snr_val = np.mean(snr[peak - pre_idx:peak + post_idx])
        beat_snr.append(beat_snr_val)

    df = pd.DataFrame({'timestamp': timestamps[peaks], 'idx': peaks, 'r_roll': r_values, 'snr': beat_snr,
                       'avg_peak': avg_height, 'peak_height': indiv_height, 'peak_change': peak_diff_percent,
                       'peak_change_abs': peak_diff_abs})
    df['valid_r'] = df['r_roll'] >= corr_thresh
    df['valid_snr'] = df['snr'] >= snr_thresh
    df['valid'] = df['valid_r'] & df['valid_snr']

    return df, window_dict


ecg_signal = ecg.filt
timestamps = ecg.ts
peaks = list(df_nk['idx_corr2'][:2500])
snr = ecg.snr
qrs_template = qrs
window_size = .4  # .6
peak_align = .4
remove_mean = True
correct_peaks = True
n_beats_template = 6
corr_thresh = .7
snr_thresh = 5
show_plot = False
sample_rate = ecg.fs

"""  """
# index offsets to centre peak in desired part of window
pre_idx = int(sample_rate * window_size * peak_align)
post_idx = int(sample_rate * window_size * (1 - peak_align))

raw_i = peaks[-1]  # last peak index
ecg_signal = ecg_signal[:raw_i + int(30 * sample_rate) if raw_i + int(30 * sample_rate) < len(ecg_signal) else -1]

# function calls ========================

qrs_use = format_qrs_template(qrs_template, window_size, sample_rate, peak_align)

peaks = correct_peak_locations(ecg_signal, pre_idx, post_idx, peaks, sample_rate=sample_rate, max_hr=220)

start_peak_idx, qrs_new, window_beats = create_initial_template_consecutive(ecg_signal=ecg_signal, peaks=peaks, template=qrs_use,
                                                                            remove_mean=True, n_beats_template=6)

df_test, templates = rolling_template2(peaks=peaks, ecg_signal=ecg_signal,
                                      start_peak_idx=start_peak_idx, window_beats=window_beats,
                                      corr_thresh=corr_thresh, timestamps=timestamps)

# for invalid peaks, calculates difference between peak loc and mid-point of surrounding valid peaks
df_test['invalid_peak_diff'] = calculate_invalidpeak_loc_difference(df=df_test)

df_test['valid2'] = df_test['valid'].copy()
df_test.loc[(df_test['invalid_peak_diff'] < 250/10) & (~df_test['valid'] & (df_test['peak_change'] <= 33)), 'valid2'] = True  # 1/6th of second ~ QRS width

df_test_invalid1 = df_test.loc[~df_test['valid']]  # invalid from SNR/correlation
df_test_invalid2 = df_test.loc[~df_test['valid2']]  # removed invalids
df_valid2 = df_test.loc[df_test['valid2']]
df_corrected = df_test.loc[(~df_test['valid']) & (df_valid2['valid2'])]
df_final = df_test.loc[df_test['valid2']]
df_final['rr'] = calculate_valid_rr(df=df_final, sample_rate=sample_rate, validity_colname='valid2')

df_check = df_test.loc[(df_test['timestamp'] >= '2022-05-30 14:36:28') & (df_test['timestamp'] <= '2022-05-30 14:36:32')]
max_i = df_test.iloc[-1]['idx']

fig, ax = plt.subplots(5, figsize=(12, 8), sharex='col', gridspec_kw={'height_ratios': [1, .5, .5, .5, 1]})

ax[0].plot(ecg.ts[:max_i], ecg.filt[:max_i], color='black')
ax[0].scatter(df_test['timestamp'], ecg.filt[df_test['idx']], marker='o', color='dodgerblue', label=f'all_peaks (n={df_test.shape[0]})')
ax[0].scatter(df_test_invalid1['timestamp'], [ecg.filt[i] + 250 for i in df_test_invalid1['idx']], marker='x', color='purple', label=f'invalid1 (n={df_test_invalid1.shape[0]})')
ax[0].scatter(df_test_invalid2['timestamp'], [ecg.filt[i] + 500 for i in df_test_invalid2['idx']], marker='x', color='red', label=f'invalid2 (n={df_test_invalid2.shape[0]})')
ax[0].scatter(df_corrected['timestamp'], [ecg.filt[i] + 750 for i in df_corrected['idx']], marker='v', color='orange', label=f'corrected (n={df_corrected.shape[0]})')
ax[0].scatter(df_final['timestamp'], [ecg.filt[i] + 1000 for i in df_final['idx']], marker='v', color='limegreen', label=f'final (n={df_final.shape[0]})')
ax[0].legend(loc='upper right')

"""for key in templates.keys():
    template = templates[key]
    t = ecg.ts[key - pre_idx:key + post_idx]

    ax[0].plot(t[::2], template[::2], color='red')"""

ax[1].scatter(df_test_invalid1['timestamp'], df_test_invalid1['snr'], color='purple', marker='x')
ax[1].scatter(df_test_invalid2['timestamp'], df_test_invalid2['snr'], color='red', marker='x')
ax[1].scatter(df_corrected['timestamp'], df_corrected['snr'], color='orange', marker='v')
ax[1].scatter(df_final['timestamp'], df_final['snr'], color='limegreen', marker='v')
ax[1].axhspan(xmin=0, xmax=1, ymin=5, ymax=ax[1].get_ylim()[1], color='limegreen', alpha=.2)
ax[1].set_ylabel("SNR")

ax[2].scatter(df_test_invalid1['timestamp'], df_test_invalid1['peak_change'], color='purple', marker='x')
ax[2].scatter(df_test_invalid2['timestamp'], df_test_invalid2['peak_change'], color='red', marker='x')
ax[2].scatter(df_corrected['timestamp'], df_corrected['peak_change'], color='orange', marker='v')
ax[2].scatter(df_final['timestamp'], df_final['peak_change'], color='limegreen', marker='v')
ax[2].axhspan(xmin=0, xmax=1, ymin=0, ymax=33, color='limegreen', alpha=.2)
ax[2].set_ylabel("RR diff\n(% change)")
ax[2].grid()
ax[2].set_ylim(0, )

ax[3].scatter(df_test_invalid1['timestamp'], df_test_invalid1['r_roll'], color='purple', marker='x')
ax[3].scatter(df_test_invalid2['timestamp'], df_test_invalid2['r_roll'], color='red', marker='x')
ax[3].scatter(df_corrected['timestamp'], df_corrected['r_roll'], color='orange', marker='v')
ax[3].scatter(df_final['timestamp'], df_final['r_roll'], color='limegreen', marker='v')
ax[3].axhspan(xmin=0, xmax=1, ymin=corr_thresh, ymax=1, color='limegreen', alpha=.2)
ax[3].grid()
ax[3].set_ylabel("Template\ncorrelation")

ax[4].scatter(df_final['timestamp'], df_final['rr'], color='limegreen', marker='v')
ax[4].set_ylabel("B2B HR (bpm)")
ax[4].grid()
ax[4].axhspan(xmin=0, xmax=1, ymin=40, ymax=220, color='limegreen', alpha=.2)

ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S.%f"))
plt.tight_layout()

""" SCREEN V1 """
"""
# valid based on template correlation and snr
df_test_valid = df_test.loc[df_test['valid']].reset_index(drop=True)
df_test_valid['rr'] = calculate_valid_rr(df=df_test_valid, sample_rate=sample_rate)

# invalid based on template correlation and snr
df_test_invalid = df_test.loc[~df_test['valid']].reset_index(drop=True)

# valid HRs based on 40 < HR < 220 bpm
df_test_valid_hr = df_test_valid.loc[(df_test_valid['rr'] > 30) & (df_test_valid['rr'] <= 220)].reset_index(drop=True)

# invalid HRs based on HR < 40 OR HR > 220
df_test_invalid_hr = df_test_valid.loc[(df_test_valid['rr'] <= 30) | (df_test_valid['rr'] > 220)].reset_index(drop=True)

df_epoch = epoch_avg_rr(df=df_test_valid_hr, sample_rate=sample_rate, epoch_len=15)
"""

# fig = plot_data(ds_ratio=1, show_template=False)


# TODO
# peak removal for HR; recalc with next beat + keep if valid
# final RR interval check
