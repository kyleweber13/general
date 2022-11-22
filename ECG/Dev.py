import pandas as pd
from ECG.ImportFiles import ECG
from ECG.Processing import remove_peaks_during_bouts, window_beat
from ECG.PeakDetection import correct_cn_peak_locations, create_beat_template_snr_bouts, get_zncc
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
from ECG.Plotting import *
import neurokit2 as nk
from ECG.HR_Calculation import calculate_inst_hr
from tqdm import tqdm
from scipy.stats import pearsonr
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


def other_methods():
    df_nk, df_nk_epoch = run_neurokit_method(ecg_obj=ecg, df_snr=ecg.df_snr, window_size=.33, n_hq_snr_bouts=5, corr_thresh=.5)

    zncc, df_zncc, df_zncc_epoch = run_zncc_method(input_data=ecg.filt, template=qrs, zncc_thresh=.725, snr=ecg.snr,
                                                   sample_rate=ecg.fs, downsample=1, timestamps=ecg.ts,
                                                   thresholds=thresholds, show_plot=False, min_dist=int(ecg.fs/(220/60)))
    df_zncc = correct_cn_peak_locations(df_peaks=df_zncc, peaks_colname='idx',
                                        ecg_signal=ecg.filt, sample_rate=ecg.fs, window_size=.3, use_abs_peaks=True)


def import_snr_bout_file(filepath: str):
    """ Imports signal-to-noise ratio (SNR) bout file from csv and formats column data appropriately.

        arguments:
        -filepath: pathway to SNR bout file

        returns:
        -dataframe
    """

    dtype_cols = {"study_code": str, 'subject_id': str, 'coll_id': str,
                  'start_idx': pd.Int64Dtype(), 'end_idx': pd.Int64Dtype(), 'bout_num': pd.Int64Dtype(),
                  'quality': str, 'avg_snr': float}
    date_cols = ['start_timestamp', 'end_timestamp']
    df = pd.read_csv(filepath, dtype=dtype_cols, parse_dates=date_cols)

    df['duration'] = [(row.end_timestamp - row.start_timestamp).total_seconds() for row in df.itertuples()]

    # replaces strings with numeric equivalents for signal qualities
    df['quality_use'] = df['quality'].replace({'ignore': 3, 'full': 1, 'HR': 1})

    return df


def import_nw_file(filepath: str, sample_rate: float or int, start_timestamp: str):
    """ Imports and formats nonwear bouts csv file.

        arguments:
        -filepath: pathway to csv file
        -sample_rate: of ECG signal, Hz
        -start_timestamp: of ECG signal

        returns:
        -dataframe
    """

    start_timestamp = pd.to_datetime(start_timestamp)
    date_cols = ['start_time', 'end_time']
    df = pd.read_csv(filepath, parse_dates=date_cols)

    df['start_idx'] = [int((row.start_time - start_timestamp).total_seconds() * sample_rate) for row in df.itertuples()]
    df['end_idx'] = [int((row.end_time - start_timestamp).total_seconds() * sample_rate) for row in df.itertuples()]

    return df


def screen_delta_hr(df_peaks: pd.DataFrame, ecg_signal: np.array or list, sample_rate: int or float,
                    peaks_colname: str = 'idx',  absolute_hr: bool = True, delta_hr_thresh: float or int = 20,
                    use_correl_template: bool = False, correl_template=None,
                    correl_window_size: int or float = .3, correl_thresh: float = .7, quiet: bool = True):
    """ Screening function for beats that change above the specified amount (either bpm or % change).
        Each time a large HR change is detected, the beat involved is correlated with the
        next beat or the beat template and removed if it's below the correlation threshold.

        arguments:
        -df_peaks: dataframe containing peaks. required columns: ['hr', 'idx']
        -peaks_colname: column name in df_peaks to use as peak indexes

        -ecg_signal: ECG signal with indexes that correspond to those in df_peaks
        -sample_rate: of ECG signal, Hz

        -absolute_hr: boolean. If True, delta HR will be bpm. If False, will use percent change instead of bpm
        -delta_hr_thresh: bpm threshold above which beats are checked. Units are bpm or % based on absolute_hr

        -use_correl_template: boolean. If False, correlation check is run between current and previous beat.
                                       If True, correlation check is run between current beat and template.
        -correl_template: correlation template from PeakDetection.create_beat_template_snr_bouts().

        -correl_window_size: number of seconds on either side of beats that are included in correlation check
        -correl_thresh: Pearson correlation threshold below which beats are removed

        -quiet: boolean

        returns:
        -df_out: dataframe with only valid beats
        -df_rem: dataframe with removed beats
    """

    if not quiet:
        print(f"\nScreening peaks for HR changes above {delta_hr_thresh}{'bpm' if absolute_hr else '%'}...")

    df = df_peaks.copy()

    # delta HR. Values are beat and change from previous beat
    if absolute_hr:
        ch = [j - i for i, j in zip(df['hr'].iloc[:], df['hr'].iloc[1:])]
    if not absolute_hr:
        ch = [(j - i) * 100 / i for i, j in zip(df['hr'].iloc[:], df['hr'].iloc[1:])]

    ch.insert(0, None)  # padding
    df['delta_hr'] = ch

    # number of samples on each side of peak to window for correlation check
    beat_window = int(correl_window_size * sample_rate)

    # df indexes where delta_hr above threshold
    # only positive delta HR is used to try and reject false positives (beats detected too soon --> increase HR)
    check_idx = list(df.loc[df['delta_hr'] >= delta_hr_thresh].index)
    remove_idx = []  # row indexes to be removed
    r_vals = []  # r values for removed beats
    valid_idx = []
    r_vals_all = np.array([None] * df.shape[0])

    for idx in tqdm(df.index[:-1]):

        if idx not in check_idx:
            valid_idx.append(idx)

        if idx in check_idx:
            curr_beat = df.loc[idx]  # current beat's row
            next_beat = df.loc[idx + 1]  # next beat's row

            # ecg data window for current beat
            curr_data = ecg_signal[curr_beat[peaks_colname] - beat_window:curr_beat[peaks_colname] + beat_window]

            # ecg data window for next beat
            next_data = ecg_signal[next_beat[peaks_colname] - beat_window:next_beat[peaks_colname] + beat_window]

            if not use_correl_template or correl_template is None:
                r_next = pearsonr(curr_data, next_data)[0]  # pearson correlation between beats
                valid_r_next = (r_next >= correl_thresh)

            if use_correl_template and correl_template is not None:
                # r_next = pearsonr(curr_data, correl_template)[0]  # pearson correlation between beats
                r_next = pearsonr(next_data, correl_template)[0]  ##
                valid_r_next = (r_next >= correl_thresh)

            # flags next beat for removal if correlation below threshold
            if not valid_r_next:

                # last valid beat window
                idx_last = df.loc[valid_idx[-1]]['idx']
                last_data = ecg_signal[idx_last - beat_window:idx_last + beat_window]

                if not use_correl_template or correl_template is None:  ##
                    r_last = pearsonr(curr_data, last_data)[0]  # pearson correlation between beats
                if use_correl_template and correl_template is not None:
                    r_last = pearsonr(next_data, last_data)[0]  # pearson correlation between beats

                valid_r_last = (r_last >= correl_thresh)

                if not valid_r_last:
                    remove_idx.append(idx + 1)
                    r_vals.append(r_next)

                    r_vals_all[idx] = r_next  ##

                if valid_r_last:
                    valid_idx.append(idx)

    df['r'] = r_vals_all  ##

    df_out = df.copy().drop(remove_idx)

    df_rem = df.loc[remove_idx]
    df_rem['r'] = r_vals

    df_out.reset_index(drop=True, inplace=True)

    # recalculates beat-to-beat HR
    df_out['hr'] = calculate_inst_hr(sample_rate=sample_rate, peak_colname=peaks_colname,
                                     df_peaks=df_out, min_quality=3, max_break=3, quiet=True)

    # recalculates beat-to-beat change in HR
    ch = [j - i for i, j in zip(df_out['hr'].iloc[:], df_out['hr'].iloc[1:])]
    ch.insert(0, None)
    df_out.loc[:, 'delta_hr'] = ch

    return df_out, df_rem


def run_deltahr_screen(df_peaks: pd.DataFrame, ecg_signal: np.array or list, sample_rate: int or float,
                       peaks_colname: str = 'idx', absolute_hr: bool = True, delta_hr_thresh: float or int = 30,
                       max_iters: int = 10, correl_window_size: float or int = .2,
                       use_correl_template: bool = False, correl_template: bool = None,
                       corr_thresh: float = .7, quiet: bool = True):
    """ Function that calls screen_delta_hr() in a loop to iteratively remove beats whose HR changes too quickly.

        -df_peaks: dataframe containing peaks. required columns: ['hr', 'idx']
        -peaks_colname: column name in df_peaks to use as peak indexes

        -ecg_signal: ECG signal with indexes that correspond to those in df_peaks
        -sample_rate: of ECG signal, Hz

        -absolute_hr: boolean. If True, delta HR will be bpm. If False, will use percent change instead of bpm
        -delta_hr_thresh: bpm threshold above which beats are checked. Units are bpm or % based on absolute_hr

        -use_correl_template: boolean. If False, correlation check is run between current and previous beat.
                                       If True, correlation check is run between current beat and template.
        -correl_template: correlation template from PeakDetection.create_beat_template_snr_bouts().

        -correl_window_size: number of seconds on either side of beats that are included in correlation check
        -correl_thresh: Pearson correlation threshold below which beats are removed

        -max_iters: maximum number of loops to run. Will stop if no more beats to remove before this number is reached.

        -quiet: boolean

        returns:
        -df_out: dataframe with only valid beats
        -df_rem: dataframe with removed beats
    """

    if not quiet:
        print(f"\nScreening peaks data for beats whose HR changes by >{delta_hr_thresh}{'bpm' if absolute_hr else '%'}")
        print(f"-Using a correlation threshold of r>{corr_thresh} in {correl_window_size}-second windows")
        print(f"-Running <{max_iters} iterations")

    start_len = df_peaks.shape[0]

    if use_correl_template and correl_template is not None:
        # ensures template length matches specified window size
        max_i = np.argmax(correl_template)
        pad_i = int(sample_rate * correl_window_size)
        template = correl_template[int(max_i - pad_i):int(max_i + pad_i)]

    if not use_correl_template or correl_template is None:
        template = None

    loop_i = 1

    df_final, df_rem = screen_delta_hr(df_peaks=df_peaks, peaks_colname=peaks_colname,
                                       ecg_signal=ecg_signal, sample_rate=sample_rate,
                                       delta_hr_thresh=delta_hr_thresh, absolute_hr=absolute_hr,
                                       use_correl_template=use_correl_template, correl_template=template,
                                       correl_window_size=correl_window_size, correl_thresh=corr_thresh)

    df_rem['loop'] = [loop_i] * df_rem.shape[0]

    end_len = df_final.shape[0]

    keep_looping = start_len != end_len

    while keep_looping:
        loop_i += 1
        start_len2 = df_final.shape[0]
        df_final, df_rem2 = screen_delta_hr(df_peaks=df_final, peaks_colname=peaks_colname,
                                            ecg_signal=ecg_signal, sample_rate=sample_rate,
                                            delta_hr_thresh=delta_hr_thresh,
                                            correl_window_size=correl_window_size, correl_thresh=corr_thresh)

        df_rem2['loop'] = [loop_i] * df_rem2.shape[0]

        df_rem = pd.concat([df_rem, df_rem2])

        end_len = df_final.shape[0]
        keep_looping = (start_len2 != end_len) and (max_iters > loop_i)

    if not quiet:
        print(f"\nRemoved {start_len - end_len}/{start_len} beats with {loop_i} iteration{'s' if loop_i != 1 else ''}")

    df_final.reset_index(drop=True, inplace=True)
    df_rem.sort_values(peaks_colname, inplace=True)
    df_rem.reset_index(drop=True, inplace=True)

    return df_final, df_rem


def readd_highly_correlated_peaks(df_removed_beats, df_valid, sample_rate, ecg_signal, timestamps, correl_template,
                                  zncc_thresh=.7, window_size=.2, min_amp=200, plot_data=False):
    """ Goes through peaks that have been removed to check for more appropriate peaks to re-include in their
        vicinity. For each peak, a zero-normalized cross-correlation is run

        arguments:
        -df_removed_beats: df_rem output from run_deltahr_screen()
        -df_valid: df_out output from run_deltahr_screen(). This df is combined with the output of this algorithm,
                   checked for duplicate peaks, and sorted into chronological order.

        -ecg_signal: ECG signal to use (filtered)
        -sample_rate: of ecg_signal, Hz
        -timestamps: of ecg_signal

        -correl_template: average QRS template from PeakDetection.create_beat_template_snr_bouts()
        -zncc_thresh: threshold for ZNCC value for peak to be included

        -window_size: window size in seconds around each peak to look for a more highly-correlated peak
        -min_amp: minimum peak amplitude for re-inclusion

        returns:
        -dataframe that combines df_valid with re-added peaks.
    """

    add_peaks = []
    fs = int(sample_rate)

    for row in df_removed_beats.itertuples():

        # zncc between w and qrs template
        curr_window = ecg_signal[row.idx - fs:row.idx + fs]
        z = get_zncc(curr_window, correl_template)

        # zncc output padding
        z = np.insert(z, 0, np.zeros(int(fs / (1 / window_size))))

        # indexes of zncc output to check for other peaks
        check_z = z[int(fs - fs * window_size):int(fs + fs * window_size)]
        max_zi = np.argmax(check_z) + int(fs - fs * window_size)  # index of peak with highest correlation
        offset = int(max_zi - fs)  # peak index relative to initial peak
        check_val = z[max_zi]  # max zncc value
        height = abs(ecg_signal[row.idx + offset])  # peak amplitude at max ZNCC value

        if plot_data:
            fig, ax = plt.subplots(2, sharex='col')

            ax[0].plot(curr_window, label='current', color='black')
            ax[0].scatter(ecg.fs, curr_window[int(ecg.fs)], color='limegreen', label='current peak')
            ax[0].plot(np.arange(ecg.fs - np.argmax(correl_template),
                                 ecg.fs + len(correl_template) - np.argmax(correl_template)), correl_template,
                       color='red', label='template')
            ax[1].plot(z, color='blue', label='ZNCC')
            ax[1].scatter(ecg.fs, z[int(ecg.fs)], color='limegreen', label='current peak')
            ax[1].axvspan(xmin=int(fs - fs * window_size), xmax=int(fs + fs * window_size), ymin=0, ymax=1, color='grey',
                          alpha=.25, label='zncc search')
            ax[1].scatter(max_zi, z[max_zi], color='gold', s=30, marker='s', label='max zncc')
            ax[0].legend()
            ax[1].legend()

        # data to add to df: initial index, new_index, zncc value,
        #     boolean if zncc value higher than initial peak's zncc value and if value above threshold
        add_peaks.append([row.idx, row.idx + offset, z[fs], check_val, height,
                          check_val > z[fs] and check_val >= zncc_thresh and height >= min_amp])

    df_zncc = pd.DataFrame(add_peaks, columns=['idx', 'idx_corr', 'r_idx', 'r_idx_corr', 'height', 'valid'])
    df_zncc['start_time'] = timestamps[df_zncc['idx_corr']]

    df_readd = df_zncc.loc[df_zncc['valid']]
    df_readd = df_readd[['start_time', 'idx_corr']]
    df_readd.reset_index(drop=True, inplace=True)
    df_readd.columns = ['start_time', 'idx']

    df_not_readd = df_zncc.loc[~df_zncc['valid']]

    df_out = pd.concat([df_valid[['start_time', 'idx']], df_readd])
    df_out.drop_duplicates(inplace=True)
    df_out.sort_values("idx", inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    # beat-to-beat HR for new data
    df_out['hr'] = calculate_inst_hr(sample_rate=fs, df_peaks=df_out, peak_colname='idx',
                                     min_quality=3, max_break=3)

    df_out['height'] = ecg_signal[df_out['idx']]
    df_not_readd['height'] = ecg_signal[df_not_readd['idx']]

    return df_out, df_not_readd


def screen_peak_amplitudes(peaks, peak_heights, sample_rate, timestamps, ecg_signal,
                           abs_change=False, change_thresh=50, use_last_n_beats=3,
                           corr_thresh=.7, correl_window_size=.25):

    sample_rate = int(sample_rate)
    peaks = list(peaks)
    peak_heights = list(peak_heights)

    valid_height_idx = [False] * use_last_n_beats
    diffs = [None] * use_last_n_beats
    roll_means = [None] * (use_last_n_beats - 1)
    use_heights = peak_heights[:use_last_n_beats]
    r_vals = [None] * use_last_n_beats
    valid_rs = [False] * use_last_n_beats

    for idx in range(use_last_n_beats, len(peaks)):

        # average height of last use_last_n_peaks valid peaks
        prev_height = use_heights[-use_last_n_beats:]
        roll_mean_height = np.mean(prev_height)
        roll_means.append(roll_mean_height)

        curr_height = peak_heights[idx]  # height of current peak

        # difference in peak heights either as percent or absolute voltage
        diff = abs(curr_height - roll_mean_height) if abs_change else \
            abs((curr_height - roll_mean_height) * 100 / roll_mean_height)
        diffs.append(diff)

        valid_height = diff < change_thresh
        valid_height_idx.append(valid_height)

        r_valid = True

        if not valid_height:
            prev = window_beat(idx=peaks[valid_height_idx[-1]], ecg_signal=ecg_signal, sample_rate=sample_rate,
                               window_size=correl_window_size)
            curr = window_beat(idx=peaks[idx], ecg_signal=ecg_signal, sample_rate=sample_rate,
                               window_size=correl_window_size)

            r = pearsonr(curr, prev)[0]
            r_vals.append(r)

            r_valid = r >= corr_thresh
            valid_rs.append(r_valid)

        if valid_height:
            valid_rs.append(None)
            r_vals.append(None)

        if valid_height and r_valid:
            use_heights.append(curr_height)

    df_height = pd.DataFrame({'start_time': timestamps[peaks], 'idx': peaks,
                              'height': [round(i, 1) for i in peak_heights],
                              'diff': diffs, 'valid_height': valid_height_idx,
                              'r': r_vals, 'valid_r': valid_rs})

    df_height['valid'] = [False] * df_height.shape[0]
    df_height.loc[(~df_height['valid_height']) & (df_height['valid_r']), 'valid'] = True
    df_height.loc[df_height['valid_height'], 'valid'] = True

    df_valid = df_height.loc[df_height['valid']]
    # df_valid.drop('valid', axis=1, inplace=True)
    # df_valid.reset_index(drop=True, inplace=True)
    df_valid['hr'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_valid, peak_colname='idx',
                                       min_quality=3, max_break=3, quiet=True)

    df_invalid = df_height.loc[~df_height['valid']]
    # df_invalid.reset_index(drop=True, inplace=True)

    return df_valid, df_invalid


def readd_based_on_location(sample_rate, df_invalid, df_valid, ecg_signal, margin=.1, amplitude_thresh=500):
    margin_samples = int(margin * sample_rate)
    readd_idx = []
    row_idx = []

    for row in df_invalid.dropna().itertuples():
        last_beat = df_valid.loc[df_valid.index < row.Index].iloc[-1, :]
        next_beat = df_valid.loc[df_valid.index > row.Index].iloc[0, :]

        gap_window = (next_beat.start_time - last_beat.start_time).total_seconds()
        n_beats = next_beat.name - last_beat.name
        exp_ts = last_beat.start_time + timedelta(seconds=gap_window / n_beats * (row.Index - last_beat.name))

        error = abs((row.start_time - exp_ts).total_seconds())

        if error < margin:
            curr_amp = ecg_signal[row.idx]
            mean_amp = np.mean([ecg_signal[last_beat.idx], ecg_signal[next_beat.idx]])
            valid_amp = abs(curr_amp - mean_amp) < amplitude_thresh

            if valid_amp:
                readd_idx.append(row.idx - margin_samples +
                                 np.argmax(filt[row.idx - margin_samples:row.idx + margin_samples]))
                row_idx.append(row.Index)

    df_readd = df_invalid.loc[row_idx]
    df_final = pd.concat([df_readd, df_valid])
    df_final.sort_values('idx', inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    df_final['hr'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_final, peak_colname='idx',
                                       min_quality=3, max_break=3)

    return df_final, df_invalid.loc[[i not in row_idx for i in df_invalid.index]]


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

    df_epoch = pd.DataFrame({"start_time": ecg.ts[epoch_idxs], "idx": epoch_idxs, 'n_beats': beats, 'hr': hrs})

    return df_epoch


def plot_analysis_stages(ecg_signal, timestamps, subj_id, ds_ratio=1, df_snr=pd.DataFrame(),
                         data_dict=None, hr_plot_type='scatter', hr_markers=None):
    """ Function to plot ecg signal with peaks and HR from multiple stages of analysis and SNR data.

        arguments:
        -ecg_signal: arraylike
        -timestamps: of ecg_signal
        -subj_id: str for plot title
        -ds_ratio: downsample ratio applied to ecg_signal

        -df_snr: dataframe of SNR bouts. Bouts are plotted as horizontal lines spanning the bout
        -data_dict: dictionary containing data to plot. Each key is a str which becomes the data's label.
                    Each value is a list with the data in the following indexes:
                        -0: peaks dataframe
                        -1: peak column name in peaks dataframe ('idx' or 'idx_corr')
                        -2: colour for plotting this data
                        -3: scatterplot marker
                        -4: float/int by which scatterplot markers are raised relative to the peak value in ecg_signal
                        -5: boolean whether to use this dataframe to plot beat-to-beat HR

        -hr_plot_type: 'scatter' or 'line'

        returns:
        -figure
    """

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': [1, .66, .33]})
    plt.suptitle(f"{subj_id}")

    ax[0].plot(timestamps[:len(ecg_signal)][::ds_ratio], ecg_signal[::ds_ratio], color='black', zorder=0)

    offset = 500
    for key in data_dict.keys():
        key_data = data_dict[key]
        ax[0].scatter(timestamps[key_data[0][key_data[1]]], ecg_signal[key_data[0][key_data[1]]] + offset * key_data[4],
                      marker=key_data[3], color=key_data[2],
                      label=f"{key}\n(n={key_data[0].shape[0]})")

        if 'hr' in key_data[0].columns and key_data[5]:
            if hr_plot_type == 'scatter':
                ax[1].scatter(key_data[0]['start_time'], key_data[0]['hr'],
                              color=key_data[2], marker=key_data[3], label=key)
            if hr_plot_type == 'line':
                ax[1].plot(key_data[0]['start_time'], key_data[0]['hr'],
                           color=key_data[2], marker=key_data[3] if hr_markers is not None else None,
                           markerfacecolor=key_data[2], label=key)

    ax[0].legend(fontsize=8, loc='lower right')
    ax[1].legend(loc='lower right')
    ax[1].grid()
    ax[1].set_ylabel("HR")

    c = {0: 'grey', 1: 'limegreen', 2: 'dodgerblue', 3: 'red', 'ignore': 'grey', 'full': 'limegreen', 'HR': 'dodgerblue'}
    for row in df_snr.itertuples():
        ax[2].plot([row.start_timestamp, row.end_timestamp], [row.avg_snr, row.avg_snr], color=c[row.quality])

    ax[2].set_ylabel("SNR (dB)")

    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.%f"))
    plt.tight_layout()
    plt.subplots_adjust(hspace=.04)

    return fig


""" ===================================== SAMPLE RUN ===================================== """

# data import --------------
full_id = 'OND09_0005'
# full_id = 'OND09_0030'
thresholds = (5, 18)


ecg = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/", ecg_fname=f"{full_id}_01_BF36_Chest.edf",
          bandpass=(1.5, 25), thresholds=thresholds)

# data cropping for faster testing
end_idx = int(24*60*60*ecg.ecg.signal_headers[ecg.ecg.get_signal_index("ECG")]['sample_rate'])
# end_idx = len(ecg.ecg.signals[ecg.ecg.get_signal_index("ECG")])
ecg.ecg.signals[0] = ecg.ecg.signals[0][:end_idx]
ecg.ecg.filt = ecg.ecg.filt[:end_idx]
filt = ecg.ecg.filt.copy()

# nonwear data import
ecg.df_nw = import_nw_file(filepath=f"C:/Users/ksweber/Desktop/ECG_nonwear_dev/FinalBouts_SNR/{full_id}_01_BF36_Chest_NONWEAR.csv",
                           start_timestamp=ecg.start_stamp, sample_rate=ecg.fs)

# signal quality (SNR) data import -----------
# bouts appropriate for HR/rhythm analysis
ecg.df_snr = import_snr_bout_file(filepath=f"C:/Users/ksweber/Desktop/SNR_dev/Bouts/{full_id}_01_snr_bouts_hr.csv")
ecg.df_snr = ecg.df_snr.loc[ecg.df_snr['end_idx'] < end_idx]

# bouts appropriate for full analysis
ecg.df_snr_q1 = import_snr_bout_file(filepath=f"C:/Users/ksweber/Desktop/SNR_dev/Bouts/{full_id}_01_snr_bouts_fullanalysis.csv")
ecg.df_snr_q1 = ecg.df_snr_q1.loc[ecg.df_snr_q1['end_idx'] < end_idx]

ecg.df_snr_all = import_snr_bout_file(filepath=f"C:/Users/ksweber/Desktop/SNR_dev/Bouts/{full_id}_01_snr_bouts.csv")
ecg.df_snr_all = ecg.df_snr_all.loc[ecg.df_snr_all['end_idx'] < end_idx]

# dataframe of lower quality bouts, for use in peak rejection ------
try:
    df_snr_ignore = ecg.df_snr.loc[ecg.df_snr['quality_use'] > 2]
except KeyError:
    ecg.df_snr['quality'] = [int(i) for i in ecg.df_snr['quality']]
    df_snr_ignore = ecg.df_snr.loc[ecg.df_snr['quality'] > 2]

# initial peak detection -----------------
peaks = nk.ecg_peaks(ecg_cleaned=filt, sampling_rate=ecg.fs, method='neurokit', correct_artifacts=False)[1]['ECG_R_Peaks']
df_peaks1 = pd.DataFrame({'start_time': ecg.ts[peaks], 'idx': peaks, 'height': filt[peaks]})
correct_cn_peak_locations(df_peaks=df_peaks1, peaks_colname='idx', ecg_signal=filt, sample_rate=ecg.fs, window_size=.3, use_abs_peaks=False)

# removes very low amplitude peaks
df_peaks1 = df_peaks1.loc[df_peaks1['height'] >= 200]
df_peaks1.reset_index(drop=True, inplace=True)
df_peaks1['hr'] = calculate_inst_hr(sample_rate=ecg.fs, df_peaks=df_peaks1, peak_colname='idx', min_quality=3, max_break=3)

# removes peaks during df_snr_ignore and ecg.df_nw periods
df_peaks2, df_rem2 = remove_peaks_during_bouts(df_peaks=df_peaks1, dfs_events_to_remove=(df_snr_ignore, ecg.df_nw))
# recalculates beat-to-beat HR
df_peaks2['hr'] = calculate_inst_hr(sample_rate=ecg.fs, df_peaks=df_peaks2, peak_colname='idx', min_quality=3, max_break=3)

# run peak amplitude screen -----

# QRS template creation -------------
# creates average QRS template using only highest quality data (Q1, >60-sec durations)
df_snr_template = ecg.df_snr_q1.loc[(ecg.df_snr_q1['quality_use'] == 1) & (ecg.df_snr_q1['duration'] >= 60)].sort_values('avg_snr', ascending=False).reset_index(drop=True)

# creation of average QRS template in 20 highest quality SNR bouts > 60 seconds
qrs, qrs_n, all_beats = create_beat_template_snr_bouts(df_snr=df_snr_template.iloc[0:20],
                                                       ecg_signal=filt, plot_data=False,
                                                       sample_rate=ecg.fs, peaks=df_peaks2['idx'],
                                                       window_size=.5, remove_outlier_amp=True,
                                                       use_median=False, peak_align=.4, remove_mean=True)

# peak screening based on abrupt changes in HR -------
# was using delta_hr_thresh=30
df_peaks3, df_rem3 = run_deltahr_screen(df_peaks=df_peaks2, ecg_signal=filt, sample_rate=ecg.fs,
                                        peaks_colname='idx', corr_thresh=.5, correl_window_size=.2,
                                        use_correl_template=True, correl_template=qrs,
                                        absolute_hr=True, delta_hr_thresh=20, max_iters=5, quiet=False)

# Re-adds removed peaks that meet correlation criteria/tests for new peaks around removed peaks
# df_rem3a = df_rem3.loc[df_rem3['start_time'] >= pd.to_datetime('2021-11-09 22:17:43')].iloc[:1]
df_peaks4, df_peaks4_ignored = readd_highly_correlated_peaks(df_removed_beats=df_rem3, df_valid=df_peaks3,
                                                             ecg_signal=filt, sample_rate=ecg.fs, timestamps=ecg.ts,
                                                             correl_template=qrs, zncc_thresh=.5, window_size=.25,
                                                             min_amp=200, plot_data=False)

# runs this again after peaks were re-added
df_peaks5, df_rem5 = run_deltahr_screen(df_peaks=df_peaks4, ecg_signal=filt, sample_rate=ecg.fs,
                                        peaks_colname='idx', corr_thresh=.5, correl_window_size=.2,
                                        use_correl_template=True, correl_template=qrs,
                                        absolute_hr=True, delta_hr_thresh=15, max_iters=5, quiet=False)

df_peaks6, df_rem6 = screen_peak_amplitudes(peaks=df_peaks5['idx'], peak_heights=df_peaks5['height'],
                                            ecg_signal=filt, sample_rate=ecg.fs, timestamps=ecg.ts,
                                            correl_window_size=.25, corr_thresh=.7,
                                            abs_change=True, change_thresh=750,
                                            use_last_n_beats=3)

df_peaks7, df_peaks7_ignored = readd_based_on_location(ecg_signal=filt,sample_rate=ecg.fs,
                                                       df_invalid=df_rem6, df_valid=df_peaks6, margin=.1, amplitude_thresh=500)

df_peaks8, df_rem8 = run_deltahr_screen(df_peaks=df_peaks7, ecg_signal=filt, sample_rate=ecg.fs,
                                        peaks_colname='idx', corr_thresh=.5, correl_window_size=.2,
                                        use_correl_template=True, correl_template=qrs,
                                        absolute_hr=True, delta_hr_thresh=15, max_iters=5, quiet=False)

df_epoch = jumping_epoch_hr(sample_rate=ecg.fs, timestamps=ecg.ts[:len(filt)], epoch_len=15, peaks=df_peaks8['idx'])

# Plotting stages of analysis ---------
fig = plot_analysis_stages(ecg_signal=filt, timestamps=ecg.ts, subj_id=full_id, df_snr=ecg.df_snr,
                           ds_ratio=2, hr_plot_type='line', hr_markers=None,
                           data_dict={
                                      "original": [df_peaks1, 'idx', 'orange', 'v', 1, True],
                                      # 'snr/nw': [df_peaks2, 'idx', 'gold', 'v', 2, False],
                                      # 'snr/nw rem.': [df_rem2, 'idx', 'gold', 'x', 2, False],
                                      #'deltahr1': [df_peaks3, 'idx', 'dodgerblue', 'v', 3, True],
                                      #'deltahr1_rem': [df_rem3, 'idx', 'dodgerblue', 'x', 3, False],
                                      're-add_corr': [df_peaks4, 'idx', 'limegreen', 'v', 4, False],
                                      'not re-add_corr': [df_peaks4_ignored, 'idx', 'limegreen', 'x', 4, False],
                                      'deltahr2': [df_peaks5, 'idx', 'purple', 'v', 5, True],
                                      'deltahr2_rem': [df_rem5, 'idx', 'purple', 'x', 5, False],
                                      'peak_amp': [df_peaks6, 'idx', 'fuchsia', 'v', 6, True],
                                      'peak_amp_rem': [df_rem6, 'idx', 'fuchsia', 'x', 6, False],
                                      're_add_loc': [df_peaks7, 'idx', 'cyan', 'v', 7, True],
                                      're_add_loc_ign.': [df_peaks7_ignored, 'idx', 'cyan', 'x', 7, False],
                                      'deltahr3': [df_peaks8, 'idx', 'green', 'o', 8, True],
                                      'deltahr3_rem': [df_rem8, 'idx', 'green', 'x', 8, False],
                                      })

# working on readd_location() function
fig.axes[1].plot(df_epoch['start_time'], df_epoch['hr'], color='black')
