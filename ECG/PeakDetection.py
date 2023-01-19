from ECG.ImportFiles import *
import neurokit2 as nk
from ECG.HR_Calculation import *
import scipy.stats
from tqdm import tqdm
from ECG.Processing import get_zncc, find_snr_bouts


def detect_peaks_initial(ecg_signal: np.array or list or tuple,
                         sample_rate: float or int,
                         timestamps: np.array or list or tuple,
                         correct_locations: bool = True,
                         correction_windowsize: float or int = 0.3,
                         min_height: float or int or None = None,
                         absolute_peaks: bool = False,
                         neurokit_method: str = 'neurokit'):
    """ Runs ECG peak detection using specific parameters. Optional peak location correction based on local
        maxima/minima and minimum peak voltage height requirement.

        arguments:
        -ecg_signal: timeseries ECG signal
        -sample_rate: sample rate of ecg_signal, Hz
        -timestamps: timestamps for ecg_signal

        -correct_locations: boolean to correct peak locations. If True, will use values specified in
                            correction_windowsize and absolute_peaks to determine most appropriate peak location.
        -correction_windowsize: number of seconds in which the peak location correction function checks on either
                                side of the input peak
        -absolute_peaks: if True, runs peak location correction using absolute values instead of local maxima
        -min_height:
        -neurokit_method: peak detection method called by neurkit.ecg_peaks(method)

        returns:
        -df_peaks: dataframe containing peak information and timestamps

    """

    # neurokit peak detection
    peaks = nk.ecg_peaks(ecg_cleaned=ecg_signal, sampling_rate=sample_rate,
                         method=neurokit_method, correct_artifacts=False)[1]['ECG_R_Peaks']

    # output df
    df_peaks = pd.DataFrame({'start_time': timestamps[peaks], 'idx': peaks, 'height': ecg_signal[peaks]})

    if correct_locations:
        df_peaks = correct_peak_locations(df_peaks=df_peaks, peaks_colname='idx', ecg_signal=ecg_signal,
                                          sample_rate=sample_rate, window_size=correction_windowsize,
                                          use_abs_peaks=absolute_peaks)

    # removes peaks with amplitude < min_height if min_height given, else skips
    if min_height is not None:
        df_peaks = df_peaks.loc[df_peaks['height'] >= min_height]

    df_peaks.reset_index(drop=True, inplace=True)

    # calculates beat-to-beat HR on original indexes or corrected indexes
    df_peaks['hr'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_peaks,
                                       peak_colname='idx' if not correct_locations else 'idx_corr',
                                       min_quality=3,  max_break=3)

    return df_peaks


def create_beat_template_snr_bouts(df_snr: pd.DataFrame,
                                   ecg_signal: np.array or list or tuple,
                                   sample_rate: int or float,
                                   peaks: np.array or list or tuple or pd.Series,
                                   window_size: float = 0.2,
                                   remove_outlier_amp: bool = False,
                                   remove_mean: bool = True,
                                   use_median: bool = False,
                                   plot_data: bool = False,
                                   quiet: bool = True):
    """From sections of data specified in df_snr, calculates a 'beat template' of the average heartbeat by
       averaging windows on either side of each given peak.

        arguments:
        -df_snr: output from create_snr_bouts(). RECOMMENDED to crop to only a few high-quality segments of data.
        -ecg_signal: timeseries ECG signal
        -sample_rate: sample rate of ecg_signal, Hz
        -peaks: indexes of peaks corresponding to ecg_signal

        -window_size: window size of windows and beat template, in seconds
        -remove_outlier_amp: boolean whether to remove peaks whose amplitude is deemed an outlier
                             for the template generation (median + 3*IQR)
        -remove_mean: if True, removes mean from each beat window to center at 0
        -use_median: method for template creation. If True, takes median datapoint at each index. If False, uses mean

        -plot_data: boolean --> slow function if True
            -Plots segments of data specified in df_snr with gaps between segments (time series) on top subplot
             and all heartbeats with the template on the bottom subplot

        -quiet: whether to print processing progress to console

         returns:
        -qrs: QRS template using specified parameters
        -qrs_norm: QRS template using specified paramters, normalized to have a voltage range of 1
        -qrs_crop: QRS template cropped to contain a smaller window
        -all_beats: list of lists of each beat window used in template creation
    """

    if not quiet:
        print(f"\nComputing QRS template using {df_snr.shape[0]} high-quality data segments "
              f"totaling {int(df_snr['duration'].sum())} seconds of data")
        print(f"-Using a window size of +- {window_size} seconds and "
              f"{'not' if not remove_outlier_amp else ''}removing large amplitude peaks")

    # alignment indexes based on peak_align and window_size
    win_idx = int(sample_rate * window_size)

    if plot_data:
        fig, ax = plt.subplots(2, figsize=(12, 8))

    peaks = np.array(peaks)
    used_peaks_idx = []
    all_beats = []

    # adds all peaks to used_peaks as a list for each signal quality bout
    for row in df_snr.itertuples():
        # only peaks contained in current event in df_snr
        template_peaks = peaks[(peaks > row.start_idx) & (peaks < row.end_idx)]

        if plot_data:
            t = np.arange(row.start_idx, row.end_idx)
            ax[0].plot(t / sample_rate, ecg_signal[row.start_idx:row.end_idx], color='black')

        for j in template_peaks:
            used_peaks_idx.append(j)

    med_peak_amp = np.median(ecg_signal[used_peaks_idx])  # median peak amplitude

    # stats for outlier detection. Outlier = median + 3*IQR
    q3 = np.percentile(ecg_signal[used_peaks_idx], 75)
    q1 = np.percentile(ecg_signal[used_peaks_idx], 25)
    outlier_peak_max = sorted([med_peak_amp + 3 * (q3 - q1), med_peak_amp - 3 * (q3 - q1)])

    # all used peaks
    final_peaks = used_peaks_idx

    # only inlier peaks
    if remove_outlier_amp:
        # final_peaks = [i for i in used_peaks_idx if abs(ecg_signal[i]) <= outlier_peak_max]
        final_peaks = [i for i in used_peaks_idx if outlier_peak_max[0] <= ecg_signal[i] <= outlier_peak_max[1]]

    # specified processing and addition to output list for each peak
    for peak in final_peaks:
        # data window for each peak
        w = ecg_signal[peak - win_idx:peak + win_idx]

        # removes mean from window to center at 0
        if remove_mean:
            mean_w = np.mean(w)
            w = [i - mean_w for i in w]

        all_beats.append(w)

    c = ['dodgerblue' if outlier_peak_max[0] <= ecg_signal[i] <= outlier_peak_max[1] else 'red' for i in used_peaks_idx]

    if plot_data:
        ax[0].scatter([i / sample_rate for i in used_peaks_idx], ecg_signal[used_peaks_idx], color=c)

    if not quiet:
        print(f"-Template being generated from {len(all_beats)} heartbeats")

    if not use_median:
        qrs = np.mean(np.array(all_beats), axis=0)
    if use_median:
        qrs = np.median(np.array(all_beats), axis=0)

    if plot_data:

        for i, beat in enumerate(all_beats):
            if i > 0:
                ax[1].plot(np.linspace(-win_idx/sample_rate, win_idx/sample_rate, len(beat)), beat, color='black')
            if i == 0:
                ax[1].plot(np.linspace(-win_idx/sample_rate, win_idx/sample_rate, len(beat)), beat, color='black',
                           label=f'AllBeats (n={len(all_beats)})')

        ax[1].plot(np.linspace(-win_idx/sample_rate, win_idx/sample_rate, len(qrs)), qrs,
                   color='red', lw=2, label='Template')
        ax[1].axvline(0, color='grey', linestyle='dashed', label='Peaks')
        ax[1].legend()
        ax[1].set_xlabel("Seconds")

        ax[1].grid()
        plt.tight_layout()

    # Normalizing QRS to have a voltage range of 1
    qrs_min = min(qrs)
    qrs_max = max(qrs)

    qrs_norm = [(i - qrs_min) / (qrs_max - qrs_min) for i in qrs]
    mean_val = np.mean(qrs_norm)
    qrs_norm = [i - mean_val for i in qrs_norm]

    qrs_crop = crop_template(template=qrs, sample_rate=sample_rate, window_size=window_size/2)

    return qrs, qrs_norm, qrs_crop, all_beats


def correct_peak_locations(df_peaks: pd.DataFrame,
                           peaks_colname: str,
                           ecg_signal: np.array or list or tuple,
                           sample_rate: float or int,
                           window_size: float or int = 0.15,
                           use_abs_peaks: bool = False,
                           quiet: bool = True):
    """Adjusts peak indexes to correspond to highest amplitude value in window surrounding each beat.

        arguments:
        -df_peaks: dataframe containing peaks, timestamps, etc.
        -peaks_colname: column name in df_peaks corresponding to peak indexes
        -ecg_signal: timeseries ECG signal
        -sample_rate: sample rate of ecg_signal, Hz
        -window_size: number of seconds included in window on either side of each peak that is checked for
                      more appropriate peak location
        -use_abs_peaks: if True, uses absolute values. If False, only looks at positive values for possible peaks.
        -quiet: whether to print processing progress to console

        returns:
        -copy of df_peaks with new 'idx_corr' column (correct indexes for peaks)
    """

    if not quiet:
        print(f"\nAdjusting peak locations using a window size of +- {window_size} seconds and "
              f"{'positive' if not use_abs_peaks else 'largest amplitude'} peaks...")

    df_out = df_peaks.copy()

    peaks = []

    # Loops through peaks
    for peak in list(df_peaks[peaks_colname]):

        # segment of data: peak +- window_size duration
        window = ecg_signal[int(peak - window_size * sample_rate):int(peak + window_size * sample_rate)]

        # index of largest value/absolute value in the window
        if use_abs_peaks:
            p = np.argmax(np.abs(window))
        if not use_abs_peaks:
            p = np.argmax(window)

        # converts window's index to whole collection's index
        peaks.append(p + peak - int(window_size * sample_rate))

    # difference in seconds between peak and corrected peak
    diff = [(i - j) / sample_rate for i, j in zip(peaks, list(df_peaks[peaks_colname]))]

    df_out['idx_corr'] = peaks
    df_out['idx_diff'] = diff

    # drops duplicates in case two beats get corrected to the same location (unlikely)
    df_out.drop_duplicates(subset='idx_corr', keep='first', inplace=True)
    df_out = df_out.reset_index(drop=True)

    return df_out


"""========================================= NOT CHECKED ========================================="""


def screen_peaks_corr(ecg_signal, df_peaks, peaks_colname, qrs_temp, corr_thresh=-1.0, drop_invalid=False):
    """Calculates correlation between QRS template and each beat. Deems peaks as valid/invalid based on
       given correlation threshold.

        arguments:
        -ecg_signal: array on which the QRS template was generated
        -peaks: peak indexes corresponding to ecg_signal
        -sample_rate: Hz, of ecg_signal
        -qrs_temp: QRS template array
        -timestamps: of ecg_signal
        -window_size: number of seconds that is windowed on each side of each beat in the correlation.
        -corr_threshold: value between -1 and 1. Peaks with correlation < threshold are deemed invalid.
            -To not reject any peaks, set to -1

        returns:
        -df
    """

    print(f"\nAnalyzing detection peaks using a template correlation threshold of r={corr_thresh}...")

    """ ============= SET-UP ==========="""
    df_out = df_peaks.copy()

    if len(qrs_temp) % 2 == 1:
        qrs_temp = np.append(qrs_temp, qrs_temp[-1])

    # win_size = int(len(qrs_temp) / 2)
    r_vals = []
    peaks = list(df_peaks[peaks_colname])

    pre_idx = np.argmax(qrs_temp)
    post_idx = len(qrs_temp) - pre_idx

    """============= LOOPING THROUGH BEATS ========== """

    """ finds first beat that exceeds correlation threshold --> first good beat """
    for peak in tqdm(peaks):

        # curr_beat = ecg_signal[peak - win_size:peak + win_size]  # window for current beat
        curr_beat = ecg_signal[peak - pre_idx:peak + post_idx]  # window for current beat

        i = min([len(curr_beat), len(qrs_temp)])
        r = scipy.stats.pearsonr(curr_beat[:i], qrs_temp[:i])
        r_vals.append(r[0])

    df_out['r'] = r_vals
    df_out['valid'] = df_out['r'] > corr_thresh

    # Removes any potential duplicate beats due to something I can't figure out
    df_out = df_out.drop_duplicates(subset=peaks_colname, keep='first', inplace=False).reset_index(drop=True)

    print(f"\nSUMMARY:")
    print("-{}/{} beats did not meet correlation threshold of {}".format(df_out['valid'].value_counts()[False],
                                                                         df_out.shape[0], corr_thresh))
    print("-{} beats remain ({}% of input beats)".format(df_out['valid'].value_counts()[True],
                                                         round(df_out['valid'].value_counts()[True]/
                                                               df_out.shape[0]*100, 1)))

    if drop_invalid:
        print("Dropping invalid beats.")
        df_out = df_peaks.loc[df_peaks['valid']].reset_index(drop=True)

    return df_out


def run_zncc_method(input_data, template, min_dist, timestamps, snr, sample_rate=250, downsample=2, zncc_thresh=.7,
                    show_plot=False, thresholds=(5, 20), epoch_len=30, min_quality=2):

    # Runs zero-normalized cross correlation
    correl = get_zncc(x=template, y=input_data)

    print("\nDetecting peaks in ZNCC signal...")
    c_peaks = peakutils.indexes(y=correl, thres_abs=True, thres=zncc_thresh, min_dist=min_dist)
    r = correl[c_peaks]

    print(f"-Found {len(c_peaks)} heartbeats")

    if show_plot:
        print("\nGenerating plot...")

        fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))

        axes[0].plot(np.arange(len(input_data))[::downsample] / sample_rate, input_data[::downsample],
                     color='black', label="Filtered")

        axes[0].legend(loc='lower left')
        axes[0].set_title("Filtered data with overlayed template on peaks")
        axes[0].set_ylabel("Voltage")

        x = np.arange(len(template) / 2, len(correl) + len(template) / 2) / 250
        axes[1].plot(x[::downsample], correl[::downsample], color='dodgerblue', label="ZNCC")
        axes[1].scatter(x[c_peaks], [correl[i] * 1.1 for i in c_peaks], marker="v", color='red', label="ZNCCPeaks")
        axes[1].axhline(y=zncc_thresh, color='limegreen', linestyle='dotted', label="ZNCC_Thresh")
        axes[1].legend(loc='lower left')
        axes[1].set_ylabel("ZNCC")
        axes[1].set_xlabel("Seconds")

        plt.tight_layout()

    df_out = pd.DataFrame({"timestamp": timestamps[c_peaks], 'idx': c_peaks, 'r': r})

    df_out = calculate_beat_snr(df_peaks=df_out, peak_colname='idx', snr_data=snr,
                                thresholds=thresholds, sample_rate=sample_rate, window_width=1)

    """Beat-to-beat HR calculations"""
    df_out['inst_hr_q1'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_out, max_break=3, min_quality=1)
    df_out['inst_hr_q2'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_out, max_break=3, min_quality=2)
    df_out['inst_hr_q3'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_out, max_break=3, min_quality=3)

    """Epoching HR in jumping windows. Excludes peaks below given quality requirement"""
    df_epoch = calculate_epoch_hr_jumping(df_peaks=df_out, min_quality=min_quality, epoch_len=epoch_len,
                                          ecg_signal=input_data, ecg_timestamps=timestamps, sample_rate=sample_rate)

    return df_out, df_epoch, correl

