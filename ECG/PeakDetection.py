from ECG.ImportFiles import *
import neurokit2 as nk
from ECG.HR_Calculation import *
import scipy.stats
from tqdm import tqdm
from ECG.Processing import get_zncc, find_snr_bouts


def run_cardiacnavigator_method(cn_file, ecg_obj, df_snr, sample_rate, timestamps,
                                n_hq_snr_bouts=5, thresholds=(5, 20), version_key_epoching='v4',
                                corr_thresh=.5, window_size=.33, epoch_len=30):

    """ Cardiac Navigator data """
    df_peaks_cn_orig = import_cn_beat_file(filename=cn_file, start_time=ecg_obj.start_stamp, sample_rate=ecg_obj.fs)
    df_peaks_cn = df_peaks_cn_orig.copy()[['timestamp', 'idx', 'rate']]

    """ corrects CN peaks to local maxima on R peak """
    df_peaks2 = correct_cn_peak_locations(df_peaks=df_peaks_cn, peaks_colname='idx',
                                          ecg_signal=ecg_obj.filt, sample_rate=ecg_obj.fs,
                                          window_size=window_size, use_abs_peaks=True)

    snr_thresh = round(np.percentile(ecg_obj.snr, 90), 1)
    df_snr_template = find_snr_bouts(df_snr=df_snr, min_snr=snr_thresh,
                                     min_duration=60, n_bouts=n_hq_snr_bouts, min_total_minutes=60)

    # df_snr_template = df_snr.loc[(df_snr['avg_snr'] >= snr_thresh) & (df_snr['duration'] >= 60)].sort_values('duration', ascending=False).reset_index(drop=True)

    qrs_template = create_beat_template_snr_bouts(df_snr=df_snr_template, ecg_signal=ecg_obj.filt,
                                                  sample_rate=ecg_obj.fs, peaks=df_peaks2['idx_corr'], plot_data=True,
                                                  window_size=window_size, remove_outlier_amp=True, use_median=False,
                                                  peak_align=.4, remove_mean=True)

    qrs_data = {'avg': qrs_template[0], 'avg_norm': qrs_template[1], 'beats': qrs_template[2]}

    """Removal of peaks based on QRS template"""
    df_peaks3 = screen_peaks_corr(ecg_signal=ecg_obj.filt, df_peaks=df_peaks2, peaks_colname='idx_corr',
                                  qrs_temp=qrs_data['avg'], corr_thresh=corr_thresh, drop_invalid=False)

    df_peaks3 = calculate_beat_snr(df_peaks=df_peaks3, peak_colname='idx_corr', snr_data=ecg_obj.snr,
                                   thresholds=thresholds, sample_rate=ecg_obj.fs, window_width=1)

    """Beat-to-beat HR calculations"""
    df_peaks3['inst_hr_q1'] = calculate_inst_hr(sample_rate=ecg_obj.fs, df_peaks=df_peaks3, max_break=3, min_quality=1)
    df_peaks3['inst_hr_q2'] = calculate_inst_hr(sample_rate=ecg_obj.fs, df_peaks=df_peaks3, max_break=3, min_quality=2)
    df_peaks3['inst_hr_q3'] = calculate_inst_hr(sample_rate=ecg_obj.fs, df_peaks=df_peaks3, max_break=3, min_quality=3)

    output_dict = {'cn': df_peaks_cn, 'v2': df_peaks2, 'v3': df_peaks3}

    df_peaks4 = filter_rr(df_peaks=df_peaks3, peaks_colname='idx_corr', timestamps=timestamps, threshold=30,
                          sample_rate=sample_rate, max_iters=10, plot_hr=False)

    output_dict['v4'] = df_peaks4

    """Epoching HR in jumping windows. Excludes peaks below given quality requirement"""
    cn_epoch = calculate_epoch_hr_jumping(df_peaks=output_dict[version_key_epoching], min_quality=2, epoch_len=epoch_len,
                                          ecg_signal=ecg_obj.filt, ecg_timestamps=ecg_obj.ts, sample_rate=ecg_obj.fs)
    output_dict['epoch'] = cn_epoch

    return output_dict, qrs_data


def run_neurokit_method(ecg_obj, df_snr, window_size=.33, n_hq_snr_bouts=5,
                        corr_thresh=.5, epoch_len=30, thresholds=(5, 20)):

    print("\nRunning condensed version!!!")
    """ NeuroKit peak data """
    df_peaks_nk = create_df_peaks(timestamps=ecg_obj.ts, peaks=nk.ecg_findpeaks(ecg_cleaned=ecg_obj.filt,
                                                                                sampling_rate=ecg_obj.fs,
                                                                                method='neurokit',
                                                                                show=False)['ECG_R_Peaks'])

    df_peaks_nk2 = correct_cn_peak_locations_centre(df_peaks=df_peaks_nk, peaks_colname='idx', signal=ecg_obj.filt,
                                                    sample_rate=ecg_obj.fs, window_size=window_size)

    df_snr_template = df_snr.loc[df_snr['quality'] == 1].sort_values('duration', ascending=False).reset_index(drop=True)

    qrs, qrs_n, all_beats = create_beat_template_snr_bouts(df_snr=df_snr_template.iloc[0:n_hq_snr_bouts], ecg_signal=ecg_obj.filt,
                                                           sample_rate=ecg_obj.fs, peaks=df_peaks_nk2['idx'], plot_data=False,
                                                           window_size=window_size, remove_outlier_amp=True,
                                                           use_median=False, peak_align=.4, remove_mean=True)

    """Removal of peaks based on QRS template"""
    # df_peaks_nk3 = screen_peaks_corr(ecg_signal=ecg_obj.filt, df_peaks=df_peaks_nk2, peaks_colname='idx', qrs_temp=qrs, corr_thresh=corr_thresh, drop_invalid=False)
    df_peaks_nk3 = df_peaks_nk2.copy()  # remove me

    df_peaks_nk3 = calculate_beat_snr(df_peaks=df_peaks_nk3, peak_colname='idx', snr_data=ecg_obj.snr,
                                      thresholds=thresholds, sample_rate=ecg_obj.fs, window_width=1)

    df_peaks_nk3['inst_hr_q1'] = calculate_inst_hr(sample_rate=ecg_obj.fs, df_peaks=df_peaks_nk3,
                                                   max_break=3, min_quality=1)
    df_peaks_nk3['inst_hr_q2'] = calculate_inst_hr(sample_rate=ecg_obj.fs, df_peaks=df_peaks_nk3,
                                                   max_break=3, min_quality=2)
    df_peaks_nk3['inst_hr_q3'] = calculate_inst_hr(sample_rate=ecg_obj.fs, df_peaks=df_peaks_nk3,
                                                   max_break=3, min_quality=3)

    """Epoching HR in jumping windows. Excludes peaks below given quality requirement"""
    cn_epoch = calculate_epoch_hr_jumping(df_peaks=df_peaks_nk3, min_quality=2, epoch_len=epoch_len,
                                          ecg_signal=ecg_obj.filt, ecg_timestamps=ecg_obj.ts, sample_rate=ecg_obj.fs)

    return df_peaks_nk3, cn_epoch, qrs


def correct_cn_peak_locations(df_peaks, peaks_colname, ecg_signal, sample_rate, window_size=.3, use_abs_peaks=False):
    """Adjusts peak indexes to correspond to highest absolute value in window surrounding each beat.

        arguments:
        -peaks_idx: array
        -ecg_signal: signal against which peaks_idx are checked
        -sample_rate: of ecg_signal
        -window_size: beat +- window_size is considered for peak adjustment, seconds
        -use_abs_peaks: boolean. If True, peaks are corrected to index of highest absolute value. If False,
                peak relocation is limited to positive deflections. If True, may cause issues if positive and
                negative R-peak deflections are similar amplitude as it will sometimes use the positive peak and
                other times, the negative
    """

    print("\nAdjusting peak locations using a "
          "window size of +- {} seconds and {} peaks...".format(window_size,
                                                                "positive" if not use_abs_peaks else "largest amplitude"))

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

    df_out.drop_duplicates(subset='idx_corr', keep='first', inplace=True)
    df_out = df_out.reset_index(drop=True)

    return df_out


def correct_cn_peak_locations_centre(df_peaks, peaks_colname, sample_rate, signal, window_size=.3, use_abs_peaks=False):
    """Adjusts peak indexes to correspond to index of the midpoint between the local maxima and minima
       in window surrounding each beat.

        arguments:
        -peaks_idx: array
        -ecg_signal: signal against which peaks_idx are checked
        -sample_rate: of ecg_signal
        -window_size: beat +- window_size is considered for peak adjustment, seconds
        -use_abs_peaks: boolean. If True, peaks are corrected to index of highest absolute value. If False,
                peak relocation is limited to positive deflections. If True, may cause issues if positive and
                negative R-peak deflections are similar amplitude as it will sometimes use the positive peak and
                other times, the negative
    """

    print("\nAdjusting peak locations using a "
          "window size of +- {} seconds and {} peaks...".format(window_size,
                                                                "positive" if not use_abs_peaks else "largest amplitude"))

    df_out = df_peaks.copy()

    peaks = list(df_peaks[peaks_colname])
    maxes = []
    mins = []

    for peak in peaks:
        window_start = int(peak - window_size * sample_rate)
        window_end = int(peak + window_size * sample_rate)
        window = signal[window_start:window_end]

        loc_max = np.argmax(window) + window_start
        maxes.append(loc_max)

        loc_min = np.argmin(window) + window_start
        mins.append(loc_min)

    mids = [int(np.mean([i, j])) for i, j in zip(maxes, mins)]

    diff = [(i - j) / sample_rate for i, j in zip(peaks, mids)]

    df_out['idx_corr2'] = mids
    df_out['diff2'] = diff

    df_out.drop_duplicates(subset='idx_corr2', keep='first', inplace=True)
    df_out = df_out.reset_index(drop=True)

    return df_out


def create_beat_template_snr_bouts(df_snr, ecg_signal, sample_rate, peaks, remove_outlier_amp=False,
                                   window_size=.5, plot_data=False, use_median=False,
                                   peak_align=.5, remove_mean=True):
    """From sections of data specified in df_snr, calculates a 'beat template' of the average heartbeat by
       averaging +- 250ms windows on either side of each given peak..

        arguments:
        -df_snr: output from create_snr_bouts(). RECOMMENDED to crop to only a few high-quality segments of data.
        -ecg_signal: array
        -sample_rate: Hz, of ecg_signal
        -peaks: indexes of peaks corresponding to ecg_signal
        -plot_data: boolean --> slow function if True
            -Plots segments of data specified in df_snr with gaps between segments (time series) on top subplot
             and all heartbeats with the template on the bottom subplot
        -peak_align: float from .0 to 1.0 to determine where in the template window the peak is located
                -e.g.: .5 means centered, .4 means the peak is 40% of the way in

        returns:
        -QRS template
        -mean peak amplitude
        -SD of peak amplitudes
    """

    print("\nComputing QRS template using {} high-quality "
          "data segments totaling {} seconds of data".format(df_snr.shape[0], int(df_snr['duration'].sum())))
    print("-Using a window size of {} seconds and {}"
          "removing large amplitude peaks".format(window_size, "not " if not remove_outlier_amp else ""))
    print(f"-Peak alignment set to {int(peak_align * 100)}%")

    pre_idx = int(sample_rate * window_size * peak_align)
    post_idx = int(sample_rate * window_size * (1-peak_align))

    if plot_data:
        fig, ax = plt.subplots(2, figsize=(12, 8))

    used_peaks_idx = []
    all_beats = []

    # adds all peaks to used_peaks as a list for each signal quality bout
    for row in df_snr.itertuples():
        template_peaks = [i for i in peaks if row.start_idx < i < row.end_idx]

        if plot_data:
            t = np.arange(row.start_idx, row.end_idx)
            ax[0].plot(t / sample_rate, ecg_signal[row.start_idx:row.end_idx], color='black')

        for j in template_peaks:
            used_peaks_idx.append(j)

    med_peak_amp = np.median(ecg_signal[used_peaks_idx])
    q3 = np.percentile(ecg_signal[used_peaks_idx], 75)
    q1 = np.percentile(ecg_signal[used_peaks_idx], 25)
    outlier_peak_max = med_peak_amp + 3 * (q3 - q1)

    if remove_outlier_amp:
        final_peaks = [i for i in used_peaks_idx if ecg_signal[i] <= outlier_peak_max]
    if not remove_outlier_amp:
        final_peaks = used_peaks_idx

    for peak in final_peaks:
        w = ecg_signal[peak - pre_idx:peak + post_idx]

        if remove_mean:
            mean_w = np.mean(w)
            w = [i - mean_w for i in w]

        all_beats.append(w)

    c = ['dodgerblue' if ecg_signal[i] <= outlier_peak_max else 'red' for i in used_peaks_idx]

    if plot_data:
        ax[0].scatter([i / sample_rate for i in used_peaks_idx], ecg_signal[used_peaks_idx], color=c)

    if not use_median:
        qrs = np.mean(np.array(all_beats), axis=0)
    if use_median:
        qrs = np.median(np.array(all_beats), axis=0)

    print(f"-Template being generated from {len(all_beats)} heartbeats")

    if plot_data:

        for i, beat in enumerate(all_beats):
            if i > 0:
                ax[1].plot(np.linspace(-pre_idx/sample_rate, post_idx/sample_rate, len(beat)), beat, color='black')
            if i == 0:
                ax[1].plot(np.linspace(-pre_idx/sample_rate, post_idx/sample_rate, len(beat)), beat, color='black',
                           label=f'AllBeats (n={len(all_beats)})')

        ax[1].plot(np.linspace(-pre_idx/sample_rate, post_idx/sample_rate, len(qrs)), qrs, color='red', lw=2, label='Template')
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

    qrs_crop = crop_template(template=qrs, sample_rate=sample_rate, window_size=.2)

    return qrs, qrs_norm, qrs_crop


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


def detect_peaks_initial(ecg_signal, sample_rate, timestamps, correct_locations=True, min_height=None,
                         correction_windowsize=.3, absolute_peaks=False, neurokit_method='neurokit'):

    peaks = nk.ecg_peaks(ecg_cleaned=ecg_signal, sampling_rate=sample_rate,
                         method=neurokit_method, correct_artifacts=False)[1]['ECG_R_Peaks']

    df_peaks = pd.DataFrame({'start_time': timestamps[peaks], 'idx': peaks, 'height': ecg_signal[peaks]})

    if correct_locations:
        df_peaks = correct_cn_peak_locations(df_peaks=df_peaks, peaks_colname='idx', ecg_signal=ecg_signal,
                                             sample_rate=sample_rate, window_size=correction_windowsize,
                                             use_abs_peaks=absolute_peaks)

    if min_height is not None:
        df_peaks = df_peaks.loc[df_peaks['height'] >= min_height]

    df_peaks.reset_index(drop=True, inplace=True)
    df_peaks['hr'] = calculate_inst_hr(sample_rate=sample_rate, df_peaks=df_peaks, peak_colname='idx',
                                       min_quality=3,  max_break=3)

    return df_peaks
