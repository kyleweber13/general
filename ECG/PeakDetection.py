import ECG.Processing as Processing
import ECG.HR_Calculation as HR_Calculation
import ECG.PeakScreening as PeakScreening

import neurokit2 as nk
import numpy as np
import pandas as pd

""" ===== CHECKED ===== """

""" ===== NOT CHECKED ===== """


def detect_peaks_initial(ecg_signal: np.array or list or tuple,
                         sample_rate: float or int,
                         timestamps: np.array or list or tuple,
                         correct_locations: bool = True,
                         correction_windowsize: float or int = 0.3,
                         min_height: float or int or None = None,
                         absolute_peaks: bool = False,
                         use_correlation: bool = True,
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
        -use_correlation: if True, only changes peak location if new peak is more highly-correlated with the previous
                          peak than the original peak
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
    df_peaks_corr = None

    if correct_locations:
        df_peaks_corr = PeakScreening.correct_peak_locations(df_peaks=df_peaks,
                                                             peaks_colname='idx',
                                                             ecg_signal=ecg_signal,
                                                             sample_rate=sample_rate,
                                                             window_size=correction_windowsize,
                                                             use_abs_peaks=absolute_peaks,
                                                             use_correlation=use_correlation,
                                                             timestamps=timestamps)

        df_use = df_peaks_corr

    # removes peaks with amplitude < min_height if min_height given, else skips
    if min_height is not None:
        df_use = df_use.loc[df_use['height'] >= min_height]

    df_use.reset_index(drop=True, inplace=True)

    # calculates beat-to-beat HR on original indexes or corrected indexes
    df_peaks['hr'] = HR_Calculation.calculate_inst_hr(sample_rate=sample_rate,
                                                      df_peaks=df_peaks,
                                                      peak_colname='idx',
                                                      min_quality=3,
                                                      max_break=3)

    df_use['hr'] = HR_Calculation.calculate_inst_hr(sample_rate=sample_rate,
                                                    df_peaks=df_use,
                                                    peak_colname='idx' if not correct_locations else 'idx_corr',
                                                    min_quality=3,
                                                    max_break=3)

    return df_peaks, df_peaks_corr


def create_beat_template_snr_bouts(df_snr: pd.DataFrame,
                                   ecg_signal: np.array or list or tuple,
                                   sample_rate: int or float,
                                   peaks: np.array or list or tuple or pd.Series,
                                   window_size: float = 0.2,
                                   remove_outlier_amp: bool = False,
                                   remove_mean: bool = True,
                                   use_median: bool = False,
                                   plot_data: bool = False,
                                   centre_on_absmax_peak: bool = False,
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

    qrs_crop = Processing.crop_template(template=qrs,
                                        sample_rate=sample_rate,
                                        window_size=window_size/2,
                                        centre_on_absmax_peak=centre_on_absmax_peak)

    return qrs, qrs_norm, qrs_crop, all_beats
