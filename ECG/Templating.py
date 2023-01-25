import numpy as np
import scipy.stats

""" ===== CHECKED ===== """


def find_first_highly_correlated_beat_notemplate(ecg_signal: list or np.array,
                                                 peaks: list or np.array,
                                                 sample_rate: int or float = 250,
                                                 correl_window_size: float = .2,
                                                 correl_thresh: float = .7,
                                                 n_consec: int = 5):
    """Looks for first peak that matches correlation to template criteria.
       Requires a streak of n_consec consecutive peaks that meet correlation threshold.

        arguments:
        -ecg_signal: timeseries ECG signal
        -sample_rate: of ecg_signal, in Hz
        -sample_rate: of ecg_signal, Hz
        -peaks: array-like of peak indexes corresponding to ecg_signal
        -correl_thresh: Pearson correlation threshold required for "valid" beat
        -n_consec: number of consecutive beats above correlation threshold

        returns:
        -index of first beat in the sequence of n_consecutive beats that exceed correlation threshold
    """

    window_samples = int(correl_window_size * sample_rate)

    beats = []
    for peak in peaks[:n_consec]:
        if peak - window_samples >= 0 and peak + window_samples < len(ecg_signal):
            beats.append(ecg_signal[peak - window_samples:peak + window_samples])

    template = np.mean(np.array(beats), axis=0)

    all_r = []
    for idx, peak in enumerate(peaks[n_consec:]):
        if peak + window_samples < len(ecg_signal):
            window = ecg_signal[peak - window_samples:peak + window_samples]
            r = scipy.stats.pearsonr(window, template)[0]

            all_r.append(r >= correl_thresh)

            if len(all_r) >= n_consec:
                if sum(all_r[-n_consec:]) == n_consec:
                    return idx, template, beats

            if r < correl_thresh:
                beats = beats[1:]
                beats.append(window)
                template = np.mean(np.array(beats), axis=0)

    return 0, []


def crop_template(template: np.array or list, sample_rate: int or float,
                  window_size: int or float, centre_on_absmax_peak: bool = False):
    """Crops QRS template to shorter length for use in other analyses.

        arguments:
        -template: output from Templating.find_first_highly_correlated_beat_notemplate()
        -sample_rate: of ECG signal used to generate template, in Hz
        -window_size: duration on either side of each peak that is used in the window, in seconds
        -centre_on_absmax_peak: if True, the windowed is centered on the templates's maximum absolute value. If False,
                                left as-is.

        returns:
        -cropped QRS template
    """

    # ensures template length matches specified window size
    if centre_on_absmax_peak:
        max_i = np.argmax(abs(template))

    if not centre_on_absmax_peak:
        max_i = int(len(template)/2)

    pad_i = int(sample_rate * window_size)
    template = template[int(max_i - pad_i) if int(max_i - pad_i) >= 0 else 0:int(max_i + pad_i)]

    return template


""" ===== NOT CHECKED ==== """
