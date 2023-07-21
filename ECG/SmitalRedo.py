import neurokit2 as nk
import numpy as np
import pandas as pd
np.seterr(invalid='ignore')
import pywt
import matplotlib.pyplot as plt
from ECG.physionetdb_smital import create_epoched_df, generate_timeseries_noise_value
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, stft, resample, wiener
from ECG.physionetdb_smital import shade_noise_on_plot, create_epoched_df
from nwecg.awwf import WwfParams, wwf
from nwecg.ecg_quality import pad_to_length
import bottleneck
import datetime


""" ==================================== PRE-PROCESSING ==================================== """


def resample_signal(signal, old_rate, new_rate):

    print(f"\nResampling signal from {old_rate} to {new_rate} Hz...")

    ratio = new_rate / old_rate
    out = resample(x=signal, num=int(len(signal) * ratio))

    return out


def filter_highpass(signal: np.ndarray, sample_rate: float, cutoff_low: float = .67, order: int = 100):

    nyquist_freq = 0.5 * sample_rate
    sos = butter(N=order, Wn=[cutoff_low/nyquist_freq], btype="highpass", output="sos")

    return sosfiltfilt(sos, x=signal, axis=0)


def filter_lowpass(signal: np.ndarray, sample_rate: float, cutoff_high: float = 30, order: int = 100):

    nyquist_freq = 0.5 * sample_rate
    sos = butter(N=order, Wn=[cutoff_high/nyquist_freq], btype="lowpass", output="sos")

    return sosfiltfilt(sos, x=signal, axis=0)


def filter_notch(signal: np.ndarray, sample_rate: float, freq: int or float = 60):

    b, a = iirnotch(w0=freq, fs=sample_rate, Q=30.0)

    return filtfilt(b, a, x=signal)


def find_rr_intervals(ecg_raw: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculate the RR intervals for the ECG signal, based on the ecgdetectors package

    :param ecg_raw: the raw ecg signal
    :param fs: the sampling rate for the ecg signal
    """

    try:
        r_peaks = nk.ecg_peaks(ecg_cleaned=ecg_raw, sampling_rate=fs)[1]['ECG_R_Peaks']
    except IndexError:
        r_peaks = []

    return r_peaks


def flag_rr_windows(r_peaks, quiet=True):
    """ Creates 2D array for windows around detected R peaks. The start of each window is the midpoint between
        beat[i] and beat[i-1], while the end of each window is the midpoint between beat[i] and beat[i+1]
    """

    if not quiet:
        print(f"-Flagging {len(r_peaks)} windows that each contain 1 heartbeat...")

    windows = [[0, int(np.mean([r_peaks[0], r_peaks[1]]))]]
    for p1, p2, p3 in zip(r_peaks[:], r_peaks[1:], r_peaks[2:]):
        rr_pre = p2 - int(np.ceil((p2 - p1) / 2))  ##
        rr_post = p2 + int(np.floor((p3 - p2) / 2))  ##

        windows.append([rr_pre, rr_post])

    return windows


""" ==================================== WAVELET PROCESSING ==================================== """


def calculate_wavelet_bands(sample_rate, levels):

    print(f"\nSample rate of {sample_rate}Hz and {levels} decomposition levels:")
    for i in range(1, levels + 1):
        low = sample_rate / 2 ** (i+1)
        high = sample_rate / 2 ** i - .01
        print(f"Level #{i}: {low} - {high} Hz")


def swt1(xn, levels=4, wavelet='bior2.2'):

    # SWT requires that the signal has length which is a multiple of 2 ** level.
    n = 2 ** max(levels, 5)
    required_length = int(np.ceil((len(xn) + 120) / n) * n)
    x_padded, padding_left, padding_right = pad_to_length(xn, required_length, mode='reflect')

    max_level = pywt.swt_max_level(len(x_padded))

    swc = pywt.swt(x_padded, level=np.min([levels, max_level]), wavelet=wavelet, start_level=0)

    return np.asarray(swc), padding_left, padding_right


def iswt1(output_from_H, wavelet='bior2.2'):

    return pywt.iswt(output_from_H, wavelet=wavelet)


def swt2(signal, levels=4, wavelet='bior2.2', pad_signal=True):

    # SWT requires that the signal has length which is a multiple of 2 ** level.
    n = 2 ** max(levels, 5)
    required_length = int(np.ceil((len(signal) + 120) / n) * n)

    if pad_signal:
        x_padded, padding_left, padding_right = pad_to_length(signal, required_length, mode='reflect')
    if not pad_signal:
        x_padded = signal
        padding_right = 0
        padding_left = 0

    max_level = pywt.swt_max_level(len(x_padded))

    swc = pywt.swt(x_padded, level=np.min([levels, max_level]), wavelet=wavelet, start_level=0)

    return np.asarray(swc), padding_left, padding_right


def iswt2(lambda_ymn, wavelet='bior2.2'):

    return pywt.iswt(lambda_ymn, wavelet=wavelet)


def H(ymn, windows, pad_left=0, tm=2.8, apply_threshold=True):

    coefs = np.asarray(ymn)
    coefs_shape = coefs.shape
    data_len = len(coefs[0]) if coefs_shape[0] > 1 else len(coefs)
    n_levels = coefs_shape[0]

    if apply_threshold:
        # threshold calculation -------------
        threshes = []

        for decomp_level in range(n_levels):
            thresh_level = np.zeros(pad_left)
            coef = np.abs(coefs[decomp_level])

            for window_i, window in enumerate(windows):
                # Smital et al. (2013), eq. 2 and 3 (combined)
                data_window = coef[window[0]:window[1]]
                med_val = np.median(data_window)
                t = med_val / .6745 * tm
                window_thresh = np.array([t] * int(window[1] - window[0]))
                thresh_level = np.append(thresh_level, window_thresh)

            if len(thresh_level) < data_len:
                thresh_level = np.append(thresh_level, np.zeros(data_len - len(thresh_level)))

            threshes.append(thresh_level)

        ymn_threshed = np.array([pywt.threshold(data=c, value=np.asarray(threshes[i]), mode='garrote') for
                                 i, c in enumerate(ymn)])

    if not apply_threshold:
        ymn_threshed = ymn
        threshes = np.zeros(data_len)

    return np.asarray(threshes), np.asarray(ymn_threshed)


def WH(umn_cA, umn_cD, umn_ca_noise, umn_cd_noise, tm):
    """ 'Wiener filter in the wavelet domain.' """

    gm_ca = []
    gm_cd = []

    for level in range(len(umn_cA)):
        # squared coefficients
        cA_sq = np.square(umn_cA[level])
        cD_sq = np.square(umn_cD[level])

        cA_noise_sq = np.square(umn_ca_noise[level]/tm)
        cD_noise_sq = np.square(umn_cd_noise[level]/tm)

        # Smital et al. (2013) eq. 4
        cA = cA_sq / (cA_sq + cA_noise_sq)
        cA = np.nan_to_num(x=cA, nan=0)

        cD = cD_sq / (cD_sq + cD_noise_sq)
        cD = np.nan_to_num(x=cD, nan=0)
        gm_ca.append(cA)
        gm_cd.append(cD)

    return np.asarray(gm_ca), np.asarray(gm_cd)


def remove_decomp_levels(coefs, keep_levels, quiet=True):

    if isinstance(coefs, np.ndarray):
        x = coefs.copy()

        # indexes that get set to 0
        remove_levels = [i for i in range(x.shape[0]) if i not in keep_levels]

        if len(coefs) == 3:
            x[[remove_levels], :, :] = np.zeros(x.shape[2])

        elif len(coefs) == 2:
            x[[remove_levels], :] = np.zeros(x.shape[1])

    elif isinstance(coefs, list):
        remove_levels = [i for i in range(len(coefs)) if i not in keep_levels]

        x = []
        z = np.zeros(len(coefs[0][0]))
        for level in range(len(coefs)):
            if level in remove_levels:
                x.append([z, z])
            if level not in remove_levels:
                x.append([coefs[level][0], coefs[level][1]])

    if not quiet and len(remove_levels) > 0:
        print(f"-Data from decomposition level{'s' if len(remove_levels) > 1 else ''} {remove_levels} "
              f"ha{'ve' if len(remove_levels) > 1 else 's'} been removed")

    return x


""" ==================================== UTILITIES ==================================== """


class Smital:

    def __init__(self, ecg_filt, sample_rate,
                 upper_ca_thresh=True, upper_cd_thresh=True,
                 lower_ca_thresh=True, lower_cd_thresh=True,
                 use_ca=True, use_cd=True,
                 upper_wavelet='bior4.4', lower_wavelet='sym4',
                 use_snr_sum=False,
                 n_decomp_levels=4,
                 use_decomp_levels: list or tuple or None = None,
                 ):

        self.sample_rate = sample_rate
        self.data_upper = {'xn': ecg_filt, 'ymn': [], 'crop': [0, 0], 'r_peaks': [], 'rr_windows': [],
                           'ymn_cA_threshes': [], 'ymn_cA_threshed': [], 'ymn_cD_threshes': [], 'ymn_cD_threshed': [],
                           'ymn_threshed': [], 's^': [], 'u^mn_cA': [], 'u^mn_cD': [], 'snr': [],
                           'valid_beats': True}
        self.data_lower = {'ymn_cA': [], 'ymn_cD': [], 'crop': [0, 0], 'g^mn_cA': [], 'g^mn_cD': [],
                           'lambda_ymn_cA': [], 'lambda_ymn_cD': [], 'lambda_ymn': [], 'yn': [], 'snr': []}
        self.data_awwf = {'zn': [], 'snr_segments': [], 'params': [], 'snr': []}
        self.df_med = pd.DataFrame()

        self.upper_ca_thresh = upper_ca_thresh
        self.upper_cd_thresh = upper_cd_thresh
        self.lower_ca_thresh = lower_ca_thresh
        self.lower_cd_thresh = lower_cd_thresh
        self.use_ca = use_ca
        self.use_cd = use_cd
        self.upper_wavelet = upper_wavelet
        self.lower_wavelet = lower_wavelet
        self.use_snr_sum = use_snr_sum
        self.n_decomp_level = n_decomp_levels
        self.use_decomp_levels = sorted(use_decomp_levels) if \
            use_decomp_levels is not None else np.arange(n_decomp_levels)

        self.universal_params = WwfParams.universal()

    def run_process(self):

        self.data_upper['r_peaks'] = find_rr_intervals(ecg_raw=self.data_upper['xn'], fs=self.sample_rate)
        self.data_upper['rr_windows'] = flag_rr_windows(r_peaks=self.data_upper['r_peaks'])

        # padding starts for all signals except data_upper['xn']
        wwf_upper_path(data_upper=self.data_upper,
                       n_decomp_levels=self.n_decomp_level,
                       use_decomp_levels=self.use_decomp_levels,
                       tm=self.universal_params.threshold_multiplier,
                       wavelet=self.upper_wavelet,
                       threshold_ca=self.upper_ca_thresh,
                       threshold_cd=self.upper_cd_thresh,
                       quiet=False)

        self.data_lower = wwf_lower_path(data_upper=self.data_upper,
                                         n_decomp_levels=self.n_decomp_level,
                                         use_decomp_levels=self.use_decomp_levels,
                                         tm=self.universal_params.threshold_multiplier,
                                         wavelet=self.lower_wavelet,
                                         threshold_ca=self.lower_ca_thresh,
                                         threshold_cd=self.lower_cd_thresh,
                                         use_ca=self.use_ca,
                                         use_cd=self.use_cd,
                                         quiet=False)

        crop_dict_data(data_upper=self.data_upper, data_lower=self.data_lower)

        self.data_upper['snr'] = get_rolling_snr(x=self.data_upper['xn'], s=self.data_upper['s^'],
                                                 window=int(2 * self.sample_rate), use_sum=self.use_snr_sum)[1]
        self.data_lower['snr'] = get_rolling_snr(x=self.data_upper['xn'], s=self.data_lower['yn'],
                                                 window=int(2 * self.sample_rate), use_sum=self.use_snr_sum)[1]
        """
        self.data_awwf = adaptive_wwf(signal=ecg_signal,
                                      data_lower=self.data_lower,
                                      sample_rate=self.sample_rate,
                                      min_snr_seg_len=5,
                                      lower_ca_thresh=self.lower_ca_thresh,
                                      lower_cd_thresh=self.lower_cd_thresh,
                                      use_ca=self.use_ca,
                                      use_cd=self.use_cd,
                                      quiet=True)

        self.data_awwf['snr'] = get_rolling_snr(x=self.data_upper['xn'], s=self.data_awwf['zn'],
                                                window=int(2 * self.sample_rate), use_sum=self.use_snr_sum)[1]

        self.df_med = self.calculate_median_snr_by_stage()"""

    def calculate_median_snr_by_stage(self):

        df_snr_upper = create_epoched_df(snr_dict=self.data_upper, sample_rate=self.sample_rate, keys=['snr'])
        med_upper = df_snr_upper.groupby("noise")['snr'].describe()[['50%']]
        med_upper.loc['diff'] = [med_upper['50%'].iloc[0] - med_upper['50%'].iloc[1]]

        df_snr_lower = create_epoched_df(snr_dict=self.data_lower, sample_rate=self.sample_rate, keys=['snr'])
        med_lower = df_snr_lower.groupby("noise")['snr'].describe()[['50%']]
        med_lower.loc['diff'] = [med_lower['50%'].iloc[0] - med_lower['50%'].iloc[1]]

        df_snr_awwf = create_epoched_df(snr_dict=self.data_awwf, sample_rate=self.sample_rate, keys=['snr'])
        med_awwf = df_snr_awwf.groupby("noise")['snr'].describe()[['50%']]
        med_awwf.loc['diff'] = [med_awwf['50%'].iloc[0] - med_awwf['50%'].iloc[1]]

        df_med = pd.DataFrame({'upper': med_upper['50%'].values,
                               'lower': med_lower['50%'].values,
                               "awwf": med_awwf['50%'].values},
                              index=['clean', 'noise', 'diff'])

        return df_med

    def plot_results(self, data_key, original_dict=None, shade_rr_windows=False, mitbih=False,
                     plot_upper=True, plot_lower=True, plot_awwf=True):

        fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))

        ax[0].plot(np.arange(len(self.data_upper['xn']))/self.sample_rate, self.data_upper['xn'],
                   label=f'input sig.', color='black', lw=2.5, zorder=0)

        if 'r_peaks' in self.data_awwf.keys():
            peaks = self.data_awwf['r_peaks'][np.argwhere(self.data_awwf['r_peaks'] < len(self.data_upper['xn']))]
            ax[0].scatter(np.arange(len(self.data_upper['xn']))[peaks]/self.sample_rate,
                          self.data_upper['xn'][peaks],
                          color='limegreen', marker='v', zorder=1, label='peaks')

        if plot_upper:
            ax[0].plot(np.arange(len(self.data_upper['s^']))/self.sample_rate, self.data_upper['s^'],
                       label='s^', color='red', lw=2, zorder=0)

            ax[1].plot(np.arange(len(self.data_upper['snr']))/self.sample_rate, self.data_upper['snr'],
                       color='red', label='upper', lw=2)

        if plot_lower:
            ax[0].plot(np.arange(len(self.data_lower['yn']))/self.sample_rate, self.data_lower['yn'],
                       label='yn', color='dodgerblue', lw=1.5, zorder=0)

            ax[1].plot(np.arange(len(self.data_lower['snr']))/self.sample_rate, self.data_lower['snr'],
                       color='dodgerblue', label='lower', lw=1.5)

        if plot_awwf:
            ax[0].plot(np.arange(len(self.data_awwf['zn']))/sample_rate, self.data_awwf['zn'], label='zn', color='orange')

            ax[1].plot(np.arange(len(self.data_awwf['snr']))/sample_rate, self.data_awwf['snr'],
                       color='orange', label='awwf')

        ax[0].legend(loc='upper left')
        ax[0].set_title("Noise-free signal estimates")

        if original_dict is not None:
            ax[1].plot(np.arange(len(original_dict['roll_snr']))/360, original_dict['roll_snr'],
                       color='black', label='original')

        if mitbih:
            shade_noise_on_plot(ax[1], self.sample_rate, 'seconds')
            ax[1].axhline(y=int(data_key) if data_key != "_6" else -6, color='grey', linestyle='dashed', label='clean SNR')
            ax[1].axhline(y=24, color='darkgrey', linestyle='dashed', label='noise SNR')

        ax[1].legend(loc='upper left')

        if shade_rr_windows:
            for i in self.data_upper['rr_windows'][::2]:
                ax[0].axvspan(i[0]/self.sample_rate, i[1]/self.sample_rate, 0, 1, color='pink', alpha=.2)

        fig.axes[1].grid()

        plt.tight_layout()

        return fig


def calculate_snr(x: np.ndarray, s: np.ndarray):
    """ SNR calculation: Smital et al. (2020), eq. 5

        Parameters
        ----------
        x
            Input signal
        s
            Noise-free signal estimate

        Returns
        -------
        array of SNR values
    """

    # Estimate noise component, w[n].
    w = x - s

    # Equation (5), Smital 2020.
    return 10 * np.log10(np.sum(np.square(s)) / np.sum(np.square(w)))


def get_rolling_snr(x: np.ndarray, s: np.ndarray, window: int, use_sum: bool = True, replace_inf_val: float or int = 30):
    """ Calculate the rolling signal-to-noise ratio for the given signal.

        Parameters
        ----------
        x
            ECG signal
        s
            Noise-free signal estimate
        window
            Size of rolling window in samples


    :param x: 1xn array representing signal.
    :param s: 1xn array representing noise-free estimate of signal.
    :param window: Size of rolling window.
    :return: 1xn array representing rolling signal-to-noise ratio.
    """

    # Estimate noise component, w[n].
    w = x - s

    # Equation (5), Smital 2020.
    if use_sum:
        rolling_s_energy = bottleneck.move_sum(a=np.square(s), window=window, axis=0)[window - 1:]  # Trim first few NaN values.
        rolling_w_energy = bottleneck.move_sum(a=np.square(w), window=window, axis=0)[window - 1:]

    if not use_sum:
        rolling_s_energy = bottleneck.move_var(a=np.square(s), window=window, axis=0)[window - 1:]  # Trim first few NaN values.
        rolling_w_energy = bottleneck.move_var(a=np.square(w), window=window, axis=0)[window - 1:]

    rolling_snr = 10 * np.log10(rolling_s_energy / rolling_w_energy)

    rolling_snr[np.isinf(rolling_snr)] = replace_inf_val

    # Apply Short-Time Averaging (STA) to SNR.
    rolling_snr_sta = bottleneck.move_mean(a=rolling_snr, window=window, axis=0)[window - 1:]
    rolling_snr_sta, *_ = pad_to_length(x=rolling_snr_sta, length=len(s), mode='edge')

    return rolling_snr, rolling_snr_sta


def get_threshold_crossings(x: np.ndarray, thresholds: list or np.array = (-5, 10, 20, 35, 45)):
    """ Creates list of indexes corresponding to where signal 'x' crosses any threshold specified in 'thresholds'

        Parameters
        ----------
        x
            input signal
        thresholds
            list of thresholds

        Returns
        -------
        np.array of indexes where threshold was crossed
    """

    # Segment given signal by the given thresholds.
    annotations = np.zeros(np.shape(x), dtype=np.int8)
    for i, threshold in enumerate(sorted(thresholds)):
        annotations[x > threshold] = i + 1

    # Get first differences (i.e., x[n + 1] - x[n]).
    first_diff = np.diff(annotations)

    # Changes occur where the first difference is non-zero.
    diffs = np.append(np.where(first_diff)[0] + 1, len(x))
    diffs = np.insert(arr=diffs, obj=0, values=[0])

    return diffs


def get_wavelet_parameters_for_snr(noise: float):
    """
    Classify wavelet parameters for each SNR level.
    Based directly on the values from Smital et al. 2013 paper (table III)
    """
    # Round to nearest 5.
    noise = int(round(noise / 5) * 5)

    if noise <= -5:
        level1 = 4
        level2 = 4
        threshold_multiplier = 3.6
        wavelet1 = 'rbio3.3'
        wavelet2 = 'rbio4.4'

    elif noise <= 10:
        level1 = 4
        level2 = 4
        threshold_multiplier = 3.4
        wavelet1 = 'rbio1.3'
        wavelet2 = 'rbio4.4'

    elif noise <= 20:
        level1 = 4
        level2 = 4
        threshold_multiplier = 3.1
        wavelet1 = 'db4'
        wavelet2 = 'sym4'

    elif noise <= 35:
        level1 = 3
        level2 = 3
        threshold_multiplier = 2.8
        wavelet1 = 'bior4.4'
        wavelet2 = 'sym4'

    elif noise <= 45:
        level1 = 3
        level2 = 3
        threshold_multiplier = 2.5
        wavelet1 = 'bior3.9'
        wavelet2 = 'sym4'

    else:
        level1 = 2
        level2 = 2
        threshold_multiplier = 2.3
        wavelet1 = 'sym6'
        wavelet2 = 'bior3.3'

    dict_out = {'level1': level1, 'level2': level2, 'thresh_type': 'garrote',
                'tm': threshold_multiplier, 'wavelet1': wavelet1, 'wavelet2': wavelet2}

    return dict_out


def adaptive_wwf(signal,
                 data_lower,
                 sample_rate,
                 min_snr_seg_len=5,
                 upper_ca_thresh=True,
                 upper_cd_thresh=False,
                 lower_ca_thresh=True,
                 lower_cd_thresh=False,
                 use_ca=True,
                 use_cd=True,
                 use_decomp_levels: int or None = None,
                 fixed_n_levels: int or None = None,
                 quiet: bool = True):

    if not quiet:
        print("========== Running AWWF pathway ==========")

    snr_segs = get_threshold_crossings(data_lower['snr'], WwfParams.snr_thresholds())

    awwf_dict = {'zn': np.zeros_like(signal),
                 'snr_segments': snr_segs,
                 'params': [],
                 'r_peaks': np.array([])}

    idx_nan = np.argwhere(np.isnan(data_lower['snr']))
    data_lower['snr'][idx_nan] = 0

    idx_inf = np.isinf(data_lower['snr']).transpose()
    data_lower['snr'][idx_inf] = 0

    pad = int(2 * sample_rate)
    sig_len = len(signal)

    start_idx = 0

    for end_idx in awwf_dict['snr_segments'][1:]:
        # For each segment, apply WWF with parameters based on the SNR of the segment.
        # Smital et al. (2013) table III
        seg_params = WwfParams.for_snr(data_lower['snr'][start_idx])

        awwf_dict['params'].append([start_idx, seg_params.wavelet1, seg_params.wavelet2,
                                    seg_params.level1, seg_params.level2])

        # window indexing to keep indexes within bounds of data ---------
        # RR-window +/- 2-second padding to account for rolling calculations

        start = start_idx - pad if start_idx - pad >= 0 else 0
        start_pad = start_idx - start

        end = end_idx + pad if end_idx + pad <= sig_len else sig_len
        end_pad = end - end_idx if end - end_idx != 0 else 1

        seg_upper = {'xn': signal[start:end]}

        seg_upper['r_peaks'] = find_rr_intervals(ecg_raw=seg_upper['xn'], fs=sample_rate)

        awwf_dict['r_peaks'] = np.concatenate([awwf_dict['r_peaks'],
                                               np.array([i + start for i in seg_upper['r_peaks']])])

        if len(seg_upper['r_peaks']) >= 3:
            seg_upper['rr_windows'] = flag_rr_windows(r_peaks=seg_upper['r_peaks'])

            if fixed_n_levels is None:
                upper_levels = seg_params.level1
                lower_levels = seg_params.level2

            if fixed_n_levels is not None:
                upper_levels = fixed_n_levels
                lower_levels = fixed_n_levels

            wwf_upper_path(data_upper=seg_upper,
                           tm=seg_params.threshold_multiplier,
                           n_decomp_levels=upper_levels,
                           use_decomp_levels=use_decomp_levels if use_decomp_levels is not None else \
                               np.arange(upper_levels),
                           wavelet=seg_params.wavelet1,
                           threshold_ca=upper_ca_thresh,
                           threshold_cd=upper_cd_thresh,
                           quiet=quiet)

            seg_dict = wwf_lower_path(data_upper=seg_upper,
                                      tm=seg_params.threshold_multiplier,
                                      n_decomp_levels=lower_levels,
                                      use_decomp_levels=use_decomp_levels if \
                                          use_decomp_levels is not None else np.arange(lower_levels),
                                      wavelet=seg_params.wavelet2,
                                      threshold_ca=lower_ca_thresh,
                                      threshold_cd=lower_cd_thresh,
                                      use_ca=use_ca,
                                      use_cd=use_cd,
                                      quiet=quiet)

            seg_dict['yn'] = seg_dict['yn'][seg_dict['crop'][0]:seg_dict['crop'][1]]

            try:
                awwf_dict['zn'][start_idx:end_idx] = seg_dict['yn'][start_pad:-end_pad]
            except ValueError:
                awwf_dict['zn'][start_idx:end_idx-1] = seg_dict['yn'][start_pad:-end_pad]

        if len(seg_upper['r_peaks']) < 3:
            awwf_dict['zn'][start_idx:end_idx] = np.zeros(end_idx - start_idx)

        start_idx = end_idx

    awwf_dict['r_peaks'] = awwf_dict['r_peaks'].astype(int)

    return awwf_dict


def wwf_upper_path(data_upper: dict,
                   n_decomp_levels: int = 4,
                   use_decomp_levels: int or None = None,
                   tm: int or float = 2.8,
                   wavelet: str = 'bior2.2',
                   threshold_cd: bool = True,
                   threshold_ca: bool = True,
                   quiet: bool = True):
    """ Upper path in Smital et al. (2013) figure 2. Calls swt1(), H(), iswt1(), and swt2().

        Parameters
        ----------
        data_upper
            dictionary containing filtered ECG signal 'xn'
        n_decomp_levels
            number of decompositions to perform in SWT
        use_decomp_levels
            used to specify which levels from swt1 to use in signal reconstruction
            -'all' for all levels, or an integer to specific one specific level
        tm
            'threshold multipllier'; constant by which thresholds in Smital et al. (2013) eq. 3 are multiplied
        wavelet
            wavelet to use for decomposition and reconstruction
        threshold_cd
            boolean whether or not to apply thresholding to detail coefficients
        threshold_ca
            boolean whether or not to apply thresholding to approximation coefficients
        quiet
            boolean to print parameters/progress to console

        Returns
        -------
        dict_upper: dictinoary with lots of data (see below)

        Keys
        -----
        -xn: highpass + notch-filtered ECG signal
            -"corrupted input signal"

        -crop: indexes used to undo signal padding that is required for SWT

        -r_peaks: QRS peak indexes corresponding to indexes in 'xn'
        -rr_windows: windows containing 1 QRS complex (midpoints between beats) with indexes corresponding to 'xn'
        -rr_windows_pad: windows containing 1 QRS complex (midpoints between beats) with indexes
                         corresponding to padded signals

        -ymn: SWT coefficients of 'xn', output from swt1() function
        -ymn_cA_threshes: array of calculated thresholds from Smital et al. (2013) eq. 3 corresponding to
                          approximation coefficients in each decomposition level
        -ymn_cA_threshed: ymn approximation coefficients after applying thresholds
        -ymn_cD_threshes: array of calculated thresholds from Smital et al. (2013) eq. 3 corresponding to
                          detail coefficients in each decomposition level
        -ymn_cD_threshed: ymn detail coefficients after applying thresholds
        -ymn_threshed: combined array of ymn_cA_threshed and ymn_cD_threshed for input into inverse SWT function

        -s^: estimate of the noise-free signal
        -u^mn_cA: approximation coefficients of estimated noise-free signal 's^'
        -u^mn_cD: detail coefficients of estimated noise-free signal 's^'
    """

    if not quiet:
        print(" ========== Running WWF upper pathway ==========")
        print("Settings:")
        print(f"    -Decomposition levels = {n_decomp_levels}")
        print(f"    -Wavelet = {wavelet}")
        print(f"    -Threshold multiplier = {tm}")
        print(f"    -Threshold cA = {threshold_ca}, threshold cD = {threshold_cd}")

    # SWT1
    # Signal gets padded since SWT input signal needs to be a multiple of 2 ** wavelet_level
    data_upper['ymn'], padding_left, padding_right = swt1(xn=data_upper['xn'],
                                                          levels=n_decomp_levels,
                                                          wavelet=wavelet)

    data_upper['crop'] = [padding_left, -1 if padding_right == 0 else -padding_right]

    # adjust rr_window indexes to account for padding from above
    # data_upper['r_peaks'] = data_upper['r_peaks'] + data_upper['crop'][0]
    data_upper['windows'] = [[i[0] + data_upper['crop'][0], i[1] + data_upper['crop'][0]] for
                             i in data_upper['rr_windows']]

    # Smital et al. (2013) eq. 2 and 3
    # estimates noise using median of wavelet coefficients' median value and two constants
    # thresholds wavelet coefficients using calculated thresholds
    data_upper['ymn_cA_threshes'], data_upper['ymn_cA_threshed'] = H(ymn=data_upper['ymn'][:, 0],
                                                                     windows=data_upper['windows'],
                                                                     tm=tm,
                                                                     pad_left=data_upper['crop'][0],
                                                                     apply_threshold=threshold_ca)

    data_upper['ymn_cD_threshes'], data_upper['ymn_cD_threshed'] = H(ymn=data_upper['ymn'][:, 1],
                                                                     windows=data_upper['windows'],
                                                                     tm=tm,
                                                                     pad_left=data_upper['crop'][0],
                                                                     apply_threshold=threshold_cd)

    # combined ymn_cA_threshed and ymn_cD_threshed into format needed for inverse SWT
    data_upper['ymn_threshed'] = [[data_upper['ymn_cA_threshed'][i], data_upper['ymn_cD_threshed'][i]] for
                                  i in range(len(data_upper['ymn_cA_threshed']))]

    data_upper['ymn_threshed'] = remove_decomp_levels(coefs=data_upper['ymn_threshed'],
                                                      keep_levels=use_decomp_levels,
                                                      quiet=False)

    # noise-free signal estimate using inverse SWT, used to design Wiener filter in lower pathway
    data_upper['s^'] = pywt.iswt(data_upper['ymn_threshed'], wavelet=wavelet)

    # estimates wavelet coefficients of noise-free signal
    data_upper['u^mn'], padding_left, padding_right = swt2(signal=data_upper['s^'],
                                                           levels=n_decomp_levels,
                                                           wavelet=wavelet,
                                                           pad_signal=False)

    data_upper['u^mn_cA'] = np.array([coef[0] for coef in data_upper['u^mn']])
    data_upper['u^mn_cD'] = np.array([coef[1] for coef in data_upper['u^mn']])
    del data_upper['u^mn']


def wwf_lower_path(data_upper: dict,
                   n_decomp_levels: int = 4,
                   use_decomp_levels: int or None = None,
                   tm: int or float = 2.8,
                   wavelet: str = 'bior2.2',
                   threshold_cd: bool = True,
                   threshold_ca: bool = True,
                   use_ca: bool = True,
                   use_cd: bool = True,
                   quiet: bool = True):
    """ Lower path in Smital et al. (2013) figure 2. Calls swt2(), HW(), and iswt2().

        Parameters
        ----------
        data_upper
            dictionary returned from wwf_upper_path
        n_decomp_levels
            number of decompositions to perform in SWT
        tm
            'threshold multipllier'; constant by which thresholds in Smital et al. (2013) eq. 3 are multiplied
        wavelet
            wavelet to use for decomposition and reconstruction
        threshold_cd
            boolean whether or not to apply thresholding to detail coefficients
        threshold_ca
            boolean whether or not to apply thresholding to approximation coefficients
        quiet
            boolean to print parameters/progress to console

        Returns
        -------
        data_lower: dictionary with lots of data (see below)

        Keys
        ----
        -ymn_cA:
        -ymn_cD:
        -ymn_cA_noise:
        -ymn_cD_noise:
        -g^mn_cA:
        -g^mn_cD:
        -lambda_ymn_cA:
        -lambda_ymn_cD:
        -lambda_ymn:
        -yn:

    """

    if not quiet:
        print(" ========== Running WWF lower pathway ==========")
        print("Settings:")
        print(f"    -Decomposition levels = {n_decomp_levels}")
        print(f"    -Wavelet = {wavelet}")
        print(f"    -Threshold multiplier = {tm}")
        print(f"    -Threshold cA = {threshold_ca}, threshold cD = {threshold_cd}")

    data_lower = {}

    ymn, padding_left, padding_right = swt2(signal=data_upper['xn'],
                                            levels=n_decomp_levels,
                                            wavelet=wavelet,
                                            pad_signal=True)

    # splits ymn above into approximation and detail coefficients
    data_lower['ymn_cA'] = ymn[:, 0]
    data_lower['ymn_cD'] = ymn[:, 1]

    # crop indexes to undo signal padding (required for SWT)
    data_lower['crop'] = [padding_left, -1 if padding_right == 0 else -padding_right]

    # Smital et al. (2013) eq. 4 -------------------
    # Wiener correction factor based on thresholds
    # tm set to 1 since it is set to 1 in the call to H() for umn_cA_noise/umn_cD_noise estimates
    data_lower['g^mn_cA'], data_lower['g^mn_cD'] = WH(umn_cA=data_upper['u^mn_cA'],
                                                      umn_cD=data_upper['u^mn_cD'],
                                                      umn_ca_noise=data_upper['ymn_cA_threshes']/tm,
                                                      umn_cd_noise=data_upper['ymn_cD_threshes']/tm,
                                                      tm=1)

    # Smital et al. (2013) eq. 5
    # Applies Wiener correction factor to wavelet coefficients ymn_cA/ymn_cD --> 'modified coefficients'
    data_lower['lambda_ymn_cA'] = data_lower['ymn_cA'] * data_lower['g^mn_cA']
    data_lower['lambda_ymn_cD'] = data_lower['ymn_cD'] * data_lower['g^mn_cD']

    # Smital et al. (2013) eq. 5
    data_lower['lambda_ymn'] = [[a, d] for a, d in zip(data_lower['lambda_ymn_cA'], data_lower['lambda_ymn_cD'])]

    data_lower['lambda_ymn'] = remove_decomp_levels(coefs=data_lower['lambda_ymn'],
                                                    keep_levels=use_decomp_levels,
                                                    quiet=False)

    if use_ca and not use_cd:
        data_lower['lambda_ymn'] = np.asarray(data_lower['lambda_ymn'])[:, 0]
    if not use_ca and use_cd:
        data_lower['lambda_ymn'] = np.asarray(data_lower['lambda_ymn'])[:, 1]

    # Inverse SWT2
    data_lower['yn'] = iswt2(lambda_ymn=data_lower['lambda_ymn'], wavelet=wavelet)

    return data_lower


def crop_dict_data(data_upper, data_lower):

    data_len = len(data_upper['xn'])

    # Crop upper data --------------------------------
    for key in ['ymn', 'ymn_cA_threshes', 'ymn_cA_threshed', 'ymn_cD_threshes',
                'ymn_cD_threshed', 's^', 'u^mn_cA', 'u^mn_cD']:

        if key in data_upper.keys():

            if len(data_upper[key].shape) == 3:
                if len(data_upper[key][0][0]) > data_len:
                    data_upper[key] = data_upper[key][:, :, data_upper['crop'][0]:data_upper['crop'][1]]

            if len(data_upper[key].shape) == 2:
                if len(data_upper[key][0]) > data_len:
                    data_upper[key] = data_upper[key][:, data_upper['crop'][0]:data_upper['crop'][1]]

            if len(data_upper[key].shape) == 1:
                if len(data_upper[key]) > data_len:
                    data_upper[key] = data_upper[key][data_upper['crop'][0]:data_upper['crop'][1]]

    for key in ['ymn_threshed']:
        if key in data_upper.keys():
            if len(data_upper[key][0][0]) > data_len:
                data_upper[key] = [(data_upper[key][i][0][data_upper['crop'][0]:data_upper['crop'][1]],
                                    data_upper[key][i][1][data_upper['crop'][0]:data_upper['crop'][1]])
                                   for i in range(len(data_upper[key]))]

    # Crop lower data --------------------------------

    for key in ['ymn_cA', 'ymn_cD', 'g^mn_cA', 'g^mn_cD', 'lambda_ymn_cA', 'lambda_ymn_cD', 'yn', 'snr']:

        if key in data_lower.keys():

            if len(data_lower[key].shape) > 1:
                if len(data_lower[key][0]) > data_len:
                    data_lower[key] = data_lower[key][:, data_lower['crop'][0]:data_lower['crop'][1]]

            if len(data_lower[key].shape) == 1:
                if len(data_lower[key]) > data_len:
                    data_lower[key] = data_lower[key][data_lower['crop'][0]:data_lower['crop'][1]]

    for key in ['lambda_ymn']:  # 'ymn_threshed'
        if key in data_lower.keys():
            if len(data_lower[key][0][0]) > data_len:
                data_lower[key] = [(data_lower[key][i][0][data_lower['crop'][0]:data_lower['crop'][1]],
                                    data_lower[key][i][1][data_lower['crop'][0]:data_lower['crop'][1]])
                                   for i in range(len(data_lower[key]))]


def clip_array(arr, min_val=-50, max_val=30):

    return np.clip(a=arr, a_min=min_val if min_val is not None else min(arr),
                   a_max=max_val if max_val is not None else max(arr), out=arr)


def append_settings_csv(smital_obj, pathway):

    df_settings = pd.read_csv(pathway)

    df_row = pd.DataFrame({"data_key": data_key, "sample_rate": sample_rate,
                           "low_f_cut": low_f_cut, "notch_cut": notch_cut,
                           'use_ca': smital_obj.use_ca, 'use_cd': smital_obj.use_cd,
                           "upper_ca_thresh": smital_obj.upper_ca_thresh,
                           'upper_cd_thresh': smital_obj.upper_cd_thresh,
                           'upper_n_decomp': 4, 'upper_decomp_level': 'all', 'upper_wavelet': smital_obj.upper_wavelet,
                           'lower_ca_thresh': smital_obj.lower_ca_thresh, 'lower_cd_thresh': smital_obj.lower_cd_thresh,
                           'lower_n_decomp': 4, 'lower_decomp_level': 'all', 'lower_wavelet': smital_obj.lower_wavelet,
                           'use_snr_sum': smital_obj.use_snr_sum,
                           'upper_clean_snr': round(smital_obj.df_med.loc['clean']['upper'], 2),
                           'upper_noise_snr': round(smital_obj.df_med.loc['noise']['upper'], 2),
                           'lower_clean_snr': round(smital_obj.df_med.loc['clean']['lower'], 2),
                           'lower_noise_snr': round(smital_obj.df_med.loc['noise']['lower'], 2),
                           'awwf_clean_snr': round(smital_obj.df_med.loc['clean']['awwf'], 2),
                           'awwf_noise_snr': round(smital_obj.df_med.loc['noise']['awwf'], 2)},
                          index=[df_settings.shape[0]])

    try:
        if (df_row == df_settings.iloc[-1]).iloc[0].sum() != df_settings.shape[1]:
            df_settings = pd.concat([df_settings, df_row])
    except IndexError:
        df_settings = pd.concat([df_settings, df_row])

    df_settings.to_csv(pathway, index=False)
    print("-Settings file updated")

    return df_settings


def plot_stft(data, sample_rate, f, t, Zxx):
    fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 8))
    ax1.plot(np.arange(0, len(data)) / sample_rate, data, color='black')
    ax1.grid()

    pcm = ax2.pcolormesh(t, f, np.abs(Zxx), cmap='turbo', shading='auto')

    cbaxes = fig.add_axes([.91, .11, .03, .35])
    cb = fig.colorbar(pcm, ax=ax2, cax=cbaxes)

    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Seconds')
    plt.subplots_adjust(top=0.95, bottom=0.075, left=0.075, right=0.9, hspace=0.2, wspace=0.2)

    return fig


""" ==================================== FUNCTION CALLS ==================================== """

sample_rate = 500
data_key = '00'
low_f_cut = .67
notch_cut = 60

# Preprocessing ---------

t0 = datetime.datetime.now()
"""
import nimbalwear
subj = 'OND09_SBH0336'

ecg = nimbalwear.Device()
ecg.import_edf(f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{subj}_01_BF36_Chest.edf")
ecg_signal = resample_signal(signal=ecg.signals[0][:int(250*3600/2)], old_rate=250, new_rate=sample_rate)
ecg_signal = filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=low_f_cut, order=100)
ecg_signal = filter_lowpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_high=40, order=100)
# ecg_signal = filter_notch(signal=ecg_signal, sample_rate=sample_rate, freq=notch_cut)

snr_orig = nimbalwear.Device()
snr_orig.import_edf(f"W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/{subj}_01_snr.edf")
"""

# resamples signal to 512 Hz to match Smital et al. (2013)
"""
ecg_signal = resample_signal(signal=nst_data[data_key]['ecg'], old_rate=360, new_rate=sample_rate)
ecg_signal = filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=low_f_cut, order=100)
ecg_signal = filter_notch(signal=ecg_signal, sample_rate=sample_rate, freq=notch_cut)
noise = generate_timeseries_noise_value(signal=ecg_signal, noise_key=data_key, sample_rate=sample_rate, noise_indexes=(300, 540, 780, 1020, 1260, 1500, 1740))
"""


# plt.close("all")

best_settings = {"upper_ca_thresh": True, "upper_cd_thresh": False,
                 "lower_ca_thresh": True, 'lower_cd_thresh': False,
                 'use_ca': True, 'use_cd': True,
                 'upper_wavelet': 'bior2.2', 'lower_wavelet': 'bior2.2',
                 "use_decomp_levels": [0, 1, 2, 3],
                 "use_snr_sum": False}

# ecg_signal[500000:500500] = 100
self = Smital(ecg_filt=ecg_signal, sample_rate=sample_rate,
              upper_ca_thresh=True, upper_cd_thresh=True,
              lower_ca_thresh=True, lower_cd_thresh=True,
              use_ca=True, use_cd=True,
              upper_wavelet='bior2.2', lower_wavelet='bior2.2',
              n_decomp_levels=4,
              use_decomp_levels=None,
              use_snr_sum=False)
self.run_process()

self.data_awwf = adaptive_wwf(signal=self.data_upper['xn'],
                              data_lower=self.data_lower,
                              sample_rate=self.sample_rate,
                              min_snr_seg_len=5,
                              upper_ca_thresh=self.upper_ca_thresh,
                              upper_cd_thresh=self.upper_cd_thresh,
                              #upper_cd_thresh=False,
                              lower_ca_thresh=self.lower_ca_thresh,
                              lower_cd_thresh=self.lower_cd_thresh,
                              # lower_cd_thresh=False,
                              use_ca=self.use_ca,
                              use_cd=self.use_cd,
                              fixed_n_levels=None,
                              quiet=True)

self.data_awwf['snr'] = get_rolling_snr(x=self.data_upper['xn'], s=self.data_awwf['zn'],
                                        window=int(2 * self.sample_rate), use_sum=self.use_snr_sum)[1]

self.df_med = self.calculate_median_snr_by_stage()

# df_settings = append_settings_csv(smital_obj=data, pathway="C:/Users/ksweber/Desktop/smital_settings.csv")

t1 = datetime.datetime.now()
dt = (t1 - t0).total_seconds()
print(f"\n ==========  Processing time ==========")
print(f"-Processing time = {dt:.1f} seconds")
print(f"     = {dt / (len(ecg_signal)/sample_rate/3600):.1f} sec/hour of data")

print()
print(self.df_med.round(1))

#for arr in [self.data_upper['snr'], self.data_lower['snr'], self.data_awwf['snr']]:
#    arr = clip_array(arr=arr, min_val=None, max_val=50)

"""
fig = self.plot_results(data_key=data_key, original_dict=None, mitbih=False, shade_rr_windows=True,
                        plot_upper=True, plot_lower=True, plot_awwf=True)

try:
    r = sample_rate / 250
    l = len(self.data_upper['xn'])
    fig.axes[1].plot(np.arange(int(l / r))/250, snr_orig.signals[0][:int(l / r)], color='black', label='original')
except NameError:
    pass

try:
    fig.axes[1].plot(np.arange(len(noise))/sample_rate, noise, color='black', label='true')
except NameError:
    pass

fig.axes[1].legend(loc='upper left')
"""

# TODO
# add option to only threshold certain levels
"""
df_cn = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/cardiac_navigator_screened/{subj}_01_CardiacNavigator_Screened.csv")
df_cn_raw = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/CardiacNavigator/{subj}_Events.csv", delimiter=';')
df_cn_raw = df_cn_raw.loc[(df_cn_raw['Type'] != 'Sinus') & (df_cn_raw['Msec'] < (len(ecg_signal) / sample_rate) * 1000)]
for row in df_cn_raw.itertuples():
    fig.axes[1].axvspan(row.Msec/1000, row.Msec/1000 + row.Length/1000, 0, 1, color='grey', alpha=.2)
"""

"""
fig, ax = plt.subplots(3, figsize=(10, 6), sharex='col')

ax[0].plot(self.data_upper['xn'], color='black', label='xn')
ax[0].plot(self.data_upper['ymn'][1, 0, :], color='red', label='cA[1]')
ax[0].plot(self.data_upper['ymn_cA_threshed'][1, :], color='dodgerblue', label='cA[1]_threshed')

ax[1].plot(self.data_upper['ymn_cA_threshes'][1], color='dodgerblue', label='cA[1] thresholds')
ax[1].plot(self.data_upper['ymn'][1, 0, :], color='red', label='cA[1]')

ax[2].plot(self.data_upper['ymn_cA_threshed'][1, :], color='red', label='cA[1]_threshed')

for axis in ax:
    axis.legend(loc='upper left')

for w in self.data_upper['rr_windows'][::2]:
    ax[1].axvspan(w[0], w[1], 0, 1, color='grey', alpha=.2)

plt.tight_layout()
"""
