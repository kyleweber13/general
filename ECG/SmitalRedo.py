import neurokit2 as nk
import numpy as np
import pandas as pd
np.seterr(invalid='ignore')
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, stft, resample
from ECG.physionetdb_smital import shade_noise_on_plot, create_epoched_df
from ECG.SmitalRedoPlotting import *
from nwecg.awwf import WwfParams
from nwecg.ecg_quality import pad_to_length
import bottleneck
import datetime
from tqdm import tqdm

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
        # print(f"Level #{i}: {low} - {high} Hz")
        print(f"Level #{i}: cA = {low}Hz lowpass; cD = {low}-{high}Hz")


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


def H_beatwindows(ymn, windows, pad_left=0, tm=2.8, apply_threshold=True):

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


def H_fixedroll(ymn: np.ndarray,
                sample_rate: int,
                apply_threshold: bool = True,
                tm: float or int = 2.8,
                n_sec: int or float = 0.6) -> np.ndarray:
    """
    Wavelet-domain operation corresponding to block H. Uses rolling median with fixed window
    lengths to create wavelet thresholds

    Ignore the zero division error, the number will get clipped anyways in the threshold (according to pywt code on github)
    """

    coefs = np.asarray(ymn)
    coefs_shape = coefs.shape
    data_len = len(coefs[0]) if coefs_shape[0] > 1 else len(coefs)
    n_levels = coefs_shape[0]

    window = int(n_sec * sample_rate)

    if apply_threshold:
        threshes = []

        for decomp_level in range(n_levels):

            # pads window with zeros that get removed post-calculation
            coef = np.concatenate([np.zeros(window), np.abs(coefs[decomp_level])])

            # Equation (3), Smital 2020, using sliding window
            # thresh level padded by window size; does not need padding with upper_data['crop'][0]
            thresh_level = bottleneck.move_median(coef, window + 1, axis=0)[window:] / 0.6745 * tm

            threshes.append(thresh_level)

        ymn_threshed = np.array([pywt.threshold(data=c, value=np.asarray(threshes[i]), mode='garrote') for
                                 i, c in enumerate(ymn)])

        for i in ymn_threshed:
            i[np.argwhere(np.isnan(i))] = 0

    if not apply_threshold:
        ymn_threshed = ymn
        threshes = np.zeros(data_len)

    return np.asarray(threshes), np.asarray(ymn_threshed)


def HW(umn_cA, umn_cD, umn_ca_noise, umn_cd_noise, tm):
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

    def __init__(self,
                 subj, study_code,
                 ecg_filt, sample_rate,
                 clip_snr=True,
                 upper_ca_thresh=True, upper_cd_thresh=True,
                 lower_ca_thresh=True, lower_cd_thresh=True,
                 use_ca=True, use_cd=True,
                 swt1_wavelet='bior2.2', swt2_wavelet='bior2.2',
                 swt3_wavelet='bior4.4', swt4_wavelet='sym4',
                 use_snr_sum=False,
                 n_decomp_levels=4,
                 use_decomp_levels: list or tuple or None = None,
                 fixed_awwf_levels: int or None = None,
                 use_rr_windows: bool = True,
                 roll_window_sec: int or float = 0.6,
                 snr_roll_window_sec: int or float = 2,
                 fixed_tm: int or float or None = None,
                 fixed_awwf_tm: int or float or None = None,
                 ):

        self.subj = subj
        self.study_code = study_code
        self.sample_rate = sample_rate
        self.clip_snr = clip_snr

        self.data_upper = {'xn': ecg_filt, 'ymn': [], 'crop': [0, 0],
                           'r_peaks': [], 'rr_windows': [], 'windows': [],
                           'ymn_cA_threshes': [], 'ymn_cA_threshed': [], 'ymn_cD_threshes': [], 'ymn_cD_threshed': [],
                           'ymn_threshed': [], 's^': [], 'u^mn_cA': [], 'u^mn_cD': [],
                           'snr_raw': [], 'snr_sta': []}
        self.data_lower = {'ymn_cA': [], 'ymn_cD': [], 'crop': [0, 0], 'g^mn_cA': [], 'g^mn_cD': [],
                           'lambda_ymn_cA': [], 'lambda_ymn_cD': [], 'lambda_ymn': [],
                           'yn': [], 'snr_raw': [], 'snr_sta': []}
        self.data_awwf = {'zn': [], 'snr_segments': [], 'params': [], 'snr_raw': [], 'snr_sta': []}
        self.df_med = pd.DataFrame()

        self.upper_ca_thresh = upper_ca_thresh
        self.upper_cd_thresh = upper_cd_thresh
        self.lower_ca_thresh = lower_ca_thresh
        self.lower_cd_thresh = lower_cd_thresh
        self.use_ca = use_ca
        self.use_cd = use_cd
        self.swt1_wavelet = swt1_wavelet
        self.swt2_wavelet = swt2_wavelet
        self.swt3_wavelet = swt3_wavelet
        self.swt4_wavelet = swt4_wavelet
        self.use_snr_sum = use_snr_sum
        self.n_decomp_level = n_decomp_levels
        self.use_decomp_levels = sorted(use_decomp_levels) if \
            use_decomp_levels is not None else np.arange(n_decomp_levels)
        self.fixed_awwf_levels = fixed_awwf_levels
        self.use_rr_windows = use_rr_windows
        self.roll_window_sec = roll_window_sec
        self.snr_roll_window_sec = snr_roll_window_sec
        self.fixed_tm = fixed_tm
        self.fixed_awwf_tm = fixed_awwf_tm

        self.universal_params = WwfParams.universal()

        self.df_snr_awwf = pd.DataFrame()

        self.df_stats = {}

    def run_process(self, time_processing=True):

        t0 = datetime.datetime.now()

        if self.use_rr_windows:
            self.data_upper['r_peaks'] = find_rr_intervals(ecg_raw=self.data_upper['xn'], fs=self.sample_rate)
            self.data_upper['rr_windows'] = flag_rr_windows(r_peaks=self.data_upper['r_peaks'])

        # padding starts for all signals except data_upper['xn']
        wwf_upper_path(data_upper=self.data_upper,
                       n_decomp_levels=self.n_decomp_level,
                       use_decomp_levels=self.use_decomp_levels,
                       tm=self.universal_params.threshold_multiplier if \
                           self.fixed_tm is None else self.fixed_tm,
                       swt1_wavelet=self.swt1_wavelet,
                       swt2_wavelet=self.swt2_wavelet,
                       threshold_ca=self.upper_ca_thresh,
                       threshold_cd=self.upper_cd_thresh,
                       use_rr_windows=self.use_rr_windows,
                       roll_window_sec=self.roll_window_sec,
                       sample_rate=self.sample_rate,
                       quiet=False)

        self.data_lower = wwf_lower_path(data_upper=self.data_upper,
                                         n_decomp_levels=self.n_decomp_level,
                                         use_decomp_levels=self.use_decomp_levels,
                                         tm=self.universal_params.threshold_multiplier if \
                                             self.fixed_tm is None else self.fixed_tm,
                                         swt2_wavelet=self.swt2_wavelet,
                                         threshold_ca=self.lower_ca_thresh,
                                         threshold_cd=self.lower_cd_thresh,
                                         use_ca=self.use_ca,
                                         use_cd=self.use_cd,
                                         use_rr_windows=self.use_rr_windows,
                                         roll_window_sec=self.roll_window_sec,
                                         quiet=False)

        crop_dict_data(data_upper=self.data_upper, data_lower=self.data_lower)

        # first signal-to-noise ratio estimates; used to adjust parameters in AWWF
        self.data_upper['snr_raw'], self.data_upper['snr_sta'] = get_rolling_snr(x=self.data_upper['xn'],
                                                                                 s=self.data_upper['s^'],
                                                                                 window=int(self.snr_roll_window_sec *
                                                                                            self.sample_rate),
                                                                                 use_sum=self.use_snr_sum)

        self.data_lower['snr_raw'], self.data_lower['snr_sta'] = get_rolling_snr(x=self.data_upper['xn'],
                                                                                 s=self.data_lower['yn'],
                                                                                 window=int(self.snr_roll_window_sec *
                                                                                            self.sample_rate),
                                                                                 use_sum=self.use_snr_sum)

        if self.clip_snr:
            self.data_upper['snr_raw'] = clip_array(self.data_upper['snr_raw'], min_val=-20, max_val=30)
            self.data_upper['snr_sta'] = clip_array(self.data_upper['snr_sta'], min_val=-20, max_val=30)
            self.data_lower['snr_raw'] = clip_array(self.data_lower['snr_raw'], min_val=-20, max_val=30)
            self.data_lower['snr_sta'] = clip_array(self.data_lower['snr_sta'], min_val=-20, max_val=30)

        self.data_awwf = adaptive_wwf(signal=self.data_upper['xn'],
                                      data_lower=self.data_lower,
                                      sample_rate=self.sample_rate,
                                      upper_ca_thresh=self.upper_ca_thresh,
                                      upper_cd_thresh=self.upper_cd_thresh,
                                      lower_ca_thresh=self.lower_ca_thresh,
                                      lower_cd_thresh=self.lower_cd_thresh,
                                      use_ca=self.use_ca,
                                      use_cd=self.use_cd,
                                      use_decomp_levels=self.use_decomp_levels,
                                      use_rr_windows=self.use_rr_windows,
                                      fixed_tm=self.fixed_awwf_tm,
                                      fixed_n_levels=self.fixed_awwf_levels if \
                                          self.fixed_awwf_levels is not None else None,
                                      roll_window_sec=self.roll_window_sec,
                                      snr_roll_window_sec=self.snr_roll_window_sec,
                                      quiet=True)

        self.data_awwf['snr_raw'], self.data_awwf['snr_sta'] = get_rolling_snr(x=self.data_upper['xn'],
                                                                               s=self.data_awwf['zn'],
                                                                               window=int(self.snr_roll_window_sec *
                                                                                          self.sample_rate),
                                                                               use_sum=self.use_snr_sum)

        if self.clip_snr:
            self.data_awwf['snr_raw'] = clip_array(self.data_awwf['snr_raw'], min_val=-20, max_val=30)
            self.data_awwf['snr_sta'] = clip_array(self.data_awwf['snr_sta'], min_val=-20, max_val=30)

        if time_processing:
            t1 = datetime.datetime.now()
            dt = (t1 - t0).total_seconds()
            print(f"\n ==========  Processing time ==========")
            print(f"-Processing time = {dt:.1f} seconds")
            print(f"     = {dt / (len(self.data_upper['xn']) / sample_rate / 3600):.1f} sec/hour of data\n")

    def calculate_snr_stat_by_stage(self, describe_colname='50%'):

        df_snr_upper = create_epoched_df(snr_dict=self.data_upper, sample_rate=self.sample_rate, keys=['snr_sta'])
        upper = df_snr_upper.groupby("noise")['snr_sta'].describe()[[describe_colname]]
        upper.loc['diff'] = [upper[describe_colname].iloc[0] - upper[describe_colname].iloc[1]]

        df_snr_lower = create_epoched_df(snr_dict=self.data_lower, sample_rate=self.sample_rate, keys=['snr_sta'])
        lower = df_snr_lower.groupby("noise")['snr_sta'].describe()[[describe_colname]]
        lower.loc['diff'] = [lower[describe_colname].iloc[0] - lower[describe_colname].iloc[1]]

        self.df_snr_awwf = create_epoched_df(snr_dict=self.data_awwf, sample_rate=self.sample_rate, keys=['snr_sta'])
        awwf = self.df_snr_awwf.groupby("noise")['snr_sta'].describe()[[describe_colname]]
        awwf.loc['diff'] = [awwf[describe_colname].iloc[0] - awwf[describe_colname].iloc[1]]

        df_out = pd.DataFrame({'upper': upper[describe_colname].values,
                               'lower': lower[describe_colname].values,
                               "awwf": awwf[describe_colname].values},
                              index=['clean', 'noise', 'diff'])

        return df_out

    def plot_results(self, data_key,
                     original_dict=None,
                     shade_rr_windows=False,
                     mitbih=False,
                     plot_upper=True,
                     plot_lower=True,
                     plot_awwf=True):

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

            ax[1].plot(np.arange(len(self.data_upper['snr_sta']))/self.sample_rate, self.data_upper['snr_sta'],
                       color='red', label='upper', lw=2)

        if plot_lower:
            ax[0].plot(np.arange(len(self.data_lower['yn']))/self.sample_rate, self.data_lower['yn'],
                       label='yn', color='dodgerblue', lw=1.5, zorder=0)

            ax[1].plot(np.arange(len(self.data_lower['snr_sta']))/self.sample_rate, self.data_lower['snr_sta'],
                       color='dodgerblue', label='lower', lw=1.5)

        if plot_awwf:
            ax[0].plot(np.arange(len(self.data_awwf['zn']))/sample_rate, self.data_awwf['zn'], label='zn', color='orange')

            ax[1].plot(np.arange(len(self.data_awwf['snr_sta']))/sample_rate, self.data_awwf['snr_sta'],
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


def calculate_snr(x: np.ndarray,
                  s: np.ndarray):
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


def get_rolling_snr(x: np.ndarray,
                    s: np.ndarray, window: int,
                    use_sum: bool = True,
                    replace_inf_val: float or int = 0):
    """ Calculate the rolling signal-to-noise ratio for the given signal.

        Parameters
        ----------
        x
            pre-processed ECG signal
        s
            Noise-free signal estimate
        window
            Size of rolling window in samples
        use_sum
            boolean. If True, uses sum to calculate SNR. If False, uses variance
        replace_inf_val
            value used to replace infinite values in SNR calculation

        Returns
        -------
        snr
            'raw' (non-averaged) SNR values
        snr_sta
            Short Time Averaged SNR values
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
    rolling_snr, *_ = pad_to_length(x=rolling_snr, length=len(s), mode='edge')

    return rolling_snr, rolling_snr_sta


def get_threshold_crossings(x: np.ndarray,
                            thresholds: list or np.array = (-5, 10, 20, 35, 45)):
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


def adaptive_wwf(signal: np.array or list or tuple,
                 data_lower: dict,
                 sample_rate: int,
                 replace_inf_value: int or float = 0,
                 upper_ca_thresh: bool = True,
                 upper_cd_thresh: bool = True,
                 lower_ca_thresh: bool = True,
                 lower_cd_thresh: bool = True,
                 use_ca: bool = True,
                 use_cd: bool = True,
                 use_decomp_levels: int or None = None,
                 fixed_n_levels: int or None = None,
                 use_rr_windows: bool = True,
                 fixed_tm: int or float or None = None,
                 roll_window_sec: int or float = .6,
                 snr_roll_window_sec: int or float = 2,
                 quiet: bool = True):

    if not quiet:
        print("========== Running AWWF pathway ==========")

    snr_segs = get_threshold_crossings(data_lower['snr_sta'], WwfParams.snr_thresholds())

    awwf_dict = {'zn': np.zeros_like(signal),
                 'snr_segments': snr_segs,
                 'params': [],
                 'r_peaks': np.array([])}

    idx_nan = np.argwhere(np.isnan(data_lower['snr_sta']))
    data_lower['snr_sta'][idx_nan] = 0

    idx_inf = np.isinf(data_lower['snr_sta']).transpose()
    data_lower['snr_sta'][idx_inf] = replace_inf_value

    pad = int(snr_roll_window_sec * sample_rate)
    sig_len = len(signal)

    start_idx = 0

    for end_idx in tqdm(awwf_dict['snr_segments'][1:]):
        # For each segment, apply WWF with parameters based on the SNR of the segment.
        # Smital et al. (2013) table III
        seg_params = WwfParams.for_snr(data_lower['snr_sta'][start_idx])

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
            if use_rr_windows:
                seg_upper['rr_windows'] = flag_rr_windows(r_peaks=seg_upper['r_peaks'])

            if fixed_n_levels is None:
                upper_levels = seg_params.level1
                lower_levels = seg_params.level2

            if fixed_n_levels is not None:
                upper_levels = fixed_n_levels
                lower_levels = fixed_n_levels

            wwf_upper_path(data_upper=seg_upper,
                           tm=seg_params.threshold_multiplier if fixed_tm is None else fixed_tm,
                           n_decomp_levels=upper_levels if fixed_n_levels is None else fixed_n_levels,
                           use_decomp_levels=use_decomp_levels if use_decomp_levels is not None else \
                               np.arange(upper_levels),
                           swt1_wavelet=seg_params.wavelet1,
                           swt2_wavelet=seg_params.wavelet2,
                           threshold_ca=upper_ca_thresh,
                           threshold_cd=upper_cd_thresh,
                           use_rr_windows=use_rr_windows,
                           sample_rate=sample_rate,
                           roll_window_sec=roll_window_sec,
                           quiet=quiet)

            seg_dict = wwf_lower_path(data_upper=seg_upper,
                                      tm=seg_params.threshold_multiplier if fixed_tm is None else fixed_tm,
                                      n_decomp_levels=lower_levels if fixed_n_levels is None else fixed_n_levels,
                                      use_decomp_levels=use_decomp_levels if \
                                          use_decomp_levels is not None else np.arange(lower_levels),
                                      swt2_wavelet=seg_params.wavelet2,
                                      threshold_ca=lower_ca_thresh,
                                      threshold_cd=lower_cd_thresh,
                                      use_ca=use_ca,
                                      use_cd=use_cd,
                                      use_rr_windows=use_rr_windows,
                                      roll_window_sec=roll_window_sec,
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
                   swt1_wavelet: str = 'bior2.2',
                   swt2_wavelet: str = 'bior2.2',
                   threshold_cd: bool = True,
                   threshold_ca: bool = True,
                   use_rr_windows: bool = True,
                   sample_rate: int = 500,
                   roll_window_sec: int or float = 0.6,
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
        swt1_wavelet
            wavelet to use for decomposition and reconstruction in swt1()
        swt2_wavelet
            wavelet to use for decomposition and reconstruction in swt2()
        threshold_cd
            boolean whether or not to apply thresholding to detail coefficients
        threshold_ca
            boolean whether or not to apply thresholding to approximation coefficients
        roll_window_sec
            Length of rolling window used to calculate wavelet thresholds in seconds. If use_rr_windows is True,
            not used
        use_rr_windows
            boolean to use single-beat windowing. If False, uses rolling window of length specified by roll_window_sec
        sample_rate
            sample rate of input signal, Hz
        quiet
            boolean to print parameters/progress to console

        Returns
        -------
        dict_upper: dictionary with lots of data (see below)

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
        -u^mn_cA_threshes:
        -u^mn_cD_threshes:

        -snr
        -windows
        -snr_raw
    """

    if not quiet:
        print(" ========== Running WWF upper pathway ==========")
        print("Settings:")
        print(f"    -Decomposition levels = {n_decomp_levels}")
        print(f"    -Wavelets = {swt1_wavelet}/{swt2_wavelet}")
        print(f"    -Threshold multiplier = {tm}")
        print(f"    -Threshold cA = {threshold_ca}, threshold cD = {threshold_cd}")

    # SWT1
    # Signal gets padded since SWT input signal needs to be a multiple of 2 ** wavelet_level
    data_upper['ymn'], padding_left, padding_right = swt1(xn=data_upper['xn'],
                                                          levels=n_decomp_levels,
                                                          wavelet=swt1_wavelet)

    data_upper['crop'] = [padding_left, -1 if padding_right == 0 else -padding_right]

    # Method that calculates thresholds using windows that contain 1 heartbeat --------
    if use_rr_windows:

        # adjust rr_window indexes to account for padding from above
        data_upper['windows'] = [[i[0] + data_upper['crop'][0], i[1] + data_upper['crop'][0]] for
                                 i in data_upper['rr_windows']]

        # Smital et al. (2013) eq. 2 and 3
        # estimates noise using median of wavelet coefficients' median value and two constants
        # thresholds wavelet coefficients using calculated thresholds
        data_upper['ymn_cA_threshes'], data_upper['ymn_cA_threshed'] = H_beatwindows(ymn=data_upper['ymn'][:, 0],
                                                                                     windows=data_upper['windows'],
                                                                                     tm=tm,
                                                                                     pad_left=data_upper['crop'][0],
                                                                                     apply_threshold=threshold_ca)

        data_upper['ymn_cD_threshes'], data_upper['ymn_cD_threshed'] = H_beatwindows(ymn=data_upper['ymn'][:, 1],
                                                                                     windows=data_upper['windows'],
                                                                                     tm=tm,
                                                                                     pad_left=data_upper['crop'][0],
                                                                                     apply_threshold=threshold_cd)

    if not use_rr_windows:

        data_upper['ymn_cA_threshes'], data_upper['ymn_cA_threshed'] = H_fixedroll(ymn=data_upper['ymn'][:, 0],
                                                                                   sample_rate=sample_rate,
                                                                                   apply_threshold=threshold_ca,
                                                                                   tm=tm,
                                                                                   n_sec=roll_window_sec)

        data_upper['ymn_cD_threshes'], data_upper['ymn_cD_threshed'] = H_fixedroll(ymn=data_upper['ymn'][:, 1],
                                                                                   sample_rate=sample_rate,
                                                                                   apply_threshold=threshold_ca,
                                                                                   tm=tm,
                                                                                   n_sec=roll_window_sec)

    # combined ymn_cA_threshed and ymn_cD_threshed into format needed for inverse SWT
    data_upper['ymn_threshed'] = [[data_upper['ymn_cA_threshed'][i], data_upper['ymn_cD_threshed'][i]] for
                                  i in range(len(data_upper['ymn_cA_threshed']))]

    data_upper['ymn_threshed'] = remove_decomp_levels(coefs=data_upper['ymn_threshed'],
                                                      keep_levels=use_decomp_levels,
                                                      quiet=False)

    # noise-free signal estimate using inverse SWT, used to design Wiener filter in lower pathway
    data_upper['s^'] = pywt.iswt(data_upper['ymn_threshed'], wavelet=swt1_wavelet)

    # estimates wavelet coefficients of noise-free signal
    data_upper['u^mn'], padding_left, padding_right = swt2(signal=data_upper['s^'],
                                                           levels=n_decomp_levels,
                                                           wavelet=swt2_wavelet,
                                                           pad_signal=False)

    data_upper['u^mn_cA'] = np.array([coef[0] for coef in data_upper['u^mn']])
    data_upper['u^mn_cD'] = np.array([coef[1] for coef in data_upper['u^mn']])
    del data_upper['u^mn']

    if use_rr_windows:

        # adjust rr_window indexes to account for padding from above
        data_upper['windows'] = [[i[0] + data_upper['crop'][0], i[1] + data_upper['crop'][0]] for
                                 i in data_upper['rr_windows']]

        # Smital et al. (2013) eq. 2 and 3
        # estimates noise using median of wavelet coefficients' median value and two constants
        # thresholds wavelet coefficients using calculated thresholds
        data_upper['u^mn_cA_threshes'], _ = H_beatwindows(ymn=data_upper['u^mn_cA'],
                                                          windows=data_upper['windows'],
                                                          tm=1,  # tm=1 applies Eq. 2, not 3
                                                          pad_left=data_upper['crop'][0],
                                                          apply_threshold=threshold_ca)

        data_upper['u^mn_cD_threshes'], _ = H_beatwindows(ymn=data_upper['u^mn_cD'],
                                                          windows=data_upper['windows'],
                                                          tm=1,  # tm=1 applies Eq. 2, not 3
                                                          pad_left=data_upper['crop'][0],
                                                          apply_threshold=threshold_ca)

    if not use_rr_windows:

        data_upper['u^mn_cA_threshes'], _ = H_fixedroll(ymn=data_upper['u^mn_cA'],
                                                        sample_rate=sample_rate,
                                                        apply_threshold=threshold_ca,
                                                        tm=1,
                                                        n_sec=roll_window_sec)

        data_upper['u^mn_cD_threshes'], _ = H_fixedroll(ymn=data_upper['u^mn_cD'],
                                                        sample_rate=sample_rate,
                                                        apply_threshold=threshold_ca,
                                                        tm=1,
                                                        n_sec=roll_window_sec)


def wwf_lower_path(data_upper: dict,
                   n_decomp_levels: int = 4,
                   use_decomp_levels: int or None = None,
                   tm: int or float = 2.8,
                   swt2_wavelet: str = 'bior2.2',
                   threshold_cd: bool = True,
                   threshold_ca: bool = True,
                   use_ca: bool = True,
                   use_cd: bool = True,
                   use_rr_windows: bool = True,
                   roll_window_sec: int or float = 0.6,
                   quiet: bool = True):
    """ Lower path in Smital et al. (2013) figure 2. Calls swt2(), HW(), and iswt2().

        Parameters
        ----------
        data_upper
            dictionary returned from wwf_upper_path
        n_decomp_levels
            number of decompositions to perform in SWT
        use_decomp_levels
        tm
            'threshold multipllier'; constant by which thresholds in Smital et al. (2013) eq. 3 are multiplied
        swt2_wavelet
            wavelet to use for decomposition and reconstruction in SWT2()
        threshold_cd
            boolean whether or not to apply thresholding to detail coefficients
        threshold_ca
            boolean whether or not to apply thresholding to approximation coefficients
        use_ca
        use_cd
        roll_window_sec
            Length of rolling window used to calculate wavelet thresholds in seconds. If use_rr_windows is True,
            not used
        use_rr_windows
            boolean to use single-beat windowing. If False, uses rolling window of length specified by roll_window_sec
        quiet
            boolean to print parameters/progress to console

        Returns
        -------
        data_lower: dictionary with lots of data (see below)

        Keys
        ----
        -ymn_cA:
        -ymn_cD:
        -ymn_cA_threshes:
        -ymn_cD_threshes:
        -crop:
        -g^mn_cA:
        -g^mn_cD:
        -lambda_ymn_cA:
        -lambda_ymn_cD:
        -lambda_ymn:
        -yn:
        -snr_raw:
        -snr:

    """

    if not quiet:
        print(" ========== Running WWF lower pathway ==========")
        print("Settings:")
        print(f"    -Decomposition levels = {n_decomp_levels}")
        print(f"    -Wavelet = {swt2_wavelet}")
        print(f"    -Threshold multiplier = {tm}")
        print(f"    -Threshold cA = {threshold_ca}, threshold cD = {threshold_cd}")

    data_lower = {}

    ymn, padding_left, padding_right = swt2(signal=data_upper['xn'],
                                            levels=n_decomp_levels,
                                            wavelet=swt2_wavelet,
                                            pad_signal=True)

    # splits ymn above into approximation and detail coefficients
    data_lower['ymn_cA'] = ymn[:, 0]
    data_lower['ymn_cD'] = ymn[:, 1]

    if use_rr_windows:

        # Smital et al. (2013) eq. 2 and 3
        # estimates noise using median of wavelet coefficients' median value and two constants
        # thresholds wavelet coefficients using calculated thresholds
        data_lower['ymn_cA_threshes'], _ = H_beatwindows(ymn=data_lower['ymn_cA'],
                                                         windows=data_upper['rr_windows'],
                                                         tm=1,  # tm=1 applies Smital et al. 2013 Eq. 2, not 3
                                                         pad_left=data_upper['crop'][0],
                                                         apply_threshold=threshold_ca)

        data_lower['ymn_cD_threshes'], _ = H_beatwindows(ymn=data_lower['ymn_cD'],
                                                         windows=data_upper['rr_windows'],
                                                         tm=1,  # tm=1 applies Smital et al. 2013 Eq. 2, not 3
                                                         pad_left=data_upper['crop'][0],
                                                         apply_threshold=threshold_ca)

    if not use_rr_windows:

        data_lower['ymn_cA_threshes'], _ = H_fixedroll(ymn=data_lower['ymn_cA'],
                                                       sample_rate=sample_rate,
                                                       apply_threshold=threshold_ca,
                                                       tm=1,  # tm=1 applies Smital et al. 2013 Eq. 2, not 3
                                                       n_sec=roll_window_sec)

        data_lower['ymn_cD_threshes'], _ = H_fixedroll(ymn=data_lower['ymn_cD'],
                                                       sample_rate=sample_rate,
                                                       apply_threshold=threshold_ca,
                                                       tm=1,  # tm=1 applies Smital et al. 2013 Eq. 2, not 3
                                                       n_sec=roll_window_sec)

    # crop indexes to undo signal padding (required for SWT)
    data_lower['crop'] = [padding_left, -1 if padding_right == 0 else -padding_right]

    # Smital et al. (2013) eq. 4 -------------------
    # Wiener correction factor based on thresholds
    # tm set to 1 since it is set to 1 in the call to H() for umn_cA_noise/umn_cD_noise estimates

    # what I've been running mostly ##
    data_lower['g^mn_cA'], data_lower['g^mn_cD'] = HW(umn_cA=data_upper['u^mn_cA'],
                                                      umn_cD=data_upper['u^mn_cD'],
                                                      #umn_ca_noise=data_upper['u^mn_cA_threshes'],  #ymn/tm
                                                      #umn_cd_noise=data_upper['u^mn_cD_threshes'],  #ymn/tm
                                                      umn_ca_noise=data_lower['ymn_cA_threshes'],
                                                      umn_cd_noise=data_lower['ymn_cD_threshes'],
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
    data_lower['yn'] = iswt2(lambda_ymn=data_lower['lambda_ymn'], wavelet=swt2_wavelet)

    return data_lower


def crop_dict_data(data_upper: dict, data_lower: dict):

    data_len = len(data_upper['xn'])

    # Crop upper data --------------------------------
    for key in ['ymn', 'ymn_cA_threshes', 'ymn_cA_threshed', 'ymn_cD_threshes',
                'ymn_cD_threshed', 's^', 'u^mn_cA', 'u^mn_cD', 'u^mn_cA_threshes', 'u^mn_cD_threshes']:

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

    for key in ['ymn_cA', 'ymn_cD', 'ymn_cA_threshes', 'ymn_cD_threshes',
                'g^mn_cA', 'g^mn_cD', 'lambda_ymn_cA', 'lambda_ymn_cD', 'yn', 'snr_sta']:

        if key in data_lower.keys():

            if len(data_lower[key].shape) > 1:
                if len(data_lower[key][0]) > data_len:
                    data_lower[key] = data_lower[key][:, data_lower['crop'][0]:data_lower['crop'][1]]

            if len(data_lower[key].shape) == 1:
                if len(data_lower[key]) > data_len:
                    data_lower[key] = data_lower[key][data_lower['crop'][0]:data_lower['crop'][1]]

    for key in ['lambda_ymn']:
        if key in data_lower.keys():
            if len(data_lower[key][0][0]) > data_len:
                data_lower[key] = [(data_lower[key][i][0][data_lower['crop'][0]:data_lower['crop'][1]],
                                    data_lower[key][i][1][data_lower['crop'][0]:data_lower['crop'][1]])
                                   for i in range(len(data_lower[key]))]


def clip_array(arr: np.array,
               min_val: int or float = -50,
               max_val: int or float = 30):

    return np.clip(a=arr, a_min=min_val if min_val is not None else min(arr),
                   a_max=max_val if max_val is not None else max(arr), out=arr)


def append_settings_csv(smital_obj,
                        pathway: str):

    df_settings = pd.read_csv(pathway)

    df_row = pd.DataFrame({"study_code": smital_obj.study_code,
                           "subject_id": smital_obj.subj,
                           "data_key": data_key, "sample_rate": sample_rate,
                           "low_f_cut": low_f_cut, "notch_cut": notch_cut,
                           'use_ca': smital_obj.use_ca, 'use_cd': smital_obj.use_cd,
                           "upper_ca_thresh": smital_obj.upper_ca_thresh,
                           'upper_cd_thresh': smital_obj.upper_cd_thresh,
                           'tm': smital_obj.universal_params.threshold_multiplier if \
                               smital_obj.fixed_tm is None else smital_obj.fixed_tm,
                           'awwf_tm': smital_obj.fixed_awwf_tm,
                           'upper_n_decomp': smital_obj.n_decomp_level, 'upper_decomp_level': 'all',
                           'lower_ca_thresh': smital_obj.lower_ca_thresh, 'lower_cd_thresh': smital_obj.lower_cd_thresh,
                           'lower_n_decomp': smital_obj.n_decomp_level, 'lower_decomp_level': 'all',
                           'swt1_wavelet': smital_obj.swt1_wavelet, 'swt2_wavelet': smital_obj.swt2_wavelet,
                           'swt3_wavelet': smital_obj.swt3_wavelet, 'swt4_wavelet': smital_obj.swt4_wavelet,
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


def calculate_snr_beats(df, snr, window_samples):

    window_samples = int(window_samples)
    snr_len = len(snr)
    s = []

    for row in df.itertuples():
        start_idx = row.idx - window_samples if row.idx - window_samples >= 0 else 0
        end_idx = row.idx + window_samples if row.idx + window_samples < snr_len else -1

        w = snr[start_idx:end_idx]
        m = np.mean(w)

        s.append(m)

    df['snr'] = s


def arrhythmia_snr_describe(smital_obj, sample_rate, snr_key='snr_raw', stage='lower', show_plot=True):

    fig = None

    obj_dict = {'lower': smital_obj.data_lower, 'upper': smital_obj.data_upper, 'awwf': smital_obj.data_awwf}

    ignore_events = ['[', '!', ']', '(', ')', 'p', 't', 'u', '`', "'", '~', "+", 's', 'T', '*', 'D', '=', '"', '@']
    calculate_snr_beats(df=nst_data[data_key]['annots'],
                        snr=obj_dict[stage][snr_key],
                        window_samples=sample_rate * .05)
    df_arr_use = nst_data[data_key]['annots'].loc[~nst_data[data_key]['annots']['beat_type'].isin(ignore_events)]
    df_arr_use['true_snr'] = gs_snr[df_arr_use['idx']]
    df_arr_use['sinus'] = df_arr_use['beat_type'] == 'N'

    df_beats_desc = df_arr_use.groupby(["true_snr", "sinus"])['snr'].describe()
    df_beattype_desc = df_arr_use.groupby(["true_snr", "beat_type"])['snr'].describe()

    if show_plot:
        fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, .5]}, sharey='row', figsize=(12, 8))
        df_arr_use.boxplot(by=['true_snr', 'beat_type'], column='snr', ax=ax[0])
        df_arr_use.boxplot(by=['true_snr'], column='snr', ax=ax[1])
        ax[0].set_ylabel("SNR")

        try:
            plt.suptitle(f"MITBIH{subj} ({int(data_key)}dB noise): Sinus vs. Arrhythmic Beats ({snr_key}, {stage} pathway)")
        except ValueError:
            plt.suptitle(f"MITBIH{subj} (multi-dB noise): Sinus vs. Arrhythmic Beats ({snr_key}, {stage} pathway)")

        for i in range(2):
            ax[i].set_title("")
            try:
                ax[i].axhline(int(data_key), color='red', linestyle='dashed')
            except ValueError:
                pass

            ax[i].axhline(24, color='limegreen', linestyle='dashed')
            ax[i].set_xlabel("")

    return df_beats_desc.reset_index(), df_beattype_desc.reset_index(), df_arr_use, fig


def peak_validation(input_signal, test_signal, fs, testsig_label='zn', show_plot=True, min_height=0.5):

    r = find_rr_intervals(ecg_raw=test_signal, fs=fs)
    r = r[np.argwhere(np.abs(test_signal)[r] >= min_height).flatten()]
    rr = 60 * (np.diff(r) / sample_rate)

    r_ref = find_rr_intervals(ecg_raw=input_signal, fs=fs)
    r_ref = r_ref[np.argwhere(np.abs(input_signal)[r_ref] >= min_height).flatten()]
    rr_ref = 60 * (np.diff(r_ref) / sample_rate)

    c_dict = {'s^': 'red', 'yn': 'limegreen', 'zn': 'orange'}

    if show_plot:
        fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

        ax[0].plot(input_signal, color='black', label='input')
        ax[0].scatter(r_ref, input_signal[r_ref] + .2, marker='v', color='red')

        ax[1].plot(test_signal, color=c_dict[testsig_label], label=testsig_label)
        ax[1].scatter(r, test_signal[r] + .2, marker='v', color='red')

        ax[2].scatter(r[:-1], rr, label=testsig_label, color=c_dict[testsig_label], marker='o')
        ax[2].scatter(r_ref[:-1], rr_ref, label='bandpassed', color='black', marker='v')

        ax[2].set_ylabel("HR")
        ax[2].grid()

        for a in ax:
            a.legend(loc='upper left')
            shade_noise_on_plot(ax=a, sample_rate=sample_rate, units='datapoint')

    plt.tight_layout()


def validate_butqdb(annots, snr_arr, q2q3=5, q1q2=18):

    gs_annot = np.ones(len(snr_arr))

    annots = annots.copy()
    annots = annots.loc[annots['start'] < len(snr_arr)]
    end_idx = annots.iloc[-1]['start']

    gs_annot = gs_annot[:end_idx]
    snr_arr = snr_arr[:end_idx]

    for row in annots.loc[annots['quality'] > 1].itertuples():
        gs_annot[row.start:row.end] = row.quality

    q = np.ones(end_idx)
    q[snr_arr < q2q3] = 3
    q[(snr_arr >= q2q3) & (snr_arr < q1q2)] = 2

    perc_acc = (q == gs_annot).sum() / (len(q)/100)
    overest = (q < gs_annot).sum() / (len(q)/100)
    underest = (q > gs_annot).sum() / (len(q)/100)
    print(f"Analysis agrees with expertly annotated quality categorization {perc_acc:.2f}% of the time")
    print(f"    -Overestimates quality {overest:.2f}% of the time")
    print(f"    -Underestimates quality {underest:.2f}% of the time")

    return gs_annot, q


def snr_by_beats_wholedataset():
    sinus_lower = {}
    arrs_lower = {}
    sinus_awwf = {}
    arrs_awwf = {}

    for data_key in ['_6', '00', '06', '12', '24']:
        ecg_signal = resample_signal(signal=nst_data[data_key]['ecg'], old_rate=360, new_rate=sample_rate)
        ecg_signal = filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=low_f_cut, order=100)
        # ecg_signal = filter_notch(signal=ecg_signal, sample_rate=sample_rate, freq=notch_cut)

        if data_key != 'test':
            gs_snr = generate_timeseries_noise_value(signal=ecg_signal, noise_key=data_key, sample_rate=sample_rate,
                                                     noise_indexes=(300, 540, 780, 1020, 1260, 1500, 1740))
        if data_key == 'test':
            gs_snr = nst_data['test']['gs_snr']

        nst_data[data_key]['annots']['idx'] = [int(i * sample_rate / 360) for i in nst_data[data_key]['annots']['idx']]

        try:
            nst_snr_og[data_key]['snr'] = resample_signal(signal=nst_snr_og[data_key]['snr'], old_rate=360,
                                                          new_rate=sample_rate)
        except KeyError:
            pass

        self = Smital(study_code=study_code,
                      subj=subj,
                      clip_snr=False,
                      ecg_filt=ecg_signal, sample_rate=sample_rate,
                      upper_ca_thresh=True, upper_cd_thresh=True,
                      lower_ca_thresh=True, lower_cd_thresh=True,
                      use_ca=True, use_cd=True,
                      swt1_wavelet='bior2.2', swt2_wavelet='bior2.2',
                      n_decomp_levels=4,
                      use_decomp_levels=None,
                      use_snr_sum=False,
                      fixed_awwf_levels=None,
                      fixed_tm=None,
                      fixed_awwf_tm=None,
                      use_rr_windows=False,
                      roll_window_sec=1,
                      snr_roll_window_sec=2,
                      )
        self.run_process(time_processing=True)

        self.df_med = self.calculate_snr_stat_by_stage().round(2)

        """
        df_cn = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/cardiac_navigator_screened/{subj}_01_CardiacNavigator_Screened.csv")
        df_cn_raw = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/CardiacNavigator/{subj}_Events.csv", delimiter=';')
        df_cn_raw = df_cn_raw.loc[(df_cn_raw['Type'] != 'Sinus') & (df_cn_raw['Msec'] < (len(ecg_signal) / sample_rate) * 1000)]
        for row in df_cn_raw.itertuples():
            fig.axes[2].axvspan(row.Msec/1000*sample_rate, row.Msec/1000*sample_rate + row.Length/1000*sample_rate, 0, 1, color='red', alpha=.2)
            print(f"{row.Msec/1000*sample_rate} || {row.Type}")
        """

        # gs_annots, q = validate_butqdb(annots=data_out['annot'], snr_arr=self.data_lower['snr_sta'])

        # plot_thresholds(coef='cA', level=1)
        # fig = plot_results(smital_obj=self, sample_rate=sample_rate, data_key=data_key, nst_data=nst_data, nst_snr=nst_snr, gs_snr=gs_snr, snr_key='snr_raw')

        df_beats_desc, df_beattype_desc, df_arr_use, fig_lower = arrhythmia_snr_describe(smital_obj=self,
                                                                                         sample_rate=sample_rate,
                                                                                         snr_key='snr_sta',
                                                                                         stage='lower', show_plot=False)
        sinus_lower[data_key] = df_beats_desc
        arrs_lower[data_key] = df_beattype_desc

        df_beats_desc, df_beattype_desc, df_arr_use, fig_awwf = arrhythmia_snr_describe(smital_obj=self,
                                                                                        sample_rate=sample_rate,
                                                                                        snr_key='snr_raw', stage='awwf',
                                                                                        show_plot=False)
        sinus_awwf[data_key] = df_beats_desc
        arrs_awwf[data_key] = df_beattype_desc

        # fig_lower.axes[0].set_ylim(-55, 40)
        # fig_awwf.axes[0].set_ylim(-55, 40)

        # peak_validation(input_signal=ecg_signal, test_signal=self.data_awwf['zn'], testsig_label='zn', fs=sample_rate, min_height=.5)

        # df_settings = append_settings_csv(smital_obj=self, pathway="C:/Users/ksweber/Desktop/smital_settings.csv")

    df_sinuslow = sinus_lower['_6'].copy()
    df_arrslow = arrs_lower['_6'].copy()
    df_sinusawwf = sinus_awwf['_6'].copy()
    df_arrsawwf = arrs_awwf['_6'].copy()

    for key in ['00', '06', '12']:
        df_sinuslow = pd.concat([df_sinuslow, sinus_lower[key].loc[sinus_lower[key]['true_snr'] != 24]])
        df_arrslow = pd.concat([df_arrslow, arrs_lower[key].loc[arrs_lower[key]['true_snr'] != 24]])
        df_sinusawwf = pd.concat([df_sinusawwf, sinus_awwf[key].loc[sinus_awwf[key]['true_snr'] != 24]])
        df_arrsawwf = pd.concat([df_arrsawwf, arrs_awwf[key].loc[arrs_awwf[key]['true_snr'] != 24]])
    df_sinuslow = pd.concat([df_sinuslow, sinus_lower['24']])
    df_arrslow = pd.concat([df_arrslow, arrs_lower['24']])
    df_sinusawwf = pd.concat([df_sinusawwf, sinus_awwf['24']])
    df_arrsawwf = pd.concat([df_arrsawwf, arrs_awwf['24']])

    c_dict = {-6: 'red', 0: 'orange', 6: 'gold', 12: 'dodgerblue', 24: 'purple'}
    for i in [-6, 0, 6, 12, 24]:
        d = df_arrsawwf.loc[df_arrsawwf['true_snr'] == i]
        plt.scatter(d['beat_type'], d['50%'], label=f"{i}dB_awwf", color=c_dict[i])
        d = df_arrslow.loc[df_arrslow['true_snr'] == i]
        plt.scatter(d['beat_type'], d['50%'], label=f"{i}dB_lower", color=c_dict[i], marker='x')
        plt.axhline(y=i, color=c_dict[i], linestyle='dashed')
    plt.legend(loc='upper left')
    plt.xlabel("beat type")
    plt.xlim(-1.5, )
    plt.ylabel('median SNR')
    plt.ylim(-20, 30)
    plt.grid()
    plt.title(f"MITBIH-NST_{subj}")


""" ==================================== FUNCTION CALLS ==================================== """

# for lower pathway SNR
best_settings = {'low_f_cut': 3,
                 "upper_ca_thresh": True, "upper_cd_thresh": True,
                 "lower_ca_thresh": True, 'lower_cd_thresh': True,
                 'use_ca': True, 'use_cd': True,
                 'swt1_wavelet': 'bior2.2', 'swt2_wavelet': 'bior2.2',
                 'n_decomp_levels': 4, "use_decomp_levels": None,
                 "use_snr_sum": False,
                 'use_rr_windows': False,
                 'fixed_awwf_levels': None,
                 'fixed_tm': None,
                 'fixed_awwf_tm': 1,
                 'roll_window_sec': 1,
                 'snr_roll_window_sec': 2,
                 }

study_code = 'MITBIHNST'
sample_rate = 500
data_key = 'test'
low_f_cut = 3  # .67
notch_cut = 60

# Preprocessing ---------

"""
import nimbalwear
subj = 'OND09_SBH0336'
n_hours = 2

ecg = nimbalwear.Device()
ecg.import_edf(f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{subj}_01_BF36_Chest.edf")
ecg.signals[0] = ecg.signals[0][:int(250*3600*n_hours)]

from ECG.RunSmital_kyle import process_snr
snr = process_snr(ecg_signal=ecg.signals[0],
                  sample_rate=250,
                  start_timestamp=ecg.header['start_datetime'],
                  progress_bar=True,
                  window_len=3600,
                  overlap_secs=2,
                  thresholds=(5, 18),
                  data_keys=None)

ecg_signal = resample_signal(signal=ecg.signals[0], old_rate=250, new_rate=sample_rate)
ecg_signal = filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=low_f_cut, order=100)
# ecg_signal = filter_lowpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_high=40, order=100)
# ecg_signal = filter_notch(signal=ecg_signal, sample_rate=sample_rate, freq=notch_cut)
"""

"""
snr_orig = nimbalwear.Device()
snr_orig.import_edf(f"W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/{subj}_01_snr.edf")
snr_orig = resample_signal(signal=snr_orig.signals[0][:int(250*3600*n_hours)], old_rate=250, new_rate=sample_rate)
"""

# resamples signal to 512 Hz to match Smital et al. (2013)

"""
ecg_signal = resample_signal(signal=nst_data[data_key]['ecg'], old_rate=360, new_rate=sample_rate)
ecg_signal = filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=low_f_cut, order=100)
# ecg_signal = filter_notch(signal=ecg_signal, sample_rate=sample_rate, freq=notch_cut)

if data_key != 'test':
    gs_snr = generate_timeseries_noise_value(signal=ecg_signal, noise_key=data_key, sample_rate=sample_rate, noise_indexes=(300, 540, 780, 1020, 1260, 1500, 1740))
if data_key == 'test':
    gs_snr = nst_data['test']['gs_snr']

max_annot_idx = nst_data[data_key]['annots']['idx'].max()
if max_annot_idx < len(ecg_signal):
    nst_data[data_key]['annots']['idx'] = [int(i * sample_rate / 360) for i in nst_data[data_key]['annots']['idx']]

try:
    nst_snr_og[data_key]['snr'] = resample_signal(signal=nst_snr_og[data_key]['snr'], old_rate=360, new_rate=sample_rate)
except KeyError:
    pass
"""

self = Smital(study_code=study_code,
              subj=subj,
              clip_snr=False,
              ecg_filt=ecg_signal, sample_rate=sample_rate,
              upper_ca_thresh=True, upper_cd_thresh=True,
              lower_ca_thresh=True, lower_cd_thresh=True,
              use_ca=True, use_cd=True,
              swt1_wavelet='bior2.2', swt2_wavelet='bior2.2',
              n_decomp_levels=4,
              use_decomp_levels=None,
              use_snr_sum=False,
              fixed_awwf_levels=None,
              fixed_tm=None,
              fixed_awwf_tm=None,
              use_rr_windows=False,
              roll_window_sec=1,
              snr_roll_window_sec=2,
              )
self.run_process(time_processing=True)

self.df_med = self.calculate_snr_stat_by_stage().round(2)

"""
df_cn = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/cardiac_navigator_screened/{subj}_01_CardiacNavigator_Screened.csv")
df_cn_raw = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/CardiacNavigator/{subj}_Events.csv", delimiter=';')
df_cn_raw = df_cn_raw.loc[(df_cn_raw['Type'] != 'Sinus') & (df_cn_raw['Msec'] < (len(ecg_signal) / sample_rate) * 1000)]
for row in df_cn_raw.itertuples():
    fig.axes[2].axvspan(row.Msec/1000*sample_rate, row.Msec/1000*sample_rate + row.Length/1000*sample_rate, 0, 1, color='red', alpha=.2)
    print(f"{row.Msec/1000*sample_rate} || {row.Type}")
"""

# gs_annots, q = validate_butqdb(annots=data_out['annot'], snr_arr=self.data_lower['snr_sta'])

# plot_thresholds(smital_obj=self, coef='cA', level=1)
# fig = plot_results(smital_obj=self, sample_rate=sample_rate, data_key=data_key, nst_data=None, nst_snr=None, gs_snr=None, snr_key='snr_sta')
# fig = plot_results(smital_obj=self, sample_rate=sample_rate, data_key=data_key, nst_data=nst_data, nst_snr=nst_snr, gs_snr=gs_snr, snr_key='snr_raw')

# df_beats_desc, df_beattype_desc, df_arr_use, fig_lower = arrhythmia_snr_describe(snr_key='snr_raw', stage='awwf', show_plot=True, smital_obj=self, sample_rate=sample_rate)
