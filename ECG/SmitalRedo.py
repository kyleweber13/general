import neurokit2 as nk
import numpy as np
import pywt
import matplotlib.pyplot as plt
from ECG.physionetdb_smital import create_epoched_df
from typing import Callable, Iterable, Tuple, List
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from ECG.physionetdb_smital import shade_noise_on_plot
from nwecg.signal_utils import get_threshold_crossings
from nwecg.awwf import WwfParams, wwf
from nwecg.ecg_quality import get_rolling_snr, pad_to_length
from nwecg.awwf import pad_to_length
import bottleneck

""" ==================================== PRE-PROCESSING ==================================== """


def resample_signal(signal, old_rate, new_rate):

    from scipy.signal import resample

    print(f"\nResampling signal from {old_rate} to {new_rate} Hz...")

    ratio = new_rate / old_rate
    out = resample(x=signal, num=int(len(signal) * ratio))

    return out


def filter_highpass(signal: np.ndarray, sample_rate: float, cutoff_low: float = .67, order: int = 100):

    nyquist_freq = 0.5 * sample_rate
    sos = butter(N=order, Wn=[cutoff_low/nyquist_freq], btype="highpass", output="sos")

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

    r_peaks = nk.ecg_peaks(ecg_cleaned=ecg_raw, sampling_rate=fs)[1]['ECG_R_Peaks']

    delta_rr = np.diff(r_peaks)
    # delta_rr = np.concatenate([np.array([delta_rr[0]]), delta_rr])

    return r_peaks


def flag_rr_windows(r_peaks):
    """ Creates 2D array for windows around detected R peaks. The start of each window is the midpoint between
        beat[i] and beat[i-1], while the end of each window is the midpoint between beat[i] and beat[i+1]
    """

    print(f"-Flagging {len(r_peaks)} windows that each contain 1 heartbeat...")

    windows = [[0, int(np.mean([r_peaks[0], r_peaks[1]]))]]  # []
    for p1, p2, p3 in zip(r_peaks[:], r_peaks[1:], r_peaks[2:]):
        rr_pre = p2 - int((p2 - p1) / 2)
        rr_post = p2 + int((p3 - p2) / 2)

        windows.append([rr_pre, rr_post])

    return windows


""" ==================================== PROCESSING ==================================== """


def swt1(xn, level=4, wavelet='bior2.2'):

    # SWT requires that the signal has length which is a multiple of 2 ** level.
    n = 2 ** max(level, 5)
    required_length = int(np.ceil((len(xn) + 120) / n) * n)
    x_padded, padding_left, padding_right = pad_to_length(xn, required_length, mode='reflect')

    max_level = pywt.swt_max_level(len(x_padded))

    swc = pywt.swt(x_padded, level=np.min([level, max_level]), wavelet=wavelet, start_level=0)

    return np.asarray(swc), padding_left, padding_right


def iswt1(output_from_H, wavelet='bior2.2'):

    return pywt.iswt(output_from_H, wavelet=wavelet)


def swt2(signal, level=4, wavelet='bior2.2', pad_signal=True):

    # SWT requires that the signal has length which is a multiple of 2 ** level.
    n = 2 ** max(level, 5)
    required_length = int(np.ceil((len(signal) + 120) / n) * n)

    if pad_signal:
        x_padded, padding_left, padding_right = pad_to_length(signal, required_length, mode='reflect')
    if not pad_signal:
        x_padded = signal
        padding_right = 0
        padding_left = 0

    max_level = pywt.swt_max_level(len(x_padded))

    swc = pywt.swt(x_padded, level=np.min([level, max_level]), wavelet=wavelet, start_level=0)

    return np.asarray(swc), padding_left, padding_right


def iswt2(lambda_ymn, wavelet='bior2.2'):

    return pywt.iswt(lambda_ymn, wavelet=wavelet)


def H(ymn, rr_windows, pad_left=0, tm=2.8, apply_threshold=True):
    coefs = np.asarray(ymn)
    coefs_shape = coefs.shape
    data_len = len(coefs[0]) if coefs_shape[0] > 1 else len(coefs)
    n_levels = coefs_shape[0]

    # threshold calculation -------------
    threshes = []
    for decomp_level in range(n_levels):
        thresh_level = np.zeros(pad_left)
        coef = np.abs(coefs[decomp_level])

        for window in rr_windows:
            # Smital et al. (2013), eq. 2 and 3 (combined)
            data_window = np.abs(coef[window[0]:window[1]])
            med_val = np.median(data_window)
            t = med_val / .6745 * tm
            thresh_level = np.append(thresh_level, np.array([t] * int(window[1] - window[0])))

        if len(thresh_level) < data_len:
            thresh_level = np.append(thresh_level, np.zeros(data_len - len(thresh_level)))

        threshes.append(thresh_level)

    if apply_threshold:
        ymn_threshed = np.array([pywt.threshold(data=c, value=np.asarray(threshes[i]), mode='garrote') for
                                 i, c in enumerate(ymn)])

    if not apply_threshold:
        ymn_threshed = np.array([])

    return np.asarray(threshes), np.asarray(ymn_threshed)


def WH(umn_cA, umn_cD, umn_ca_noise, umn_cd_noise, tm):
    """ 'Wiener filter in the wavelet domain.' """

    print("HW block. figure out correct sigma vm values (what data??)")
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


def wwf_upper_path(data_upper: dict, wavelet_level: int = 4, tm: int or float = 2.8, wavelet: str = 'bior2.2'):
    """ Upper path in Smital et al. (2013) figure 2. Calls swt1(), H(), iswt1(), and swt2().

        Parameters
        ----------
        data_upper
            dictionary containing filtered ECG signal 'xn'
        wavelet_level
            number of decompositions to perform in SWT
        tm
            'threshold multipllier'; constant by which thresholds in Smital et al. (2013) eq. 3 are multiplied
        wavelet
            wavelet to use for decomposition and reconstruction

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

    print("-Running WWF upper pathway...")

    # SWT1
    # Signal gets padded since SWT input signal needs to be a multiple of 2 ** wavelet_level
    data_upper['ymn'], padding_left, padding_right = swt1(xn=data_upper['xn'],
                                                          level=wavelet_level,
                                                          wavelet=wavelet)
    data_upper['crop'] = [padding_left, -1 if padding_right == 0 else -padding_right]

    # peak detection used for wavelet thresholding windows
    data_upper['r_peaks'] = find_rr_intervals(ecg_raw=data_upper['xn'], fs=sample_rate)
    data_upper['rr_windows'] = flag_rr_windows(r_peaks=data_upper['r_peaks'])

    data_upper['r_peaks_pad'] = data_upper['r_peaks'] + data_upper['crop'][0]
    data_upper['rr_windows_pad'] = [[i[0] + data_upper['crop'][0], i[1] + data_upper['crop'][0]] for
                                    i in data_upper['rr_windows']]

    # Smital et al. (2013) eq. 2 and 3
    # estimates noise using median of wavelet coefficients' median value and two constants
    # thresholds wavelet coefficients using calculated thresholds
    data_upper['ymn_cA_threshes'], data_upper['ymn_cA_threshed'] = H(ymn=data_upper['ymn'][:, 0],  # cA
                                                                     rr_windows=data_upper['rr_windows_pad'],
                                                                     tm=tm,
                                                                     pad_left=data_upper['crop'][0],
                                                                     apply_threshold=True)

    data_upper['ymn_cD_threshes'], data_upper['ymn_cD_threshed'] = H(ymn=data_upper['ymn'][:, 1],  # cD
                                                                     rr_windows=data_upper['rr_windows_pad'],
                                                                     tm=tm,
                                                                     pad_left=data_upper['crop'][0],
                                                                     apply_threshold=True)

    # combined ymn_cA_threshed and ymn_cD_threshed into format needed for inverse SWT
    data_upper['ymn_threshed'] = [(cA_level, cD_level) for cA_level, cD_level in
                                  zip(data_upper['ymn_cA_threshed'], data_upper['ymn_cD_threshed'])]

    # noise-free signal estimate using inverse SWT, used to design Wiener filter in lower pathway
    data_upper['s^'] = pywt.iswt(data_upper['ymn_threshed'], wavelet=wavelet)

    # estimates wavelet coefficients of noise-free signal
    data_upper['u^mn'], padding_left, padding_right = swt2(signal=data_upper['s^'],
                                                           level=wavelet_level,
                                                           wavelet=wavelet,
                                                           pad_signal=False)
    data_upper['u^mn_cA'] = np.array([coef[0] for coef in data_upper['u^mn']])
    data_upper['u^mn_cD'] = np.array([coef[1] for coef in data_upper['u^mn']])
    del data_upper['u^mn']

    # data cropping to undo signal padding that is required for SWT-----
    print("Sort out padding")
    """
    data_upper['ymn_cA_threshes'] = data_upper['ymn_cA_threshes'][:, data_upper['crop1'][0]:data_upper['crop1'][1]]
    data_upper['ymn_cD_threshes'] = data_upper['ymn_cD_threshes'][:, data_upper['crop1'][0]:data_upper['crop1'][1]]
    data_upper['ymn_cA_threshed'] = data_upper['ymn_cA_threshed'][:, data_upper['crop1'][0]:data_upper['crop1'][1]]
    data_upper['ymn_cD_threshed'] = data_upper['ymn_cD_threshed'][:, data_upper['crop1'][0]:data_upper['crop1'][1]]
    data_upper['ymn_threshed'] = [(coefs[0][data_upper['crop1'][0]:data_upper['crop1'][1]],
                                  coefs[1][data_upper['crop1'][0]:data_upper['crop1'][1]]) for
                                 coefs in data_upper['ymn_threshed']]
    data_upper['s^'] = data_upper['s^'][data_upper['crop1'][0]:data_upper['crop1'][1]]
    data_upper['u^mn'] = [(coefs[0][data_upper['crop'][0]:data_upper['crop'][1]],
                         coefs[1][data_upper['crop'][0]:data_upper['crop'][1]]) for
                        coefs in data_upper['u^mn']]
    """


def wwf_lower_path(data_upper: dict, wavelet_level: int = 4, tm: int or float = 2.8, wavelet: str = 'bior2.2'):
    """ Lower path in Smital et al. (2013) figure 2. Calls swt2(), HW(), and iswt2().

        Parameters
        ----------
        data_upper
            dictionary returned from wwf_upper_path
        wavelet_level
            number of decompositions to perform in SWT
        tm
            'threshold multipllier'; constant by which thresholds in Smital et al. (2013) eq. 3 are multiplied
        wavelet
            wavelet to use for decomposition and reconstruction

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

    data_lower = {}

    ymn, padding_left, padding_right = swt2(signal=data_upper['xn'],
                                            level=wavelet_level,
                                            wavelet=wavelet,
                                            pad_signal=True)

    # splits ymn above into approximation and detail coefficients
    data_lower['ymn_cA'] = ymn[:, 0]
    data_lower['ymn_cD'] = ymn[:, 1]

    # crop indexes to undo signal padding (required for SWT)
    data_lower['crop'] = [padding_left, -1 if padding_right == 0 else -padding_right]

    # noise estimate using Smital (2013) eq. 2 without threshold multipler (hardcoded tm = 1)
    # used in WH block below

    ## figure out padding

    # variance estimate of noise band of ymn_cA using Smital et al. (2013) eq. 2
    # tm set to 1 since we only want the median here
    # data_lower[ymn_cA_noise] = ymn_cA
    data_upper['umn_cA_noise'] = H(ymn=data_upper['u^mn_cA'],
                                   rr_windows=data_upper['rr_windows_pad'],
                                   # pad_left=data_lower['crop1'][0],
                                   pad_left=0,
                                   tm=1,
                                   apply_threshold=False)[0]

    # variance estimate of noise band of ymn_cD using Smital et al. (2013) eq. 2
    # tm set to 1 since we only want the median here
    # data_lower[ymn_cD_noise] = ymn_cD
    data_upper['umn_cD_noise'] = H(ymn=data_upper['u^mn_cD'],
                                   rr_windows=data_upper['rr_windows_pad'],
                                   # pad_left=data_lower['crop'][0],
                                   pad_left=0,
                                   tm=1,
                                   apply_threshold=False)[0]

    # Smital et al. (2013) eq. 4
    # Wiener correction factor based on thresholds
    # tm set to 1 since it is set to 1 in the call to H() for umn_cA_noise/umn_cD_noise estimates
    data_lower['g^mn_cA'], data_lower['g^mn_cD'] = WH(umn_cA=data_upper['u^mn_cA'],
                                                      umn_cD=data_upper['u^mn_cD'],
                                                      umn_ca_noise=data_upper['umn_cA_noise'],
                                                      umn_cd_noise=data_upper['umn_cD_noise'],
                                                      tm=1)

    # Smital et al. (2013) eq. 5
    # Applies Wiener correction factor to wavelet coefficients ymn_cA/ymn_cD --> 'modified coefficients'
    data_lower['lambda_ymn_cA'] = data_lower['ymn_cA'] * data_lower['g^mn_cA']
    data_lower['lambda_ymn_cD'] = data_lower['ymn_cD'] * data_lower['g^mn_cD']

    # Smital et al. (2013) eq. 5
    data_lower['lambda_ymn'] = [(cA_level, cD_level) for cA_level, cD_level in
                                zip(data_lower['lambda_ymn_cA'], data_lower['lambda_ymn_cD'])]

    # Inverse SWT2
    data_lower['yn'] = iswt2(lambda_ymn=data_lower['lambda_ymn'], wavelet=wavelet)

    return data_lower


""" ==================================== FUNCTION CALLS ==================================== """

sample_rate = 512

# Preprocessing ---------

# resamples signal to 512 Hz to match Smital et al. (2013)
ecg_signal = resample_signal(signal=nst_data['00']['ecg'], old_rate=360, new_rate=512)

# .67Hz highpass filter + 60Hz notch filter
data_upper = {'xn': filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=.67, order=100)}
data_upper['xn'] = filter_notch(signal=data_upper['xn'], sample_rate=sample_rate, freq=60)

# WWF -------------------

wwf_upper_path(data_upper=data_upper, wavelet_level=4, tm=2.8, wavelet='bior2.2')
data_lower = wwf_lower_path(data_upper=data_upper, wavelet_level=4, tm=2.8, wavelet='bior2.2')


fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))
ax[0].plot(np.arange(len(data_upper['xn']))/sample_rate, data_upper['xn'], label='bp_filt', color='black')
ax[0].plot(np.arange(len(data_upper['s^'][data_upper['crop'][0]:]))/sample_rate, data_upper['s^'][data_upper['crop'][0]:], label='s^', color='red')
ax[1].plot(np.arange(len(data_lower['g^mn_cA'][0]))/sample_rate, data_lower['g^mn_cA'][0], label='g^mn_cA', color='dodgerblue')
shade_noise_on_plot(ax)
ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')


# TODO
# wwf_upper_path: padding/cropping --> make sure compatible with lower path
# wwf_lower_path

# figure out what exactly dyadic SWT is and implement it
