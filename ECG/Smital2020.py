import neurokit2 as nk
import numpy as np
np.seterr(invalid='ignore')
import pywt
import matplotlib.pyplot as plt
from ECG.physionetdb_smital import create_epoched_df, generate_timeseries_noise_value
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from ECG.physionetdb_smital import shade_noise_on_plot, create_epoched_df
from nwecg.awwf import WwfParams, wwf
from nwecg.ecg_quality import pad_to_length
from nwecg.awwf import pad_to_length
import bottleneck
import datetime


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


def swt(signal, level=4, wavelet='bior2.2'):

    # SWT requires that the signal has length which is a multiple of 2 ** level.
    n = 2 ** max(level, 5)
    required_length = int(np.ceil((len(signal) + 120) / n) * n)
    x_padded, padding_left, padding_right = pad_to_length(signal, required_length, mode='reflect')

    max_level = pywt.swt_max_level(len(x_padded))

    swc = pywt.swt(x_padded, level=np.min([level, max_level]), wavelet=wavelet, start_level=0)

    return np.asarray(swc), padding_left, padding_right


def iswt(coefs, wavelet='bior2.2'):

    return pywt.iswt(coefs, wavelet=wavelet)


def threshold_coefs(coefs, window_samples, tm=2.8):

    rolling_abs_median = bottleneck.move_median(np.abs(coefs), int(window_samples + 1), axis=0) / .6745 * tm
    rolling_abs_median[:window_samples] = 0

    return rolling_abs_median


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


def get_rolling_snr(x: np.ndarray, s: np.ndarray, window: int, use_sum: bool = True):
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
        rolling_s_energy = bottleneck.move_sum(np.square(s), window, axis=0)[window - 1:-1]  # Trim first few NaN values.
        rolling_w_energy = bottleneck.move_sum(np.square(w), window, axis=0)[window - 1:-1]

    if not use_sum:
        rolling_s_energy = bottleneck.move_var(np.square(s), window, axis=0)[window - 1:-1]  # Trim first few NaN values.
        rolling_w_energy = bottleneck.move_var(np.square(w), window, axis=0)[window - 1:-1]

    rolling_snr = 10 * np.log10(rolling_s_energy / rolling_w_energy)

    # Apply Short-Time Averaging (STA) to SNR.
    rolling_snr_sta = bottleneck.move_mean(rolling_snr, window, axis=0)[window - 1:-1]
    rolling_snr_sta, *_ = pad_to_length(x=rolling_snr_sta, length=len(s), mode='edge')

    return rolling_snr, rolling_snr_sta


data_key = '00'
sample_rate = 512

# import data

# resample to 512Hz
data = {"x": resample_signal(signal=nst_data[data_key]['ecg'], old_rate=360, new_rate=sample_rate)}

# preprocess: .67Hz highpass + 60Hz notch filters
data['xn'] = filter_highpass(signal=data['x'], sample_rate=sample_rate, cutoff_low=.67, order=100)
data['xn'] = filter_notch(signal=data['xn'], sample_rate=sample_rate, freq=60)

# SWT: db4, level 4, nonnegative garotte thresholding
# umn
data['umn'], pad_left, pad_right = swt(signal=data['xn'], level=4, wavelet='db4')
data['umn_cA'] = data['umn'][:, 0]
data['umn_cD'] = data['umn'][:, 1]
del data['umn']

# thresholds coefficients
data['umn_cA_thresh'] = np.asarray([threshold_coefs(coefs=i, window_samples=2*sample_rate, tm=1) for i in data['umn_cA']])
data['umn_cD_thresh'] = np.asarray([threshold_coefs(coefs=i, window_samples=2*sample_rate, tm=1) for i in data['umn_cD']])

umn_cA_threshed = np.array([pywt.threshold(data=c, value=t, mode='garrote') for
                            c, t in zip(data['umn_cA'], data['umn_cA_thresh'])])
# umn_cA_threshed = data['umn_cA']

umn_cD_threshed = np.array([pywt.threshold(data=c, value=t, mode='garrote') for c, t in zip(data['umn_cD'], data['umn_cD_thresh'])])
# umn_cD_threshed = data['umn_cD']

# data['umn_threshed'] = np.asarray([(ca, cd) for ca, cd in zip(umn_cA_threshed, umn_cD_threshed)])
data['umn_threshed'] = np.asarray([(ca, cd) for ca, cd in zip(umn_cA_threshed, umn_cD_threshed)])

t = iswt(coefs=data['umn_threshed'][:, 0, :], wavelet='db4')
# t = iswt(coefs=data['umn_threshed'], wavelet='db4')

snr = get_rolling_snr(x=data['xn'], s=t[pad_left:-pad_right], window=int(2*sample_rate), use_sum=True)

fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))
ax[0].plot(data['xn'], color='black', label='xn')
ax[0].plot(t[pad_left:-pad_right], color='red', label='iswt')
ax[0].legend()
ax[1].plot(data['umn_cA'][-1], label='cA')
ax[1].plot(data['umn_cD'][-1], label='cD')
ax[2].plot(snr[1], color='dodgerblue')
ax[2].axhline(y=24, color='black')
ax[2].axhline(y=0, color='black')

# Wiener correction factor
# ymn: SWT (sym4, level 4) --> no thresholding??
# sigmamn = median(|ymn|)/.6745 in rolling window
# gmn = u2mn / (u2mn + sigma2mn)
# y^mn = ymn * gmn
# s^n = ISWT(y^mn)

# SNR calculation
# w^ = xn - s^n
# snr_tn = 10 * log10(s^2 / w^2)