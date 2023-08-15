import matplotlib.pyplot as plt
import numpy as np
from ECG.physionetdb_smital import shade_noise_on_plot


def plot_stft(data: list or tuple or np.array,
              sample_rate: float or int,
              f: np.array,
              t: np.array,
              Zxx: np.array):
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


def plot_pathway_annoted_snr(smital_obj, sample_rate, nst_snr_og,
                             data_key, nst_data, gs_snr, snr_key='snr_raw'):
    fig, ax = plt.subplots(3, sharex='col', figsize=(10, 6))
    ax[0].plot(smital_obj.data_upper['xn'], color='black', label='xn')
    ax[0].plot(smital_obj.data_lower['yn'], color='limegreen', label='yn')

    ax[1].plot(smital_obj.data_upper['xn'], color='black', label='xn')
    ax[1].plot(smital_obj.data_awwf['zn'], color='orange', label='zn')

    try:
        ax[2].plot(nst_snr_og[data_key]['snr'], color='fuchsia', label='Old code')
    except KeyError:
        pass
    ax[2].plot(smital_obj.data_lower[snr_key], color='limegreen', label='First SNR est.')
    ax[2].plot(smital_obj.data_awwf[snr_key], color='orange', label='AWWF SNR est.')
    ax[2].plot(gs_snr, color='purple', linestyle='dashed', label='true SNR')
    ax[2].set_ylabel("dB")

    # flags annotated beats as 'normal' or not normal
    df_n = nst_data[data_key]['annots'].loc[nst_data[data_key]['annots']['beat_type'] == 'N']
    ax[0].scatter(df_n['idx'], smital_obj.data_upper['xn'][df_n['idx']] + .5, color='dodgerblue', marker='v', label='sinus', s=50)
    df_arr = nst_data[data_key]['annots'].loc[nst_data[data_key]['annots']['beat_type'] != 'N']
    ax[0].scatter(df_arr['idx'], smital_obj.data_upper['xn'][df_arr['idx']] + .5, color='red', marker='x', label='arrhythmic', s=50)
    del df_n, df_arr

    for i in ax:
        i.legend(loc='upper left')

    shade_noise_on_plot(ax, sample_rate, 'datapoints')
    plt.tight_layout()


def plot_thresholds(smital_obj, coef, level):
    fig, ax = plt.subplots(3, figsize=(10, 6), sharex='col')

    ax[0].plot(smital_obj.data_upper['xn'], color='black', label='xn')
    ax[0].plot(smital_obj.data_upper['ymn'][level, 0 if coef == 'cA' else 1, :], color='red', label=f'{coef}[{level}]')
    ax[0].plot(smital_obj.data_upper[f'ymn_{coef}_threshed'][level, :], color='dodgerblue', label=f'{coef}[{level}]_threshed')

    ax[1].plot(smital_obj.data_upper[f'ymn_{coef}_threshes'][level], color='orange', label=f'{coef}[{level}]_thresholds')
    ax[1].plot(smital_obj.data_upper['ymn'][level, 0 if coef == 'cA' else 1, :], color='red', label=f'{coef}[{level}]')

    ax[2].plot(smital_obj.data_upper[f'ymn_{coef}_threshed'][level, :], color='dodgerblue', label=f'{coef}[{level}]_threshed')

    for axis in ax:
        axis.legend(loc='upper left')

    for w in smital_obj.data_upper['rr_windows'][::2]:
        ax[1].axvspan(w[0], w[1], 0, 1, color='grey', alpha=.2)

    plt.tight_layout()

    return fig


def plot_results(smital_obj, sample_rate, data_key, nst_snr, nst_data, gs_snr, snr_key='snr_raw'):

    fig, ax = plt.subplots(4, sharex='col', figsize=(12, 8))
    ax[0].plot(smital_obj.data_upper['xn'], color='black', label='preproc.')
    ax[1].plot(smital_obj.data_upper['xn'], color='black', label='preproc.')
    ax[2].plot(smital_obj.data_upper['xn'], color='black', label='preproc.')

    ax[0].plot(smital_obj.data_upper['s^'], color='dodgerblue', label='upper_s^')
    ax[1].plot(smital_obj.data_lower['yn'], color='red', label='lower_yn')
    ax[2].plot(smital_obj.data_awwf['zn'], color='orange', label='awwf_zn')

    ax[3].plot(smital_obj.data_upper[snr_key], color='dodgerblue', label='upper')
    ax[3].plot(smital_obj.data_lower[snr_key], color='red', label='lower')
    ax[3].plot(smital_obj.data_awwf[snr_key], color='orange', label='awwf')
    ax[3].grid()

    try:
        ax[3].plot(np.arange(len(nst_snr[data_key]['1s'])) * sample_rate,
                   nst_snr[data_key]['1s'], color='fuchsia', label='original')

        # flags annotated beats as 'normal' or not normal
        df_n = nst_data[data_key]['annots'].loc[nst_data[data_key]['annots']['beat_type'] == 'N']
        df_arr = nst_data[data_key]['annots'].loc[nst_data[data_key]['annots']['beat_type'] != 'N']

        for ax_i, sig in enumerate([smital_obj.data_upper['xn'], smital_obj.data_lower['yn'], smital_obj.data_awwf['zn']]):
            ax[ax_i].scatter(df_n['idx'], sig[df_n['idx']] + .5, color='limegreen', marker='v', label='sinus')
            ax[ax_i].scatter(df_arr['idx'], sig[df_arr['idx']] + .5, color='red', marker='x', label='arrhythmic')

    except:
        pass

    ax[-1].axhspan(0, 1, 5, 18, color='green', alpha=.15)

    try:
        ax[3].plot(gs_snr, color='black', label='true_snr')
    except (NameError, ValueError):
        pass

    shade_noise_on_plot(ax, sample_rate, 'datapoint')

    for ax_i, val in enumerate(['upper path', 'lower path', 'awwf', 'snr']):
        ax[ax_i].legend(loc='upper left')
        ax[ax_i].set_ylabel(val)
    plt.tight_layout()

    return fig


def add_annotated_beats(nst_data, data_key, ax, signal):

    # flags annotated beats as 'normal' or not normal
    df_n = nst_data[data_key]['annots'].loc[nst_data[data_key]['annots']['beat_type'] == 'N']
    ax.scatter(df_n['idx'], signal[df_n['idx']] + .5, color='dodgerblue', marker='v', label='sinus', s=50)
    df_arr = nst_data[data_key]['annots'].loc[nst_data[data_key]['annots']['beat_type'] != 'N']
    ax.scatter(df_arr['idx'], signal[df_arr['idx']] + .5, color='red', marker='x', label='arrhythmic', s=50)
