import matplotlib.pyplot as plt
from ECG.physionetdb_smital import shade_noise_on_plot

# for MITBIH-NST119
xlim = (147775.27432011697, 152554.09322562272)


def plot_orig_snr_issue(nst_snr, nst_data):

    c_dict = {"_6": 'red', '00': 'orange', '06': 'gold',
              '12': 'dodgerblue', '18': 'limegreen', '24': 'purple'}

    fig, ax = plt.subplots(nrows=len(nst_snr.keys())+1, sharex='col', figsize=(12, 8))

    for idx, key in enumerate(nst_snr.keys()):
        ax[0].plot(nst_snr[key]['roll_snr'], color=c_dict[key], label=key if key != '_6' else '-6')
        ax[idx+1].plot(nst_data[key]['ecg'], color=c_dict[key])
        ax[idx+1].set_ylabel("voltage")

    ax[0].set_ylabel("measured SNR (dB)")
    ax[0].legend(loc='upper left')

    shade_noise_on_plot(ax=ax, sample_rate=360, units='datapoints')
    plt.tight_layout()


def plot_thresholding_new(smital_obj, level=1):

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

    ax[0].plot(smital_obj.data_upper['xn'], color='black', label='preproc.')

    ax[1].plot(smital_obj.data_upper['ymn'][level, 0, :], color='black', label=f'cA#{level}_raw')
    ax[1].plot(smital_obj.data_upper['ymn_cA_threshes'][level], color='red', label='thresholds')
    ax[1].plot(smital_obj.data_upper['ymn_cA_threshed'][level], color='dodgerblue', label='post-thresh')

    ax[2].plot(smital_obj.data_upper['s^'], color='grey', label='est. noise-free')

    for a in ax:
        a.legend(loc='upper left')

    plt.tight_layout()

    return fig


def plot_thresholding_orig(ecg_signal, nst_snr, level=1):

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

    ax[0].plot(ecg_signal, color='black', label='preproc.')

    ax[1].plot(nst_snr['cD_x'][level], color='black', label=f'cD#{level}_raw')
    ax[1].plot(nst_snr['cD_x_thresholds'][level], color='red', label='thresholds')
    ax[1].plot(nst_snr['cD_x_threshed'][level], color='dodgerblue', label='post-thresh')

    ax[2].plot(nst_snr['s'], color='grey', label='est. noise-free')

    for a in ax:
        a.legend(loc='upper left')

    plt.tight_layout()

    return fig


# fig = plot_thresholding_orig(ecg_signal=nst_data['00']['ecg'], nst_snr=nst_snr['00'], level=1)
# fig.axes[-1].set_xlim(xlim[0]/(500/360), xlim[1]/[500/360])


def compare_thresholding_reconstruction(smital_thresh, smital_raw):

    fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))

    ax[0].plot(smital_thresh.data_upper['xn'], color='black', label='preproc.')

    ax[1].plot(smital_thresh.data_upper['s^'], color='limegreen', label='cA/cD threshed', lw=2)
    ax[1].plot(smital_raw.data_upper['s^'], color='red', label='cD threshed', alpha=.75)

    for a in ax:
        a.legend(loc='upper left')

    plt.tight_layout()

    return fig


# fig = compare_thresholding_reconstruction(smital_thresh=self, smital_raw=self2)
# fig.axes[-1].set_xlim(xlim)


def compare_tm(smital1, smital2):

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

    ax[0].plot(smital1.data_upper['xn'], color='black', label='preproc.')
    ax[0].plot(smital1.data_lower['yn'], color='limegreen', label=f'yn (tm={smital1.fixed_tm})')

    ax[1].plot(smital2.data_upper['xn'], color='black', label='preproc.')
    ax[1].plot(smital2.data_lower['yn'], color='red', label=f'yn (tm={smital2.fixed_tm})')

    ax[2].plot(smital1.data_lower['snr_raw'], color='limegreen', label=f'tm={smital1.fixed_tm}')
    ax[2].plot(smital2.data_lower['snr_raw'], color='red', label=f'tm={smital2.fixed_tm}')
    ax[2].set_ylabel("SNR")

    for a in ax:
        a.legend(loc='upper left')

    plt.tight_layout()

    return fig


# fig = compare_tm(self, self2)
# fig.axes[-1].set_xlim(xlim)
# fig.axes[-1].grid()
