import pandas as pd
from ECG.ImportFiles import ECG
from ECG.PeakDetection import run_cardiacnavigator_method, run_neurokit_method, run_zncc_method,\
    correct_cn_peak_locations, correct_cn_peak_locations_centre, create_beat_template_snr_bouts, screen_peaks_corr
from ECG.Processing import remove_low_quality_signal
import matplotlib.pyplot as plt
from ECG.Plotting import *


def other_methods():
    df_nk, df_nk_epoch = run_neurokit_method(ecg_obj=ecg, df_snr=ecg.df_snr, window_size=.33, n_hq_snr_bouts=5, corr_thresh=.5)

    zncc, df_zncc, df_zncc_epoch = run_zncc_method(input_data=ecg.filt, template=qrs, zncc_thresh=.725, snr=ecg.snr,
                                                   sample_rate=ecg.fs, downsample=1, timestamps=ecg.ts,
                                                   thresholds=thresholds, show_plot=False, min_dist=int(ecg.fs/(220/60)))
    df_zncc = correct_cn_peak_locations(df_peaks=df_zncc, peaks_colname='idx',
                                        ecg_signal=ecg.filt, sample_rate=ecg.fs, window_size=.3, use_abs_peaks=True)


"""
full_id = 'OND09_0114'
thresholds = (5, 18)

ecg = ECG(
          # edf_folder="W:/NiMBaLWEAR/OND06/processed/cropped_device_edf/BITF/", ecg_fname=f"{full_id}_01_BITF_Chest.edf",
          edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/", ecg_fname=f"{full_id}_01_BF36_Chest.edf",
          # smital_filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/{full_id}_01_BF36_Chest.pickle",
          # smital_filepath=f"W:/NiMBaLWEAR/OND06/analytics/ecg/smital/snr_timeseries/{full_id}_01_BITF_Chest.pickle",
          smital_filepath=f"C:/Users/ksweber/Desktop/SNR_dev/{full_id}_01_snr_bouts_fixed.csv",
          bandpass=(.67, 25), thresholds=thresholds)


df_cn, qrs = run_cardiacnavigator_method(cn_file=f"W:/NiMBaLWEAR/OND09/analytics/ecg/CardiacNavigator/{full_id}_Beats.csv",
                                         ecg_obj=ecg, df_snr=ecg.df_snr, sample_rate=ecg.fs, timestamps=ecg.ts,
                                         window_size=0.33, corr_thresh=.5, epoch_len=30,
                                         version_key_epoching='v3', n_hq_snr_bouts=-1)

# CHANGE IDX
end_idx = int(24*60*60*ecg.ecg.signal_headers[ecg.ecg.get_signal_index("ECG")]['sample_rate'])
ecg.ecg.signals[0] = ecg.ecg.signals[0][:end_idx]
ecg.filt = ecg.filt[:end_idx]
signal = remove_low_quality_signal(ecg_signal=ecg.filt, df_snr=ecg.df_snr, min_quality=2, min_duration=15)  # normal run

# not needed
# peaks = nk.ecg_findpeaks(ecg_cleaned=signal, sampling_rate=ecg.fs, method='neurokit', show=False)['ECG_R_Peaks']
# df_peaks = pd.DataFrame({'timestamp': ecg.ts[peaks], 'idx': peaks, 'valid': [True] * len(peaks)})
# df_peaks.to_csv(f"O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/CSEP_Abstract/Data/ECG_Peaks/{full_id}_ecg_peaks_raw.csv", index=False)

df_nk, df_epoch, qrs = run_neurokit_method(ecg_obj=ecg, df_snr=ecg.df_snr, window_size=.33, n_hq_snr_bouts=5, corr_thresh=.5, epoch_len=30, thresholds=(5, 20))
"""

"""
df_zncc, df_zncc_epoch, zncc = run_zncc_method(input_data=ecg.filt, template=qrs, min_dist=int(ecg.fs/2), timestamps=ecg.ts,
                                               snr=ecg.snr, sample_rate=250, downsample=2, zncc_thresh=.7, show_plot=False,
                                               thresholds=(5, 20), epoch_len=30, min_quality=2)
"""

# plot_corrected_peaks(ecg_signal=ecg.filt, timestamps=ecg.ts, og_peaks=df_cn['v2']['idx'], corr_peaks=df_cn['v2']['idx_corr'], ds_ratio=3)


# TODO:
# remove beats in non-wear periods
