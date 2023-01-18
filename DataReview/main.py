import os
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nimbalwear.activity

import Filtering
from ECG.PeakDetection import create_df_peaks
from ECG.HR_Calculation import calculate_epoch_hr_jumping
from DataReview.DataImport import import_data, check_filenames
from DataReview.Analysis import *
from DataReview.Plotting import *
import pandas as pd
# from ECG_Organized.ImportTabular import import_cn_beat_file, epoch_hr
# from ECG.main import ECG
import Run_FFT
import neurokit2 as nk

if __name__ == "__main__":
    subjs = ['SBH0273', 'SBH0316']
    site_code = ''
    visit_num = '01'
    subj = subjs[1]
    cutpoint_age_cutoff = 60

    edf_folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"

    file_dict = {
                 'clin_insights_file': "W:/OND09 (HANDDS-ONT)/handds_clinical_insights.xlsx",
                 'gait_file': "W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{}{}_{}_GAIT_BOUTS.csv",
                 'steps_file': "W:/NiMBaLWEAR/OND09/analytics/gait/steps/OND09_{}{}_{}_GAIT_STEPS.csv",
                 'activity_log_file': "W:/OND09 (HANDDS-ONT)/Digitized logs/handds_activity_log.xlsx",
                 'sleep_output_file':  f"W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/OND09_{subj}_{visit_num}_SPTW.csv",
                 'sleep_window_file': f"W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/OND09_{subj}_{visit_num}_SPTW.csv",
                 'epoch_folder': "W:/NiMBaLWEAR/OND09/analytics/activity/epochs/",
                 'epoch_file': f"W:/NiMBaLWEAR/OND09/analytics/activity/epochs/OND09_{site_code}{subj}_{visit_num}_ACTIVITY_EPOCHS.csv",
                 'edf_folder': edf_folder,
                 'cn_beat_file': f"W:/NiMBaLWEAR/OND09/analytics/ecg/CardiacNavigator/OND09_{subj}_Beats.csv",
                 'posture': "W:/NiMBaLWEAR/OND09/analytics/activity/posture/OND09_{}{}_{}_posture_bouts.csv",
                 'devices': "W:/NiMBaLWEAR/OND09/pipeline/devices.csv"}

    file_dict = check_filenames(file_dict=file_dict, subj=subj, site_code='SBH',
                                nw_bouts_folder="W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_standard/",
                                data_review_df_folder="W:/OND09 (HANDDS-ONT)/Data Review/", visit_num=visit_num)

    cutpoints = {"PowellDominant": [51*1000/30/15, 68*1000/30/15, 142*1000/30/15],
                 "PowellNon-dominant": [47*1000/30/15, 64*1000/30/15, 157*1000/30/15],
                 'FraysseDominant': [62.5, 92.5, 10000],
                 'FraysseNon-dominant': [42.5, 98, 10000]}

    df_epoch, epoch_len, df_clin, df_posture, df_act, df_sleep_alg, \
    df_sleep, df_gait, df_steps, df_act_log, ankle_nw, wrist_nw, ankle, wrist = import_data(file_dict=file_dict, subj=subj, visit_num=visit_num,
                                                                                            site_code=site_code, load_raw=True)

    """
    thresholds = (5, 20)
    
    ecg = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/", ecg_fname=f"OND09_{subj}_01_BF36_Chest.edf",
              # smital_filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/OND09_{subj}_01_BF36_Chest.pickle",
              smital_filepath="",
              bandpass=(.67, 40), thresholds=thresholds)
    
    df_peaks_nk = create_df_peaks(timestamps=ecg.ts, peaks=nk.ecg_findpeaks(ecg_cleaned=ecg.filt, sampling_rate=ecg.fs,
                                                                            method='neurokit', show=False)['ECG_R_Peaks'])
    df_peaks_nk['quality'] = [1] * df_peaks_nk.shape[0]
    hr_epoch_len = 15
    df_hr_epoch = calculate_epoch_hr_jumping(df_peaks=df_peaks_nk, sample_rate=ecg.fs, ecg_signal=ecg.filt,
                                             ecg_timestamps=ecg.ts, min_quality=1, epoch_len=hr_epoch_len)
    df_hr_epoch['hr'] = Filtering.filter_signal(data=df_hr_epoch['hr'].fillna(method='bfill'), sample_f=1/hr_epoch_len,
                                                filter_type='lowpass', filter_order=3, low_f=1/300)
    """

    # hand_dom = calculate_hand_dom(df_clin=df_clin, wrist_file=file_dict['wrist_file'])
    hand_dom = calculate_hand_dom2(subj=subj,
                                   subjs_csv_file="W:/NiMBaLWEAR/OND09/pipeline/collections.csv",
                                   devices_csv_file="W:/NiMBaLWEAR/OND09/pipeline/devices.csv")

    age = df_clin['Age'].iloc[0]
    use_author = 'Fraysse' if age >= cutpoint_age_cutoff else 'Powell'

    df_epoch['cadence'] = epoch_cadence(epoch_timestamps=df_epoch['Timestamp'], df_steps=df_steps)

    df_epoch['intensity'] = epoch_intensity(df_epoch=df_epoch, column='avm', cutpoints_dict=cutpoints, cutpoint_key='Dominant' if hand_dom else 'Non-dominant', author=use_author)
    df_epoch['sleep_mask'] = flag_sleep_epochs(df_epoch=df_epoch, df_sleep_alg=df_sleep_alg)

    df_daily = combine_df_daily(ignore_sleep=True, df_epoch=df_epoch, cutpoints=cutpoints, df_gait=df_gait, df_act=df_act, epoch_len=15, hand_dom=hand_dom)
    print()
    walk_avm_str = print_walking_intensity_summary(df_gait=df_gait, df_epoch=df_epoch, min_dur=30, cutpoints=cutpoints[use_author + "Dominant" if hand_dom else use_author + 'Non-dominant'])

    gen_relationship_graph(daily_df=df_daily.loc[(df_daily['Day_dur'] == 1440)], df_gait=df_gait, author=use_author)
    summary_plot(df_daily=df_daily.loc[(df_daily['Day_dur'] == 1440)], author=use_author, subj=subj, df_sleep=df_sleep)

    print("")
    fig = plot_raw(subj=subj, wrist=wrist, ankle=ankle, ecg=None,
                   highpass_accel=False, intensity_markers=True,
                   cutpoints=cutpoints, dominant=hand_dom, author=use_author, wrist_nw=wrist_nw, wrist_gyro=False,
                   ankle_gyro=True, ankle_nw=ankle_nw,
                   df_epoch=df_epoch, df_sleep_alg=df_sleep_alg, df_gait=df_gait, df_act_log=df_act_log,
                   df_hr=None, df_posture=None, df_steps=df_steps,
                   shade_gait_bouts=True, min_gait_dur=15, mark_steps=True, bout_steps_only=True,
                   show_activity_log=True, shade_sleep_windows=True,
                   alpha=.35, ds_ratio=2)
    df_act_log = calculated_logged_intensity(epoch_len=15, df_epoch=df_epoch, df_act_log=df_act_log, hours_offset=0, df_steps=df_steps, quiet=True)
    print_activity_log(df_act_log)
    print_quick_summary(subj=subj, df_clin=df_clin, df_daily=df_daily.loc[df_daily['Day_dur'] == 1440], walk_avm_str=walk_avm_str,
                        cutpoint_author=use_author, dominant=hand_dom, df_gaitbouts=df_gait, df_epoch=df_epoch)
    print_medical_summary(df_clin=df_clin)

    # compare_cutpoints(df_daily=df_daily)

    # fft_fig, df_fft, dom_f, fft_idx = freq_analysis(obj=wrist, subj=subj, channel='Accelerometer x', ts='2022-11-19 2:36:00', lowpass=None, highpass=.25, sample_rate=None, n_secs=30, stft_mult=5, stft=False, show_plot=True)

    # calculate_window_cadence(start=df_act_log.loc[10]['start_time'], stop=df_act_log.loc[10]['start_time'] + timedelta(minutes=df_act_log.loc[10]['duration']), df_steps=df_steps, ankle_obj=ankle, show_plot=True, axis="Gyroscope y")
    # check_intensity_window(df_epoch=df_epoch, start=df_act_log.loc[5]['start_time'], end=df_act_log.loc[5]['start_time'] + timedelta(minutes=df_act_log.loc[5]['duration']))
    # plot_gait_histogram(df_gait, 'step_count', np.arange(0, 600, 10))
    # calculate_logged_intensity_individual(start_timestamp='2022-10-29 12:08:42', end_timestamp='2022-10-29 12:31:47', df_epoch=df_epoch, epoch_len=15)

