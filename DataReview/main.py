import os
import matplotlib.pyplot as plt
from DataReview.DataImport import import_data, check_filenames
from DataReview.Analysis import *
from DataReview.Plotting import *
import pandas as pd
from ECG_Organized.ImportTabular import import_cn_beat_file, epoch_hr
from ECG_Organized import ImportECG
import Run_FFT

subj = '0175'
cutpoint_age_cutoff = 60

edf_folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"

file_dict = {'clin_insights_file': "W:/OND09 (HANDDS-ONT)/handds_clinical_insights.xlsx",
             'gait_file': f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/OND09_{subj}_01_GAIT_BOUTS.csv",
             'steps_file': f"W:/NiMBaLWEAR/OND09/analytics/gait/steps/OND09_{subj}_01_GAIT_STEPS.csv",
             'activity_log_file': f"W:/OND09 (HANDDS-ONT)/Digitized logs/handds_activity_log.xlsx",
             'sleep_output_file':  f"W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/OND09_{subj}_01_SPTW.csv",
             'sleep_window_file': f"W:/NiMBaLWEAR/OND09/analytics/sleep/sptw/OND09_{subj}_01_SPTW.csv",
             'epoch_folder': "W:/NiMBaLWEAR/OND09/analytics/activity/epochs/",
             'edf_folder': edf_folder,
             'cn_beat_file': f"W:/NiMBaLWEAR/OND09/analytics/ecg/CardiacNavigator/OND09_{subj}_Beats.csv",
             'posture': "W:/NiMBaLWEAR/OND09/analytics/activity/posture/OND09_{subj}_01_posture_bouts.csv"}

file_dict = check_filenames(file_dict=file_dict, subj=subj,
                            nw_bouts_folder="W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_standard/",
                            data_review_df_folder="W:/OND09 (HANDDS-ONT)/Data Review/")

cutpoints = {"PowellDominant": [51*1000/30/15, 68*1000/30/15, 142*1000/30/15],
             "PowellNon-dominant": [47*1000/30/15, 64*1000/30/15, 157*1000/30/15],
             'FraysseDominant': [62.5, 92.5, 10000],
             'FraysseNon-dominant': [42.5, 98, 10000]}

df_epoch, epoch_len, df_clin, df_posture, df_act, df_sleep_alg, df_sleep, df_gait, df_steps, df_act_log, ankle_nw, wrist_nw, ankle, wrist = import_data(file_dict=file_dict, subj=subj)


ecg = ImportECG.ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                    ecg_fname=f"OND09_{subj}_01_BF36_Chest.edf", bandpass=(.67, 80),
                    smital_filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/{subj}_01_BF36_Chest.pickle")

# cardiac navigator beat file
df_cn_beats = import_cn_beat_file(filename=file_dict['cn_beat_file'], start_time=ecg.start_stamp)
df_hr = epoch_hr(df_beats=df_cn_beats, start_time=ecg.ts[0], end_time=ecg.ts[-1], avg_period=30, centre=False)

hand_dom = calculate_hand_dom(df_clin=df_clin)
age = df_clin['Age'].iloc[0]
use_author = 'Fraysse' if age >= cutpoint_age_cutoff else 'Powell'
print_clinical_summary(subj=subj, df_clin=df_clin)

df_epoch['intensity'] = epoch_intensity(df_epoch=df_epoch, column='avm', cutpoints_dict=cutpoints, cutpoint_key='Dominant' if hand_dom else 'Non-dominant', author=use_author)
df_epoch['sleep_mask'] = flag_sleep_epochs(df_epoch=df_epoch, df_sleep_alg=df_sleep_alg)

df_daily = combine_df_daily(ignore_sleep=True, df_epoch=df_epoch, cutpoints=cutpoints, df_gait=df_gait, df_act=df_act, epoch_len=15, hand_dom=hand_dom)

df_act_log = calculated_logged_intensity(epoch_len=15, df_epoch=df_epoch, df_act_log=df_act_log, hours_offset=0)

# summary_plot(df_daily=df_daily, author=use_author, subj=subj, df_sleep=df_sleep)
# gen_relationship_graph(daily_df=df_daily, df_gait=df_gait, author=use_author)

fig = plot_raw(subj=subj, wrist=wrist, ankle=ankle, ecg=None,
               cutpoints=cutpoints, dominant=hand_dom, author=use_author, ds_ratio=2,
               wrist_gyro=False, ankle_gyro=False, highpass_accel=False,
               wrist_nw=wrist_nw, ankle_nw=ankle_nw,
               df_epoch=df_epoch, df_sleep_alg=df_sleep_alg, df_gait=df_gait, df_act_log=df_act_log,
               df_hr=None, df_posture=None, df_steps=df_steps,
               shade_gait_bouts=True, min_gait_dur=20, mark_steps=False, bout_steps_only=True,
               show_activity_log=True, shade_sleep_windows=True,
               alpha=.35)

# compare_cutpoints(df_daily=df_daily)
# fft_fig, df_fft = freq_analysis(obj=ecg, subj=subj, channel='Accelerometer z', ts="2022-05-04 17:35:00", lowpass=None, highpass=None, sample_rate=None, n_secs=120, stft_mult=20, stft=False, show_plot=True)
