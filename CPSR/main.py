from CPSR.DataImport import import_ax_data, create_df_mask
from CPSR.EpochData import epoch_data, combine_epoched
# from CPSR.Plotting import *
from CPSR.DataProcessing import *
import matplotlib.pyplot as plt

epoch_len = 15

lw, lw_fs, lw_ts = import_ax_data(filename="C:/Users/ksweber/Desktop/GRASP_LeftWrist.cwa")
rw, rw_fs, rw_ts = import_ax_data(filename="C:/Users/ksweber/Desktop/GRASP_RightWrist.cwa")

full_id = 'OND09_0060'

# lw, lw_fs, lw_ts = import_ax_data(f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{full_id}_01_AXV6_LWrist.edf")
# rw, rw_fs, rw_ts = import_ax_data(f"W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/{full_id}_01_AXV6_LAnkle.edf")  # stand-in since no right wrist

start_ts, end_ts = calculate_collection_startstop(obj1=lw, obj2=rw)

lw_df = epoch_data(data_object=lw, use_accel=True, epoch_len=1,
                   use_activity_counts=True, use_sd=False,
                   bandpass_epoching=False, remove_baseline=True,
                   start=start_ts, end=end_ts)

rw_df = epoch_data(data_object=rw, use_accel=True, epoch_len=1,
                   use_activity_counts=True, use_sd=False,
                   bandpass_epoching=False, remove_baseline=True,
                   start=start_ts, end=end_ts)

df15 = combine_epoched(df_dom=lw_df, df_nondom=rw_df, epoch_len=epoch_len)

df_mask, df_gait, df_sleep, df_nw = create_df_mask(start_stamp=start_ts,
                                                   coll_dur=int((end_ts - start_ts).total_seconds()),
                                                   gait_filepath="Moose",
                                                   sleep_filepath="Steak",
                                                   nw_filepath="fakepath")

"""df_mask, df_gait, df_sleep, df_nw = create_df_mask(start_stamp=start_ts,
                                                   coll_dur=int((end_ts - start_ts).total_seconds()),
                                                   gait_filepath=f"W:/NiMBaLWEAR/OND09/analytics/gait/bouts/{full_id}_01_GAIT_BOUTS.csv",
                                                   sleep_filepath=f"W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/{full_id}_01_SLEEP_BOUTS.csv",
                                                   nw_filepath=f"W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{full_id}_01_AXV6_LWrist_NONWEAR.csv")"""

df15 = epoch_context(df_epoch=df15, df_mask=df_mask, remove_sleep_activity=True)

df15_v2, df_totals = calculate_activity_totals(df_epoch=df15, dom_cutpoints=(62.5, 92.5), nondom_cutpoints=(42.5, 98.0),
                                               exclude_nw=True, exclude_gait=False, exclude_sleep=True)

# Activity totals during given time period
# df_totals_crop = analyze_data_section(df_epoch=df15, start="2021-11-15 13:46:49", end="2022-11-15 14:18:58", show_plot=True)

plot_epoched_context(df_epoch=df15)

"""
fig, ax = plt.subplots(4, sharex='col', figsize=(12, 8))
plot_device(data_object=lw, raw_timestamps=lw_ts, df_epoch=lw_df, ax1=ax[0], ax2=ax[1], label='LWrist')
plot_device(data_object=rw, raw_timestamps=rw_ts, df_epoch=rw_df, ax1=ax[2], ax2=ax[3], label='RWrist')
"""

"""
fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

ax[0].plot(lw_df['timestamp'], lw_df['value'], color='dodgerblue')
ax[1].plot(rw_df['timestamp'], rw_df['value'], color='red')
ax[2].plot(lw_df['timestamp'], rw_df['value'] / lw_df['value'], color='purple')
ax[2].xaxis.set_major_formatter(xfmt)
"""