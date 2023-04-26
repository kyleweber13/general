from Other.Nami.Nami_EventViewer import import_edf, import_steps_files, import_bout_files
from Other.Nami.Nami import find_bouts, import_newsteps_files, import_boutfiles_new, import_sptw_files, flag_steps_gaitboutnum
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.%f")
import numpy as np
from utilities.Filtering import filter_signal
import peakutils
os.chdir("W:/NiMBaLWEAR/STEPS/analytics/gait/")
from Other.utils import get_zncc
from datetime import timedelta


def findpeaks_in_bouts(df_bouts, min_steps, ankle_signal, sample_rate, raw_timestamps, side="", peak_thresh=.25):

    start_timestamp = pd.to_datetime(raw_timestamps[0])

    df_bouts_use = df_bouts.loc[df_bouts['step_count'] >= min_steps]

    all_peaks = np.array([])
    bout_num = np.array([])

    for bout in df_bouts_use.itertuples():
        start_idx = int((bout.start_time - start_timestamp).total_seconds() * sample_rate - sample_rate/4)
        end_idx = int((bout.end_time - start_timestamp).total_seconds() * sample_rate + sample_rate/4)

        sig = ankle_signal[start_idx:end_idx]

        peaks = peakutils.indexes(y=sig, thres=peak_thresh, min_dist=sample_rate*2/3, thres_abs=True)
        peaks = np.array([i + start_idx for i in peaks])
        all_peaks = np.append(all_peaks, peaks)

        bout_boutnum = [bout.gait_bout_num] * len(peaks)
        bout_num = np.append(bout_num, bout_boutnum)

    all_peaks = all_peaks.astype(int)

    df_out = pd.DataFrame({'step_time': raw_timestamps[all_peaks],
                           'idx': all_peaks,
                           'gait_bout_num': bout_num,
                           'foot': [side] * len(all_peaks)})

    return df_out


# IDs to use: ignore STEPS_8856
use_ids = ['STEPS_1611', 'STEPS_6707', 'STEPS_2938', 'STEPS_7914', 'STEPS_8856']

df_assess_new = pd.read_excel("O:/OBI/Personal Folders/Namiko Huynh/steps_assessment_timestamps_new.xlsx")
df_assess_new.columns = ['subject_id', 'start_time', 'end_time', 'comments',
                         'walk1_6m_start', 'walk1_6m_end', 'walk2_6m_start', 'walk2_6m_end',
                         'walk3_6m_start', 'walk3_6m_end', '6mwt_start', '6mwt_end']

# step/bout data
# dict_steps = import_steps_files(subjs=use_ids)
# dict_bouts = import_bout_files(subjs=use_ids)

dict_steps = import_newsteps_files(folder="O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/",
                                   subjs=use_ids[:-1], file_suffix="all_steps")

dict_steps_old = import_steps_files(subjs=use_ids[:-1])

#for key in dict_steps.keys():
#    dict_steps[key].sort_values("step_num", inplace=True)

dict_bouts = import_boutfiles_new(folder="O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/",
                                  subjs=use_ids[:-1], file_suffix="newbouts")

flag_steps_gaitboutnum(dict_steps=dict_steps, dict_bouts=dict_bouts)

dict_sptw = import_sptw_files(use_ids[:-1])


subj = 'STEPS_1611'

la, ra = import_edf(la_file=f"Z:/NiMBaLWEAR/STEPS/wearables/device_edf_cropped/{subj}_01_GNOR_LAnkle.edf",
                    ra_file=f"Z:/NiMBaLWEAR/STEPS/wearables/device_edf_cropped/{subj}_01_GNOR_RAnkle.edf")

la_y_filt = filter_signal(data=la.signals[la.get_signal_index('Accelerometer y')], sample_f=75, high_f=.05, filter_type='highpass')
ra_y_filt = filter_signal(data=ra.signals[ra.get_signal_index('Accelerometer y')], sample_f=75, high_f=.05, filter_type='highpass')

la_y_abs = abs(la_y_filt)
ra_y_abs = abs(ra_y_filt)

# la_y_abs = abs(filter_signal(data=la_y_abs, sample_f=75, high_f=10, filter_type='highpass'))
# ra_y_abs = abs(filter_signal(data=ra_y_abs, sample_f=75, high_f=10,  filter_type='highpass'))
la_y_abs = abs(filter_signal(data=la_y_abs, sample_f=75, low_f=1, filter_type='lowpass'))
ra_y_abs = abs(filter_signal(data=ra_y_abs, sample_f=75, low_f=1, filter_type='lowpass'))


proc_dict = {"STEPS_1611": {'thresh': .1, 'la': la_y_abs, 'ra': ra_y_abs},
         # "STEPS_1611": {'thresh': .25, 'la': la_y_filt, 'ra': ra_y_filt},
         # "STEPS_6707": {'thresh': .1, 'la': la_y_abs, 'ra': ra_y_abs},
         "STEPS_6707": {'thresh': .1, 'la': la_y_abs, 'ra': ra_y_abs},
         # "STEPS_2938": {'thresh': .25, 'la': la_y_filt, 'ra': ra_y_filt},
         "STEPS_2938": {'thresh': .125, 'la': la_y_abs, 'ra': ra_y_abs},
         # "STEPS_7914": {'thresh': .25, 'la': la_y_filt, 'ra': ra_y_filt},
         "STEPS_7914": {'thresh': .1, 'la': la_y_abs, 'ra': ra_y_abs},
         "STEPS_8856": {'thresh': .1, 'la': la_y_abs, 'ra': ra_y_abs}}

"""la_peaks = findpeaks_in_bouts(df_bouts=dict_bouts[subj], min_steps=5, ankle_signal=proc_dict[subj]['la'], side='left',
                          sample_rate=la.signal_headers[0]['sample_rate'], raw_timestamps=la.ts,
                          peak_thresh=proc_dict[subj]['thresh'])"""
la_peaks_all = peakutils.indexes(y=proc_dict[subj]['la'], thres=proc_dict[subj]['thresh'], min_dist=la.signal_headers[0]['sample_rate'] * 2 / 3, thres_abs=True)
la_peaks_all = pd.DataFrame({"step_time": la.ts[la_peaks_all], 'idx': la_peaks_all, 'gait_bout_num': [0] * len(la_peaks_all),
                             'peak_height': proc_dict[subj]['la'][la_peaks_all], 'foot': ['left'] * len(la_peaks_all)})

"""ra_peaks = findpeaks_in_bouts(df_bouts=dict_bouts[subj], min_steps=5, ankle_signal=proc_dict[subj]['ra'], side='right',
                          sample_rate=ra.signal_headers[0]['sample_rate'], raw_timestamps=ra.ts,
                          peak_thresh=proc_dict[subj]['thresh'])"""
ra_peaks_all = peakutils.indexes(y=proc_dict[subj]['ra'], thres=proc_dict[subj]['thresh'], min_dist=la.signal_headers[0]['sample_rate'] * 2 / 3, thres_abs=True)
ra_peaks_all = pd.DataFrame({"step_time": ra.ts[ra_peaks_all], 'idx': ra_peaks_all, 'gait_bout_num': [0] * len(ra_peaks_all),
                             'peak_height': proc_dict[subj]['ra'][ra_peaks_all], 'foot': ['right'] * len(ra_peaks_all)})

# df_steps_bouts = pd.concat([la_peaks, ra_peaks])
# df_steps_bouts.sort_values("step_time", inplace=True)

df_steps_all = pd.concat([la_peaks_all, ra_peaks_all])
df_steps_all.sort_values("step_time", inplace=True)
df_steps_all.reset_index(drop=True, inplace=True)

# print(f"Original = {dict_steps[subj].shape[0]} steps, new (bouts) = {df_steps_bouts.shape[0]} steps, new (all) = {df_steps_all.shape[0]}")

# df_steps_bouts.to_csv(f"O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/{subj}_originalbouts_steps.csv", index=False)
# df_steps_all.to_csv(f"O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/{subj}_all_steps.csv", index=False)

# ====================

# TODO
# step removal based on proximity to previous step

# verify 6m walk start/end times crop off accel/decel steps
# symmetry measure

df_steps_all = pd.concat([la_peaks_all, ra_peaks_all])
df_steps_all.sort_values("step_time", inplace=True)
df_steps_all.reset_index(drop=True, inplace=True)

df_bill = pd.read_excel("C:/Users/ksweber/Desktop/STEPS_1611.xlsx")

y = pd.read_csv("W:/STEPS/Gait_symmetry_dev/STEPS_1611_ACCy_clinical.csv")
end_idx = int((df_assess_new.loc[df_assess_new['subject_id'] == subj]['6mwt_end'].iloc[0] - la.ts[0]).total_seconds() * 75 + 75 * 300)
z = get_zncc(x=la.signals[1][:end_idx], y=y['Left_y'])
maxz_idx = np.argmax(z)

"""
fig, ax = plt.subplots(2, sharex='col', figsize=(12, 8))
ax[0].plot(la.signals[1][:end_idx], color='black', label="'raw'")

ax[0].plot(np.arange(maxz_idx, maxz_idx + len(y)), y['Left_y'], color='red', label='cropped')
ax[0].legend(loc='lower right')
ax[1].plot(z, label='zncc', color='dodgerblue')
ax[1].legend(loc='lower right')
plt.tight_layout()
"""

start_ts = la.ts[maxz_idx]
df_bill['start_time'] = [start_ts + timedelta(seconds=row.seconds) for row in df_bill.itertuples()]

fig, ax = plt.subplots(2, figsize=(12, 8), sharex='col')
ax[0].plot(la.ts[:250000], la_y_filt[:250000], color='red', zorder=0, label='Left')

ax[1].plot(ra.ts[:250000], ra_y_filt[:250000], color='dodgerblue', zorder=0, label='Right')

event_6m1 = df_assess_new.loc[df_assess_new['subject_id'] == subj]
ax[0].axvspan(event_6m1.walk1_6m_start, event_6m1.walk1_6m_end, 0, 1, color='orange', alpha=.25)
ax[0].axvline(event_6m1.walk1_6m_start, color='orange')

for i in range(2):
    ax[i].axvspan(event_6m1.walk1_6m_start, event_6m1.walk1_6m_end, 0, 1, color='orange', alpha=.25)
    ax[i].axvline(event_6m1.walk1_6m_start, color='orange')
    ax[i].axvspan(event_6m1.walk2_6m_start, event_6m1.walk2_6m_end, 0, 1, color='orange', alpha=.25)
    ax[i].axvline(event_6m1.walk2_6m_start, color='orange')
    ax[i].axvspan(event_6m1.walk3_6m_start, event_6m1.walk3_6m_end, 0, 1, color='orange', alpha=.25)
    ax[i].axvline(event_6m1.walk3_6m_start, color='orange')

for row in df_bill.loc[df_bill['foot'] == 'left'][['start_time', 'Swing']].dropna().itertuples():
    fig.axes[0].axvline(row.start_time, color='grey')

for row in df_bill.loc[df_bill['foot'] == 'left'][['start_time', 'Stance']].dropna().itertuples():
    fig.axes[0].axvline(row.start_time, color='grey', linestyle='dashed')


for row in df_bill.loc[df_bill['foot'] == 'right'][['start_time', 'Swing']].dropna().itertuples():
    fig.axes[1].axvline(row.start_time, color='grey')

for row in df_bill.loc[df_bill['foot'] == 'right'][['start_time', 'Stance']].dropna().itertuples():
    fig.axes[1].axvline(row.start_time, color='grey', linestyle='dashed')

ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')

ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.\n%f"))
plt.tight_layout()
