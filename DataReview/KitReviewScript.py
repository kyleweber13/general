from pathlib import Path
from datetime import timedelta

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import Patch as mpatch
from scipy.signal import butter, sosfilt

from nwdata import NWData
from nwdata.nwfiles import EDFFile


# device paths
wrist_edf_path = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/device_edf_crop/OND09_0085_01_AXV6_RWrist.edf")
ankle_edf_path = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/device_edf_crop/OND09_0085_01_AXV6_RAnkle.edf")
chest_edf_path = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/device_edf_crop/OND09_0085_01_BF36_Chest.edf")

chest_full_edf_path = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/device_edf_sync/OND09_0085_01_BF36_Chest.edf")

# even paths
gait_bouts_csv = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/events/OND09_0085_01_gait_bouts.csv")
posture_bouts_csv = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/events/OND09_0085_01_posture_bouts.csv")
activity_bouts_csv = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/events/OND09_0085_01_activity_bouts.csv")
heart_beats_csv = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/events/OND09_0085_01_cn_beats.csv")
heart_events_csv = Path("W:/OND09 (HANDDS-ONT)/Data Review/0085/events/OND09_0085_01_cn_events.csv")

# included arrythmias
focus_arrs = ()

# plotting color dicts
posture_color_dict = {'sit': 'red', 'stand': 'blue', 'sitstand': 'purple', 'leftside': 'orange', 'rightside': 'green',
                      'reclined': 'pink', 'prone': 'yellow', 'supine': 'cyan', 'transition': 'black', 'other': 'grey'}

activity_color_dict = {'light': 'gold', 'moderate': 'orange', 'vigorous': 'orangered'}

hr_color_dict = {'sedentary': 'royalblue', 'light': 'gold', 'moderate': 'orange',
                 'vigorous': 'orangered', 'max': 'firebrick', 'unlikely': 'gray'}

he_color_dict = {'Tachy': 'orangered', 'SVT': 'orange', 'VT': 'gold', 'Brady': 'cornflowerblue', 'Arrest': 'midnightblue',
                 'AF': 'lime',  'ST+': 'lightcoral', 'ST-': 'aquamarine', 'Block': 'plum', 'AV2/II': 'mediumpurple',
                 'AV2/III': 'rebeccapurple'}

gait_color = 'green'


device_edf_paths = [chest_edf_path, wrist_edf_path, ankle_edf_path]

# read devices
devices = []
for idx, device_edf_path in enumerate(device_edf_paths):

    # read device data
    device = NWData()
    device.import_edf(device_edf_path)

    devices.append(device)


# read events
gait = pd.read_csv(gait_bouts_csv)
gait['start_timestamp'] = pd.to_datetime(gait['start_timestamp'])
gait['end_timestamp'] = pd.to_datetime(gait['end_timestamp'])

posture = pd.read_csv(posture_bouts_csv)
posture['start_timestamp'] = pd.to_datetime(posture['start_timestamp'])
posture['end_timestamp'] = pd.to_datetime(posture['end_timestamp'])

activity = pd.read_csv(activity_bouts_csv)
activity['start_timestamp'] = pd.to_datetime(activity['start_timestamp'])
activity['end_timestamp'] = pd.to_datetime(activity['end_timestamp'])


chest_full = EDFFile(chest_full_edf_path)
chest_full.read_header()

heart_beats = pd.read_csv(heart_beats_csv, sep=';')
heart_beats.rename(columns={heart_beats.columns[0]: "msec"}, inplace=True)
heart_beats['msec'] = pd.to_timedelta(heart_beats['msec'], unit='ms')
heart_beats['timestamp'] = chest_full.header['startdate'] + heart_beats['msec']
heart_beats['rate'] = 60000 / heart_beats['RR']
heart_beats['%max'] = heart_beats['rate'] / (220 - 68)
heart_beats['intensity'] = None
heart_beats['intensity'].loc[(heart_beats['%max'] < 0.55)] = 'sedentary'
heart_beats['intensity'].loc[(heart_beats['%max'] >= 0.55) & (heart_beats['%max'] < 0.65)] = 'light'
heart_beats['intensity'].loc[(heart_beats['%max'] >= 0.65) & (heart_beats['%max'] < 0.75)] = 'moderate'
heart_beats['intensity'].loc[(heart_beats['%max'] >= 0.75) & (heart_beats['%max'] < 0.90)] = 'vigorous'
heart_beats['intensity'].loc[(heart_beats['%max'] >= 0.90) & (heart_beats['%max'] < 1.10)] = 'max'
heart_beats['intensity'].loc[(heart_beats['%max'] >= 1.10)] = 'unlikely'

heart_events = pd.read_csv(heart_events_csv, sep=';')
heart_events.rename(columns={heart_events.columns[0]: "msec"}, inplace=True)
heart_events['msec'] = pd.to_timedelta(heart_events['msec'], unit='ms')
heart_events['Length'] = pd.to_timedelta(heart_events['Length'], unit='ms')
heart_events['start_timestamp'] = chest_full.header['startdate'] + heart_events['msec']
heart_events['end_timestamp'] = heart_events['start_timestamp'] + heart_events['Length']
detect_he_color_dict = { k: v for k,v in he_color_dict.items() if k in heart_events['Type'].unique() }

print('Plotting...')

fig, axs = plt.subplots(nrows=len(devices) + 2, ncols=1, sharex='all', figsize=(18,9))
titles = ['ecg', 'heart rate', 'chest accel + posture', 'wrist accel + activity', 'ankle accel + gait']
plt.subplots_adjust(left=0.04, right=0.85, bottom=0.07, top=0.96, hspace=0.2, wspace=0.2)

ecg_ds = 3

ecg_ind = devices[0].get_signal_index('ECG')
ecg = devices[0].signals[ecg_ind][::ecg_ds]

start_datetime = devices[0].header['startdate']
ecg_sample_rate = devices[0].signal_headers[ecg_ind]['sample_rate'] / ecg_ds

sos = butter(N=5, Wn=[.67, 40], btype='bandpass', analog=False, output='sos', fs=ecg_sample_rate)
ecg = sosfilt(sos, ecg)


ecg_times = mdates.date2num([start_datetime + timedelta(seconds=(i / ecg_sample_rate)) for i in range(len(ecg))])
heart_beats = heart_beats.loc[(ecg_times[0] <= mdates.date2num(heart_beats['timestamp'])) & (mdates.date2num(heart_beats['timestamp']) <= ecg_times[-1])]
heart_events = heart_events.loc[(ecg_times[0] <= mdates.date2num(heart_events['end_timestamp'])) & (mdates.date2num(heart_events['start_timestamp']) <= ecg_times[-1])]

axs[0].set_title(titles[0])
axs[0].set_ylabel('ecg (uV)')
axs[0].plot_date(ecg_times, ecg, label='ecg', fmt='', linewidth=0.25)
#axs[0].set_ylim((-2000, 2000))


axs[1].set_title(titles[1])
axs[1].set_ylabel('hr (bpm)')

for i, c in hr_color_dict.items():
    axs[1].plot_date(heart_beats['timestamp'].loc[heart_beats['intensity'] == i],
                     heart_beats['rate'].loc[heart_beats['intensity'] == i],
                     label=i, ms=2, mew=0, mfc=c, alpha=0.5)


for i, d in enumerate(devices):

    x_ind = d.get_signal_index('Accelerometer x')
    y_ind = d.get_signal_index('Accelerometer y')
    z_ind = d.get_signal_index('Accelerometer z')

    x = d.signals[x_ind]
    y = d.signals[y_ind]
    z = d.signals[z_ind]

    start_datetime = d.header['startdate']
    accel_sample_rate = d.signal_headers[x_ind]['sample_rate']

    accel_times = mdates.date2num(
        [start_datetime + timedelta(seconds=(i / accel_sample_rate)) for i in range(len(x))])

    axs[i-3].set_title(titles[i-3])
    axs[i-3].set_ylabel('acceleration (g)')
    axs[i-3].plot_date(accel_times, x, label='x', fmt='', linewidth=0.25)
    axs[i-3].plot_date(accel_times, y, label='y', fmt='', linewidth=0.25)
    axs[i-3].plot_date(accel_times, z, label='z', fmt='', linewidth=0.25)
    axs[i-3].set_ylim((-5, 5))

axs[-1].set_xlabel('datetime')

for _, r in heart_events.iterrows():

    if r['Type'] in detect_he_color_dict.keys():

        r['end_timestamp'] = r['start_timestamp'] if pd.isnull(r['end_timestamp']) else r['end_timestamp']
        r['start_timestamp'] = r['end_timestamp'] if pd.isnull(r['start_timestamp']) else r['start_timestamp']

        axs[0].axvspan(xmin=r['start_timestamp'], xmax=r['end_timestamp'], alpha=0.15, linewidth=0,
                       color=detect_he_color_dict[r['Type']])

for _, r in posture.iterrows():

    if r['posture'] in posture_color_dict.keys():

        r['end_timestamp'] = r['start_timestamp'] if pd.isnull(r['end_timestamp']) else r['end_timestamp']
        r['start_timestamp'] = r['end_timestamp'] if pd.isnull(r['start_timestamp']) else r['start_timestamp']

        axs[-3].axvspan(xmin=r['start_timestamp'], xmax=r['end_timestamp'], alpha=0.15,  linewidth=0,
                        color=posture_color_dict[r['posture']])

for _, r in activity.iterrows():

    if r['intensity'] not in ['none', 'sedentary']:
        r['end_timestamp'] = r['start_timestamp'] if pd.isnull(r['end_timestamp']) else r['end_timestamp']
        r['start_timestamp'] = r['end_timestamp'] if pd.isnull(r['start_timestamp']) else r['start_timestamp']

        axs[-2].axvspan(xmin=r['start_timestamp'], xmax=r['end_timestamp'], alpha=0.15, linewidth=0,
                       color=activity_color_dict[r['intensity']])

for _, r in gait.iterrows():

    r['end_timestamp'] = r['start_timestamp'] if pd.isnull(r['end_timestamp']) else r['end_timestamp']
    r['start_timestamp'] = r['end_timestamp'] if pd.isnull(r['start_timestamp']) else r['start_timestamp']

    axs[-1].axvspan(xmin=r['start_timestamp'], xmax=r['end_timestamp'], alpha=0.15,  linewidth=0,
                    color=gait_color)

l = axs[0].legend()
legend_handles = l.legendHandles
for e, c in detect_he_color_dict.items():
    legend_handles.append(mpatch(facecolor=c, label=e, alpha=0.15))
ncol = int(len(legend_handles) / 6) + 1
axs[0].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol)


l = axs[1].legend()
legend_handles = l.legendHandles
# for i, c in hr_color_dict.items():
#     legend_handles.append(mpatch(facecolor=c, label=i))
axs[1].legend(handles=legend_handles[::-1], loc='center left', bbox_to_anchor=(1, 0.5))


axs[-3].legend()
legend_handles, _ = plt.gca().get_legend_handles_labels()
for p, c in posture_color_dict.items():
    legend_handles.append(mpatch(facecolor=c, label=p, alpha=0.15))
axs[-3].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

axs[-2].legend()
legend_handles, _ = plt.gca().get_legend_handles_labels()
for a, c in activity_color_dict.items():
    legend_handles.append(mpatch(facecolor=c, label=a, alpha=0.15))
axs[-2].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

axs[-1].legend()
legend_handles, _ = plt.gca().get_legend_handles_labels()
legend_handles.append(mpatch(facecolor=gait_color, label='walking', alpha=0.15))
axs[-1].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
