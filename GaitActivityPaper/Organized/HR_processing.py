import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
from datetime import timedelta
from tqdm import tqdm


def import_subject_processed(subj, root_dir):

    out_dict = {}

    out_dict['wrist_1s'] = pd.read_csv(f"{root_dir}EpochedWrist/{subj}_EpochedWrist.csv")
    out_dict['wrist_15s'] = pd.read_csv(f"{root_dir}EpochedWrist/{subj}_EpochedWrist15.csv")
    out_dict['steps'] = pd.read_csv(f"{root_dir}Steps/{subj}_01_GAIT_STEPS.csv")
    out_dict['processed_bouts'] = pd.read_csv(f"{root_dir}WalkingBouts/{subj}_ProcessedBouts.csv")
    out_dict['walking_epochs'] = pd.read_csv(f"{root_dir}WalkingEpochs/{subj}_WalkEpochs.csv")
    try:
        out_dict['ecg_peaks'] = pd.read_csv(f"{root_dir}ECG_Peaks/{subj}_ecg_peaks_raw.csv")

        b2b_hr = [60 / ((int(j) - int(i)) / 250) for i, j in zip(out_dict['ecg_peaks']['idx'][:], out_dict['ecg_peaks']['idx'][1:])]
        b2b_hr.append(None)
        out_dict['ecg_peaks']['hr'] = b2b_hr

    except FileNotFoundError:
        out_dict['ecg_peaks'] = pd.DataFrame()

    for key in out_dict.keys():
        for column in out_dict[key].columns:
            if 'time' in column:
                out_dict[key][column] = pd.to_datetime(out_dict[key][column])

    return out_dict


def plot_results(data, age=None, incl_walking_intensity=False, cadence_thresh=None, use_b2b_hr=True,
                 use_epoch_hr=True, wrist_epoch_len=15):

    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))

    ax[0].plot(data[f'wrist_{wrist_epoch_len}s']['start_time'], data[f'wrist_{wrist_epoch_len}s']['avm'], color='black', zorder=1)

    if incl_walking_intensity:
        c_dict = {"sedentary": 'grey', 'light': 'limegreen', 'moderate': 'orange'}
        for intensity in ['sedentary', 'light', 'moderate']:
            d = data['walking_epochs'].loc[data['walking_epochs']['intensity'] == intensity]
            ax[0].scatter(d['start_time'], d['avm'], color=c_dict[intensity])

    ax[0].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Fraysse_light.', zorder=0)
    ax[0].axhline(y=92.5, color='orange', linestyle='dashed', label='Fraysse_mod.', zorder=0)
    ax[0].legend(loc='upper right')
    ax[0].set_ylim(0, data[f'wrist_{wrist_epoch_len}s']['avm'].max() + 25)

    ax[0].set_ylabel("Wrist AVM")
    ax[1].bar(data['walking_epochs']['start_time'], data['walking_epochs']['cadence'], width=15/86400,
              color='grey', edgecolor='black', align='edge')

    if cadence_thresh is not None:
        ax[1].axhline(cadence_thresh, color='orange', linestyle='dashed')
    ax[1].set_ylabel("steps/min")

    if use_b2b_hr:
        ax[2].plot(data['ecg_peaks']['timestamp'], data['ecg_peaks']['hr'], color='black')
    if use_epoch_hr:
        for row in data['walking_epochs'].itertuples():
            ax[2].plot([row.start_time, row.start_time + timedelta(seconds=15)],
                       [row.hr, row.hr], color='black' if not use_b2b_hr else 'dodgerblue')

    if age is not None:
        ax[2].axhline(y=207-.7*age, color='grey', linestyle='dashed', label=f"100% HRmax ({207-.7*age:.0f}bpm)")
        ax[2].axhline(y=.7*(207-.7*age), color='red', linestyle='dashed', label=f'70% HRmax ({.7*(207-.7*age):.0f}bpm)')
        ax[2].axhline(y=.5*(207-.7*age), color='orange', linestyle='dashed', label=f'50% HRmax ({.5*(207-.7*age):.0f}bpm)')
        ax[2].axhline(y=.4*(207-.7*age), color='limegreen', linestyle='dashed', label=f'40% HRmax ({.4*(207-.7*age):.0f}bpm)')

    ax[2].legend(loc='upper right')
    ax[2].set_ylabel("Beat-to-beat\nheart rate (bpm)")

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig


def epoch_hr(df):

    hrs = []

    for row in tqdm(df.itertuples()):
        start = row.start_time
        end = row.start_time + timedelta(seconds=15)

        df_hr = data['ecg_peaks'].loc[(data['ecg_peaks']['timestamp'] >= start) &
                                      (data['ecg_peaks']['timestamp'] < end)].reset_index(drop=True)

        try:
            n_beats = df_hr.shape[0]
            dt = (df_hr['timestamp'].iloc[-1] - df_hr['timestamp'].iloc[0]).total_seconds()

            if n_beats >= 10:
                hrs.append((n_beats - 1) / dt * 60)
            else:
                hrs.append(None)
        except:
            hrs.append(None)

    return hrs


def calculate_all_epoch_cadence(data):

    cads = []
    for start, end in zip(data['wrist_15s']['start_time'].iloc[0:], data['wrist_15s']['start_time'].iloc[1:]):
        df_steps = data['steps'].loc[(data['steps']['step_time'] >= start) & (data['steps']['step_time'] < end)]

        if 'OND06' in full_id:
            n_steps = df_steps.shape[0]
        if 'OND09' in full_id:
            n_steps = df_steps.shape[0] * 2
        cad = n_steps / 15 * 60
        cads.append(cad)

    cads.append(0)

    return cads


full_id = 'OND06_9525'
age = 71
root_dir = "O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/Calibrated/Data/"
df_demos = pd.read_excel("O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/CSEP_Abstract/Data/SummaryData/totals_all.xlsx")

data = import_subject_processed(subj=full_id, root_dir=root_dir)

data['walking_epochs']['hr'] = epoch_hr(data['walking_epochs'])

# fig = plot_results(data=data, age=age, incl_walking_intensity=True, cadence_thresh=80, use_b2b_hr=True, use_epoch_hr=True, wrist_epoch_len=1)

cadence = calculate_all_epoch_cadence(data)
hr15 = epoch_hr(data['wrist_15s'])
data['all_epochs'] = pd.DataFrame({'start_time': data['wrist_15s']['start_time'],
                                   'wrist_15s': data['wrist_15s']['avm'],
                                   'cadence': cadence,
                                   'hr': hr15})

pred_target_hr = {"max": round(207 - .7*age, 0), '40%': round(.4*(207-.7*age), 0),
                  '50%': round(.5*(207-.7*age), 0), '70%': round(.7*(207-.7*age), 0)}


def plot_section(data, start=None, stop=None, wrist_epochlen=15, incl_wrist_scatter=False, abs_time=False,
                 shade_long_walks=False, use_hrmax=True, use_hrr=False,
                 use_epoch_hr=False, use_b2b_hr=True):

    fig, ax = plt.subplots(3, sharex='col', figsize=(13, 6))

    if start is not None:
        start = pd.to_datetime(start)
    if stop is not None:
        stop = pd.to_datetime(stop)

    if start is None:
        start = data['wrist_1s']['start_time'].iloc[0]
    if stop is None:
        stop = data['wrist_1s']['start_time'].iloc[-1]

    df_wrist = data[f'wrist_{wrist_epochlen}s'].loc[(data[f'wrist_{wrist_epochlen}s']['start_time'] >= start) &
                                                    (data[f'wrist_{wrist_epochlen}s']['start_time'] <= stop)]

    df_walkepochs = data['walking_epochs'].loc[(data['walking_epochs']['start_time'] >= start) &
                                               (data['walking_epochs']['start_time'] <= stop)]

    df_allepochs = data['all_epochs'].loc[(data['all_epochs']['start_time'] >= start) &
                                          (data['all_epochs']['start_time'] <= stop)]

    df_b2b = data['ecg_peaks'].loc[(data['ecg_peaks']['timestamp'] >= start) &
                                   (data['ecg_peaks']['timestamp'] <= stop)]

    if abs_time:
        df_wrist['start_time'] = [(row.start_time - start).total_seconds() / 60 for row in df_wrist.itertuples()]
        df_walkepochs['start_time'] = [(row.start_time - start).total_seconds() / 60 for row in df_walkepochs.itertuples()]
        df_allepochs['start_time'] = [(row.start_time - start).total_seconds() / 60 for row in df_allepochs.itertuples()]
        df_b2b['timestamp'] = [(row.timestamp - start).total_seconds() / 60 for row in df_b2b.itertuples()]

    ax[0].plot(df_wrist['start_time'], df_wrist['avm'],  color='black', zorder=1)

    if incl_wrist_scatter:
        c_dict = {"sedentary": 'grey', 'light': 'limegreen', 'moderate': 'orange'}
        for intensity in ['sedentary', 'light', 'moderate']:
            d = df_walkepochs.loc[df_walkepochs['intensity'] == intensity]
            ax[0].scatter(d['start_time'], d['avm'], color=c_dict[intensity], zorder=2)

    ax[0].axhline(y=92.5, color='orange', linestyle='dashed', label='Fraysse mod.', zorder=0)
    ax[0].axhline(y=62.5, color='limegreen', linestyle='dashed', label='Fraysse light', zorder=0)
    ax[0].legend(loc='upper right')
    ax[0].set_ylabel(f"Wrist AVM ({wrist_epochlen}-sec)")
    ax[0].set_ylim(0, )

    ax[1].plot(df_allepochs['start_time'], df_allepochs['cadence'], color='dodgerblue')
    ax[1].axhline(y=80, color='orange', linestyle='dashed', label="Mod. threshold (Jeng et al., 2020)")
    ax[1].set_ylabel("Cadence (steps/min)")
    ax[1].legend()

    if use_epoch_hr:
        ax[2].plot(df_allepochs['start_time'], df_allepochs['hr'], color='black')
    if use_b2b_hr:
        ax[2].plot(df_b2b['timestamp'], df_b2b['hr'], color='black')

    if use_hrmax:
        # ax[2].axhline(pred_target_hr['max'], color='grey', linestyle='dashed', label='HRmax')
        ax[2].axhline(pred_target_hr['70%'], color='red', linestyle='dashed', label='70% HRmax')
        ax[2].axhline(pred_target_hr['50%'], color='orange', linestyle='dashed', label='50% HRmax')
        ax[2].axhline(pred_target_hr['40%'], color='limegreen', linestyle='dashed', label='40% HRmax')

    if use_hrr:
        ax[2].axhline(.6 * (pred_target_hr['max'] - 60) + 60, color='red', label='60% HRR')
        ax[2].axhline(.4 * (pred_target_hr['max'] - 60) + 60, color='orange', label='40% HRR')
        ax[2].axhline(.3 * (pred_target_hr['max'] - 60) + 60, color='limegreen', label='30% HRmax')

    ax[2].legend(loc='upper right')
    ax[2].set_ylabel("HR (bpm)")

    if shade_long_walks:
        for row in data['walking_epochs'].itertuples():
            ax[1].axvspan(xmin=row.start_time, xmax=row.start_time + timedelta(seconds=14.5), ymin=0, ymax=1,
                          color='gold', alpha=.15)

    if not abs_time:
        ax[-1].xaxis.set_major_formatter(xfmt)
    if abs_time:
        ax[-1].set_xlabel("Minutes")

    plt.tight_layout()

    return fig


fig = plot_section(data=data, start='2020-03-03 15:28:45', stop='2020-03-03 15:35:46', abs_time=True,
                   wrist_epochlen=15, incl_wrist_scatter=False,
                   shade_long_walks=False, use_hrmax=True, use_hrr=False,
                   use_epoch_hr=False, use_b2b_hr=True)
# sedentary
fig.axes[-1].set_xlim(0, 7)
fig.axes[0].set_ylim(0, 150)
fig.axes[1].set_ylim(-1, 175)
fig.axes[2].set_ylim(50, 115)
# plt.savefig("C:/Users/ksweber/Desktop/CSEP_plots/Raw/sample_window.tiff", dpi=200)

""""
ax[-1].set_xlim(pd.to_datetime('2020-03-05 13:40:00'), pd.to_datetime('2020-03-05 14:05:00'))
ax[0].set_ylim(0, 150)
ax[1].set_ylim(0, 175)
ax[2].set_ylim(50, 115)
plt.savefig("C:/Users/ksweber/Desktop/CSEP_plots/Raw/sedwrist_gaitmod_hrmixed.tiff", dpi=150)

plt.savefig("C:/Users/ksweber/Desktop/CSEP_plots/Raw/sedwrist_gaitmod_hrlight_hrr.tiff", dpi=150)

# mixed intensity
ax[-1].set_xlim(pd.to_datetime('2020-03-04 13:48:33'), pd.to_datetime('2020-03-04 14:04:30'))
ax[0].set_ylim(0, 150)
ax[1].set_ylim(0, 175)
ax[2].set_ylim(60, 115)
plt.savefig("C:/Users/ksweber/Desktop/CSEP_plots/Raw/mixed_intensity_hrr.tiff", dpi=150)
"""
