import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.dates import num2date as n2d
from matplotlib.backend_bases import MouseButton
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import os
import numpy as np
from tqdm import tqdm
from Other.Nami.Nami import import_bout_files, import_steps_files
import nimbalwear
from peakutils import indexes as find_peaks
from utilities.Filtering import filter_signal
import peakutils


def plot_subj(subj_id):

    print(f"\nPrepping data for {subj_id}")

    event = df_assess.loc[df_assess['subject_id'] == subj_id].iloc[0]
    event_new = df_assess_new.loc[df_assess_new['subject_id'] == subj_id].iloc[0]

    steps = dict_steps[subj_id]
    bouts = dict_bouts[subj_id]
    bouts['cadence'] = bouts['step_count'] * 60 / bouts['duration']

    # creates steps mask with 100ms resolution
    start_ts = steps.iloc[0]['step_time']
    stop_ts = steps.iloc[-1]['step_time']
    s = np.zeros(int((stop_ts - start_ts).total_seconds() * 10))
    s[[int((row.step_time - start_ts).total_seconds() * 10) for row in steps.iloc[:-1].itertuples()]] = 1
    ts = pd.date_range(start=start_ts, end=stop_ts, freq='100ms')[:-1]

    # cadence in 10-second windows
    window_ts = ts[::100]  # 100ms[::10] = 10 seconds
    cads = []
    for i in tqdm(range(len(window_ts) - 1)):
        t1 = window_ts[i]
        t2 = window_ts[i+1]
        df_epoch = steps.loc[(steps['step_time'] >= t1) & (steps['step_time'] < t2)]
        if df_epoch.shape[0] >= 4:
            n = df_epoch.shape[0]
            cads.append(60 * n / 10)
        else:
            cads.append(0)
    df_epoch = pd.DataFrame({'start_time': window_ts[:-1], 'cadence': cads})

    # PLOT =========================
    fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8))
    ax[0].set_title(f"Steps data for subject {subj_id}")
    ax[0].plot(ts, s, color='dodgerblue')
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['rest', 'step'])

    for row in bouts.itertuples():
        ax[1].plot([row.start_time, row.end_time], [row.cadence, row.cadence], color='red', zorder=1)
        ax[2].plot([row.start_time, row.end_time], [row.step_count, row.step_count],
                   color='gold' if row.step_count < 20 else 'purple')

    ax[1].plot(df_epoch['start_time'], df_epoch['cadence'], color='black')
    ax[1].grid(zorder=0)
    ax[1].set_ylim(0, )
    ax[1].set_ylabel("Cadence")

    ax[2].set_ylabel("Step Counts")
    ax[2].set_ylim(0, )

    try:
        ax[0].axvspan(event_new['walk1_6m_start'], event_new['walk1_6m_end'], color='red', label='6m_1', alpha=.2)
    except:
        pass

    try:
        ax[0].axvspan(event_new['walk2_6m_start'], event_new['walk2_6m_end'], color='orange', label='6m_2', alpha=.2)
    except:
        pass

    try:
        ax[0].axvspan(event_new['walk3_6m_start'], event_new['walk3_6m_end'], color='gold', label='6m_3', alpha=.2)
    except:
        pass

    try:
        ax[0].axvspan(event_new['6mwt_start'], event_new['6mwt_end'], color='dodgerblue', label='6mwt', alpha=.2)
    except:
        pass

    for ax_i in range(1, len(ax)):

        # start/stop for event(s)
        try:
            ax[ax_i].axvline(event.start_time, color='limegreen')
        except:
            pass

        try:
            ax[ax_i].axvline(event.end_time, color='red')
        except:
            pass

        try:
            ax[ax_i].axvspan(event.start_time, event.end_time, 0, 1, color='dodgerblue', alpha=.25)
        except (TypeError, AttributeError, ValueError):
            pass

        # bout shading
        for row in bouts.loc[bouts['step_count'] < 20].itertuples():
            ax[ax_i].axvspan(row.start_time, row.end_time, 0, 1, color='gold', alpha=.25)
        for row in bouts.loc[bouts['step_count'] >= 20].itertuples():
            ax[ax_i].axvspan(row.start_time, row.end_time, 0, 1, color='purple', alpha=.25)

    ax[0].legend(loc='lower right')

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig


def import_edf(la_file, ra_file):

    la = nimbalwear.Device()
    la.import_edf(la_file)

    ra = nimbalwear.Device()
    ra.import_edf(ra_file)

    la.ts = pd.date_range(start=la.header['start_datetime'], periods=len(la.signals[0]),
                          freq=f"{1000 / la.signal_headers[0]['sample_rate']:.6f}ms")
    ra.ts = pd.date_range(start=la.header['start_datetime'], periods=len(ra.signals[0]),
                          freq=f"{1000 / ra.signal_headers[0]['sample_rate']:.6f}ms")

    return la, ra


def detect_6mwt_turns(la_obj, ra_obj, df_steps, df_events, ankle_axis='x', filter_cut=.1, pad_sec=10):

    chn_idx = la_obj.get_signal_index(f'Accelerometer {ankle_axis}')
    fs = la_obj.signal_headers[chn_idx]['sample_rate']

    idx = [int((df_events['6mwt_start'].iloc[0] - la_obj.header['start_datetime']).total_seconds() * fs) - int(pad_sec*fs),
           int((df_events['6mwt_end'].iloc[0] - la_obj.header['start_datetime']).total_seconds() * fs) + int(pad_sec*fs)]

    ts = la_obj.ts[idx[0]:idx[1]]

    # sig = ankle_obj.signals[chn_idx][idx[0]:idx[1]] + (other_ankle.signals[chn_idx][idx[0]:idx[1]] * -1)

    # .05Hz highpass filter to remove gravity
    hp_left = filter_signal(data=la_obj.signals[chn_idx][idx[0]:idx[1]], sample_f=fs, high_f=.05, filter_type='highpass')
    hp_right = filter_signal(data=ra_obj.signals[chn_idx][idx[0]:idx[1]], sample_f=fs, high_f=.05, filter_type='highpass')
    # hp = filter_signal(data=sig, sample_f=fs, high_f=.05, filter_type='highpass')

    # .2Hz lowpass filter on absolute value of highpass signal
    le_left = filter_signal(data=abs(hp_left), sample_f=fs, low_f=filter_cut, filter_type='lowpass')
    le_right = filter_signal(data=abs(hp_right), sample_f=fs, low_f=filter_cut, filter_type='lowpass')

    la_le_inv = le_left * -1
    la_le_inv_peaks = find_peaks(y=la_le_inv, min_dist=fs*5)

    ra_le_inv = le_right * -1
    ra_le_inv_peaks = find_peaks(y=ra_le_inv, min_dist=fs*5)

    df_steps_cropped = df_steps.loc[(df_steps['step_time'] >= ts[0]) & (df_steps['step_time'] <= ts[-1])]
    df_steps_cropped.sort_values("step_time", inplace=True)

    def mouse_click(event):
        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            print(n2d(x).strftime("%Y-%m-%d %H:%M:%S.%f"))

    fig, ax = plt.subplots(4, sharex='col', figsize=(12, 8))

    ax[0].plot(ts, la_obj.signals[chn_idx][idx[0]:idx[1]], color='black', label='left')
    ax[0].plot(ts, ra_obj.signals[chn_idx][idx[0]:idx[1]], color='limegreen', label='right')
    # ax[0].plot(ts, sig, color='red', linestyle='dotted')

    ax[0].legend(loc='lower right')

    ax[1].plot(ts, hp_left, color='black', label='left_highpassed')
    ax[1].plot(ts, hp_right, color='limegreen', label='right_highpassed')
    ax[1].legend(loc='lower right')

    ax[2].plot(ts, abs(hp_left), color='black', label='left_FWR')
    ax[2].plot(ts, abs(hp_right), color='limegreen', label='right_FWR')
    ax[2].legend(loc='lower right')

    ax[3].plot(ts, le_left, color='black', label='left_LE')
    ax[3].plot(ts, le_right, color='limegreen', label='right_LE')
    ax[3].legend(loc='lower right')

    for i in range(4):
        ax[i].axvspan(df_events['6mwt_start'].iloc[0], df_events['6mwt_end'].iloc[0],
                      color='dodgerblue', label='6mwt', alpha=.1)
        ax[i].scatter(df_steps_cropped['step_time'], [ax[i].get_ylim()[1] * 1.1] * df_steps_cropped.shape[0],
                      color='dodgerblue', s=20, marker='v')
        for peak in la_le_inv_peaks:
            ax[i].axvline(ts[peak], color='black', linestyle='dashed')
        for peak in ra_le_inv_peaks:
            ax[i].axvline(ts[peak], color='limegreen', linestyle='dashed')

    ax[3].xaxis.set_major_formatter(xfmt)
    plt.connect('button_press_event', mouse_click)
    plt.tight_layout()

    df_turns_left = pd.DataFrame({"turn_time": ts[la_le_inv_peaks]})
    df_turns_right = pd.DataFrame({"turn_time": ts[ra_le_inv_peaks]})

    return df_turns_left, df_turns_right, df_steps_cropped, fig


def verify_turns(ankle_obj, other_ankle, ankle_axis, df_turns, pad_sec=10):

    chn_idx = ankle_obj.get_signal_index(f'Accelerometer {ankle_axis}')
    fs = ankle_obj.signal_headers[chn_idx]['sample_rate']

    idx = [int((df_turns['turn_time'].iloc[0] - ankle_obj.header['start_datetime']).total_seconds() * fs) - int(pad_sec*fs),
           int((df_turns['turn_time'].iloc[-1] - ankle_obj.header['start_datetime']).total_seconds() * fs) + int(pad_sec*fs)]

    ts = ankle_obj.ts[idx[0]:idx[1]]

    fig, ax = plt.subplots(1, sharex='col', figsize=(12, 6))

    ax.plot(ts, ankle_obj.signals[chn_idx][idx[0]:idx[1]], color='black', label='used_raw')
    ax.plot(ts, other_ankle.signals[chn_idx][idx[0]:idx[1]], color='green', label='other_raw')

    for row in df_turns.itertuples():
        ax.axvline(row.turn_time, color='grey', linestyle='dashed')

    ax.xaxis.set_major_formatter(xfmt)
    plt.tight_layout()


def process_straight_walks(df_turns, df_steps, crop_nsteps=2):
    bouts = []

    for row_idx in range(df_turns.shape[0] - 1):
        df_straight = df_steps.loc[(df_steps['step_time'] >= df_turns.iloc[row_idx]['turn_time']) &
                                   (df_steps['step_time'] <= df_turns.iloc[row_idx + 1]['turn_time'])]

        df_steady = df_straight.iloc[crop_nsteps:-crop_nsteps]
        bouts.append([row_idx + 1, df_steady.shape[0], list(df_steady['step_time'])])

    df_bouts = pd.DataFrame(bouts, columns=['bout_number', 'n_steps', 'step_timestamp'])

    df_bouts['step_times'] = [list(pd.Series(df_bouts.iloc[i]['step_timestamp']).diff()) for
                              i in range(df_bouts.shape[0])]
    df_bouts['avg_step_times'] = [pd.Series(df_bouts.iloc[i]['step_times']).dropna().mean() for
                                  i in range(df_bouts.shape[0])]

    return df_bouts


def flag_step_side():
    for bout_num in dict_steps[subj]['gait_bout_num'].unique():
        bout = dict_steps[subj].loc[dict_steps[subj]['gait_bout_num'] == bout_num]
        d = np.diff(bout['step_idx'])

        try:
            ch_idx = np.argwhere(d < 0)[0][0]
            bout1 = bout.iloc[:ch_idx+1]
            bout2 = bout.iloc[ch_idx+1:]

            dict_steps[subj].loc[bout1.index, 'leg'] = 'right'
            dict_steps[subj].loc[bout2.index, 'leg'] = 'left'

        except:
            pass

    dict_steps[subj].sort_values('step_time', inplace=True)

    df_steps = dict_steps[subj].loc[~dict_steps[subj]['leg'].isnull()]


if __name__ == '__main__':

    os.chdir("W:/NiMBaLWEAR/STEPS/analytics/gait/")

    # IDs to use: ignore STEPS_8856
    use_ids = ['STEPS_1611', 'STEPS_6707', 'STEPS_2938', 'STEPS_7914', 'STEPS_8856']

    # step/bout data
    dict_steps = import_steps_files(subjs=use_ids)
    dict_bouts = import_bout_files(subjs=use_ids)

    for key in dict_steps.keys():
        dict_steps[key].sort_values("step_num", inplace=True)

    # event file: CHANGE TO MAC FORMAT
    df_assess = pd.read_excel("O:/OBI/Personal Folders/Namiko Huynh/steps_assessment_timestamps.xlsx")
    df_assess.columns = ['Participant ID', 'start_time', 'end_time', 'comments']
    df_assess.insert(loc=0, column='subject_id', value=[f"STEPS_{i}" for i in df_assess['Participant ID']])
    df_assess['date'] = [dict_steps[row.subject_id].iloc[0]['step_time'].date() for row in df_assess.itertuples()]
    df_assess['start_time'] = [pd.to_datetime(f"{row.date} {row.start_time}") if not pd.isna(row.start_time) else None for row in df_assess.itertuples()]
    df_assess['end_time'] = [pd.to_datetime(f"{row.date} {row.end_time}") if not pd.isna(row.end_time) else None for row in df_assess.itertuples()]
    df_assess.drop(['Participant ID', 'date'], axis=1, inplace=True)

    # Change to Mac format: /Volumes/....
    df_assess_new = pd.read_excel("O:/OBI/Personal Folders/Namiko Huynh/steps_assessment_timestamps_new.xlsx")
    df_assess_new.columns = ['subject_id', 'start_time', 'end_time', 'comments',
                             'walk1_6m_start', 'walk1_6m_end', 'walk2_6m_start', 'walk2_6m_end',
                             'walk3_6m_start', 'walk3_6m_end', '6mwt_start', '6mwt_end']
    df_assess_new = df_assess_new.loc[df_assess_new['subject_id'] != 'STEPS_8856']

    fig = plot_subj(subj_id='STEPS_1611')
    # df_6m = calculate_6m_walk_stepcount()
    # df_6m.to_excel("C:/Users/ksweber/Desktop/STEPS_6m_walk_stepcounts.xlsx", index=False)
