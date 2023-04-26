import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
from Other.Nami.Nami import import_newsteps_files, import_boutfiles_new, import_sptw_files, flag_steps_gaitboutnum
from Other.Nami.fl_describe import remove_sleep_and_clinical


def crop_bouts_edge(dict_steps, min_steps=0, max_steps=10000, n_steps=2):

    print(f"\nRemoving {n_steps} steps from start/end of all bouts between {min_steps}-{max_steps} steps in duration...")
    dict_steps_out = {}

    for subj in tqdm(dict_steps.keys()):
        df = pd.DataFrame(columns=dict_steps[subj].columns)

        df_subj = dict_steps[subj].copy()
        df_subj = df_subj.loc[df_subj['gait_bout_num'] != 0]

        bout_tally = df_subj['gait_bout_num'].value_counts()
        use_bouts = bout_tally.loc[(bout_tally >= min_steps) & (bout_tally <= max_steps)]
        use_bouts.sort_index(inplace=True)

        for bout_num in use_bouts.index:
            df_bout = df_subj.loc[df_subj['gait_bout_num'] == bout_num].iloc[n_steps:-n_steps]

            df = pd.concat([df, df_bout])

        df = df.reset_index(drop=True)

        dict_steps_out[subj] = df

    return dict_steps_out


def flaggedsteps_to_bouts(dict_steps):

    dict_bouts_out = {}

    for subj in tqdm(dict_steps.keys()):
        df_subj = []

        tally = dict_steps[subj]['gait_bout_num'].value_counts()
        tally = tally.loc[tally.index != 0]
        tally.sort_index(inplace=True)

        for bout_num in tally.index:
            bout = dict_steps[subj].loc[dict_steps[subj]['gait_bout_num'] == bout_num]

            df_subj.append([bout_num,
                            bout.iloc[0]['step_time'], bout.iloc[-1]['step_time'],
                            bout.shape[0],
                            (bout.iloc[-1]['step_time'] - bout.iloc[0]['step_time']).total_seconds(),
                            bout.loc[bout['foot'] == 'left'].shape[0], bout.loc[bout['foot'] == 'right'].shape[0]])

        df_subj = pd.DataFrame(df_subj, columns=['gait_bout_num', 'start_time', 'end_time', 'step_count',
                                                 'duration', 'n_left', 'n_right'])
        df_subj['cadence'] = df_subj['step_count'] * 60 / df_subj['duration']
        df_subj['step_diff'] = [abs(l-r) for l, r in zip(df_subj['n_left'], df_subj['n_right'])]

        dict_bouts_out[subj] = df_subj

    return dict_bouts_out


def calculate_rl_bout_steptimes(dict_bouts, dict_steps, sample_rate=75):
    for subj in tqdm(dict_bouts.keys()):
        l_diffs = []
        r_diffs = []

        for row in dict_bouts[subj].itertuples():
            bout_steps = dict_steps[subj].loc[dict_steps[subj]['gait_bout_num'] == row.gait_bout_num]

            l_steps = bout_steps.loc[bout_steps['foot'] == 'left']
            r_steps = bout_steps.loc[bout_steps['foot'] == 'right']

            # step time from index to seconds
            l_diffs.append(list(l_steps['idx'].diff() / sample_rate)[1:])
            r_diffs.append(list(r_steps['idx'].diff() / sample_rate)[1:])

        dict_bouts[subj]['right_steptimes'] = r_diffs
        dict_bouts[subj]['left_steptimes'] = l_diffs
        dict_bouts[subj]['left_mean_steptime'] = [np.mean(row.left_steptimes) for row in dict_bouts[subj].itertuples()]
        dict_bouts[subj]['right_mean_steptime'] = [np.mean(row.right_steptimes) for row in
                                                   dict_bouts[subj].itertuples()]
        dict_bouts[subj]['rl_ratio'] = dict_bouts[subj]['right_mean_steptime']/dict_bouts[subj]['left_mean_steptime']


def remove_nonalternating_bouts(dict_bouts, dict_steps):
    for subj in dict_bouts.keys():
        dict_bouts[subj] = dict_bouts[subj].loc[dict_bouts[subj]['step_diff'] <= 1]

        alternating = []
        for row in dict_bouts[subj].itertuples():
            bout_steps = dict_steps[subj].loc[dict_steps[subj]['gait_bout_num'] == row.gait_bout_num]
            alternating.append(len(bout_steps.iloc[::2]['foot'].unique()) == 1 and
                               bout_steps.iloc[::2]['foot'].unique()[0] == bout_steps.iloc[0]['foot'] and
                               len(bout_steps.iloc[1::2]['foot'].unique()) == 1 and
                               bout_steps.iloc[1::2]['foot'].unique()[0] == bout_steps.iloc[1]['foot'])

        dict_bouts[subj]['alternating_feet'] = alternating

        dict_bouts[subj] = dict_bouts[subj].loc[dict_bouts[subj]['alternating_feet']]


def classify_boutlen(dict_bouts):
    for subj in dict_bouts.keys():
        # maps label to values in dict_bouts['column'] that fall into bins
        dict_bouts[subj]['stepdur_category'] = pd.cut(x=dict_bouts[subj]['step_count'],
                                                      bins=(0, 11, 40, 200, np.inf),
                                                      labels=['s', 'ms', 'ml', 'l'])


def get_6mwalk1_steps(events, dict_steps, crop_6m_steps_dict):

    dict_out = {}
    for key in dict_steps.keys():
        event = events.loc[events['subject_id'] == key].iloc[0]
        print(key, crop_6m_steps_dict[key])
        dict_out[key] = dict_steps[key].loc[(dict_steps[key]['step_time'] >= event.walk1_6m_start) &
                                            (dict_steps[key]['step_time'] <= event.walk1_6m_end)]

        if crop_6m_steps_dict[key][0] is not None:
            dict_out[key] = dict_out[key].iloc[crop_6m_steps_dict[key][0]:]
        if crop_6m_steps_dict[key][1] is not None:
            dict_out[key] = dict_out[key].iloc[:-crop_6m_steps_dict[key][1]]

    return dict_out


def sliding_window_analysis(subj, dict_steps, dict_bouts, max_steps, window_size):

    df_steps = dict_steps[subj].copy()
    df_bouts = dict_bouts[subj].copy()
    df_bouts = df_bouts.loc[(df_bouts['step_count'] >= window_size) & (df_bouts['step_count'] <= max_steps)]

    data_out = []

    for bout in df_bouts.itertuples():
        bout_steps = df_steps.loc[(df_steps['step_time'] >= bout.start_time) &
                                  (df_steps['step_time'] <= bout.end_time)]
        bout_steps.reset_index(drop=True, inplace=True)

        df_l = bout_steps.loc[bout_steps['foot'] == 'left']
        df_r = bout_steps.loc[bout_steps['foot'] == 'right']

        df_l['diff'] = df_l['idx'].diff() / 75
        df_r['diff'] = df_r['idx'].diff() / 75

        bout_steps = pd.concat([df_l, df_r]).sort_values("step_time")

        for i in range(0, bout_steps.shape[0] - window_size, 1):
            window = bout_steps.loc[i:i+window_size]
            l = window.loc[window['foot'] == 'left'].dropna()
            r = window.loc[window['foot'] == 'right'].dropna()

            data_out.append([bout.gait_bout_num, i, l.shape[0], r.shape[0],
                             round(l['diff'].mean(), 5), round(r['diff'].mean(), 5),
                             round(r['diff'].mean()/l['diff'].mean(), 5)])

    df_out = pd.DataFrame(data_out, columns=['gait_bout_num', 'step_num', 'n_left', 'n_right',
                                             'mean_steptime_l', 'mean_steptime_r', 'rl_ratio'])

    return df_out


use_ids = ['STEPS_1611', 'STEPS_6707', 'STEPS_2938', 'STEPS_7914', 'STEPS_8856']

crop_6m_steps_dict = {"STEPS_1611": [1, 3],
                      'STEPS_6707': [1, 2],
                      'STEPS_7914': [1, None],
                      'STEPS_2938': [1, None]}

df_6m = pd.read_excel("O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/STEPS_6m_walk_stepcounts_new.xlsx")

df_assess_new = pd.read_excel("O:/OBI/Personal Folders/Namiko Huynh/steps_assessment_timestamps_new.xlsx")
df_assess_new.columns = ['subject_id', 'start_time', 'end_time', 'comments',
                         'walk1_6m_start', 'walk1_6m_end', 'walk2_6m_start', 'walk2_6m_end',
                         'walk3_6m_start', 'walk3_6m_end', '6mwt_start', '6mwt_end']
df_assess_new = df_assess_new.loc[df_assess_new['subject_id'] != 'STEPS_8856']

dict_steps = import_newsteps_files(folder="O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/",
                                   subjs=use_ids[:-1], file_suffix="all_steps")

dict_steps_6m = get_6mwalk1_steps(df_assess_new, dict_steps, crop_6m_steps_dict)

dict_bouts = import_boutfiles_new(folder="O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/",
                                  subjs=use_ids[:-1], file_suffix="newbouts")

flag_steps_gaitboutnum(dict_steps=dict_steps, dict_bouts=dict_bouts)

dict_sptw = import_sptw_files(use_ids[:-1])

dict_bouts, dict_steps = remove_sleep_and_clinical(dict_sptw, dict_steps, dict_bouts, df_events=df_assess_new)

# remove 2 steps from start and end of bouts
dict_steps_crop = crop_bouts_edge(dict_steps=dict_steps, min_steps=11, max_steps=40, n_steps=2)

# gets stats for bouts (duration, cadence, start/end times, step_count, R/L steps) on cropped bout data
dict_bouts_crop = flaggedsteps_to_bouts(dict_steps_crop)

# flags each bout as short/medium-short/medium-long/long
classify_boutlen(dict_bouts_crop)

# removes bouts whose R/L step counts differ by more than one step
# also removes bouts where step detection led to non-alternating steps
remove_nonalternating_bouts(dict_bouts_crop, dict_steps_crop)

# calculates left and right step times within each bout
calculate_rl_bout_steptimes(dict_bouts=dict_bouts_crop, dict_steps=dict_steps_crop, sample_rate=75)

# sliding window analysis
dict_slide = {}
for subj in use_ids[:-1]:
    dict_slide[subj] = sliding_window_analysis(subj=subj, dict_steps=dict_steps_crop, dict_bouts=dict_bouts,
                                               max_steps=1000, window_size=df_6m.loc[df_6m['subject_id'] == subj]['walk1'].iloc[0])
