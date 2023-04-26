import os
import pandas as pd
import numpy as np
from Other.Nami.Nami import import_bout_files, import_steps_files, import_dailysteps_files, import_sptw_files
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm


def characterize_freeliving_gait(steps_dict,
                                 bouts_dict,
                                 classify_bout_col='duration',
                                 classify_bout_thresh=(),
                                 classify_bout_labels=('short', 'med', 'long'),
                                 ):

    data_out = []

    for subj in use_ids:

        # uses daily summary data to determine which days were full collection days: first and last
        startday = dict_daily[subj]['date'].min()
        endday = dict_daily[subj]['date'].max()

        n_days = (endday - startday).days

        # crops steps df to only include steps on full days

        all_steps = steps_dict[subj].loc[(steps_dict[subj]['step_time'] >= pd.to_datetime(startday)) &
                                         (steps_dict[subj]['step_time'] < pd.to_datetime(endday) + timedelta(days=1))]
        fl_steps = all_steps.shape[0]

        # uses function arguments to classify gait bout on their length (time or steps)
        bouts_dict[subj]['bout_dur_class'] = pd.cut(x=bouts_dict[subj][classify_bout_col],
                                                    bins=classify_bout_thresh,
                                                    labels=classify_bout_labels)

        # crops bouts df to only include steps on full days
        bouts = bouts_dict[subj].loc[(bouts_dict[subj]['start_time'] >= pd.to_datetime(startday)) &
                                     (bouts_dict[subj]['end_time'] <= pd.to_datetime(endday) + timedelta(days=1))]

        # number of steps in bouts
        # bouted_steps = bouts_dict[subj]['step_count'].sum()
        bouted_steps = all_steps.loc[all_steps['gait_bout_num'] != 0].shape[0]

        # total time walking = sum of bout durations
        total_time = bouts['duration'].sum()

        # counts bout classification values and makes into percent
        bout_dur_percents = bouts['bout_dur_class'].value_counts(normalize=1)*100
        bout_dur_tally = bouts['bout_dur_class'].value_counts()

        # participant's data as list
        subj_data = [subj, fl_steps, bouted_steps, round(bouted_steps/n_days, 1),
                     round(total_time, 0), round(total_time/n_days, 1)]

        # output column names
        cols = ['subject_id', 'all_steps', 'bouted_steps', 'boutedsteps_perday', 'walk_dur_sec', 'walk_dur_sec_perday']

        for bout_cat in classify_bout_labels:
            # percent of bouts
            if bout_cat in bout_dur_percents.index.categories:
                subj_data.append(round(bout_dur_percents.loc[bout_cat], 2))

            if bout_cat not in bout_dur_percents.index.categories:
                bout_dur_percents.loc[bout_cat] = 0
                subj_data.append(0)
            cols.append(f"perc_{bout_cat}_walks")

            # tally
            if bout_cat in bout_dur_tally.index.categories:
                subj_data.append(round(bout_dur_tally.loc[bout_cat], 2))

            if bout_cat not in bout_dur_tally.index.categories:
                bout_dur_tally.loc[bout_cat] = 0
                subj_data.append(0)
            cols.append(f"{bout_cat}_walks")

        # list of lists for output df
        data_out.append(subj_data)

    df_out = pd.DataFrame(data_out, columns=cols)

    df_out['walk_dur_mins'] = df_out['walk_dur_sec'] / 60
    df_out['walk_dur_mins_perday'] = df_out['walk_dur_sec_perday'] / 60

    df_out['total_walks'] = df_out['short_walks'] + df_out['medshort_walks'] + \
                            df_out['medlong_walks'] + df_out['long_walks']

    df_out = df_out[['subject_id', 'all_steps', 'bouted_steps', 'boutedsteps_perday',
                     'walk_dur_sec', 'walk_dur_sec_perday',
                     'walk_dur_mins', 'walk_dur_mins_perday',
                     'perc_short_walks', 'perc_medshort_walks', 'perc_medlong_walks', 'perc_long_walks',
                     'short_walks', 'medshort_walks', 'medlong_walks', 'long_walks', 'total_walks']]

    return df_out


def remove_sleep_and_clinical(dict_sptw, dict_steps, dict_bouts, df_events):

    dict_bouts_use = {}
    dict_steps_use = {}

    for subj in tqdm(dict_sptw.keys()):
        steps = dict_steps[subj].copy()
        bouts = dict_bouts[subj].copy()
        sleep = dict_sptw[subj].copy()
        events = df_events.loc[df_events['subject_id'] == subj].iloc[0]

        df_mask = pd.DataFrame({'start_time': pd.date_range(start=steps.iloc[0]['step_time'].round(freq='1S'),
                                                            end=steps.iloc[-1]['step_time'].round("1S"), freq='1S')})
        start_stamp = df_mask.iloc[0]['start_time']
        df_mask['sleep_mask'] = [0] * df_mask.shape[0]
        df_mask['clin_mask'] = [0] * df_mask.shape[0]

        for row in sleep.itertuples():
            start_i = int(np.floor((row.start_time - start_stamp).total_seconds()))
            end_i = int(np.ceil((row.end_time - start_stamp).total_seconds()))
            df_mask.loc[start_i:end_i, 'sleep_mask'] = 1

        for cols in [['walk1_6m_start', 'walk1_6m_end'],
                     ['walk2_6m_start', 'walk2_6m_end'],
                     ['walk3_6m_start', 'walk3_6m_end'],
                     ['6mwt_start', '6mwt_end']]:
            start_i = int(np.floor((events[cols[0]] - start_stamp).total_seconds()))
            end_i = int(np.ceil((events[cols[1]] - start_stamp).total_seconds()))

            df_mask.loc[start_i:end_i, 'clin_mask'] = 1

        bouts['has_clin'] = [False] * bouts.shape[0]
        bouts['has_sleep'] = [False] * bouts.shape[0]
        for row in bouts.itertuples():
            df_bout = df_mask.loc[(df_mask['start_time'] >= row.start_time) & (df_mask['start_time'] < row.end_time)]
            bouts.loc[row.Index, 'has_clin'] = df_bout['clin_mask'].sum() > 0
            bouts.loc[row.Index, 'has_sleep'] = df_bout['sleep_mask'].sum() > 0

        dict_bouts_use[subj] = bouts.loc[(~bouts['has_clin']) & (~bouts['has_sleep'])]

        dict_steps_use[subj] = steps.loc[steps['gait_bout_num'].isin(list(dict_bouts_use[subj]['gait_bout_num']))]

    return dict_bouts_use, dict_steps_use


def remove_edge_steps(df_steps, df_bouts, crop_n_steps=2):

    steps_out = pd.DataFrame()

    for row in df_bouts.itertuples():

        bout_steps = df_steps.loc[df_steps['gait_bout_num'] == row.gait_bout_num]

        steps_out = pd.concat([steps_out, bout_steps.iloc[crop_n_steps:-crop_n_steps]])

    return steps_out


def get_walk(df_steps, start, end, crop_n_steps=2):

    df_out = df_steps.loc[(df_steps['step_time'] >= pd.to_datetime(start)) &
                        (df_steps['step_time'] <= pd.to_datetime(end))].iloc[crop_n_steps:-crop_n_steps]

    return df_out


if __name__ == '__main__':
    os.chdir("W:/NiMBaLWEAR/STEPS/analytics/gait/")

    # IDs to use
    use_ids = ['STEPS_1611', 'STEPS_2938', 'STEPS_6707', 'STEPS_7914']

    # dict_steps = import_steps_files(subjs=use_ids)
    # dict_bouts = import_bout_files(subjs=use_ids)
    dict_daily = import_dailysteps_files(subjs=use_ids)
    # dict_sptw = import_sptw_files(subjs=use_ids)

    df_events = pd.read_excel("O:/OBI/Personal Folders/Namiko Huynh/steps_assessment_timestamps_new.xlsx")
    df_events.columns = ['subject_id', 'start_time', 'end_time', 'comments',
                         'walk1_6m_start', 'walk1_6m_end', 'walk2_6m_start', 'walk2_6m_end',
                         'walk3_6m_start', 'walk3_6m_end', '6mwt_start', '6mwt_end']
    df_events.drop(['start_time', 'end_time', 'comments'], inplace=True, axis=1)
    df_events = df_events.loc[df_events['subject_id'] != 'STEPS_8856']

    dict_bouts_use, dict_steps_use = remove_sleep_and_clinical()

    df_desc = characterize_freeliving_gait(steps_dict=dict_steps,
                                           # steps_dict=dict_steps_use,
                                           bouts_dict=dict_bouts_use,
                                           classify_bout_col='step_count',
                                           classify_bout_thresh=(0, 10, 40, 200, np.inf),
                                           classify_bout_labels=('short', 'medshort', 'medlong', 'long'))


    # df_desc.to_csv("O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/freeliving_walking_final_descriptive.csv", index=False)

    dict_6mwalk1 = {}
    for subj in dict_steps_use.keys():
        dict_steps_use[subj] = remove_edge_steps(df_steps=dict_steps_use[subj], crop_n_steps=2,
                                                 df_bouts=dict_bouts_use[subj].loc[dict_bouts_use[subj]['bout_dur_class'] == 'medshort'])

        dict_6mwalk1[subj] = get_walk(df_steps=dict_steps[subj], crop_n_steps=2,
                                      start=df_events.loc[df_events['subject_id'] == subj]['walk1_6m_start'].iloc[0],
                                      end=df_events.loc[df_events['subject_id'] == subj]['walk1_6m_end'].iloc[0])
