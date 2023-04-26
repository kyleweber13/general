import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_6m_walk_stepcount(dict_steps):
    dict_6m = {}

    data = []
    for row in df_assess_new.itertuples():
        try:
            walk1 = dict_steps[row.subject_id].loc[(dict_steps[row.subject_id]['step_time'] >= row.walk1_6m_start) &
                                                   (dict_steps[row.subject_id]['step_time'] <= row.walk1_6m_end)]

            walk2 = dict_steps[row.subject_id].loc[(dict_steps[row.subject_id]['step_time'] >= row.walk2_6m_start) &
                                                   (dict_steps[row.subject_id]['step_time'] <= row.walk2_6m_end)]

            walk3 = dict_steps[row.subject_id].loc[(dict_steps[row.subject_id]['step_time'] >= row.walk3_6m_start) &
                                                   (dict_steps[row.subject_id]['step_time'] <= row.walk3_6m_end)]

            walks = [walk1, walk2, walk3]

        except KeyError:
            walks = [None, None, None]

        dict_6m[row.subject_id] = walks

        data.append([row.subject_id, walk1.shape[0], walk2.shape[0], walk3.shape[0]])

    df_6m = pd.DataFrame(data, columns=['subject_id', 'walk1', 'walk2', 'walk3'])

    return df_6m


def import_steps_files(subjs):
    """Imports steps files for given subjects"""

    # dictionary of steps data. Each key (subject ID) has df as its value
    dict_steps = {}
    for subj in subjs:
        # reads csv file
        dict_steps[subj] = pd.read_csv(f"steps/{subj}_01_GAIT_STEPS.csv")

        # converts step_time column to datetime
        dict_steps[subj]['step_time'] = pd.to_datetime(dict_steps[subj]['step_time'])

        dict_steps[subj] = dict_steps[subj].sort_values("step_time")

    return dict_steps


def import_newsteps_files(folder, subjs, file_suffix="all_steps"):
    """Imports steps files for given subjects"""

    # dictionary of steps data. Each key (subject ID) has df as its value
    dict_steps = {}
    for subj in subjs:
        # reads csv file
        dict_steps[subj] = pd.read_csv(f"{folder}{subj}_{file_suffix}.csv")

        # converts step_time column to datetime
        dict_steps[subj]['step_time'] = pd.to_datetime(dict_steps[subj]['step_time'])

        dict_steps[subj] = dict_steps[subj].sort_values("step_time")

    return dict_steps


def import_bout_files(subjs):
    """ Imports gait bout files for given subjects"""

    # dictionary of bout data. Each key (subject ID) has df as its value
    dict_bouts = {}
    for subj in subjs:
        # reads csv file
        dict_bouts[subj] = pd.read_csv(f"bouts/{subj}_01_GAIT_BOUTS.csv")

        # converts start_time and end_time columns to datetime
        dict_bouts[subj]['start_time'] = pd.to_datetime(dict_bouts[subj]['start_time'])
        dict_bouts[subj]['end_time'] = pd.to_datetime(dict_bouts[subj]['end_time'])

        # calculates bout duration in seconds
        dict_bouts[subj]['duration'] = [(row.end_time - row.start_time).total_seconds() for
                                        row in dict_bouts[subj].itertuples()]

    return dict_bouts


def import_boutfiles_new(folder, subjs, file_suffix="newbouts"):
    """ Imports gait bout files for given subjects"""

    # dictionary of bout data. Each key (subject ID) has df as its value
    dict_bouts = {}
    for subj in subjs:
        # reads csv file
        dict_bouts[subj] = pd.read_csv(f"{folder}/{subj}_{file_suffix}.csv")

        # converts start_time and end_time columns to datetime
        dict_bouts[subj]['start_time'] = pd.to_datetime(dict_bouts[subj]['start_time'])
        dict_bouts[subj]['end_time'] = pd.to_datetime(dict_bouts[subj]['end_time'])

        # calculates bout duration in seconds
        dict_bouts[subj]['duration'] = [(row.end_time - row.start_time).total_seconds() for
                                        row in dict_bouts[subj].itertuples()]

    return dict_bouts


def flag_steps_gaitboutnum(dict_steps, dict_bouts):

    print("\nFlagging gait bout numbers for each step...")

    for subj in tqdm(dict_bouts.keys()):
        for row in dict_bouts[subj].itertuples():
            dict_steps[subj].loc[(dict_steps[subj]['step_time'] >= row.start_time) &
                                 (dict_steps[subj]['step_time'] <= row.end_time), 'gait_bout_num'] = row.gait_bout_num


def import_dailysteps_files(subjs):
    """Imports daily step count files for given subjects"""

    # dictionary of steps data. Each key (subject ID) has df as its value
    dict_daily = {}
    for subj in subjs:
        # reads csv file
        dict_daily[subj] = pd.read_csv(f"daily/{subj}_01_GAIT_DAILY.csv")

        # converts date column to datetime
        dict_daily[subj]['date'] = [pd.to_datetime(i).date() for i in dict_daily[subj]['date']]

    return dict_daily


def import_sptw_files(subjs):
    """Imports Sleep Period Time Window files for given subjects"""

    # dictionary of steps data. Each key (subject ID) has df as its value
    data_dict = {}
    for subj in subjs:
        # reads csv file
        data_dict[subj] = pd.read_csv(f"Z:/NiMBaLWEAR/STEPS/analytics/sleep/sptw/{subj}_01_SPTW.csv")

        # converts date column to datetime
        data_dict[subj]['start_time'] = [pd.to_datetime(i) for i in data_dict[subj]['start_time']]
        data_dict[subj]['end_time'] = [pd.to_datetime(i) for i in data_dict[subj]['end_time']]

    return data_dict


def find_bouts(step_timestamps: list or tuple or pd.Series,
               min_steps: int = 3,
               max_break: int or float = 5):
    """ Bouts steps data according to given min_steps and max_break values.

        Parameter
        ---------
        step_timestamps
            timestamps for every step
        min_steps
            minimum number of steps to be considered a 'bout'
        max_break
            maximum allowed break period before a bout ends, in seconds

        Returns
        -------
        dataframe of bouts with start_time, end_time, step_count, and duration (seconds) columns

    """

    print(f"\nFinding bouts of minimum {min_steps} steps and maximum break of {max_break} seconds...")

    starts = []
    stops = []
    n_steps_list = []

    step_ts = list(sorted(step_timestamps))

    curr_ind = 0
    for i in range(len(step_timestamps)):

        if i >= curr_ind:

            prev_step = step_ts[i]
            n_steps = 1

            for j in range(i + 1, len(step_ts)):
                step_time = (step_ts[j] - prev_step).total_seconds()

                if step_time <= max_break:
                    n_steps += 1

                    prev_step = step_ts[j]

                if step_time > max_break:
                    if n_steps >= min_steps:
                        starts.append(step_ts[i])
                        curr_ind = j
                        stops.append(step_ts[j - 1])
                        n_steps_list.append(n_steps)

                    if n_steps < min_steps:
                        # if not enough steps in bout, ignore it
                        pass

                    break

    df_out = pd.DataFrame({'gait_bout_num': np.arange(1, len(starts) + 1), "start_time": starts,
                           "end_time": stops, "step_count": n_steps_list})
    df_out['duration'] = [(j - i).total_seconds() for i, j in zip(starts, stops)]

    df_out['cadence'] = 60 * df_out['step_count'] / df_out['duration']

    print(f"-Found {df_out.shape[0]} bouts.")

    return df_out.reset_index(drop=True)


def categorize_bout_length(dict_bouts, column, thresholds):
    """ Categorizes bouts using specific column and thresholds as short/medium/long."""

    # sorts in ascending order
    thresholds = sorted(thresholds)

    if 0 not in thresholds:
        thresholds.insert(0, 0)

    # adds infinity to end of list
    thresholds.append(np.inf)

    print(f"\nCategorizing gait bouts by {column} using thresholds of {thresholds}")

    # loops through each df in dict_bouts
    for subj in dict_bouts.keys():
        # maps label to values in dict_bouts['column'] that fall into bins
        dict_bouts[subj][f'{column}_category'] = pd.cut(x=dict_bouts[subj][column],
                                                        bins=thresholds,
                                                        labels=['short', 'medium', 'long'])


def tally_percentages(df, column):

    categories = list(df[list(df.keys())[0]][column].unique().categories)
    categories.insert(0, 'subject')

    df_out = pd.DataFrame(columns=categories)

    for subj in df.keys():
        d = pd.DataFrame(df[subj][column].value_counts(normalize=100))
        d *= 100
        d.loc['subject'] = [subj]
        df_out = pd.concat([df_out, d.transpose()])

    df_out.reset_index(drop=True, inplace=True)

    return df_out


if __name__ == '__main__':
    """ ================ RUNNING CODE ================"""

    # changes working directory
    os.chdir("W:/NiMBaLWEAR/STEPS/analytics/gait/")

    # IDs to use
    use_ids = ['STEPS_1611', 'STEPS_2938', 'STEPS_6707', 'STEPS_7914']

    #dict_steps = import_steps_files(subjs=use_ids)
    #dict_bouts = import_bout_files(subjs=use_ids)

    dict_steps = import_newsteps_files(folder="O:/OBI/Personal Folders/Namiko Huynh/KyleStepDetection/1hzlowpass_abs/",
                                       subjs=use_ids, file_suffix="all_steps")

    dict_bouts = {}
    for subj in dict_steps.keys():
        dict_bouts[subj] = find_bouts(step_timestamps=dict_steps[subj]['step_time'], min_steps=3, max_break=2)

    dict_daily = import_dailysteps_files(subjs=use_ids)

    categorize_bout_length(dict_bouts=dict_bouts, column='duration', thresholds=[25, 180])
    categorize_bout_length(dict_bouts=dict_bouts, column='step_count', thresholds=[15, 50])

    step_cat = tally_percentages(df=dict_bouts, column='step_count_category')
    bout_cat = tally_percentages(df=dict_bouts, column='duration_category')

    step_totals = pd.DataFrame({'subject': dict_daily.keys(),
                                'total_steps': [dict_daily[subj]['total_steps'].sum() for subj in dict_daily.keys()],
                                'daily_avg': [dict_daily[subj]['total_steps'].sum()/dict_daily[subj].shape[0] for subj in dict_daily.keys()]})

    # % of total steps accumulated in each bout category
    dict_bouts['STEPS_1611'].groupby("duration_category")['step_count'].sum()/dict_bouts['STEPS_1611']['step_count'].sum()*100

    df_6m_steps = calculate_6m_walk_stepcount(dict_steps=dict_steps)
