import matplotlib.pyplot as plt
import pandas as pd
import os
from ECG_Organized.ArrhythmiaScreening import flag_high_enough_snr, apply_arrhythmia_criteria
from ECG_Organized.ImportECG import ECG
from Run_FFT import *
import plotly.graph_objects as go
import nwdata

focus_arrs = ("Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block')


def combine_all_files(folder, files):

    df_all = pd.DataFrame(columns=['start_idx', 'end_idx', 'start_timestamp', 'end_timestamp', 'Type',
                                   'duration', 'abs_volt',
                                   # 'p0', 'p25', 'p50', 'p75', 'p100', 'avg_snr',
                                   'gait%', 'activity%', 'nw%'])

    for i, file in enumerate(files):
        print(f"File {i+1}/{len(files)}")

        if "xlsx" in file:
            df = pd.read_excel(folder + file)
        if 'csv' in file:
            df = pd.read_csv(folder + file)

        df['full_id'] = [file[:10]] * df.shape[0]

        df_all = df_all.append(df)

    return df_all


def groupby_subj(df, col):

    d = df.groupby("full_id")[col]

    df_out = pd.DataFrame({'full_id': df['full_id'].unique(),
                           'n_events': [d.get_group(subj).shape[0] for subj in df['full_id'].unique()]})

    print("{} || {} +- {} events, min = {}, max = {}".format(col, round(df_out['n_events'].mean(), 1),
                                                             round(df_out['n_events'].std(), 1),
                                                             round(df_out['n_events'].min(), 1),
                                                             round(df_out['n_events'].max(), 1)))

    return df_out


def process_all(df, snr_thresh=20,
                use_arrs=("Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block')):

    df = df.copy()

    def check_for_id(ids, check_df, check_df_col):

        out = []
        for subj in ids:
            d = check_df.loc[check_df['full_id'] == subj]

            if d.shape[0] == 0:
                out.append(0)
            if d.shape[0] > 0:
                out.append(d[check_df_col].iloc[0])

        return out

    df_out = groupby_subj(df, col='Type')
    df_out.columns = ['full_id', 'CN_output']

    if use_arrs is not None:
        df = df.loc[df['Type'].isin(use_arrs)]
        tally_critsubj = groupby_subj(df, col='Type')

        df_out['critical_events'] = check_for_id(ids=df_out['full_id'].unique(),
                                                 check_df=tally_critsubj, check_df_col='n_events')

    df_wear = df.loc[df['nw%'] == 0]
    tally_wearsubj = groupby_subj(df_wear, col='Type')
    df_out['RemNW'] = check_for_id(ids=df_out['full_id'].unique(),
                                   check_df=tally_wearsubj, check_df_col='n_events')

    # remove arrest/block events with 60Hz noise
    # df60hz = df.loc[~df['valid_freq'].isin(['False'])]
    df60hz = df_wear.loc[df_wear['dom_freq_all'] < 30]
    tally_60hz = df60hz['Type'].value_counts()
    tally_60hzsubj = groupby_subj(df60hz, col='Type')

    df_out['60HzArrest'] = check_for_id(ids=df_out['full_id'].unique(),
                                        check_df=tally_60hzsubj, check_df_col='n_events')

    # criteria CN settings can't handle
    df2 = apply_arrhythmia_criteria(df_arr=df60hz, voltage_thresh=5000)
    tally_all2 = df2['Type'].value_counts()
    tally_subj2 = groupby_subj(df2, col='Type')

    df_out['add_criteria'] = check_for_id(ids=df_out['full_id'].unique(),
                                          check_df=tally_subj2, check_df_col='n_events')

    df3 = flag_high_enough_snr(df=df2, use_percentile=0, default_thresh=snr_thresh, exceptions_dict={'Arrest': -50, 'Block': -50})
    df3 = df3.loc[df3['snr_valid']]

    tally_subj3 = groupby_subj(df3, col='Type')

    df_out[f'quality_check_{snr_thresh}db'] = check_for_id(ids=df_out['full_id'].unique(),
                                                           check_df=tally_subj3, check_df_col='n_events')

    if use_arrs is not None:
        all_arrs = list(use_arrs)

    if use_arrs is None:
        # all_arrs = list(df3['Type'].unique())
        all_arrs = list(df['Type'].unique())

    subjs_data = []
    for subj in df['full_id'].unique():
        subj_data = [subj]

        d_subj = df3.loc[df3['full_id'] == subj]
        # d_subj = df.loc[df['full_id'] == subj]
        vals = d_subj['Type'].value_counts()

        for arr in all_arrs:
            if arr in vals.index:
                subj_data.append(vals.loc[arr])
            if arr not in vals.index:
                subj_data.append(0)

        subjs_data.append(subj_data)

    all_arrs.insert(0, 'full_id')
    df_subj_tally = pd.DataFrame(subjs_data, columns=all_arrs)

    df_subj_tally['Total'] = [df_subj_tally.iloc[i, 1:].sum() for i in range(df_subj_tally.shape[0])]

    return df_out.sort_values("full_id").reset_index(drop=True), df_subj_tally, df3


def calc_subj_tally_all(df_all, all_arrs=("Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block')):
    subjs_data = []
    all_arrs = list(all_arrs)

    for subj in df_all['full_id'].unique():
        subj_data = [subj]

        d_subj = df_all.loc[df_all['full_id'] == subj]
        # d_subj = df.loc[df['full_id'] == subj]
        vals = d_subj['Type'].value_counts()

        for arr in all_arrs:
            if arr in vals.index:
                subj_data.append(vals.loc[arr])
            if arr not in vals.index:
                subj_data.append(0)

        subjs_data.append(subj_data)

    all_arrs.insert(0, 'full_id')
    df_subj_tally = pd.DataFrame(subjs_data, columns=all_arrs)
    df_subj_tally = df_subj_tally.reindex(sorted(df_subj_tally.columns), axis=1)

    # df_subj_tally['Total'] = [df_subj_tally.iloc[i, 1:].sum() for i in range(df_subj_tally.shape[0])]

    return df_subj_tally


def plot_sankey():
    values = {"All Events": 0,
              "Non-critical": df_sums['CN_output'] - df_sums['critical_events'],
              "Critical":  df_sums['critical_events'],
              "Fail60Hz": df_sums['critical_events'] - df_sums['60HzArrest'],
              "Pass60Hz": df_sums['60HzArrest'],
              'FailCriteria': df_sums['60HzArrest'] - df_sums['add_criteria'],
              "PassCriteria": df_sums['add_criteria'],
              "BadQuality": df_sums['add_criteria'] - df_sums['quality_check'],
              "GoodQuality": df_sums['quality_check']}

    fig = go.Figure(data=[go.Sankey(node=dict(pad=15,
                                              thickness=20,
                                              line=dict(color="black", width=0.5),
                                              label=["All Events (n={})".format(df_out['CN_output'].sum()),
                                                     'Non-critical (n={})'.format(values['Non-critical']),
                                                     'Critical (n={})'.format(values['Critical']),
                                                     "Fail60Hz (n={})".format(values['Fail60Hz']),
                                                     "Pass60Hz (n={})".format(values['Pass60Hz']),
                                                     'Fail Criteria(n={}))'.format(values['FailCriteria']),
                                                     'Pass Criteria (n={})'.format(values['PassCriteria']),
                                                     'Bad Quality (n={})'.format(values['BadQuality']),
                                                     'Good Quality (n={})'.format(values['GoodQuality'])],
                                              color=["red", 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green']),
                                    link=dict(source=[0, 0, 0,
                                                      2, 2,
                                                      4, 4,
                                                      6, 6],
                                              target=[1, 1, 2,
                                                      4, 3,
                                                      5, 6,
                                                      7, 8],
                                              value=list(values.values())))])

    fig.show()


def plot_sankey2():
    values = {"All Events": 0,
              "Non-critical": df_sums['CN_output'] - df_sums['critical_events'],
              "Critical":  df_sums['critical_events'],
              'DeviceNW': df_sums['critical_events'] - df_sums['RemNW'],
              "DeviceWear": df_sums['RemNW'],
              "Fail60Hz": df_sums['RemNW'] - df_sums['60HzArrest'],
              "Pass60Hz": df_sums['60HzArrest'],
              'FailCriteria': df_sums['60HzArrest'] - df_sums['add_criteria'],
              "PassCriteria": df_sums['add_criteria'],
              "BadQuality": df_sums['add_criteria'] - df_sums['quality_check_20db'],
              "GoodQuality": df_sums['quality_check_20db']}

    fig = go.Figure(data=[go.Sankey(node=dict(pad=15,
                                              thickness=20,
                                              line=dict(color="black", width=0.5),
                                              label=["All Events (n={})".format(df_out['CN_output'].sum()),
                                                     'Non-critical (n={})'.format(values['Non-critical']),
                                                     'Critical (n={})'.format(values['Critical']),
                                                     'DeviceNW (n={})'.format(values['DeviceNW']),
                                                     'DeviceWear (n={})'.format(values['DeviceWear']),
                                                     "Fail60Hz (n={})".format(values['Fail60Hz']),
                                                     "Pass60Hz (n={})".format(values['Pass60Hz']),
                                                     'Fail Criteria(n={}))'.format(values['FailCriteria']),
                                                     'Pass Criteria (n={})'.format(values['PassCriteria']),
                                                     'Bad Quality (n={})'.format(values['BadQuality']),
                                                     'Good Quality (n={})'.format(values['GoodQuality'])],
                                              color=["red", 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green']),
                                    link=dict(source=[0, 0, 0,
                                                      2, 2,
                                                      4, 4,
                                                      6, 6,
                                                      8, 8],
                                              target=[1, 1, 2,
                                                      4, 3,
                                                      5, 6,
                                                      7, 8,
                                                      9, 10],
                                              value=list(values.values())))])

    fig.show()


plot_sankey2()


def plot_subj_tally(df, col, bin_size=None):

    fig, ax = plt.subplots(1, figsize=(10, 6))

    ax.hist(df[col], bins=None if bin_size is None else np.arange(0, max(df[col])*1.1, bin_size), color='grey', edgecolor='black')
    ax.set_xlim(0, )
    ax.grid()
    ax.set_ylabel("Number of participants")
    ax.set_xlabel("Number of events")
    ax.set_title(col)


def plot_dual_histogram(df, col, bin_size=2):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex='col')
    ax[0].hist(df[col], color='grey', edgecolor='black',
               bins=np.arange(0, df[col].max()+bin_size, bin_size), weights=100*np.ones(df.shape[0]) / df.shape[0])
    ax[1].hist(df[col], color='grey', edgecolor='black',
               bins=np.arange(0, df[col].max()+bin_size, bin_size), weights=100*np.ones(df.shape[0]) / df.shape[0], cumulative=True)
    ax[0].grid()
    ax[0].set_ylabel("% of events")
    ax[1].set_ylabel("% of events")
    ax[1].set_yticks(np.arange(0, 101, 10))
    ax[1].grid()
    ax[0].set_title("Histogram")
    ax[1].set_title("Cumulative Histogram")
    plt.suptitle(f"Distribution of {col}")

    plt.tight_layout()

    return fig


def plot_nw_tally_bar():
    for arr in df['Type'].unique():
        d_arr = df.loc[df['Type'] == arr]
        no_nw = d_arr.loc[d_arr['nw%'] == 0]
        nw = d_arr.loc[d_arr['nw%'] > 0]

        plt.bar([arr], 100 * nw.shape[0] / d_arr.shape[0], color='red', edgecolor='black')
        plt.bar([arr], 100 * no_nw.shape[0] / d_arr.shape[0], color='grey', bottom=100 * nw.shape[0] / d_arr.shape[0],
                edgecolor='black')

    plt.ylim(0, 100)
    plt.ylabel("% of events")
    plt.grid()


folder = "C:/Users/ksweber/Desktop/Processed/Archive/VisualNW2/"
files = os.listdir(folder)
files = [i for i in files if 'all' in i]

# df = combine_all_files(folder, files)
df = pd.read_csv("C:/Users/ksweber/Desktop/CardiacNavigator/df_cn_n14.csv")

df_out, df_subj_tally, df_valid = process_all(df, use_arrs=focus_arrs)
df_subj_all = calc_subj_tally_all(df)

# df_out = pd.read_excel("C:/Users/ksweber/Desktop/tally.xlsx")
df_sums = df_out.iloc[:, 1:].sum()

# df_subj_tally = pd.read_excel("C:/Users/ksweber/Desktop/subj_tally.xlsx")

# plot_subj_tally(df_out, df_out.columns[-1], bin_size=10)
df_subj_tally.iloc[1:].boxplot()

# plot_dual_histogram(df, 'dom_freq_all', 1)


def print_independent_summary(df):

    n = df.shape[0]

    print(f"\nTotal events = {n}")

    n_crit = df.loc[df['Type'].isin(focus_arrs)].shape[0]
    n_noncrit = df.loc[~df['Type'].isin(focus_arrs)].shape[0]
    print(f"\n-Critical events = {n_crit}")
    print(f"-Non-critical events = {n_noncrit}")

    n_wear = df.loc[df['nw%'] == 0].shape[0]
    n_nw = df.loc[df['nw%'] > 0].shape[0]
    print(f"\n-During wear = {n_wear}")
    print(f"-During non-wear = {n_nw}")

    n_hq = df.loc[df['p0'] >= 20].shape[0]
    n_lq = df.loc[df['p0'] < 20].shape[0]
    print(f"\n-High quality = {n_hq}")
    print(f"-Low quality = {n_lq}")

    n_hf = df.loc[df['dom_freq_all'] >= 30].shape[0]
    n_lf = df.loc[df['dom_freq_all'] < 30].shape[0]
    print(f"\n-60hz noise = {n_hf}")
    print(f"-Minimal 60hz noise = {n_lf}")

# heatmap of df_subj_tally

df_heat = df_subj_tally.iloc[:, 1:-1]
df_heat['Block'] = [0]*df_heat.shape[0]
df_heat = df_heat.reindex(sorted(df_heat.columns), axis=1)

df_heat_all = df_subj_tally.iloc[:, 1:-1]
df_heat_all = df_heat_all[[i for i in df_heat_all.columns if i in focus_arrs]]
df_heat_all['AV2/III'] = [0]*df_heat_all.shape[0]
df_heat_all['Brady'] = [0]*df_heat_all.shape[0]
df_heat_all = df_heat_all.reindex(sorted(df_heat_all.columns), axis=1)

import seaborn as sns
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
sns.heatmap(df_heat_all, linewidth=.3, ax=ax[0], cmap='nipy_spectral', annot=True, fmt='.0f', cbar=False, vmin=0, vmax=1167)
ax[0].set_title("Raw Output")
sns.heatmap(df_heat, linewidth=.3, ax=ax[1], cmap='nipy_spectral', annot=True, fmt='.0f', vmin=0, vmax=1167)
ax[1].set_title("Processed Output")

ax[0].set_ylabel("Participants")
ax[0].set_yticklabels([f"#{i}" for i in range(1, 15)], rotation=0)

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

plt.tight_layout()