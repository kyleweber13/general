import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr, pearsonr
import matplotlib.pyplot as plt


""" ==================================== FFA DATASET ==================================== """


def format_column_values():
    for column in ['gait_fof_fall', 'gait_fof_bal', 'gait_abc_1',
                   'gait_abc_2', 'gait_abc_3', 'gait_abc_4', 'gait_abc_5', 'gait_abc_6',
                   'gait_abc_7', 'gait_abc_8', 'gait_abc_9', 'gait_abc_10', 'gait_abc_11',
                   'gait_abc_12', 'gait_abc_13', 'gait_abc_14', 'gait_abc_15',
                   'gait_abc_16', 'gait_assist_dvc_room', 'gait_assist_dvc_halls_s',
                   'gait_assist_dvc_halls_l', 'gait_assist_dvc_out']:

        vals = []
        for i, val in enumerate(df[column]):
            try:
                v = int(str(val).split("(")[1][:-1]) if not pd.isnull(val) else None
            except ValueError:
                v = int(str(val).split("(")[2][:-1]) if not pd.isnull(val) else None

            vals.append(v)

        df[column] = vals


def archive():
    df = pd.read_csv("O:/Student_Projects/Lillia_FOF/FoF Data.csv", skiprows=1)
    df.columns = [i.strip("()") for i in df.columns]

    format_column_values()

    # site ID from subject_id
    df.insert(loc=1, value=[row.subject_id.split("_")[1] for row in df.itertuples()], column='site_code')

    # split redcap_event_name components
    df.insert(loc=1, column='visit_num', value=[row.redcap_event_name.split(" (")[0].split("Visit ")[1] for row in df.itertuples()])
    df.insert(loc=2, column='visit_desc', value=[row.redcap_event_name.split(" (")[1].split(")")[0] for row in df.itertuples()])
    df.insert(loc=3, column='arm', value=[row.redcap_event_name.split("Arm ")[1].split(":")[0] for row in df.itertuples()])
    df.insert(loc=4, column='cohort', value=[row.redcap_event_name.split(": ")[1].split(")")[0] for row in df.itertuples()])

    # removes PD and ALS participants
    df = df.loc[~df['cohort'].isin(['PD', 'ALS'])]

    # add bmi
    df.insert(loc=int(np.argwhere(df.columns == 'gait_wght')[0][0] + 1), column='bmi', value=df['gait_wght'] / ((df['gait_hght']/100)**2))

    # sum ABC scores ------
    abc_columns = ['gait_abc_1', 'gait_abc_2', 'gait_abc_3', 'gait_abc_4', 'gait_abc_5', 'gait_abc_6',
                   'gait_abc_7', 'gait_abc_8', 'gait_abc_9', 'gait_abc_10', 'gait_abc_11',
                   'gait_abc_12', 'gait_abc_13', 'gait_abc_14', 'gait_abc_15', 'gait_abc_16']

    # find participants with
    df['abc_sum'] = df[abc_columns].transpose().sum()
    df['abc_hasallscores'] = 16 - df[abc_columns].transpose().isna().sum() == 16

    # remove participants without all abc scores
    # df = df.loc[df['abc_hasallscores']]


def add_hamcop_file(df):

    fname = "O:/Data/FFA/Analyzed_data/Balance_HAMCOP/FFA_all_hamcop_with Demographics.xlsx"
    df_hamcop = pd.read_excel(fname)

    df_hamcop = df_hamcop[["Subj", 'Visit', 'Stance', 'Vision', 'RMS ML(mm)', 'Sway area(mm2/s)']]
    df_hamcop.columns = ['Subject_ID', 'Visit', 'Stance', 'Vision', 'MLRMS', 'Swayarea']
    df_hamcop['Visit'] = df_hamcop['Visit'].astype(str)

    df_hamcop = df_hamcop.loc[df_hamcop['Vision'] == 'ec']
    df_hamcop['Condition'] = [f"{row.Stance.capitalize()}{row.Vision.upper()}" for row in df_hamcop.itertuples()]

    df_hamcop.drop("Stance", axis=1, inplace=True)
    df_hamcop.drop("Vision", axis=1, inplace=True)

    df_hamcop['Visit'] = [row.Visit.split("V0")[1] for row in df_hamcop.itertuples()]
    df_hamcop.insert(loc=0, column='Visit_ID',
                     value=[f"{row.Subject_ID}_{row.Visit}" for row in df_hamcop.itertuples()])

    # wide format
    d = pd.pivot(df_hamcop, index='Visit_ID', values=['MLRMS', 'Swayarea'], columns='Condition')
    d.columns = [f"{i[0]}_{i[1]}" for i in d.columns]
    d.reset_index(inplace=True)

    df_out = pd.merge(df, d, on='Visit_ID', how='outer')  # 'outer'

    return df_out


def merge_ffa_data():

    sheetnames = {"Visits": ['Visit_ID'],
                  'Demographics': ['Visit_ID', 'Visit', "Height(cm)", 'Weight(kg)', 'Age'],
                  'Diagnoses_Medications': ['Visit_ID', 'Visit', "Number_Diagnoses", 'Meds_on_record'],
                  '25ftWalk (2)': ['Visit_ID', 'Visit', "25ft_pref(sec)"],
                  'Sit_to_Stand': ['Visit_ID', 'Visit', "5xSTS_time"],
                  'Function_Strength': ['Visit_ID', 'Visit', "R_grip", 'L_grip'],
                  "FearRatings": ['Visit_ID', 'Visit', "Fear_of_Falling", "Balance_Confidence"]}

    dfs = {}  # empty dictionary to store all dataframes (sheets)

    # loops through sheet names -------------
    for sheet in sheetnames:

        # reads in each sheet and makes new key in df with sheet name
        dfs[sheet] = pd.read_excel("O:/FFA/Database/Master_exported_as_excel/Exported_access_Oct2021.xlsx",
                                   sheet_name=sheet)

        sheet_columns = sheetnames[sheet]  # gets wanted column names for given sheet (values in sheetnames dictionary)
        dfs[sheet] = dfs[sheet][sheet_columns]  # uses only columns we want

        if 'Visit' in dfs[sheet].columns:
            dfs[sheet]['Visit'] = dfs[sheet]['Visit'].astype(str)

        if 'Visit' in dfs[sheet].columns and sheet != 'Demographics':
            dfs[sheet] = dfs[sheet].drop("Visit", axis=1)

    # merges all dataframes in dfs using 'Visit_ID' column

    df = dfs[list(dfs.keys())[1]].copy()
    for key in list(dfs.keys())[2:]:
        df = pd.merge(df, dfs[key], on='Visit_ID')

    # resets index
    # index was only used to merge dataframes
    df.reset_index(drop=True, inplace=True)

    # pulls out subject ID from index
    df.insert(loc=1, column='Subject_ID', value=[i.split("_")[0] for i in df['Visit_ID']])

    # calculates 25ft walk velo in m/s and inserts into column after '25ft_pref(sec)'
    df.insert(loc=int(np.argwhere(df.columns == '25ft_pref(sec)')[0][0]+1),  # finds 25ft_pref(sec) column index + 1
              column='25ft_Vel_pref(m/s)',  # column name
              value=25*0.3048/df['25ft_pref(sec)'])  # converts 25ft to m and calculates speed

    # determines max grip strength and inserts into column after 'L_grip'
    df.insert(loc=int(np.argwhere(df.columns == 'L_grip')[0][0]+1),  # finds L_grip column index + 1
              column='Max_grip',  # column name
              value=[max([row.R_grip, row.L_grip]) for row in df.itertuples()])  # max grip strengths

    # adds additional file
    df = add_hamcop_file(df)

    # new column for whether participant has all values (False) or has missing value(s) (True)
    # transposes ('rotates') dataframe. Checks if each row has a missing value. Sums the result, checks if sum is 0
    #   if sum != 0, has missing data
    df['has_missing_data'] = pd.isnull(df.transpose()).sum() != 0

    df.insert(loc=int(np.argwhere(df.columns == 'Fear_of_Falling')[0] + 1), column='Fear_of_Falling3',
              value=pd.cut(df['Fear_of_Falling'], bins=[-1, 1, 3, 7, 10], labels=[0, 1, 2, 3]))
    df.insert(loc=int(np.argwhere(df.columns == 'Balance_Confidence')[0] + 1), column='Balance_Confidence3',
              value=pd.cut(df['Balance_Confidence'], bins=[-1, 1, 3, 7, 10], labels=[0, 1, 2, 3]))

    df = df.sort_values("Visit_ID")
    df.reset_index(drop=True, inplace=True)

    return df


def calculate_change_score(df, column):
    """ Creates df for given column's data in wide format using subject_id and Visit as grouping variables."""

    # grabs 'subject_id', 'Visit' and specified column from df
    # subject_id is the identifier/groupby column; 'Visit' used to generate columns with 'column' values
    d = df[['Subject_ID', 'Visit', column]]

    d = d.dropna()

    # long to wide format. 'Visit' used to generate columns
    d = pd.pivot(d, index='Subject_ID', columns='Visit')
    d.columns = [str(int(i[1])) for i in d.columns]
    d.insert(loc=0, column='n', value=d.shape[1] - pd.isnull(d.transpose()).sum())

    return d


def generate_change_score_files(df, keys, write_files=False):

    # dictionary of more usable sheet names
    colname_alias = {"Height(cm)": "height", 'Weight(kg)': 'weight', 'Age': 'age', '25ft_pref(sec)': '25ft_pref_sec',
                     '25ft_Vel_pref(m/s)': '25ft_Vel_pref'}

    dfs_wide = {}

    # columns in df to ignore
    # ignore_cols = ['Visit_ID', 'Subject_ID', 'Visit', 'Number_Diagnoses', 'Meds_on_record', 'has_missing_data']
    # use_cols = [col for col in df.columns if col not in ignore_cols]

    for i in keys:
        print(f"Formatting df for {i}")

        dfs_wide[i] = calculate_change_score(df=df, column=i)

        if i in colname_alias:
            use_colname = colname_alias[i]
        else:
            use_colname = i

        if write_files:
            dfs_wide[i].to_csv(f"O:/Student_Projects/Lillia_FOF/change_scores/{use_colname}.csv")

    return dfs_wide


def create_visit_change_score_files(df_wide, cols, save_files=False):

    for key in cols:
        df = df_wide[key]

        df_v12 = df.loc[~(df['1']).isnull() & ~(df['2'].isnull())]
        df_v13 = df.loc[~(df['1']).isnull() & ~(df['3'].isnull())]
        df_v123 = df.loc[~(df['1']).isnull() & ~(df['2']).isnull() & ~(df['3'].isnull())]

        if key == '25ft_Vel_pref(m/s)':
            key = '25ft_Vel_pref(ms)'

        if save_files:
            df_v12.to_csv(f"O:/Student_Projects/Lillia_FOF/Visits1_2/{key}.csv")
            df_v13.to_csv(f"O:/Student_Projects/Lillia_FOF/Visits1_3/{key}.csv")
            df_v13.to_csv(f"O:/Student_Projects/Lillia_FOF/Visits123/{key}.csv")

        print(f"{key}: Visits(1-2) {df_v12.shape[0]}, visits(1-3) {df_v13.shape[0]}, visits(1, 2, 3) {df_v123.shape[0]}")


def compare_difference_score(df, t1, t2, binsize=1.0, show_plot=True):

    df = df.copy()
    df = df[[t1, t2]]
    df = df.loc[~df.isna().any(axis=1)]

    # df[t1] = [int(i) for i in df[t1]]
    # df[t2] = [int(i) for i in df[t2]]
    df['diff'] = df[t1] - df[t2]

    if show_plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        x = np.arange(min(df[t1]), max(df[t1]) + binsize, binsize)

        ax[0].hist(df['diff'], color='grey', edgecolor='black', align='left',
                   bins=np.arange(min(df['diff']), max(df['diff']) + binsize, binsize))
        ax[0].axvline(x=0, color='red')

        ax[1].scatter(df[t2], df['diff'], color='black')

        r = np.polyfit(df[t2], df['diff'], deg=1)
        pr = pearsonr(df[t1], df['diff'])

        ax[1].plot(x, [i*r[0] + r[1] for i in x],
                   color='red', label=f"pearson={pr[0]:.3f}")
        ax[1].set_xlabel(f"Time {t1}")
        ax[1].set_ylabel(f"Difference (T{t1} - T{t2})")
        ax[1].axhline(y=0, linestyle='dashed')
        ax[1].legend()
        plt.tight_layout()

    return df


def compare_difference_scores(df_wide, var1, var2, t1, t2):

    df1 = compare_difference_score(df=df_wide[var1], t1=t1, t2=t2, show_plot=False)
    df2 = compare_difference_score(df=df_wide[var2], t1=t1, t2=t2, show_plot=False)

    df_out = pd.DataFrame(df1['diff'])

    df_out = df_out.merge(df2['diff'], how='outer', left_index=True, right_index=True)
    df_out.reset_index(drop=False, inplace=True)

    df_out.columns = ['Subject_ID', f'{var1}_diff', f'{var2}_diff']

    df_nan = df_out[df_out.isna().any(axis=1)]
    df_out = df_out.dropna()

    return {'use': df_out, 'not_use': df_nan}


def create_v1and2_master():
    df_new = dfs_ffa_wide['25ft_Vel_pref(m/s)'][['1', '2']].copy()
    df_new.columns = [f"25ft_Vel_pref(m/s)_{i}" for i in df_new.columns]
    for key in ['5xSTS_time', 'Max_grip', 'Fear_of_Falling', 'Balance_Confidence', 'MLRMS_StandardEC']:
        df_merge = dfs_ffa_wide[key][['1', '2']]
        df_merge.columns = [f"{key}_visit{i}" for i in df_merge.columns]
        df_new = pd.merge(left=df_new, right=df_merge, left_index=True, right_index=True,
                          how='inner', suffixes=[None, key])

    df_new.dropna(inplace=True)

    # df_new.to_excel("O:/Student_Projects/Lillia_FOF/FFA_V1and2_master.xlsx")
    return df_new


"""==================================== ONDRI DATA ===================================="""


def ondri_correlation_analysis(df):
    cols = ['gait_fof_bal', 'gait_fof_fall', 'gait_fof_bal3', 'gait_fof_fall3']

    values = []
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, column in enumerate(cols):
        ax_i = [int(np.floor(i/2)), i % 2]

        shapiro_result = check_normality(df=df, col_name=column)

        use_df = df.loc[~df[column].isnull()]

        spearman = spearmanr(use_df[column], use_df['abc_sum'])

        ax[ax_i[0]][ax_i[1]].scatter(use_df[column], use_df['abc_sum'], alpha=.25)
        ax[ax_i[0]][ax_i[1]].set_xlabel(column)
        ax[ax_i[0]][ax_i[1]].set_ylabel('abc_sum')

        values.append([column, spearman[0], spearman[1], shapiro_result[1]])

    plt.tight_layout()

    df_stats = pd.DataFrame(values, columns=['variable', 'spearman_r', 'spearman_p', 'shapiro_p'])
    # df_stats.to_csv("O:/Student_Projects/Lillia_FOF/FoF_correlation_stats.csv", index=False)

    return df_stats


"""==================================== STATS ===================================="""


def outlier_zscore_check(df, flag_thresh=3):

    df_out = df[['Visit_ID', 'Subject_ID', 'Visit']].copy()
    df_out_outlier = df[['Visit_ID', 'Subject_ID', 'Visit']].copy()

    for column in [i for i in df.columns if i not in df_out.columns]:

        vals = df[column]
        mean = vals.mean()
        sd = vals.std()

        df_out.loc[:, column] = [(val - mean) / sd for val in vals]
        df_out_outlier.loc[:, column] = [abs(i) >= flag_thresh if not np.isnan(i) else np.nan for i in df_out[column]]

        print(f"{column} contains {list(df_out_outlier[column]).count(True)}/{df_out_outlier.shape[0]} outlier(s)")

    return df_out, df_out_outlier


def check_normality(df, col_name, show_plot=False):
    if show_plot:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    try:
        if show_plot:
            df[col_name].plot.density(ax=ax[0])
    except TypeError:
        pass

    try:
        if show_plot:
            ax[1].boxplot(df[col_name].dropna())
    except TypeError:
        pass

    try:
        n = shapiro(df[col_name].dropna())
        if show_plot:
            plt.title(f"Shapiro: W={n[0]:.5f}, p={n[1]:.5f}")
    except TypeError:
        pass

    return n


def generate_scatterplot(df, x, y):

    df_use = df[[x, y]].dropna(how='any')
    fig, ax = plt.subplots(1, figsize=(8, 8))

    ax.scatter(df[x], df[y], color='black')
    ax.set_ylabel(y)
    ax.set_xlabel(x)

    pearson = pearsonr(df_use[x], df_use[y])
    reg = np.polyfit(df_use[x], df_use[y], deg=1)
    spearman = spearmanr(df_use[x], df_use[y])

    X = np.arange(df[x].min(), df[x].max()+1)
    Y = [i*reg[0] + reg[1] for i in X]

    ax.plot(X, Y, color='red', label=f"{y} = {reg[0]:.5f} x {x} + {reg[1]:.5f}")

    ax.set_title(f"{x} vs. {y}\nr = {pearson[0]:.3f} (p={pearson[1]:.3f}), rho={spearman[0]:.3f} (p={spearman[1]:.3f})")

    ax.legend()
    plt.tight_layout()


# ONDRI data =====================================================================

df_ondri = pd.read_csv("O:/Student_Projects/Lillia_FOF/FoFData_Cleaned_AllABCScores.csv", skiprows=0)
# conversion from 10- to 3-point scale
# df_ondri.insert(loc=int(np.argwhere(df.columns == 'gait_fof_fall')[0]+1), column='gait_fof_fall3', value=pd.cut(df_ondri['gait_fof_fall'], bins=[-1, 1, 3, 7, 10], labels=[0, 1, 2, 3]))
# df_ondri.insert(loc=int(np.argwhere(df.columns == 'gait_fof_bal')[0]+1), column='gait_fof_bal3', value=pd.cut(df_ondri['gait_fof_bal'], bins=[-1, 1, 3, 7, 10], labels=[0, 1, 2, 3]))
# df_ondri.to_csv("O:/Student_Projects/Lillia_FOF/FoFData_Cleaned_AllABCScores.csv", index=False)

# FFA data ===========================

cols_ffa = ['25ft_Vel_pref(m/s)', '5xSTS_time', 'Max_grip', 'Fear_of_Falling', 'Fear_of_Falling3',
            'Balance_Confidence', 'Balance_Confidence3', 'MLRMS_StandardEC']

# data import ------
# df_ffa = merge_ffa_data()
df_ffa = pd.read_csv("O:/Student_Projects/Lillia_FOF/FFA_data_combined.csv")
# df_ffa.to_csv("O:/Student_Projects/Lillia_FOF/FFA_data_combined.csv", index=False)

# outlier check using z-scores ------
# flag thresh: absolute value of z-score to count as outlier
# df_z, df_z_flag = outlier_zscore_check(df_ffa, flag_thresh=5)

# change scores -----
# wide format: columns = visits
dfs_ffa_wide = generate_change_score_files(df=df_ffa, keys=cols_ffa, write_files=False)

# creates files for participants with V1/2, V1/3, and V1/2/3 data
create_visit_change_score_files(df_wide=dfs_ffa_wide, cols=cols_ffa, save_files=False)

""" =============================================== INDIVIDUAL FUNCTIONS ========================================== """

# FFA data --------------------------------------

# histogram of difference score at specified timepoint for specified outcome measure
# timepoints need to be a string of an integer between 1 and 5
# index dfs_ffa_wide using a key from ffa_cols
df_diff = compare_difference_score(df=dfs_ffa_wide['Fear_of_Falling'], t1='1', t2='3')

# dataframe of difference score between two variables

df_comp = compare_difference_scores(df_wide=dfs_ffa_wide, var1='Fear_of_Falling', var2='25ft_Vel_pref(m/s)', t1='1', t2='3')['use']

# ONDRI data --------------------------------------
ondri_stats = ondri_correlation_analysis(df_ondri)

# Generic -----------------------------------------

# scatterplot. specify dataframe and two columns
# generate_scatterplot(df_ffa, x='Fear_of_Falling', y='25ft_Vel_pref(m/s)')

df_demos = pd.read_excel("O:/Data/OND01/Analyzed_data/Balance_HAMCOP/OND01_all_hamcop_with Demographics.xlsx",
                         sheet_name="Age and Gender Information")
df_demos = df_demos[['Subject', 'Visit', 'Date', 'Gender', "Age"]]
df_demos['full_id'] = [f"{row.Subject}_{row.Visit}" for row in df_demos.itertuples()]

df_data = pd.read_csv("O:/Student_Projects/Lillia_FOF/V4_V2_diff.csv")
# df_data = df_data[['subject_id', 'visit_num_x']]
df_data['full_id'] = [f"{row.subject_id}_{row.visit_num_x}" for row in df_data.itertuples()]

df_combined = pd.merge(df_data, df_demos, left_on='full_id', right_on='full_id', how='left')
df_combined = df_combined[['subject_id', 'visit_num_x', 'Date', 'Gender', 'Age']]
df_combined.rename(columns={"visit_num_x": 'visit_num'}, inplace=True)

df_combined.to_csv("O:/Student_Projects/Lillia_FOF/V4_V2_diff_demos.csv", index=False)