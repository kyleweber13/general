import Filtering
import GaitActivityPaper.Organized.WristProcessing as WristProcessing
import GaitActivityPaper.Organized.StepsProcessing as StepsProcessing
import GaitActivityPaper.Organized.BoutProcessing as BoutProcessing
import GaitActivityPaper.Organized.Plotting as Plotting
import GaitActivityPaper.Organized.DataImport as DataImport
import GaitActivityPaper.Organized.IntensityProcessing as IntensityProcessing
import GaitActivityPaper.Organized.Other as Other
import GaitActivityPaper.Organized.FreeLiving as FreeLiving
import GaitActivityPaper.Organized.Multisubject_Processing as MSP
import GaitActivityPaper.Organized.RunLoop as RunLoop
import GaitActivityPaper.Organized.Stats as Stats
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from DataReview.Analysis import freq_analysis
plt.rcParams.update({'font.serif': 'Times New Roman', 'font.family': 'serif'})
import os
import numpy as np
import seaborn as sns


def screen_subjs(df_crit, crits=(), min_bouts=5, min_bouts_criter="n_180sec", min_age=0, max_age=1000):

    print(f"\nScreening for {crits} and more than {min_bouts} bouts of {min_bouts_criter} "
          f"with an age range of {min_age} - {max_age} years...")

    d = df_crit.copy()

    try:
        for key in crits.keys():
            d = d.loc[d[key].isin(crits[key])]

        if min_bouts > 0:
            d = d.loc[d[min_bouts_criter] >= min_bouts]

    except KeyError:
        d = None
        print("Invalid key. Options are:")
        print(list(df_crit.columns))

    d = d.loc[(d['age'] >= min_age) & (d['age'] <= max_age)]

    print("{}/{} subjects meet criteri{}.".format(d.shape[0], df_crit.shape[0], "on" if len(crits) == 1 else "a"))

    return df_crit, d


cutpoints = (62.5, 92.5)
root_dir = "O:/OBI/ONDRI@Home/Papers/Kyle and Beth - Wrist Activity in NDD/CSEP_Abstract/Data/"

# df['cohort_id'] = Other.create_cohort_ids(df_demos=df)
df_demos, subjs = screen_subjs(df_crit=DataImport.import_demos(f"{root_dir}SummaryData/totals_all.xlsx"),
                               crits={'device_side': ["D", 'Both']}, min_bouts=0, min_age=65)

df_demos['gender'].replace({"Female": 'female', 'Male': 'male'}, inplace=True)
# df_demos['hand'].replace({"Right": 'right', 'Left': 'left', 'Ambidextrous': 'ambidextrous'}, inplace=True)

# df_demos = DataImport.format_df_demos(df_demos=df_demos, root_dir=root_dir, file_dict={"OND09": 'OND09_{}_01_AXV6_{}Wrist'}, edf_dict={"OND09": "W:/Beth-CSEP/nimbalwear/OND09/wearables/device_edf_cropped/"})

# ===============================
"""
failed_files, data = RunLoop.run_loop(df_demos=df_demos, full_ids=sorted(df_demos['full_id'].unique())[1:],
                                      cutpoints=cutpoints,
                                      save_files=True, root_dir=root_dir, correct_ond09_cadence=True,
                                      min_cadence=80, min_bout_dur=60, mrp=5,
                                      min_step_time=60/200, remove_edge_low_cadence=True,
                                      file_dict={"OND09": {"daily_steps": 'W:/Beth-CSEP/nimbalwear/OND09/analytics/gait/daily/{}_01_GAIT_DAILY.csv',
                                                           "sptw": 'W:/Beth-CSEP/nimbalwear/OND09/analytics/sleep/sptw/{}_01_SPTW.csv',
                                                           'nw': 'W:/Beth-CSEP/nimbalwear/OND09/analytics/nonwear/bouts_cropped/{}_01_AXV6_{}Wrist_NONWEAR.csv'}})
"""

stat_names = ["count", "mean", '25%', "50%", '75%', "min", "max", "std"]

# demographics descriptive stats
df_demos_desc = Stats.descriptive_stats(df=df_demos, column_names=['age'], stat_names=stat_names, groupby=None)

# summary of participant totals
# df_walktotals_all = MSP.combine_walktotals(root_dir=root_dir, sort_col='')

# summary of all walking epochs
# df_walk_epochs_all = MSP.combine_walkepochs(root_dir=root_dir, df_walktotals=df_walktotals_all)

# summary of free-living activity/stepping
# df_daily_all = MSP.combine_freeliving(df_walktotals=df_walktotals_all, root_dir=root_dir)

# summary of all bouts
# df_procbouts_all = Other.combine_dataframes(folder=f'{root_dir}ProcessedBouts/', keyword="WalkingBouts.csv")
# df_procbouts_all['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walktotals_all, df_new=df_procbouts_all)

# won't work until all are processed
# df_demos['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walk_epochs_all, df_new=df_demos)

# df_cp_totals = IntensityProcessing.compare_cutpoint_totals(df=df_walk_epochs_all)
# df_cp_totals['age'] = [df_demos.loc[df_demos['full_id'] == row.full_id]['age'].iloc[0] for row in df_cp_totals.itertuples()]

# =============================================== DESCRIPTIVE STATISTICS =============================================

# epoch descriptive statistics, by participant
# df_walk_desc = Stats.descriptive_stats(df=df_walk_epochs_all, column_names=['avm', 'cadence'], stat_names=stat_names, groupby='full_id')
# df_walk_desc['avm']['cov'] = df_walk_desc['avm']['std'] * 100 / df_walk_desc['avm']['mean']
# df_walktotals_all['avm_cov'] = [df_walk_desc['avm'].loc[row.full_id]['cov'] for row in df_walktotals_all.itertuples()]

# daily free-living descriptive stats, by participant, valid days only
# df_daily_desc = Stats.descriptive_stats(df=df_daily_all.loc[df_daily_all['valid_day']], stat_names=stat_names, groupby='full_id', column_names=['valid_hours', 'steps', 'mod_norm', 'light_norm', 'sed_norm', 'mod_epochs', 'light_epochs', 'sed_epochs'])

# walking bout descriptive statistics, by participant
# df_procbouts_desc = Stats.descriptive_stats(df=df_procbouts_all, column_names=['number_steps', 'duration', 'cadence'], stat_names=stat_names, groupby='cohort_id')

# descriptive statistics for participant totals (whole sample)
# df_walktotals_desc = Stats.descriptive_stats(df=df_walktotals_all, stat_names=stat_names, groupby=None, column_names=['n_walks', 'n_epochs', 'med_cadence', 'sd_cadence', 'sed%', 'light%', 'mod%', 'long_walktime', 'fl_walktime', 'perc_fl_walktime'])

#  =============================================== STATISTICAL ANALYSIS ==============================================

# fisher_p_group, df_fisher = Stats.run_fisher_exact(df_all_epochs=df_walk_epochs_all, group_active=True, alpha=.05, bonferroni=True)
# df_ttest = Stats.run_cutpoint_ttest(df=df_demos)

# df_cp_totals = df_cp_totals.sort_values("fraysse_sedp", ascending=False).reset_index(drop=True)
# df_cp_totals['cohort_id'] = Other.copy_cohort_ids(df_copy=df_walktotals_all, df_new=df_cp_totals)

# ============================================= Optional Function Calls ==============================================

# cutpoint comparison plots ----------
# cp_fig = plot_comparison_barplot(df_cp_totals, figsize=(13, 8), binary_mvpa=False, binary_activity=False, greyscale=False, greyscale_diff=True, label_fontsize=12, fontsize=10, legend_fontsize=10)
# fig = plot_cp_diff_density(df_cp_totals)
# fig = cp_diff_hist(df_cp_totals, incl_density=False)
# plt.savefig("C:/Users/ksweber/Desktop/scatter_diff.png", dpi=150)

"""
df_demos['ndd_binary'] = ['No NDD' if row.ndd_binary == 'No' else 'NDD' for row in df_demos.itertuples()]
xs = ['med_cadence']
ys = ['fraysse_sedp', 'mean_avm']
groupby = 'ndd_binary'
colors = ['black', 'white', 'dodgerblue', 'limegreen', 'grey', 'purple', 'pink']
wide = False
sharex = 'col'

for x in xs: 

    fig, ax = plt.subplots(1 if wide else len(ys), len(ys) if wide else 1, sharey='row', sharex=sharex, figsize=(10, 10.5))

    for i, y in enumerate(ys):
        if len(fig.axes) == 1:
            use_ax = ax
        if len(fig.axes) > 1:
            use_ax = ax[i]

        Plotting.plot_scatter(df=df_demos, groupby=groupby, x='med_cadence', y=y, ax=use_ax,
                              incl_ci=True, incl_reg_line=True, group_level_reg=True,
                              incl_corr_val='spearman',
                              colors=colors)
        # use_ax.set_xlabel("Median cadence (steps per minute)")

    ax[0].set_xlabel("")
    ax[0].set_ylabel("Sedentary classification\n(% of all 15-sec epochs during gait)")
    ax[0].set_ylim(-1, 101)
    ax[1].set_ylabel("Wrist AVM")
    ax[1].set_ylim(0, 175)
    ax[0].legend(loc='lower right')
    ax[-1].set_xlabel("Median cadence (steps per minute)")
    plt.suptitle("")

    ax[1].axhspan(xmin=0, xmax=1, ymin=92.5, ymax=175, color='orange', alpha=.15, label='Fraysse MVPA', zorder=0)
    ax[1].axhspan(xmin=0, xmax=1, ymin=62.5, ymax=92.5, color='limegreen', alpha=.15, label='Fraysse light', zorder=0)
    ax[1].axhspan(xmin=0, xmax=1, ymin=0, ymax=62.0, color='grey', alpha=.15, label='Fraysse sedentary', zorder=0)
    ax[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f"C:/Users/ksweber/Desktop/CSEP_plots/combined_by_{groupby}_short.tiff", dpi=125)
    plt.close("all")
"""

# TODO
# data segment: PD with fastest cadence --> look for vig cadence, sed/light avm, and HR response
    # hunt data on review script

barfig2 = Plotting.plot_comparison_barplot(df_cp_totals=df_demos.sort_values("fraysse_sedp"), incl_legend=True,
                                           figsize=(13, 8), incl_mean=False, incl_median=True, label_fontsize=15,
                                           fontsize=12, legend_fontsize=12, sharex=False)

# barfig2.savefig("C:/Users/ksweber/Desktop/CSEP_plots/barplot.tiff", dpi=200)

