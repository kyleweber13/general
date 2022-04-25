from datetime import timedelta
import pandas as pd
from Plotting import plot_epoched_context


def calculate_collection_startstop(obj1, obj2):

    print("\nChecking file start/stop times for data cropping...")
    start = max([obj1.header['startdate'], obj2.header['startdate']])

    end = min([obj1.header['startdate'] + timedelta(seconds=len(obj1.signals[obj1.get_signal_index("Accelerometer x")]) /
                                                            obj1.signal_headers[obj1.get_signal_index("Accelerometer x")]['sample_rate']),
              obj2.header['startdate'] + timedelta(seconds=len(obj2.signals[obj2.get_signal_index("Accelerometer x")]) /
                                                           obj2.signal_headers[obj2.get_signal_index("Accelerometer x")]['sample_rate'])])

    print(f"Last start = {start} || first end = {end}")

    return start, end


def calculate_activity_totals(df_epoch, dom_cutpoints=(62.5, 92.5), nondom_cutpoints=(42.5, 98),
                              exclude_sleep=True, exclude_gait=False, exclude_nw=True):

    epoch_len = (df_epoch.iloc[1]['timestamp'] - df_epoch.iloc[0]['timestamp']).total_seconds()

    # recalculate dominant wrist intensity --------
    print(f"\nCalculating activity volume for dominant wrist using cutpoints of {dom_cutpoints}...")

    # Removal of specific contexts ----
    df_epoch = df_epoch.copy()

    if exclude_sleep:
        print("-Excluding sleep epochs")
        df_epoch = df_epoch.loc[~df_epoch['sleep']]

    if exclude_gait:
        print("-Excluding gait epochs...")
        df_epoch = df_epoch.loc[~df_epoch['gait']]

    if exclude_nw:
        print("-Exluding non-wear epochs...")
        df_epoch = df_epoch.loc[~df_epoch['nonwear']]

    dom_intensity = []

    for row in df_epoch.itertuples():
        if row.dominant < dom_cutpoints[0]:
            dom_intensity.append('sedentary')
        if dom_cutpoints[0] <= row.dominant < dom_cutpoints[1]:
            dom_intensity.append('light')
        if row.dominant >= dom_cutpoints[1]:
            dom_intensity.append('mvpa')

    df_epoch['dom_intensity'] = dom_intensity

    # recalculate non-dominant wrist intensity --------
    print(f"\nCalculating activity volume for non-dominant wrist using cutpoints of {nondom_cutpoints}...")

    nd_intensity = []

    for row in df_epoch.itertuples():
        if row.nondom < nondom_cutpoints[0]:
            nd_intensity.append('sedentary')
        if nondom_cutpoints[0] <= row.nondom < nondom_cutpoints[1]:
            nd_intensity.append('light')
        if row.nondom >= nondom_cutpoints[1]:
            nd_intensity.append('mvpa')

    df_epoch['nondom_intensity'] = nd_intensity

    # activity totals ----------
    dom_tally = df_epoch['dom_intensity'].value_counts()
    dom_tally = dom_tally.reset_index()
    dom_tally['cat'] = pd.Categorical(dom_tally['index'], ['sedentary', 'light', 'mvpa'])
    dom_tally = dom_tally.sort_values("cat")

    nd_tally = df_epoch['nondom_intensity'].value_counts()
    nd_tally = nd_tally.reset_index()
    nd_tally['cat'] = pd.Categorical(nd_tally['index'], ['sedentary', 'light', 'mvpa'])
    nd_tally = nd_tally.sort_values("cat")

    df_totals = pd.DataFrame({"Dom": dom_tally['dom_intensity'] * epoch_len / 60,
                              "ND": nd_tally['nondom_intensity'] * epoch_len / 60})
    df_totals.index = ['sedentary', 'light', 'mvpa']

    df_totals['ratio'] = df_totals['Dom'] / df_totals['ND']

    return df_epoch, df_totals


def epoch_context(df_epoch, df_mask, remove_sleep_activity=True):
    """Calculates whether each epoch in df_epoch contains sleep, nonwear, or gait.
       Creates new columns with boolean values.

        arguments:
        -df_epoch: epoched dataframe with combined wrist data
        -df_mask: output from DataImport.create_df_mask()
        -remove_sleep_activity: boolean whether to flag all epochs that contain sleep as sedentary (if False,
                                does nothing)
    """

    epoch_len = int((df_epoch.iloc[1]['timestamp'] - df_epoch.iloc[0]['timestamp']).total_seconds())
    print(f"\nUsing masked data to get behaviour context for each {epoch_len}-second epoch...")

    n_gait = []
    n_sleep = []
    n_nonwear = []

    for i in range(0, df_epoch.shape[0]):

        df = df_mask.iloc[int(i*epoch_len):int(i * epoch_len + epoch_len)]

        n_gait.append(df['gait'].sum() >= 1)
        n_sleep.append(df['sleep'].sum() >= 1)
        n_nonwear.append(df['nw'].sum() >= 1)

    df_epoch['gait'] = n_gait
    df_epoch['sleep'] = n_sleep
    df_epoch['nonwear'] = n_nonwear

    if remove_sleep_activity:
        print("-Flagging all sleep epochs as 'sedentary'...")
        df_epoch['dominant'] = [row.dominant if not row.sleep else 'sedentary' for row in df_epoch.itertuples()]
        df_epoch['nondom'] = [row.nondom if not row.sleep else 'sedentary' for row in df_epoch.itertuples()]

    return df_epoch


def analyze_data_section(df_epoch, start, end, show_plot=False):
    """Analyzes epoched data in given time period contained within 'start' and 'end' arguments."""

    print(f"\nAnalysing subsection of data from {start} to {end}...")

    if type(end) == int or type(end) == float:
        print(f"-'End' value was given as a number. Assuming you wanted 'start' + {end} seconds.")
        end = pd.to_datetime(start) + timedelta(seconds=end)

    df = df_epoch.loc[(df_epoch['timestamp'] >= pd.to_datetime(start)) & (df_epoch['timestamp'] < pd.to_datetime(end))]

    df2, df_totals = calculate_activity_totals(df_epoch=df, dom_cutpoints=(62.5, 92.5), nondom_cutpoints=(42.5, 98),
                                               exclude_sleep=True, exclude_gait=False, exclude_nw=True)

    if show_plot:
        plot_epoched_context(df_epoch=df)

    print(df_totals)

    return df_totals
