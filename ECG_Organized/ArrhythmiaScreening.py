import pandas as pd
import numpy as np
from Run_FFT import run_fft


def apply_arrhythmia_criteria(df_arr, voltage_thresh=None):
    """Applies arrhythmia screening rules/criteria that can't be implemented in Cardiac Navigator, e.g. event durations

       argument:
       -df_arr: arrhythmia df from import_cn_file()
       -voltage_threshold: if set to None, does not remove further events. If given integer/float, any event
                    with a max/min voltage value that exeeds the threshold will be omitted.
    """

    print("\nRemoving Cardiac Navigator events that don't meet our criteria...")

    df = df_arr.copy()

    if voltage_thresh is not None:
        print(f"-Removing events with a voltage range >= {voltage_thresh}...")
        n = df.shape[0]
        df["valid_volt"] = df['abs_volt'] <= voltage_thresh
        df = df.loc[df["valid_volt"]]

        print("     -Removed {} high-voltage event{}".format(n-df.shape[0], "s" if n-df.shape[0] != 1 else ""))

    # V-tach >= 30 seconds ---------------
    print("-Removing VT/SVT < 30-seconds in duration...")
    n = df.shape[0]
    df_other = df.loc[~df['Type'].isin(['Tachy', 'VT', 'SVT'])]  # not tachycardia events
    df_tach = df.loc[(df['Type'].isin(['Tachy', 'VT', 'SVT'])) & (df['duration'] >= 30)]  # valid tachycardia events

    df = pd.concat([df_other, df_tach])
    df = df.sort_values("start_idx").reset_index(drop=True)
    print(f"     -Found {n - df.shape[0]} episodes")

    # bradycardia >= 60 seconds
    print("-Removing bradycardic episodes < 60-seconds in duration...")
    n = df.shape[0]
    df_other = df.loc[df['Type'] != 'Brady']
    df_brady = df.loc[(df['Type'] == 'Brady') & (df['duration'] >= 60)]

    df = pd.concat([df_other, df_brady])
    df = df.sort_values("start_idx").reset_index(drop=True)
    print(f"     -Found {n - df.shape[0]} episodes")

    # A-fib < 30 seconds
    print("-Removing AF events < 30 seconds in duration...")
    df_other = df.loc[df['Type'] != 'AF']
    df_af = df.loc[df['Type'] == 'AF']
    n = df_af.shape[0]

    df_af = df_af.loc[df_af['duration'] >= 30]

    df = pd.concat([df_other, df_af])
    df = df.sort_values("start_idx").reset_index(drop=True)
    print(f"     -Found {n - df_af.shape[0]} events")

    return df


def apply_arrhythmia_criteria_freq(df_arr, raw_ecg, sample_rate, show_plots=False):

    print("\nRemoving detected cardiac arrest/block where 60Hz is the dominant frequency above 30Hz...")
    valid_freq = []
    df_arrest = df_arr.loc[df_arr['Type'].isin(['Arrest', 'Block'])]
    df_other = df_arr.loc[~df_arr['Type'].isin(['Arrest', 'Block'])]

    # "cardiac arrest" due to electrode disconnect
    for row in df_arrest.itertuples():
        fig, df = run_fft(data=raw_ecg[row.start_idx:row.end_idx], sample_rate=sample_rate,
                          highpass=25, show_plot=show_plots)

        df_highf = df.loc[df['freq'] >= 30]
        dom_f = df_highf.loc[df_highf['power'] == df_highf['power'].max()]['freq'].iloc[0]

        valid_freq.append(not 58 <= dom_f <= 62)

    df_valid_arrest = df_arrest.loc[valid_freq]
    print(f"-Removed {df_arrest.shape[0] - df_valid_arrest.shape[0]} of {df_arrest.shape[0]} arrest/block events.")

    return df_other.append(df_valid_arrest).sort_values('start_idx').reset_index(drop=True)


def calculate_arr_snr_percentiles(df, snr_data, fs):
    """Calculates quartile SNR values for each arrhythmia. Adds columns to df.

      arguments:
      -df: arrhythmia DF
      -snr_data: SNR array
      -fs: sample rate of snr_data
    """

    print("\nCalculating SNR descriptive data for arrhythmia events...")

    p0 = []
    p25 = []
    p50 = []
    p75 = []
    p100 = []
    avg = []
    pad_len = int(2*fs)
    l = len(snr_data)

    df = df.copy()

    for row in df.itertuples():
        if row.start_idx - pad_len >= 0:
            start = row.start_idx - pad_len
        if row.start_idx - pad_len < 0:
            start = 0
        if row.end_idx + pad_len >= l:
            end = -1
        if row.end_idx + pad_len < l:
            end = row.end_idx + pad_len

        s = snr_data[start:end]

        avg.append(np.mean(s))
        p0.append(np.percentile(s, q=0))
        p25.append(np.percentile(s, q=25))
        p50.append(np.percentile(s, q=50))
        p75.append(np.percentile(s, q=75))
        p100.append(np.percentile(s, q=100))

    df['p0'] = p0
    df["p25"] = p25
    df['p50'] = p50
    df["p75"] = p75
    df['p100'] = p100
    df['avg_snr'] = avg

    return df


def calculate_arr_abs_voltage(df, signal, fs):
    """Calculates absolute voltage range for each arrhythmia. Adds columns to df.

      arguments:
      -df: arrhythmia DF
      -signal: ECG signal (raw?)
      -fs: sample rate of signal
    """

    print("\nCalculating voltage ranges for arrhythmia events...")

    abs_volt = []
    pad_len = int(2*fs)
    l = len(signal)

    df = df.copy()

    for row in df.itertuples():
        if row.start_idx - pad_len >= 0:
            start = row.start_idx - pad_len
        if row.start_idx - pad_len < 0:
            start = 0
        if row.end_idx + pad_len >= l:
            end = -1
        if row.end_idx + pad_len < l:
            end = row.end_idx + pad_len

        v = signal[start:end]

        abs_volt.append(np.abs(max(v) - min(v)))

    df['abs_volt'] = abs_volt

    return df


def flag_high_enough_snr(df, default_thresh=15, use_percentile=0, exceptions_dict=None):
    """Function to flag arrhythmia events as valid/invalid based on SNR thresholding.

       arguments:
       -df: DF of Cardiac Navigator event data
       -default_thresh: SNR threshold to use for all events, unless specified specifically in exceptions_dict
       -use_percentile: which percentile value to apply SNR thresholding to (0, 25, 50, 75, or 100)
       -exceptions_dict: optional dictionary to specific thresholds for specific arrhythmia events.
                         Leave as None to apply default_thresh argument to all

       returns copy of df
    """

    print(f"\nFlagging arrhythmia events as valid/invalid based on {use_percentile}%ile SNR >= {default_thresh} dB...")
    if exceptions_dict is not None:
        print("-Specific thresholds:")
        for key in exceptions_dict:
            print(f"     -{key} = {exceptions_dict[key]} dB")
    df = df.copy()

    snr_valid = []

    for row in range(df.shape[0]):

        if exceptions_dict is not None:
            # Handling if unique threshold for arrhythmia
            if df.iloc[row]['Type'] in exceptions_dict.keys():
                snr_valid.append(df.iloc[row][f'p{str(use_percentile)}'] >= exceptions_dict[df.iloc[row]['Type']])
            # Handling if not unique threshold for arrhythmia
            if df.iloc[row]['Type'] not in exceptions_dict.keys():
                snr_valid.append(df.iloc[row][f'p{str(use_percentile)}'] >= default_thresh)

        # Handling if no exemptions_dict --> apply default_thresh to all
        if exceptions_dict is None:
            snr_valid.append(df.iloc[row][f'p{str(use_percentile)}'] >= default_thresh)

    df['snr_valid'] = snr_valid

    valid = list(df['snr_valid']).count(True)
    invalid = df.shape[0] - valid

    print(f"-Valid events: {valid}/{df.shape[0]}")
    print(f"-Invalid events: {invalid}/{df.shape[0]}")

    return df.loc[df['snr_valid']]


def calculate_wholefile_percentile(snr_data, percentile):

    if 0 < percentile < 1:
        percentile *= 100

    print(f"\nCalculating {str(int(percentile)) if percentile > 1 else str(int(100*percentile))}th percentile of SNR data...")

    p = np.percentile(a=snr_data[~np.isnan(snr_data)], q=percentile)

    return int(p)


def print_screening_summary(df_all_arrs, df_final,
                            focus_arrs=("Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block')):

    n_original = df_all_arrs.shape[0]
    n_focus = df_all_arrs.loc[df_all_arrs['Type'].isin(focus_arrs)].shape[0]
    n_final = df_final.loc[df_final['snr_valid']].shape[0]

    n_not_focus = n_original - n_focus
    n_lowquality = n_focus - n_final

    print("\n========== Event Screening Summary ===========")
    print(f"-{n_original} total events")
    print("     -Detected: {}".format(list(df_all_arrs['Type'].unique())))
    print(f"-{n_not_focus} removed due to not being critical ({n_focus} remain)")
    print(f"     -Focus: {focus_arrs}")
    print(f"-{n_lowquality} removed for being low signal quality")
    print(f"\n{df_final.shape[0]} events remain ({df_final.shape[0] * 100 / df_all_arrs.shape[0]:.2f}%)")

    for arr in df_final['Type'].unique():
        print(f"-{df_final['Type'].value_counts().loc[arr]} {arr} events")


def remove_context(df_arr, rules=None):
    """Removes events based on context criteria range specified by keys in 'rules'.

       arguments:
       -df_arr: event dataframe
       -rules: dictionary with keys and tuples corresponding to context columns (gait%, activity%, nw%, sleep%) in df_arr
               with ranges of values within which events are NOT removed
               e.g.: rules = {"nw%": (0, 0, None), 'gait%': (0, 100, None), 'active%': (0, 0, ['VT', 'SVT'])}
                    -Removes all events where nw% > 0
                    -Removes no gait events (all will be 0-100%)
                    -Removes only VT and SVT events when activity > 0%
               -if third object in tuple is None, rule will be applied to all arrhythmias.
                If it's a str, list, or tuple, rule will be applied to that/those arrhythmia(s)

       returns:
       -df_arr with invalid events removed
    """

    # df = pd.DataFrame(columns=df_arr.columns)
    df = df_arr.copy()

    n = df_arr.shape[0]

    if rules is not None:
        print("\nApplying context rules:")

        for key in rules:
            print(f"-{key}: {rules[key]}")

            if type(rules[key][2]) is str:
                rules[key][2] = list(rules[key][2].split("kjhadsfjhg"))  # won't be separating this str, ha!

            # if rule doesn't specify arrhythmia type:
            if rules[key][2] is None:
                df = df.loc[(df[key] >= rules[key][0]) & (df[key] <= rules[key][1])]

            # if rule does specify arrhythmia:
            if rules[key][2] is not None:
                df = df.loc[~(((df[key] < rules[key][0]) | (df[key] > rules[key][1])) & (df['Type'].isin(rules[key][2])))]

    df = df.reset_index(drop=True)

    print("\nRemoved {} event{}".format(n - df.shape[0], "s" if n - df.shape[0] != 1 else ""))

    return df
