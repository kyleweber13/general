import numpy as np


def calculate_arr_context(df_arr, sample_rate, gait_mask, sleep_mask, activity_mask, nw_mask,
                          temperature_data, temp_fs):
    """For each arrhythmia event, calculates what % of its 1-sec epochs are spent walking, sleeping, and active.
       Also average temperature.

       arguments:
       -df_arr: arrhythmia DF
       -sample_rate: sample rate of ECG/SNR data, Hz
       -gait/intensity/nw/sleep_mask: mask data from create_df_mask()
       -temperature_data: temperature data array
       -temp_fs: sample rate of temperature, Hz

       returns df with new columns
    """

    print("\nAnalyzing each Cardiac Navigator events for context...")
    df_arr = df_arr.copy()

    perc_gait = []
    perc_sleep = []
    perc_active = []
    perc_nw = []
    temperature = []
    temp_range = []

    temp_ratio = int(sample_rate / temp_fs)

    for row in df_arr.itertuples():
        start_int = int(np.floor(row.start_idx / sample_rate))
        end_int = int(np.ceil(row.end_idx / sample_rate))

        l = end_int - start_int

        t = temperature_data[int(np.floor(row.start_idx / temp_ratio)):int(np.ceil(row.end_idx / temp_ratio))]
        temperature.append(round(np.mean(t), 1))
        temp_range.append([round(min(t), 1), round(max(t), 1)])

        gait = list(gait_mask[start_int:end_int])
        p = (100 * gait.count(1) / l)
        perc_gait.append(round(p, 2))

        sleep = list(sleep_mask[start_int:end_int])
        s = (100 * sleep.count(1) / l)
        perc_sleep.append(round(s, 2))

        active = list(activity_mask[start_int:end_int])
        a = 100 * (active.count(1) + active.count(2) + active.count(3)) / l
        perc_active.append(round(a, 2))

        nw = list(nw_mask[start_int:end_int])
        # n = (100 * list(nw[start_int:end_int]).count(1) / l)
        n = (100 * sum(nw) / l)
        perc_nw.append(round(n, 2))

    df_arr['avg_temp'] = temperature
    df_arr['temp_range'] = temp_range
    df_arr['gait%'] = perc_gait
    df_arr['sleep%'] = perc_sleep
    df_arr['active%'] = perc_active
    df_arr['nw%'] = perc_nw

    return df_arr
