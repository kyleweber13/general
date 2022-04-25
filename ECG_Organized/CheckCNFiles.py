import pandas as pd
from ECG_Organized.ImportTabular import import_cn_file
import os
import matplotlib.pyplot as plt


def check_all_arrhythmia_types(event_folder,
                               key_events=("Tachy", "Brady", "Arrest", "AF", "VT", "SVT", "ST+", 'AV2/II', 'AV2/III', 'Block'),
                               show_tally_barplot=False):

    event_files = [i for i in os.listdir(event_folder) if 'Events' in i]

    all_events = []

    arr_dict = {}

    df_all_events = pd.DataFrame(columns=['start_idx', 'end_idx', 'Type', 'duration'])

    for file in event_files:
        print(file)
        df_all, df_sinus, df_cn = import_cn_file(pathway=f"{event_folder}{file}",
                                                 sample_rate=250, start_time=None, use_arrs=None)
        df_all['full_id'] = [file[:10]] * df_all.shape[0]

        df_all_events = df_all_events.append(df_all[['full_id', 'start_idx', 'end_idx', 'Type', 'duration']])

        for arr in df_all['Type'].unique():

            if arr not in arr_dict.keys():
                arr_dict[arr] = df_all['Type'].value_counts().loc[arr]

            if arr in arr_dict.keys():
                arr_dict[arr] += df_all['Type'].value_counts().loc[arr]

            all_events.append(arr)

    for key in key_events:
        if key not in arr_dict.keys():
            arr_dict[key] = 0

    print("================ SUMMARY ================")
    print(f"-Analyzed {len(event_files)} participants")
    print(f"-Found {sum(arr_dict.values())} total events")
    print("     -Found {} key events {}".format(sum([arr_dict[i] for i in key_events]), key_events))
    print(f"-Found {list(set(all_events))}")

    if show_tally_barplot:
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.bar(arr_dict.keys(), arr_dict.values(), color='grey', edgecolor='black')
        ax.set_title(f"Tally of all events from {len(event_files)} participants")
        plt.tight_layout()

    return list(set(all_events)), arr_dict, df_all_events


all_arrs, values, df_all = check_all_arrhythmia_types(event_folder="C:/Users/ksweber/Desktop/CardiacNavigator/edf_cropped/CustomSettings/", show_tally_barplot=False)
