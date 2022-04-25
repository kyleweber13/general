import os
import pandas as pd
from ECG_Organized.ImportECG import *
from ECG_Organized.ImportTabular import *
from ECG_Organized.ArrhythmiaScreening import *
from Run_FFT import run_fft
import matplotlib.pyplot as plt
import datetime
from ECG_Organized.AnalyzeContext import calculate_arr_context

focus_arrs = ["Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block']

folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"
ecg_files = [folder + i for i in os.listdir(folder) if "BF36" in i]

use_ids = pd.read_csv("C:/Users/ksweber/Desktop/CardiacNavigator/df_cn_all.csv")['full_id'].unique()
use_files = [i for i in ecg_files if i.split("/")[-1][:10] in use_ids]
use_files.remove('W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_0003_02_BF36_Chest.edf')

df_nw = pd.read_excel("O:/OBI/ONDRI@Home/Data Processing/Algorithms/Bittium Faros Non-Wear/Kyle_Archive/ECG Non-Wear/OND09_VisuallyInspectedECG_Nonwear.xlsx")
ids = sorted(list(df_nw['full_id'].unique()))

start_time = datetime.datetime.now()

failed = []
for i, subj in enumerate(ids[1:]):
    print(f"{subj} || {i+1}/{len(ids)}")

    try:
        data = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                   ecg_fname=f"{subj}_01_BF36_Chest.edf", bandpass=(.67, 80),
                   smital_filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/{subj}_01_BF36_Chest.pickle")

        # Cardiac Navigator data
        df_all, df_sinus, df_cn = import_cn_file(pathway=f"C:/Users/ksweber/Desktop/CardiacNavigator/edf_cropped/CustomSettings/{subj}_Events.csv",
                                                 sample_rate=data.fs, start_time=data.start_stamp, ecg_signal=data.signal,
                                                 use_arrs=focus_arrs, timestamp_method='start')

        """dom_freqs_all = []
        dom_freqs_high = []
        for row in df_all.itertuples():
            if row.Type in focus_arrs:
                fig, df = run_fft(data=data.signal[row.start_idx:row.end_idx], sample_rate=data.fs,
                                  highpass=2, show_plot=False)

                df_highf = df.loc[df['freq'] >= 30]
                dom_f_high = df_highf.loc[df_highf['power'] == df_highf['power'].max()]['freq'].iloc[0]
                dom_freqs_high.append(dom_f_high)

                dom_f_all = df.loc[df['power'] == df['power'].max()]['freq'].iloc[0]
                dom_freqs_all.append(dom_f_all)

            if row.Type not in focus_arrs:
                dom_freqs_high.append(None)
                dom_freqs_all.append(None)

        df_all['dom_freq_all'] = dom_freqs_all
        df_all['dom_freq_high'] = dom_freqs_high
        df_all.insert(loc=0, column='full_id', value=[subj]*df_all.shape[0])

        df_all = calculate_arr_snr_percentiles(df=df_all, snr_data=data.snr, fs=data.fs)
        df_all = calculate_arr_abs_voltage(df=df_all, signal=data.signal, fs=data.fs)

        df_sinus = calculate_arr_abs_voltage(df=df_sinus, signal=data.signal, fs=data.fs)
        df_sinus.insert(loc=0, column='full_id', value=[subj]*df_sinus.shape[0])

        df_all.to_csv(f"C:/Users/ksweber/Desktop/Processed/{subj}_cn_all_processed.csv", index=False)
        df_sinus.to_csv(f"C:/Users/ksweber/Desktop/Processed/{subj}_cn_sinus_processed.csv", index=False)"""

        df_mask, df_gait, df_sleep, df_act, df_nw = create_df_mask(sample_rate=data.fs, max_i=len(data.signal),
                                                                   start_stamp=data.start_stamp,
                                                                   gait_folder="W:/NiMBaLWEAR/OND09/analytics/gait/bouts/",
                                                                   gait_file=f"{subj}_01_GAIT_BOUTS.csv",
                                                                   sleep_folder="W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/",
                                                                   sleep_file=f"{subj}_01_SLEEP_BOUTS.csv",
                                                                   activity_folder="W:/NiMBaLWEAR/OND09/analytics/activity/epochs/",
                                                                   activity_file=f"{subj}_01_ACTIVITY_EPOCHS.csv",
                                                                   nw_folder="O:/OBI/ONDRI@Home/Data Processing/Algorithms/Bittium Faros Non-Wear/Kyle_Archive/ECG Non-Wear/",
                                                                   nw_file="OND09_VisuallyInspectedECG_Nonwear.xlsx",
                                                                   full_id=subj)
        df_all = calculate_arr_context(df_arr=df_all, sample_rate=data.fs, gait_mask=df_mask['gait'],
                                       sleep_mask=df_mask['sleep'], activity_mask=df_mask['activity'],
                                       nw_mask=df_mask['nw'], temperature_data=data.temperature, temp_fs=data.temp_fs)

        df_all.to_csv(f"C:/Users/ksweber/Desktop/Processed/{subj}_cn_all_processed.csv", index=False)

    except:
        failed.append(subj)

finish_time = datetime.datetime.now()

dt = finish_time - start_time
print("==================================")
print(f"Shit took {dt.total_seconds()/3600:.1f} hours. Damn.")
