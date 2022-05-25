import pandas as pd
import CheckAxivityFile
import CheckBittiumFiles
import os
import numpy as np

subj = '0114'

ankle_file = "W:/NiMBaLWEAR/OND09/wearables/raw/OND09_{}_01_AXV6_{}Ankle.CWA"
ankle_file = ankle_file.format(subj, "L") if os.path.exists(ankle_file.format(subj, "L")) else ankle_file.format(subj, "R")
wrist_file = "W:/NiMBaLWEAR/OND09/wearables/raw/OND09_{}_01_AXV6_{}Wrist.CWA"
wrist_file = wrist_file.format(subj, "L") if os.path.exists(wrist_file.format(subj, "L")) else wrist_file.format(subj, "R")

ankle = CheckAxivityFile.CWAData(ankle_file)
df_ankle = pd.DataFrame([[ankle_file, ankle.data_obj.header['logging_start'], ankle.data_obj.header['logging_end'],
                         (ankle.data_obj.header['logging_end'] - ankle.data_obj.header['logging_start']).total_seconds(),
                         ankle.data_obj.header['sample_rate']]], columns=["filename", 'start', 'end', 'duration', 'sample_rate'])

wrist = CheckAxivityFile.CWAData(wrist_file)
df_wrist = pd.DataFrame([[wrist_file, wrist.data_obj.header['logging_start'], wrist.data_obj.header['logging_end'],
                         (wrist.data_obj.header['logging_end'] - wrist.data_obj.header['logging_start']).total_seconds(),
                         wrist.data_obj.header['sample_rate']]], columns=["filename", 'start', 'end', 'duration', 'sample_rate'])

df, fig = CheckBittiumFiles.check_files(files=["W:/NiMBaLWEAR/OND09/wearables/raw/OND09_{}_01_BF36_Chest.EDF".format(subj)], show_plot=False)

df = df.append(df_ankle)
df = df.append(df_wrist)
df.index = ['Bittium', 'Ankle', 'Wrist']
df['duration_days'] = df['duration']/86400
print(df[['start', 'end', 'duration_days']])

