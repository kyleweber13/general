import os
import pyedflib
import pandas as pd
import numpy as np
import nwdata
import msoffcrypto
import io


def import_data(file_dict, subj):

    print("\nImporting and formatting summary dataframes...")

    # EDF file details ------------------------------------
    edfs = os.listdir(file_dict['edf_folder'])
    edfs = [i for i in edfs if subj in i and "Wrist" in i]

    # List of days in collection period
    dates = []

    start_time = pyedflib.EdfReader(file_dict['edf_folder'] + edfs[0]).getStartdatetime()
    file_dur = pyedflib.EdfReader(file_dict['edf_folder'] + edfs[0]).file_duration

    # Epoched wrist data -------------------------------------------------------
    df_epoch = pd.DataFrame(columns=['Timestamp', 'avm', 'Day'])

    try:
        df_epoch = pd.read_csv(file_dict['epoch_folder'] + f"OND09_{subj}_01_ACTIVITY_EPOCHS.csv")
        epoch_len = int(file_dur / df_epoch.shape[0])
        df_epoch["Timestamp"] = pd.date_range(start=start_time, freq=f"{epoch_len}S", periods=df_epoch.shape[0])
        df_epoch = df_epoch[["Timestamp", "avm"]]

        for day in set([i.date() for i in df_epoch['Timestamp']]):
            dates.append(day)
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    # Posture data -------------------------------------------------------
    df_posture = pd.DataFrame(columns=['start_timestamp', 'end_timestamp', 'duration', 'posture'])

    try:
        df_posture = pd.read_csv(file_dict['posture'])
        df_posture['start_timestamp'] = pd.to_datetime(df_posture['start_timestamp'])
        df_posture['end_timestamp'] = pd.to_datetime(df_posture['end_timestamp'])

    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    # Sleep data (algorithm output)
    df_sleep_alg = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'sleep_bout_num', 'sptw_num',
                                         'bout_detect', 'start_time', 'end_time'])

    try:
        df_sleep_alg = pd.read_csv(file_dict['sleep_output_file'])
        df_sleep_alg["start_time"] = pd.to_datetime(df_sleep_alg["start_time"])
        df_sleep_alg["end_time"] = pd.to_datetime(df_sleep_alg["end_time"])
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    # Sleep data - summaries
    df_sleep = pd.DataFrame(columns=['Sleep', 'Date', 'TST (h)', 'Time to Bed', 'Time Out of Bed',
                                     'Night TST (h)', 'Num Walks'])

    try:
        df_sleep = pd.read_excel(file_dict['sleep_summary_file'], sheet_name=f"{subj} Summary Dataframes")
        df_sleep = df_sleep.iloc[df_sleep.loc[df_sleep["Sedentary"] == "Sleep"].index[0]:]
        df_sleep.columns = df_sleep.iloc[0]
        df_sleep = df_sleep.iloc[1:]
        df_sleep = df_sleep.dropna()
        df_sleep["Date"] = pd.to_datetime(df_sleep["Date"])
        df_sleep["Date"] = [i.date() for i in df_sleep["Date"]]
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    # Activity descriptive data by day
    df_epoch["Day"] = [row.Timestamp.date() for row in df_epoch.itertuples()]
    avm_desc = df_epoch.groupby("Day")["avm"].describe()
    avm_sum = df_epoch.groupby("Day")['avm'].sum()

    df_act = pd.DataFrame({"Date": [i for i in set([row.Timestamp.date() for row in df_epoch.itertuples()])],
                           "svm": [i*epoch_len for i in avm_sum],
                           "mean_avm": avm_desc['mean'].reset_index(drop=True),
                           "std_avm": avm_desc['std'].reset_index(drop=True)})

    # Activity log (subjective)
    df_act_log = pd.DataFrame(['coll_id', 'study_code', 'subject_id', 'activity', 'start_time',
                               'duration', 'active', 'Notes.', 'Unnamed: 8'])

    try:
        df_act_log = pd.read_excel(file_dict['activity_log_file'])
        df_act_log['subject_id'] = [int(i) if not np.isnan(i) else None for i in df_act_log['subject_id']]
        df_act_log = df_act_log.loc[df_act_log["subject_id"] == int(subj)].reset_index(drop=True)
        df_act_log = df_act_log.fillna("")
        df_act_log.columns = ['coll_id', 'study_code', 'subject_id', 'activity', 'start_time',
                              'duration', 'active', 'Notes.', 'Unnamed: 8']

        stamps = []
        for row in df_act_log.itertuples():
            if not type(row.start_time) is str:
                stamps.append(row.start_time)
            if type(row.start_time) is str:
                stamps.append("NOT LEGIBLE")

        df_act_log["start_time"] = stamps
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    # Clinical insights
    df_clin = pd.DataFrame(columns=['Date', "ID", "Sex", "Cohort", "Hand", "Locations", "Medical", "GaitAids", "Age"])

    try:
        df_clin = read_excel_pwd(file_path=file_dict['clin_insights_file'], password="@handds2021")
        df_clin = df_clin.iloc[5:, [0, 1, 2, 3, 4, 5, 11, 13, 17]]
        df_clin.columns = ['Date', "ID", "Sex", "Cohort", "Hand", "Locations", "Medical", "GaitAids", "Age"]
        df_clin = df_clin.loc[df_clin['ID'] == subj]
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    # Gait bout data
    df_gait = pd.DataFrame(columns=['start_timestamp', 'end_timestamp', 'step_count', 'duration', 'cadence', 'SpeedEst'])

    try:
        df_gait = pd.read_csv(file_dict['gait_file'])
        df_gait = df_gait[["start_timestamp", "end_timestamp", "step_count"]]
        df_gait['start_timestamp'] = pd.to_datetime(df_gait['start_timestamp'])
        df_gait['end_timestamp'] = pd.to_datetime(df_gait['end_timestamp'])
        df_gait['duration'] = [(j-i).total_seconds() for i, j in zip(df_gait['start_timestamp'], df_gait['end_timestamp'])]
        df_gait['cadence'] = 60*df_gait["step_count"]/df_gait['duration']
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    df_steps = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'step_num', 'gait_bout_num',
                                     'step_idx', 'step_time'])

    try:
        df_steps = pd.read_csv(file_dict['steps_file'])
        df_steps["step_time"] = pd.to_datetime(df_steps["step_time"])
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    # Nonwear files
    df_ankle_nw = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'device_type', 'device_location',
                                        'nonwear_bout_id', 'start_time', 'end_time'])

    try:
        df_ankle_nw = pd.read_csv(file_dict['ankle_nw_file'])
        df_ankle_nw['start_time'] = pd.to_datetime(df_ankle_nw['start_time'])
        df_ankle_nw['end_time'] = pd.to_datetime(df_ankle_nw['end_time'])
    except (FileNotFoundError, AttributeError, KeyError, ValueError):
        pass

    df_wrist_nw = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'device_type', 'device_location',
                                        'nonwear_bout_id', 'start_time', 'end_time'])

    try:
        df_wrist_nw = pd.read_csv(file_dict['wrist_nw_file'])
        df_wrist_nw['start_time'] = pd.to_datetime(df_wrist_nw['start_time'])
        df_wrist_nw['end_time'] = pd.to_datetime(df_wrist_nw['end_time'])

    except (FileNotFoundError, AttributeError, KeyError):
        pass

    print("\nImporting ankle data...")
    ankle = nwdata.NWData()
    ankle.import_edf(file_path=file_dict['ankle_file'], quiet=False)

    try:
        ankle.ts = pd.date_range(start=ankle.header["start_datetime"], periods=len(ankle.signals[0]),
                                 freq="{}ms".format(1000 / ankle.signal_headers[0]["sample_rate"]))
        ankle.temp_ts = pd.date_range(start=ankle.header["start_datetime"],
                                      periods=len(ankle.signals[ankle.get_signal_index('Temperature')]),
                                      freq="{}ms".format(1000 / ankle.signal_headers[ankle.get_signal_index('Temperature')]["sample_rate"]))
    except KeyError:
        ankle.ts = pd.date_range(start=ankle.header["startdate"], periods=len(ankle.signals[0]),
                                 freq="{}ms".format(1000 / ankle.signal_headers[0]["sample_rate"]))
        ankle.temp_ts = pd.date_range(start=ankle.header["startdate"],
                                      periods=len(ankle.signals[ankle.get_signal_index('Temperature')]),
                                      freq="{}ms".format(1000 / ankle.signal_headers[ankle.get_signal_index('Temperature')]["sample_rate"]))

    print("\nImporting wrist data...")
    wrist = nwdata.NWData()
    wrist.import_edf(file_path=file_dict['wrist_file'], quiet=False)

    try:
        wrist.ts = pd.date_range(start=wrist.header["start_datetime"], periods=len(wrist.signals[0]),
                                 freq="{}ms".format(1000 / wrist.signal_headers[0]["sample_rate"]))
        wrist.temp_ts = pd.date_range(start=wrist.header["start_datetime"],
                                      periods=len(wrist.signals[wrist.get_signal_index('Temperature')]),
                                      freq="{}ms".format(1000 / wrist.signal_headers[wrist.get_signal_index('Temperature')]["sample_rate"]))
    except KeyError:
        wrist.ts = pd.date_range(start=wrist.header["startdate"], periods=len(wrist.signals[0]),
                                 freq="{}ms".format(1000 / wrist.signal_headers[0]["sample_rate"]))
        wrist.temp_ts = pd.date_range(start=wrist.header["startdate"], periods=len(wrist.signals[wrist.get_signal_index('Temperature')]),
                                 freq="{}ms".format(1000 / wrist.signal_headers[wrist.get_signal_index('Temperature')]["sample_rate"]))

    print("Data imported.")

    return df_epoch, epoch_len, df_clin, df_posture, df_act, df_sleep_alg, df_sleep, df_gait, df_steps, \
           df_act_log, df_ankle_nw, df_wrist_nw, ankle, wrist


def read_excel_pwd(file_path, password, **kwargs):

    file = msoffcrypto.OfficeFile(open(file_path, "rb"))
    file.load_key(password=password)

    decrypted = io.BytesIO()
    file.decrypt(decrypted)

    df = pd.read_excel(decrypted, **kwargs)

    return df


def find_summary_df(subj, data_review_df_folder="W:/OND09 (HANDDS-ONT)/Data Review/"):
    folders = os.listdir(data_review_df_folder)
    folders = [i for i in folders if len(i) == 10]

    for folder in folders:
        files = os.listdir(data_review_df_folder + folder)
        files = [data_review_df_folder + folder + "/" + i for i in files if 'Summary Dataframes' in i]

        for file in files:
            try:
                d = pd.read_excel(file, sheet_name=f"{subj} Summary Dataframes")
                return file
            except ValueError:
                pass

    print("\nSUMMARY FILE NOT FOUND")


def check_filenames(subj, file_dict, nw_bouts_folder="W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_standard/",
                    data_review_df_folder="W:/OND09 (HANDDS-ONT)/Data Review/"):

    ankle_nw_file = f"{nw_bouts_folder}OND09_{subj}_01_AXV6_RAnkle_NONWEAR.csv"
    if not os.path.exists(ankle_nw_file):
        ankle_nw_file = f"{nw_bouts_folder}OND09_{subj}_01_AXV6_LAnkle_NONWEAR.csv"

    wrist_nw_file = f"{nw_bouts_folder}OND09_{subj}_01_AXV6_RWrist_NONWEAR.csv"
    if not os.path.exists(wrist_nw_file):
        wrist_nw_file = f"{nw_bouts_folder}OND09_{subj}_01_AXV6_LWrist_NONWEAR.csv"

    ankle_file = "{}OND09_{}_01_AXV6_RAnkle.edf".format(file_dict['edf_folder'], subj)
    if not os.path.exists(ankle_file):
        ankle_file = "{}OND09_{}_01_AXV6_LAnkle.edf".format(file_dict['edf_folder'], subj)

    wrist_file = "{}OND09_{}_01_AXV6_RWrist.edf".format(file_dict['edf_folder'], subj)
    if not os.path.exists(wrist_file):
        wrist_file = "{}OND09_{}_01_AXV6_LWrist.edf".format(file_dict['edf_folder'], subj)

    sleep_summary_file = find_summary_df(subj=subj, data_review_df_folder=data_review_df_folder)

    file_dict['ankle_nw_file'] = ankle_nw_file
    file_dict['wrist_nw_file'] = wrist_nw_file
    file_dict['wrist_file'] = wrist_file
    file_dict['ankle_file'] = ankle_file
    file_dict['sleep_summary_file'] = sleep_summary_file

    return file_dict
