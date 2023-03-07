import datetime
import os
import pandas as pd
from datetime import timedelta


def check_whats_processed(file="W:/NiMBaLWEAR/OND09/analytics/ecg/ecg_processing_status.xlsx"):

    edf_folder = "W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/"
    edf_files = sorted(list(set([i for i in os.listdir(edf_folder) if 'BF36' in i])))

    study_codes = [i.split("_")[0] for i in edf_files]
    all_subjs = [i.split("_")[1] for i in edf_files]
    coll_ids = [i.split("_")[2] for i in edf_files]

    nw_folder = "C:/Users/ksweber/Desktop/ECG_nonwear_dev/FinalBouts_NoSNR/"
    nw_files = os.listdir(nw_folder)
    nw_proc = [i.split("BF36")[0][:-1] for i in nw_files]

    smital_folder = "W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/"
    smital_files = os.listdir(smital_folder)
    smital_proc = [i.split("snr")[0][:-1] for i in smital_files]

    cn_folder = "W:/NiMBaLWEAR/OND09/analytics/ecg/CardiacNavigator/"
    cn_files = os.listdir(cn_folder)
    cn_beat_files = [i for i in cn_files if 'Beats' in i]
    cn_beat_proc = [i.split("Beats")[0][:-1] for i in cn_beat_files]
    cn_event_files = [i for i in cn_files if "Events" in i]
    cn_event_proc = [i.split("Events")[0][:-1] for i in cn_event_files]

    if file is None:
        df = pd.DataFrame({'study_code': study_codes, "subject_id": all_subjs, 'coll_id': coll_ids})
        df.insert(loc=0, column='full_id',
                  value=[f"{row.study_code}_{row.subject_id}_{row.coll_id}" for row in df.itertuples()])

    if file is not None:
        df = pd.read_excel(file) if 'xlsx' in file else pd.read_csv(file)
        df['coll_id'] = ["0" * len(str(row.coll_id)) + str(row.coll_id) for row in df.itertuples()]

    df['weber_nonwear'] = [subj in nw_proc for subj in df['full_id']]
    df['weber_nonwear_date'] = [get_creation_date(nw_folder + f"{row.full_id}_BF36_Chest_NONWEAR.csv") for row in df.itertuples()]

    df['smital'] = [subj in smital_proc for subj in df['full_id']]
    df['smital_date'] = [get_creation_date(smital_folder + f"{row.full_id}_snr.edf") for row in df.itertuples()]

    df['CN_beats'] = [f"{row.study_code}_{row.subject_id}" in cn_beat_proc for row in df.itertuples()]
    df['CN_beats_date'] = [get_creation_date(cn_folder + f"{row.full_id[:-3]}_Beats.csv") for row in df.itertuples()]

    df['CN_events'] = [f"{row.study_code}_{row.subject_id}" in cn_event_proc for row in df.itertuples()]
    df['CN_events_date'] = [get_creation_date(cn_folder + f"{row.full_id[:-3]}_Events.csv") for row in df.itertuples()]

    sync_folder = "W:/NiMBaLWEAR/OND09/analytics/sync/"
    sync_files = os.listdir(sync_folder)
    df['sync_chest'] = [f"{row.study_code}_{row.subject_id}_{row.coll_id}_BF36_Chest_SYNC_SEG.csv" in sync_files for row in df.itertuples()]

    autocal_folder = "W:/NiMBaLWEAR/OND09/analytics/calib/"
    autocal = os.listdir(autocal_folder)
    df['autocal_chest'] = [f"{row.study_code}_{row.subject_id}_{row.coll_id}_BF36_Chest_CALIB.csv" in autocal for row in df.itertuples()]

    df_summary = pd.DataFrame({"full_id": ['all'], 'study_code': [None], 'subject_id': [None], 'coll_id': [None],
                               'weber_nonwear': [f"{df['weber_nonwear'].value_counts()[True]}/{df.shape[0]}"],
                               'smital': [f"{df['smital'].value_counts()[True]}/{df.shape[0]}"],
                               'synchronization': [f"{df['sync_chest'].value_counts()[True]}/{df.shape[0]}"],
                               'autocalibration': [f"{df['autocal_chest'].value_counts()[True]}/{df.shape[0]}"],
                               'CN_beats': [f"{df['CN_beats'].value_counts()[True]}/{df.shape[0]}"],
                               'CN_events': [f"{df['CN_events'].value_counts()[True]}/{df.shape[0]}"],
                               'last_checked': [datetime.datetime.strftime(datetime.datetime.now(),
                                                                           "%Y/%m/%d %H:%M:%S")]})

    print("============ Processing summary ============\n")
    print(df_summary[['weber_nonwear', 'smital', 'synchronization', 'autocalibration',
                      'CN_beats', 'CN_events', 'last_checked']].loc[0])

    return df, df_summary


def get_creation_date(filename):

    start_date = pd.to_datetime('1970-01-01 00:00:00')

    try:
        unix = os.path.getmtime(filename)
        td = start_date + timedelta(seconds=unix)
        return td.date()

    except FileNotFoundError:
        return None


# df, df_summary = check_whats_processed()
# df.to_excel("W:/NiMBaLWEAR/OND09/analytics/ecg/ecg_processing_status.xlsx", index=False)
# df_summary.to_excel("W:/NiMBaLWEAR/OND09/analytics/ecg/ecg_processing_status_summary.xlsx", index=False)
# print("\nNEXT:\n", df.loc[(~df['CN_beats']) | (~df['CN_events'])].iloc[0]['full_id'])
