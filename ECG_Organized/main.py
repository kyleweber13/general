
from ImportECG import ECG
from ImportTabular import import_cn_file, create_df_mask, import_snr_bouts
from AnalyzeContext import calculate_arr_context
from Plotting import *
from Run_FFT import *
from ArrhythmiaScreening import *

""" ======================================================= SET UP ================================================="""

# study + subject ID
subj = 'OND09_0010'

# shortlist of critical arrhythmias
focus_arrs = ["Tachy", "SVT", "Brady", "Arrest", "AF", "VT", "ST+", 'AV2/II', 'AV2/III', 'Block']

# critical arrhtyhmias I don't know the code for since no one has had them yet
unknown_codes = ['torsades de pointes', 'long QT']

# raw data + SNR
data = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
           ecg_fname=f"{subj}_01_BF36_Chest.edf", bandpass=(.67, 80),
           smital_filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/{subj}_01_BF36_Chest.pickle")

# Cardiac Navigator data
df_all, df_sinus, df_cn = import_cn_file(pathway=f"C:/Users/ksweber/Desktop/CardiacNavigator/edf_cropped/CustomSettings/{subj}_Events.csv",
                                         sample_rate=data.fs, start_time=data.start_stamp, ecg_signal=data.signal,
                                         use_arrs=focus_arrs, timestamp_method='start')

# bouted SNR segments
#df_snr, snr_totals = import_snr_bouts(filepath=f"W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_bouts/Smital Bouts V0.1.5/Thresholds 5 20/{subj}_01_BF36_Chest_SmitalBouts.xlsx", sample_rate=data.fs, snr_signal=data.snr)

""" ==================================================== PROCESSING ================================================"""

df_nw = pd.read_excel("O:/OBI/ONDRI@Home/Data Processing/Algorithms/Bittium Faros Non-Wear/Kyle_Archive/ECG Non-Wear/OND09_VisuallyInspectedECG_Nonwear.xlsx")
df_nw = df_nw.loc[df_nw['full_id'] == subj]

# screen arrhythmias based on features
df_cn = apply_arrhythmia_criteria_freq(df_arr=df_cn, raw_ecg=data.signal, sample_rate=data.fs, show_plots=False)
df_cn = apply_arrhythmia_criteria(df_arr=df_cn, voltage_thresh=5000)
df_cn = calculate_arr_snr_percentiles(df=df_cn, snr_data=data.snr, fs=data.fs)
df_all = calculate_arr_snr_percentiles(df=df_all, snr_data=data.snr, fs=data.fs)

# p75 = calculate_wholefile_percentile(snr_data=data.snr, percentile=75)
p75 = 15

# screen arrhythmias based on SNR thresholding
df_cn = flag_high_enough_snr(df=df_cn, use_percentile=0, default_thresh=p75 if p75 > 20 else 20, exceptions_dict={'Arrest': -50, 'Block': -50})

print_screening_summary(df_all_arrs=df_all, df_final=df_cn, focus_arrs=focus_arrs)


df_mask = pd.DataFrame(columns=['timestamp', 'gait', "sleep", 'activity', 'nw'])

# Context/behaviour data
"""NW data read-in will need adjusting once actual code is written"""
"""df_mask, df_gait, df_sleep, df_act, df_nw = create_df_mask(sample_rate=data.fs, max_i=len(data.signal), start_stamp=data.start_stamp,
                                                           gait_folder="W:/NiMBaLWEAR/OND09/analytics/gait/bouts/",
                                                           gait_file=f"{subj}_01_GAIT_BOUTS.csv",
                                                           sleep_folder="W:/NiMBaLWEAR/OND09/analytics/sleep/bouts/",
                                                           sleep_file=f"{subj}_01_SLEEP_BOUTS.csv",
                                                           activity_folder="W:/NiMBaLWEAR/OND09/analytics/activity/epochs/",
                                                           activity_file=f"{subj}_01_ACTIVITY_EPOCHS.csv",
                                                           nw_folder="O:/OBI/ONDRI@Home/Data Processing/Algorithms/Bittium Faros Non-Wear/Kyle_Archive/ECG Non-Wear/",
                                                           nw_file="OND09_VisuallyInspectedECG_Nonwear.xlsx",
                                                           full_id=subj)

df_cn = calculate_arr_context(df_arr=df_cn, sample_rate=data.fs,
                              gait_mask=df_mask['gait'], sleep_mask=df_mask['sleep'],
                              activity_mask=df_mask['activity'], nw_mask=df_mask['nw'],
                              temperature_data=data.temperature,
                              temp_fs=data.temp_fs)

df_all = calculate_arr_context(df_arr=df_all, sample_rate=data.fs, gait_mask=df_mask['gait'], sleep_mask=df_mask['sleep'], activity_mask=df_mask['activity'], nw_mask=df_mask['nw'], temperature_data=data.temperature, temp_fs=data.temp_fs)

df_cn = remove_context(df_arr=df_cn,
                       rules={"gait%": (0, 100, None), 'sleep%': (0, 100, None),
                              'active%': (0, 100, None), 'nw%': (0, 0, None)})"""


""" ========================================== OPTIONAL FUNCTIONS ========================================== """


def optional_function_calls():

    # Plotting ECG and SNR data
    arr_fig = plot_data(df=df_cn,
                        fs=data.fs, ecg_data=data.filt, snr_data=data.snr, t=data.ts, incl_context=False,
                        incl_arrs=df_cn['Type'].unique(),
                        ds_ratio=2, gait_mask=df_mask['gait'], intensity_mask=df_mask['activity'], sleep_mask=df_mask['sleep'], nw_mask=df_mask['nw'], q1_thresh=20)

    # Frequency domain analysis
    fft_fig, df_fft = run_fft(data=data.signal[24237517-2500:24237820+2500], sample_rate=data.fs, highpass=1, show_plot=True)

    a = plot_stft(data=data.signal[24237517-2500:24237820+2500], sample_rate=data.fs, nperseg_multiplier=2, plot_data=True)

    plot_smital_bouts(ecg_signal=data.signal, sample_rate=data.fs, snr_signal=data.snr, df_snr_bouts=df_snr, ds_ratio=3)

    plot_for_nw_detection(ecg_obj=data, df_nw=df_nw.loc[df_nw['full_id'] == subj])
