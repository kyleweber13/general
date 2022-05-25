import pandas as pd
from datetime import timedelta as timedelta
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import Run_FFT
import matplotlib.pyplot as plt


def filter_signal(data, filter_type, low_f=None, high_f=None, notch_f=None, notch_quality_factor=30.0,
                  sample_f=None, filter_order=2):
    """Function that creates bandpass filter to ECG data.
    Required arguments:
    -data: 3-column array with each column containing one accelerometer axis
    -type: "lowpass", "highpass" or "bandpass"
    -low_f, high_f: filter cut-offs, Hz
    -sample_f: sampling frequency, Hz
    -filter_order: order of filter; integer
    """

    nyquist_freq = 0.5 * sample_f

    if filter_type == "lowpass":
        low = low_f / nyquist_freq
        b, a = butter(N=filter_order, Wn=low, btype="lowpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == "highpass":
        high = high_f / nyquist_freq

        b, a = butter(N=filter_order, Wn=high, btype="highpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == "bandpass":
        low = low_f / nyquist_freq
        high = high_f / nyquist_freq

        b, a = butter(N=filter_order, Wn=[low, high], btype="bandpass")
        # filtered_data = lfilter(b, a, data)
        filtered_data = filtfilt(b, a, x=data)

    if filter_type == 'notch':
        b, a = iirnotch(w0=notch_f, Q=notch_quality_factor, fs=sample_f)
        filtered_data = filtfilt(b, a, x=data)

    return filtered_data


def epoch_intensity(df_epoch, column, cutpoint_key, cutpoints_dict, author):

    df = df_epoch.copy()

    vals = []
    for i in df[column]:
        if i < cutpoints_dict[author + cutpoint_key][0]:
            vals.append('sedentary')
        if cutpoints_dict[author + cutpoint_key][0] <= i < cutpoints_dict[author + cutpoint_key][1]:
            vals.append('light')
        if i >= cutpoints_dict[author + cutpoint_key][1]:
            vals.append("moderate")

    return vals


def calculate_daily_activity(df_epoch, cutpoints_dict,
                             df_gait, df_act, epoch_len=15, column='avm',
                             side="Non-dominant", author='Powell', ignore_sleep=True):

    days = sorted([i for i in set([i.date() for i in df_epoch['Timestamp']])])
    days = pd.date_range(start=days[0], end=days[-1] + timedelta(days=1), freq='1D')

    if ignore_sleep:
        df_epoch = df_epoch.loc[df_epoch['sleep_mask'] == 0]

    daily_vals = []
    for day1, day2 in zip(days[:], days[1:]):
        df_day = df_epoch.loc[df_epoch["Day"] == day1]

        sed = df_day.loc[df_day[column] < cutpoints_dict[author + side][0]]
        sed_mins = sed.shape[0] * epoch_len / 60

        light = df_day.loc[(df_day[column] >= cutpoints_dict[author + side][0]) &
                           (df_day[column] < cutpoints_dict[author + side][1])]
        light_mins = light.shape[0] * epoch_len / 60

        mod = df_day.loc[(df_day[column] >= cutpoints_dict[author + side][1]) &
                         (df_day[column] < cutpoints_dict[author + side][2])]
        mod_mins = mod.shape[0] * epoch_len / 60

        vig = df_day.loc[df_day[column] >= cutpoints_dict[author + side][2]]
        vig_mins = vig.shape[0] * epoch_len / 60

        df_gait_day = df_gait.loc[(day1 <= df_gait['start_timestamp']) & (df_gait['start_timestamp'] < day2)]
        n_steps = df_gait_day["step_count"].sum()
        walk_mins = df_gait_day["duration"].sum()/60

        daily_vals.append([day1, sed_mins, light_mins, mod_mins, vig_mins, n_steps, walk_mins])

    df_daily = pd.DataFrame(daily_vals, columns=["Date", "Sed", "Light", "Mod", "Vig", "Steps", "MinutesWalking"])
    df_daily["Active"] = df_daily["Light"] + df_daily["Mod"] + df_daily["Vig"]
    df_daily["MVPA"] = df_daily["Mod"] + df_daily["Vig"]
    df_daily['avm'] = df_act['mean_avm']

    df_daily = df_daily.append(pd.Series({"Date": "TOTAL", "Sed": df_daily['Sed'].sum(),
                                          "Light": df_daily['Light'].sum(), "Mod": df_daily['Mod'].sum(),
                                          "Vig": df_daily['Vig'].sum(), "Steps": df_daily['Steps'].sum(),
                                          "MinutesWalking": df_daily['MinutesWalking'].sum(),
                                          "Active": df_daily['Active'].sum(), "MVPA": df_daily['MVPA'].sum(),
                                          "avm": df_daily['avm'].mean()}), ignore_index=True)

    return df_daily


def calculate_hand_dom(df_clin):

    dom = df_clin.iloc[0]['Hand']

    try:
        if np.isnan(dom):
            print("Dominance data not given in ClinicalInsights file. Assuming right-handed.")
            dom = 'Right'
    except TypeError:
        pass

    wear = df_clin.iloc[0]['Locations'].split(",")
    wear = wear[0] if 'wrist' in wear[0] else wear[1]

    if dom.capitalize() in wear.capitalize():
        dominant_wrist = True
    else:
        dominant_wrist = False

    return dominant_wrist


def print_clinical_summary(subj, df_clin):

    print(f"-Subject {subj}: {df_clin.iloc[0]['Age']} years old, cohort = {df_clin.iloc[0]['Cohort']}, gait aid use = {df_clin.iloc[0]['GaitAids']}")

    print("Medical: ")
    for i in df_clin['Medical'].iloc[0].split("."):
        print(f"-{i}")

    print()


def combine_df_daily(df_epoch, cutpoints, df_gait, df_act, epoch_len, hand_dom, ignore_sleep=True):

    fraysse = calculate_daily_activity(df_epoch=df_epoch, cutpoints_dict=cutpoints, df_gait=df_gait, df_act=df_act,
                                       epoch_len=epoch_len, column='avm', side='Dominant' if hand_dom else 'Non-dominant',
                                       author='Fraysse', ignore_sleep=ignore_sleep)
    powell = calculate_daily_activity(df_epoch=df_epoch, column='avm', df_act=df_act, df_gait=df_gait,
                                      cutpoints_dict=cutpoints, side='Dominant' if hand_dom else 'Non-dominant',
                                      author='Powell', ignore_sleep=ignore_sleep)


    df = pd.DataFrame({"Date": powell['Date'], "Sed_Powell": powell['Sed'], "Sed_Fraysse": fraysse['Sed'],
                       "Light_Powell": powell['Light'], "Light_Fraysse": fraysse['Light'],
                       "MVPA_Powell": powell['Mod'] + powell['Vig'], "MVPA_Fraysse": fraysse['Mod'],
                       "Steps": powell['Steps'], 'MinWalking': powell['MinutesWalking'],
                       'avm': powell['avm']})
    df['Active_Powell'] = df['Light_Powell'] + df['MVPA_Powell']
    df['Active_Fraysse'] = df['Light_Fraysse'] + df['MVPA_Fraysse']

    return df


def calculated_logged_intensity(df_act_log, df_epoch, epoch_len=15, hours_offset=0):
    sed = []
    light = []
    mvpa = []

    df_act_log = df_act_log.copy()

    print("Measured intensities of logged events:")
    for row in df_act_log.itertuples():

        try:
            epoch = df_epoch.loc[(df_epoch['Timestamp'] >= row.start_time + timedelta(hours=hours_offset)) &
                                 (df_epoch["Timestamp"] <= row.start_time +
                                  timedelta(hours=hours_offset) + timedelta(seconds=row.duration * 60))]

            vals = epoch['intensity'].value_counts()

            try:
                sed.append(vals['sedentary'] / (60 / epoch_len))
            except KeyError:
                sed.append(0)

            try:
                light.append(vals['light'] / (60 / epoch_len))
            except KeyError:
                light.append(0)

            try:
                mvpa.append(vals['moderate'] / (60 / epoch_len))
            except KeyError:
                mvpa.append(0)

        except TypeError:
            sed.append(None)
            light.append(None)
            mvpa.append(None)

    df_act_log['sed'] = sed
    df_act_log['light'] = light
    df_act_log['mvpa'] = mvpa

    for row in df_act_log.itertuples():
        if type(row.duration) is int:
            print(f"#{row.Index} {row.activity} || {row.start_time} ({row.duration} minutes) || sed={row.sed}, "
                  f"light={row.light}, mvpa={row.mvpa} minutes")

    return df_act_log


def freq_analysis(obj, ts, subj="", sample_rate=None, channel="", lowpass=None, highpass=None, n_secs=60, stft_mult=5, stft=False, show_plot=True):

    stamp = pd.to_datetime(ts)

    chn_idx = obj.get_signal_index(channel)

    sample_rate = sample_rate if sample_rate is not None else obj.signal_headers[chn_idx]['sample_rate']

    try:
        idx = Run_FFT.get_index_from_stamp(start=obj.header['start_datetime'], stamp=stamp,
                                           sample_rate=sample_rate)
    except KeyError:
        idx = Run_FFT.get_index_from_stamp(start=obj.header['startdate'], stamp=stamp,
                                           sample_rate=sample_rate)

    end_idx = int(idx + sample_rate * n_secs)

    d = obj.signals[chn_idx][idx:end_idx]

    if highpass is not None:
        d = filter_signal(data=d, sample_f=sample_rate, filter_type='highpass', high_f=highpass, filter_order=5)
    if lowpass is not None:
        d = filter_signal(data=d, sample_f=sample_rate, filter_type='lowpass', low_f=lowpass, filter_order=5)

    df_fft = None

    if not stft:
        fig, df_fft = Run_FFT.run_fft(data=d, sample_rate=sample_rate, remove_dc=False, show_plot=show_plot)
        if show_plot:
            fig.suptitle(f"OND09_{subj}: {channel}   ||   {stamp}")
            plt.tight_layout()

        dom_f = round(df_fft.loc[df_fft['power'] == df_fft['power'].max()]['freq'].iloc[0], 2)
        print(f"Dominant frequency is ~{dom_f}Hz")

    if stft:
        fig, f, t, Zxx = Run_FFT.plot_stft(data=d, sample_rate=sample_rate, nperseg_multiplier=stft_mult, plot_data=show_plot)
        if show_plot:
            fig.suptitle(f"OND09_{subj}: {channel}   ||   {stamp}")
            if lowpass is not None:
                fig.axes[1].set_ylim(fig.axes[1].get_ylim()[0], lowpass)

    return fig, df_fft


def flag_sleep_epochs(df_epoch, df_sleep_alg):

    sleep_mask = np.zeros(df_epoch.shape[0])

    start_time = df_epoch.iloc[0]['Timestamp']
    epoch_len = (df_epoch.iloc[1]['Timestamp'] - start_time).total_seconds()

    for row in df_sleep_alg.itertuples():
        start_i = int(np.floor((row.start_time - start_time).total_seconds() / epoch_len))
        end_i = int(np.floor((row.end_time - start_time).total_seconds() / epoch_len))

        sleep_mask[start_i:end_i] = 1

    return sleep_mask


