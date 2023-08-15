import pandas as pd


def snr_by_beats_wholedataset():
    sinus_lower = {}
    arrs_lower = {}
    sinus_awwf = {}
    arrs_awwf = {}

    for data_key in ['_6', '00', '06', '12', '24']:
        ecg_signal = resample_signal(signal=nst_data[data_key]['ecg'], old_rate=360, new_rate=sample_rate)
        ecg_signal = filter_highpass(signal=ecg_signal, sample_rate=sample_rate, cutoff_low=low_f_cut, order=100)
        # ecg_signal = filter_notch(signal=ecg_signal, sample_rate=sample_rate, freq=notch_cut)

        if data_key != 'test':
            gs_snr = generate_timeseries_noise_value(signal=ecg_signal, noise_key=data_key, sample_rate=sample_rate,
                                                     noise_indexes=(300, 540, 780, 1020, 1260, 1500, 1740))
        if data_key == 'test':
            gs_snr = nst_data['test']['gs_snr']

        nst_data[data_key]['annots']['idx'] = [int(i * sample_rate / 360) for i in nst_data[data_key]['annots']['idx']]

        try:
            nst_snr_og[data_key]['snr'] = resample_signal(signal=nst_snr_og[data_key]['snr'], old_rate=360,
                                                          new_rate=sample_rate)
        except KeyError:
            pass

        self = Smital(study_code=study_code,
                      subj=subj,
                      clip_snr=False,
                      ecg_filt=ecg_signal, sample_rate=sample_rate,
                      upper_ca_thresh=True, upper_cd_thresh=True,
                      lower_ca_thresh=True, lower_cd_thresh=True,
                      use_ca=True, use_cd=True,
                      swt1_wavelet='bior2.2', swt2_wavelet='bior2.2',
                      n_decomp_levels=4,
                      use_decomp_levels=None,
                      use_snr_sum=False,
                      fixed_awwf_levels=None,
                      fixed_tm=None,
                      fixed_awwf_tm=None,
                      use_rr_windows=False,
                      roll_window_sec=1,
                      snr_roll_window_sec=2,
                      )
        self.run_process(time_processing=True)

        self.df_med = self.calculate_snr_stat_by_stage().round(2)

        """
        df_cn = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/cardiac_navigator_screened/{subj}_01_CardiacNavigator_Screened.csv")
        df_cn_raw = pd.read_csv(f"W:/OND09 (HANDDS-ONT)/Incidental findings/CardiacNavigator/{subj}_Events.csv", delimiter=';')
        df_cn_raw = df_cn_raw.loc[(df_cn_raw['Type'] != 'Sinus') & (df_cn_raw['Msec'] < (len(ecg_signal) / sample_rate) * 1000)]
        for row in df_cn_raw.itertuples():
            fig.axes[2].axvspan(row.Msec/1000*sample_rate, row.Msec/1000*sample_rate + row.Length/1000*sample_rate, 0, 1, color='red', alpha=.2)
            print(f"{row.Msec/1000*sample_rate} || {row.Type}")
        """

        # gs_annots, q = validate_butqdb(annots=data_out['annot'], snr_arr=self.data_lower['snr_sta'])

        # plot_thresholds(coef='cA', level=1)
        # fig = plot_results(smital_obj=self, sample_rate=sample_rate, data_key=data_key, nst_data=nst_data, nst_snr=nst_snr, gs_snr=gs_snr, snr_key='snr_raw')

        df_beats_desc, df_beattype_desc, df_arr_use, fig_lower = arrhythmia_snr_describe(smital_obj=self,
                                                                                         sample_rate=sample_rate,
                                                                                         snr_key='snr_sta',
                                                                                         stage='lower', show_plot=False)
        sinus_lower[data_key] = df_beats_desc
        arrs_lower[data_key] = df_beattype_desc

        df_beats_desc, df_beattype_desc, df_arr_use, fig_awwf = arrhythmia_snr_describe(smital_obj=self,
                                                                                        sample_rate=sample_rate,
                                                                                        snr_key='snr_raw', stage='awwf',
                                                                                        show_plot=False)
        sinus_awwf[data_key] = df_beats_desc
        arrs_awwf[data_key] = df_beattype_desc

        # fig_lower.axes[0].set_ylim(-55, 40)
        # fig_awwf.axes[0].set_ylim(-55, 40)

        # peak_validation(input_signal=ecg_signal, test_signal=self.data_awwf['zn'], testsig_label='zn', fs=sample_rate, min_height=.5)

        # df_settings = append_settings_csv(smital_obj=self, pathway="C:/Users/ksweber/Desktop/smital_settings.csv")

    df_sinuslow = sinus_lower['_6'].copy()
    df_arrslow = arrs_lower['_6'].copy()
    df_sinusawwf = sinus_awwf['_6'].copy()
    df_arrsawwf = arrs_awwf['_6'].copy()

    for key in ['00', '06', '12']:
        df_sinuslow = pd.concat([df_sinuslow, sinus_lower[key].loc[sinus_lower[key]['true_snr'] != 24]])
        df_arrslow = pd.concat([df_arrslow, arrs_lower[key].loc[arrs_lower[key]['true_snr'] != 24]])
        df_sinusawwf = pd.concat([df_sinusawwf, sinus_awwf[key].loc[sinus_awwf[key]['true_snr'] != 24]])
        df_arrsawwf = pd.concat([df_arrsawwf, arrs_awwf[key].loc[arrs_awwf[key]['true_snr'] != 24]])
    df_sinuslow = pd.concat([df_sinuslow, sinus_lower['24']])
    df_arrslow = pd.concat([df_arrslow, arrs_lower['24']])
    df_sinusawwf = pd.concat([df_sinusawwf, sinus_awwf['24']])
    df_arrsawwf = pd.concat([df_arrsawwf, arrs_awwf['24']])

    c_dict = {-6: 'red', 0: 'orange', 6: 'gold', 12: 'dodgerblue', 24: 'purple'}
    for i in [-6, 0, 6, 12, 24]:
        d = df_arrsawwf.loc[df_arrsawwf['true_snr'] == i]
        plt.scatter(d['beat_type'], d['50%'], label=f"{i}dB", color=c_dict[i])
        plt.axhline(y=i, color=c_dict[i], linestyle='dashed')
    plt.legend()
    plt.xlabel("beat type")
    plt.ylabel('median SNR')
    plt.grid()

def lookup_row(df, study_code=None, subject_id=None, data_key=None, low_f_cut=None,
               swt1_wavelet=None, swt2_wavelet=None, swt3_wavelet=None, swt4_wavelet=None,
               tm=None, awwf_tm=None):
    df = df.copy()

    parameters = {'study_code': study_code, 'subject_id': subject_id, 'data_key': data_key,
                  'low_f_cut': low_f_cut, 'swt1_wavelet': swt1_wavelet, 'swt2_wavelet': swt2_wavelet,
                  'swt3_wavelet': swt3_wavelet, 'swt4_wavelet': swt4_wavelet,
                  'tm': tm, 'awwf_tm': awwf_tm}

    for parameter in parameters.keys():

        if parameters[parameter] is not None:
            df = df.loc[df[parameter] == parameters[parameter]]

    return df


df = pd.read_csv("C:/Users/ksweber/Desktop/smital_settings.csv")
df.drop(['upper_clean_snr', 'upper_noise_snr', 'subject_id.1'], inplace=True, axis=1)
df['awwf_tm'] = df['awwf_tm'].fillna("variable")

df['lower_noise_diff'] = df['lower_noise_snr'] - df['data_key']
df['lower_clean_diff'] = df['lower_clean_snr'] - 24
df['lower_diff_err'] = abs(df['lower_noise_diff']) + abs(df['lower_clean_diff'])
df['awwf_noise_diff'] = df['awwf_noise_snr'] - df['data_key']
df['awwf_clean_diff'] = df['awwf_clean_snr'] - 24
df['awwf_diff_err'] = abs(df['awwf_noise_diff']) + abs(df['awwf_clean_diff'])
df = df.round(2)

lower = pd.DataFrame(columns=df.columns)
awwf = pd.DataFrame(columns=df.columns)
for subj in df['subject_id'].unique():
    for noise_level in df['data_key'].unique():
        d = df.loc[(df['data_key'] == noise_level) & (df['subject_id'] == subj)]
        lower = lower.append(d.loc[d['lower_diff_err'] == d['lower_diff_err'].min()])
        awwf = awwf.append(d.loc[d['awwf_diff_err'] == d['awwf_diff_err'].min()])
    del d

# awwf.to_csv("C:/Users/ksweber/Desktop/smital_validation_awwf.csv", index=False)
# lower.to_csv("C:/Users/ksweber/Desktop/smital_validation_lower.csv", index=False)

df_crop = lookup_row(df, study_code=None, subject_id=None, data_key=None, low_f_cut=3,
                     swt1_wavelet='db4', swt2_wavelet='sym4', swt3_wavelet=None, swt4_wavelet=None,
                     tm=2.8, awwf_tm=2.5)