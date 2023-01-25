import matplotlib.pyplot as plt
import numpy as np

import ECG.ImportFiles as ImportFiles
import ECG.Processing as Processing
import ECG.PeakDetection as PeakDetection
import ECG.PeakScreening as PeakScreening
import ECG.HR_Calculation as HR_Calculation
import ECG.OrphanidouSignalQuality as Orphanidou
import ECG.Templating as Templating

import pandas as pd
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.%f")
from ECG.Plotting import *
from tqdm import tqdm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# data import --------------
full_id = 'OND09_SBH0300'
coll_id = '01'

snr_folder = "W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/"
thresholds = (5, 18)  # Smital SNR thresholds

# OND09

ecg = ImportFiles.ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
                      ecg_fname=f"{full_id}_{coll_id}_BF36_Chest.edf",
                      bandpass=(1.5, 25), thresholds=thresholds,
                      smital_edf_fname=f'W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/timeseries_edf/{full_id}_{coll_id}_snr.edf',
                      snr_hr_bout_filename=f"{snr_folder}bouts_heartrate/{full_id}_01_snr_bouts_heartrate.csv",
                      snr_fullanalysis_bout_filename=f"{snr_folder}bouts_fullanalysis/{full_id}_01_snr_bouts_fullanalysis.csv",
                      snr_all_bout_filename=f"{snr_folder}bouts_original/{full_id}_01_snr_bouts.csv",
                      nw_filename=f"W:/NiMBaLWEAR/OND09/analytics/nonwear/bouts_cropped/{full_id}_01_BF36_Chest_NONWEAR.csv",
                      quiet=False)

# data cropping for faster testing -----------------------------------------
n_hours = 12

end_idx = int(n_hours*3600*ecg.ecg.signal_headers[ecg.ecg.get_signal_index("ECG")]['sample_rate'])
# end_idx = len(ecg.ecg.signals[ecg.ecg.get_signal_index("ECG")])
ecg_signal = ecg.ecg.signals[0][:end_idx]
ecg.ecg.filt = ecg.ecg.filt[:end_idx]
filt = ecg.ecg.filt.copy()
ecg.snr = ecg.snr[:end_idx]

ecg.df_snr_all = Processing.crop_df(ecg.df_snr_all, start_idx=0, end_idx=end_idx)
ecg.df_snr_hr = Processing.crop_df(ecg.df_snr_hr, start_idx=0, end_idx=end_idx)
ecg.df_snr_q1 = Processing.crop_df(ecg.df_snr_q1, start_idx=0, end_idx=end_idx)
ecg.df_snr_ignore = Processing.crop_df(ecg.df_snr_ignore, start_idx=0, end_idx=end_idx)
ecg.df_nw = Processing.crop_df(ecg.df_nw, start_idx=0, end_idx=end_idx)

# dfs_orph = ImportFiles.import_orphanidou_dfs(full_id, root_dir="W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/orphanidou/dev/")
dfs_orph = None


def run_algorithm(ecg_signal: list or np.array,
                  raw_timestamps: list or np.array,
                  sample_rate: int or float,
                  epoch_len: int = 15,
                  use_corrected_peaks: bool = False,
                  abs_value_correction: bool = True,
                  correl_thresh: float = .7,
                  correl_method: str = 'neighbour',
                  correl_window_size: int or float = .2,
                  premature_beat_correl_window_size: int or float = .125,
                  premature_search_window: int or float = .3,
                  amplitude_thresh: int or float = 100,
                  delta_hr_thresh: int or float = 20,
                  location_margin: float = .1,
                  min_snr_quality: int = 2,
                  df_snr_ignore: pd.DataFrame = pd.DataFrame(),
                  df_snr_q1: pd.DataFrame() = pd.DataFrame(),
                  df_nw: pd.DataFrame = pd.DataFrame(),
                  orphanidou_dfs: dict or None = None,
                  quiet: bool = True):
    """ Runs numerous function calls and analysis steps in the running of peak detection, peak screening, signal
        quality analysis, and heart rate calculations.

        arguments:

        data
        -ecg_signal: timeseries ECG signal upon which peak detection was run
        -raw_timestamps: timestamps for ecg_signal
        -sample_rate: sampling rate of ecg_signal, Hz

        -epoch_len: epoch length in seconds over which Orphanidou signal quality is checked
            and epoched HR is calculated

        -correl_thresh: correlation threshold used in peak screening. Value between -1 and 1.
        -correl_method: correlation method for which data to run correlation upon. "template" or "neighbour"
            -"template" runs beat correlation vs. data specified with hq_qrs_template
            -"neighbour": runs beat correlation vs. next beat
        -correl_window_size: two-sided window size used in calculating beat correlations in seconds
        -premature_beat_correl_window_size: two-sided window size used in calculating beat correlations in seconds
                                            when the beat has been flagged as being possibly premature
        -premature_search_window: two-sided search window for premature beats in seconds

        -amplitude_thresh: voltage amplitude for valid beats in units that ECG was measured
        -delta_hr_thresh: beat-to-beat change in HR above which is the pair of beats used to calculate the
                          change in HR is considered to have a potentially erroneous beat
        -location_margin: max allowed margin of error in checking for a beat that was erroneously rejected in seconds.
            -E.g., if set to .1, rejected beat has to be within 100ms of its expected
             location (midpoint of neighbouring beat on each side)
        -min_snr_quality: minimum SNR quality needed for beats to be included in beat-to-beat HR calculations.
            -Values refer to Smital et al. (2020) signal qualities (1 = best, 2 = good, 3 = ignore). If data is not
             included in df_peaks' column 'quality', no beats are omitted.

        -df_snr_ignore: dataframe containing low-quality data bouts within which detected peaks are ignored
        -df_nw: dataframe containing non-wear bouts within which detected peaks are ignored
        -orphanidou_dfs: dictionary containing processed Orphanidou data.
                         Output of OrphanidouSignalQuality.import_orphanidou_dfs()
        -quiet: whether to print processing progress to console

    returns:
    -data_dict: dictionary of dataframes from each stage of peak screening

    """

    idx_key = 'idx' if not use_corrected_peaks else 'idx_corr'

    # initial peak detection -----------------
    df_peaks_originals = PeakDetection.detect_peaks_initial(ecg_signal=ecg_signal,
                                                            sample_rate=sample_rate,
                                                            timestamps=raw_timestamps,
                                                            correct_locations=True,
                                                            min_height=None,
                                                            correction_windowsize=.15,
                                                            absolute_peaks=abs_value_correction,
                                                            use_correlation=True)

    data_dict = {'original': df_peaks_originals[0],
                 'original_corr': df_peaks_originals[1]}

    last_df = 'original' if not use_corrected_peaks else 'original_corr'

    # creation of average QRS template in highest 20 minutes worth of data -----------------------
    """
    df_snr_template = df_snr_q1.loc[(df_snr_q1['avg_snr'] >= 25) & (df_snr_q1['duration'] >= 30)]
    df_snr_template.sort_values('avg_snr', ascending=False).reset_index(drop=True)

    if df_snr_template.shape[0] < 20:
        df_snr_template = df_snr_q1.sort_values(['avg_snr', 'duration'], ascending=False)
        df_snr_template = df_snr_template.loc[df_snr_template['duration'] >= 30].reset_index(drop=True)

    n_rows = min([20, df_snr_template.shape[0]])
    data_dict['snr_template'] = df_snr_template.iloc[0:n_rows]

    qrs, qrs_n, qrs_crop, all_template_beats = PeakDetection.create_beat_template_snr_bouts(df_snr=data_dict['snr_template'],
                                                                              ecg_signal=ecg_signal,
                                                                              sample_rate=sample_rate,
                                                                              peaks=data_dict['original'][idx_key],
                                                                              window_size=correl_window_size*2,
                                                                              remove_outlier_amp=True,
                                                                              use_median=False,
                                                                              remove_mean=True,
                                                                              quiet=quiet,
                                                                              centre_on_absmax_peak=abs_value_correction,
                                                                              plot_data=False)
    data_dict['qrs_template'] = qrs
    data_dict['qrs_template_crop'] = qrs_crop
    data_dict['all_template_beats'] = all_template_beats
    
    start_peak_idx = Processing.find_first_highly_correlated_beat(ecg_signal=ecg_signal,
                                                                  peaks=data_dict['original'][idx_key],
                                                                  template=data_dict['qrs_template_crop'],
                                                                  correl_thresh=correl_thresh,
                                                                  n_consec=5)
    """

    # creation of average QRS template using rolling template that is highly correlated with n_consec beats from the
    # start of the data
    x = Templating.find_first_highly_correlated_beat_notemplate(ecg_signal=ecg_signal,
                                                                peaks=data_dict[last_df][idx_key],
                                                                sample_rate=sample_rate,
                                                                correl_window_size=correl_window_size * 2,
                                                                correl_thresh=correl_thresh,
                                                                n_consec=10)

    start_peak_idx = x[0]
    qrs = x[1]

    # crops QRS template for use in subsequent analyses
    qrs_crop = Templating.crop_template(template=qrs, sample_rate=sample_rate,
                                        window_size=correl_window_size,
                                        centre_on_absmax_peak=abs_value_correction)

    all_template_beats = x[2]

    data_dict['qrs_template'] = qrs
    data_dict['qrs_template_crop'] = qrs_crop
    data_dict['all_template_beats'] = all_template_beats

    # crops to first valid peak (based on high correlation to template) -------
    data_dict['original_fail'] = data_dict[last_df].loc[:start_peak_idx]
    data_dict[last_df] = data_dict[last_df].loc[start_peak_idx:]
    data_dict[last_df].reset_index(drop=True, inplace=True)

    # change in beat-to-beat HR. value at i is difference in HR[i+1] - HR[i] -------
    data_dict[last_df]['delta_hr'] = HR_Calculation.calculate_delta_hr(hr=list(data_dict[last_df]['hr']),
                                                                       absolute_hr=True)

    # optional: remove peaks based on SNR bouts ------------------------
    df, df_rem = PeakScreening.remove_peaks_during_bouts(df_peaks=data_dict[last_df],
                                                         idx_colname=idx_key,
                                                         stage_name='snr_fail',
                                                         dfs_events_to_remove=(df_snr_ignore),
                                                         quiet=quiet)

    data_dict['snr_pass'] = df
    data_dict['snr_fail'] = df_rem

    df_last = "snr_pass"  # key for most recently processed df

    # optional: remove peaks during nonwear ---------------------------
    df, df_rem = PeakScreening.remove_peaks_during_bouts(df_peaks=data_dict[df_last],
                                                         stage_name='nonwear',
                                                         idx_colname=idx_key,
                                                         dfs_events_to_remove=(df_nw),
                                                         quiet=quiet)

    data_dict['wear'] = df
    data_dict['nonwear'] = df_rem  # not needed for anything else

    df_last = 'wear'  # key for most recently processed df

    # beat-to-beat heart rate and change in HR ---------------------------

    # re-calculates values since they would change after low-quality data/nonwear peaks removed
    data_dict[df_last]['hr'] = HR_Calculation.calculate_inst_hr(sample_rate=sample_rate,
                                                                peak_colname=idx_key,
                                                                df_peaks=data_dict[df_last],
                                                                min_quality=min_snr_quality,
                                                                max_break=3,
                                                                quiet=quiet)

    data_dict[df_last]['delta_hr'] = HR_Calculation.calculate_delta_hr(hr=list(data_dict[df_last]['hr']),
                                                                       absolute_hr=True)

    # Logical part of algorithm =====================================================

    # number of samples to include in general correlation checks (peak +- window_samples)
    window_samples = int(sample_rate * correl_window_size)

    # error margin when checking expected location of peak (peak +- location_samples)
    location_samples = int(sample_rate * location_margin)

    valid_idx = [start_peak_idx]  # indexes in df of valid peaks (pass all screening)
    invalid_idx = []  # indexes in df of invalid peaks (fail screening)

    df = data_dict[df_last].copy()  # copies last processed df so it can be edited

    processing_values = [[df.iloc[0].start_time,  # timestamp
                          df.iloc[0].idx,  # beat index
                          False,  # premature checked
                          'Not checked',  # premature result
                          None,  # premature check peaks
                          None,  # premature check peaks correlations
                          None,  # max correlation
                          True,  # valid correlation
                          round(ecg_signal[df.iloc[0][idx_key]], 1),  # peak amplitude
                          ecg_signal[df.iloc[0][idx_key]] > amplitude_thresh,  # valid amplitude
                          False,  # location checked
                          None,  # location error
                          None,  # valid location
                          True]  # valid beat
                         ]

    # prem_beat_qrs = Processing.crop_template(template=qrs, sample_rate=sample_rate, window_size=premature_beat_correl_window_size, centre_on_absmax_peak=abs_value_correction)
    prem_beat_qrs_win_size = int(sample_rate * premature_beat_correl_window_size)

    # loops through all beats ----------------
    # skips first beat; this one has already been classified as valid
    for row in tqdm(df.iloc[1:].itertuples(), total=df.shape[0]-1) if not quiet else df.iloc[1:].itertuples():

        premature_checked = False  # boolean: was beat checked due to it potentially being premature?
        premature_str = "Not checked"  # str related to premature_checked
        prem_check_peaks = None
        checked_r_vals = None  # correlation values of prem_check_peaks
        r_next_beat = None  # correlation values for next beat
        valid_r_next = None  # boolean: r_next_beat above correlation threshold?
        loc_checked = False  # boolean: is location checked?
        loc_err = None  # if beat's location is corrected, location error in samples
        valid_loc = None  # boolean: location valid?
        valid_amp = None

        try:
            peak_amp = round(ecg_signal[df.loc[row.Index + 1][idx_key]], 1)
        except KeyError:
            peak_amp = round(ecg_signal[df.loc[row.Index][idx_key]], 1)

        if row.Index not in invalid_idx:

            continue_check = True

            # if delta HR below threshold, beat is valid. No further analysis on this beat --------
            if row.delta_hr <= delta_hr_thresh:
                valid_idx.append(row.Index)

            # if delta HR above threshold, runs further checks ---------------------------
            if row.delta_hr > delta_hr_thresh:
                valid_idx.append(row.Index)

                # based on how delta HR is calculated, if there's a quick positive change in HR,
                # the 'next_beat' beat is the problem beat
                curr_beat = df.loc[row.Index]  # current beat's data row
                next_beat = df.loc[row.Index + 1]  # next beat's data row
                last_valid_beat = df.loc[valid_idx[-1]]  # last valid beat's data row

                # ECG windows ---------------------
                # ecg data window for current beat
                curr_data = ecg_signal[curr_beat[idx_key] - window_samples:curr_beat[idx_key] + window_samples]
                # ecg data window for next beat
                next_data = ecg_signal[next_beat[idx_key] - window_samples:next_beat[idx_key] + window_samples]

                # checks for premature beat and adjusts peak if appropriate --------
                premature_checked = True

                # window of last valid QRS complex
                # can use instead of average QRS to help with ECG changes over time
                prem_beat_qrs = ecg_signal[last_valid_beat[idx_key] - prem_beat_qrs_win_size:
                                           last_valid_beat[idx_key] + prem_beat_qrs_win_size]

                x = Processing.correct_premature_beat(ecg_signal=ecg_signal,
                                                      sample_rate=sample_rate,
                                                      peak_idx=next_beat.idx,
                                                      next_peak_idx=df.loc[next_beat.name+1][idx_key],
                                                      qrs_template=prem_beat_qrs,
                                                      search_window=premature_search_window,
                                                      correl_window=premature_beat_correl_window_size,
                                                      volt_thresh=amplitude_thresh,
                                                      use_abs_value=abs_value_correction)
                new_peak, new_peak_r, checked_r_vals, prem_check_peaks = x[0], x[1], x[2], x[3]

                # recalculates some values using new-found peak if new peak is not the next_beat peak already
                if new_peak != next_beat.idx:
                    # HR between idx+1 and idx+2 with new idx+1 value
                    try:
                        new_hr_next = 60 / (df.loc[next_beat.name + 1]['start_time'] -
                                            raw_timestamps[new_peak]).total_seconds()
                    except ZeroDivisionError:
                        print(df.loc[next_beat.name + 1]['start_time'], raw_timestamps[new_peak], 'issue')

                    # current beat's HR with new peak for next_beat
                    new_hr_curr = 60 / (raw_timestamps[new_peak] - curr_beat.start_time).total_seconds()

                    # delta HR for current beat using new peak
                    new_delta_hr_curr = new_hr_next - new_hr_curr

                    # next_beat valid if new delta HR below threshold
                    if new_delta_hr_curr < delta_hr_thresh:
                        valid_idx.append(row.Index + 1)

                        # adjusts values for next beat's data
                        df.loc[next_beat.name, "start_time"] = raw_timestamps[new_peak]
                        df.loc[next_beat.name, "idx"] = new_peak
                        df.loc[next_beat.name, 'height'] = round(ecg_signal[new_peak], 1)
                        df.loc[next_beat.name, "hr"] = new_hr_next
                        df.loc[next_beat.name, 'delta_hr'] = new_delta_hr_curr

                        # adjusts current beat's data
                        df.loc[curr_beat.name, 'delta_hr'] = new_delta_hr_curr
                        df.loc[curr_beat.name, 'hr'] = new_hr_curr

                        premature_str = "Changed"

                        ##
                        next_data = ecg_signal[new_peak - window_samples:new_peak + window_samples]

                        continue_check = False

                    if new_delta_hr_curr >= delta_hr_thresh:
                        premature_str = 'Not changed'

                # additional processing if premature beat detection did not change anything -------------------------
                if continue_check:
                    # correlation check ------------------------------------------------------------

                    # option 1: correlates current beat with next beat
                    if correl_method in ['neighbour', 'neighbor']:
                        r_next_beat = pearsonr(curr_data, next_data)[0]  # correlation between beats
                        valid_r_next = (r_next_beat >= correl_thresh)

                    # option 2: correlates current beat with template
                    if correl_method == 'template':
                        r_next_beat = pearsonr(qrs_crop, next_data)[0]  # correlation with template
                        valid_r_next = (r_next_beat >= correl_thresh)

                        if pd.to_datetime("2022-11-22 15:29:19") <= curr_beat.start_time <= pd.to_datetime("2022-11-22 15:29:20"):
                            fig, ax = plt.subplots(1)
                            ax.plot(curr_data, label=f'curr {curr_beat.start_time}')
                            ax.plot(next_data, label=f"next {next_beat.start_time}")
                            ax.plot(qrs_crop, label='template')
                            ax.legend()
                            print(f"rtemplate = {r_next_beat:.3f}, rneighbour = {pearsonr(curr_data, next_data)[0]:.3f}")

                    valid_amp = True

                    # if valid correlation, checks voltage amplitude
                    if valid_r_next:
                        valid_amp = abs(peak_amp) >= amplitude_thresh

                        # if valid voltage amplitude, peak is valid
                        if valid_amp:
                            valid_idx.append(row.Index + 1)

                    # if correlation test was failed OR if correlation test past but amplitude test failed
                    if not valid_r_next or not valid_amp:

                        loc_checked = True

                        # checks location --------------
                        # checks to see if next beat falls into the expected location
                        # expected location: current beat location + last RR interval +- margin of error
                        last_beat_interval = (curr_beat[idx_key] - last_valid_beat[idx_key])
                        expected_location = curr_beat[idx_key] + last_beat_interval
                        loc_err = abs(next_beat[idx_key] - expected_location)
                        valid_loc = loc_err < location_samples

                        # if wrong location --> reject
                        if not valid_loc:
                            invalid_idx.append(row.Index + 1)

                        # if right location --> accept
                        if valid_loc:
                            valid_idx.append(row.Index + 1)

            new_data_row = [row.start_time,
                            row.idx if not use_corrected_peaks else row.idx_corr,
                            premature_checked,
                            premature_str,
                            prem_check_peaks,
                            checked_r_vals,
                            r_next_beat, valid_r_next,
                            peak_amp, valid_amp, loc_checked,
                            loc_err, valid_loc,
                            row.Index + 1 == valid_idx[-1]]

            processing_values.append(new_data_row)

    df_columns = ['start_time', 'idx', 'premature_checked', 'premature_result', 'premature_check_peaks',
                  'premature_check_r', 'r', 'valid_r', 'height', 'valid_height', 'loc_check',
                  'loc_err', 'valid_loc', 'valid_beat']
    data_dict['detailed_check'] = pd.DataFrame(processing_values, columns=df_columns)

    data_dict['detailed_pass'] = df.loc[sorted(set(valid_idx))]
    data_dict['detailed_pass'].reset_index(drop=True, inplace=True)
    data_dict['detailed_pass']['hr'] = HR_Calculation.calculate_inst_hr(sample_rate=sample_rate,
                                                                        df_peaks=data_dict['detailed_pass'],
                                                                        peak_colname=idx_key,
                                                                        min_quality=min_snr_quality,
                                                                        max_break=3,
                                                                        quiet=quiet)

    data_dict['detailed_fail'] = data_dict[df_last].loc[sorted(set(invalid_idx))]

    data_dict['epoch'] = HR_Calculation.jumping_epoch_hr(sample_rate=sample_rate,
                                                         timestamps=raw_timestamps,
                                                         epoch_len=epoch_len,
                                                         peaks=data_dict['detailed_pass'][idx_key])

    # Orphanidou et al. 2015 signal quality processing ==========================================

    # Runs analysis if processed files not given
    if orphanidou_dfs is None:
        orph_data = Orphanidou.run_orphanidou(signal=ecg_signal,
                                              sample_rate=sample_rate,
                                              peaks=data_dict['detailed_pass'][idx_key],
                                              timestamps=raw_timestamps,
                                              window_size=.2,
                                              override_rr_ratio=False,
                                              override_r_value=.8,
                                              epoch_len=epoch_len,
                                              volt_thresh=amplitude_thresh,
                                              corr_thresh=.66,  # hardcoded threshold: from Orphanidou paper
                                              rr_thresh=3,
                                              rr_ratio_thresh=3,
                                              quiet=quiet)

        data_dict['orph_epochs'] = orph_data['orph_epochs']
        data_dict['orph_valid'] = orph_data['orph_valid']
        data_dict['orph_invalid'] = orph_data['orph_invalid']
        data_dict['orph_bout'] = orph_data['orph_bout']

        data_dict['orph_valid']['hr'] = HR_Calculation.calculate_inst_hr(sample_rate=sample_rate,
                                                                         df_peaks=data_dict['orph_valid'],
                                                                         peak_colname='idx',  # no 'idx_corr' in this df
                                                                         min_quality=3,  # redudant if < 3
                                                                         max_break=3,
                                                                         quiet=quiet)

    # variable redeclaration if processed files given
    if orphanidou_dfs is not None:
        data_dict['orph_epochs'] = orphanidou_dfs['orph_epochs']
        data_dict['orph_valid'] = orphanidou_dfs['orph_valid']
        data_dict['orph_invalid'] = orphanidou_dfs['orph_invalid']
        data_dict['orph_bout'] = orphanidou_dfs['orph_bout']

    return data_dict


""" ===================================== SAMPLE RUN ===================================== """

data = run_algorithm(ecg_signal=filt, raw_timestamps=ecg.ts, epoch_len=15,
                     sample_rate=ecg.fs,
                     use_corrected_peaks=True,
                     abs_value_correction=False,
                     correl_window_size=.2, correl_thresh=.66,
                     correl_method='neighbour',  # ['template', 'neighbour']
                     amplitude_thresh=100,  # 250,
                     delta_hr_thresh=15,  # 20
                     location_margin=.1, premature_beat_correl_window_size=.125, premature_search_window=.4,
                     min_snr_quality=2, df_snr_ignore=ecg.df_snr_ignore, df_snr_q1=ecg.df_snr_q1,
                     df_nw=ecg.df_nw,
                     orphanidou_dfs=dfs_orph,
                     quiet=False)


# Orphanidoiu.export_orphanidou_dfs(full_id=full_id, data_dict=data, use_keys=['orph_epochs', 'orph_valid', 'orph_invalid', 'orph_bout'], root_dir="W:/NiMBaLWEAR/OND09/analytics/ecg/signal_quality/orphanidou/dev/")

fig = plot_results(data_dict=data, ecg_signal=filt, ecg_timestamps=ecg.ts, subj=full_id,
                   peak_cols=(
                              'original_fail', 'original_corr',
                              # 'snr_pass', 'snr_fail',
                              'wear',
                              'detailed_pass', 'detailed_fail',
                              'orph_valid', # 'orph_invalid'
                              ),
                   hr_cols=[
                            # 'original_corr', 'wear',
                            'orph_valid', 'orph_epochs'
                            ],
                   orphanidou_bouts=data['orph_epochs'].loc[data['orph_epochs']['valid_period'].isin([False])],
                   smital_quality=ecg.df_snr_hr.loc[ecg.df_snr_hr['quality_use'] == 3],
                   smital_raw=ecg.snr, df_nw=ecg.df_nw, ds_ratio=1,
                   corrected_peaks=True,
                   hr_markers=True)

fig.axes[0].axvline(pd.to_datetime('2022-11-22 15:29:19.6'), color='red', linestyle='dashed')
# fig.axes[1].set_ylim(30, 250)
# fig.axes[-1].set_xlim(pd.to_datetime("2022-11-22 15:29:17"), pd.to_datetime("2022-11-22 15:29:23"))
# fig.axes[0].set_ylim(-500, 1250)
fig.axes[-1].xaxis.set_major_formatter(xfmt)
# for row in data['snr_template'].itertuples():
#     fig.axes[0].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, color='green', alpha=.25)

fig.axes[0].axvline(ecg.ts[4440899], color='red')

"""
data['orph_epochs'].loc[(~data['orph_epochs']['valid_rr_ratio']) & (data['orph_epochs']['correlation'] >= .8), 'valid_period'] = True

from ECG.Processing import row_by_timestamp
from ECG.OrphanidouSignalQuality import plot_orphanidou_epochs
r = row_by_timestamp(df=data['orph_epochs'], n_rows=2, timestamp="2022-11-22 15:29:05")
r.reset_index(drop=True, inplace=True)
fig2 = plot_orphanidou_epochs(df=r, ecg_signal=filt, sample_rate=ecg.fs, timestamps=ecg.ts, epoch_len=15)
"""
