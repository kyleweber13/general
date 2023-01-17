import pandas as pd
from ECG.ImportFiles import ECG, import_nw_file
from ECG.Processing import remove_peaks_during_bouts, crop_template, find_first_highly_correlated_beat, correct_premature_beat
from ECG.PeakDetection import correct_cn_peak_locations, create_beat_template_snr_bouts, detect_peaks_initial
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.%f")
from ECG.Plotting import *
from ECG.HR_Calculation import calculate_inst_hr, calculate_delta_hr, jumping_epoch_hr
from tqdm import tqdm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from datetime import timedelta
from ECG.OrphanidouSignalQuality import run_orphanidou


def run_algorithm(ecg_signal: list or np.array, raw_timestamps: list or np.array,
                  df_peaks: pd.DataFrame, sample_rate: int or float,
                  start_peak_idx: int, hq_qrs_template: list or np.array = None, epoch_len: int = 15,
                  correl_thresh: float = .7, correl_method: str = 'neighbour',
                  correl_window_size: int or float = .2,
                  premature_beat_correl_window_size: int or float = .125,
                  premature_search_window: int or float = .3,
                  amplitude_thresh: int or float = 250, delta_hr_thresh: int or float = 20,
                  location_margin: float = .1, min_snr_quality: int = 2,
                  df_snr_ignore: pd.DataFrame = pd.DataFrame(), df_nw: pd.DataFrame = pd.DataFrame()):

    print("\nRunning peak screening algorithm ==================================")

    # dictionary with DFs from each processing stage
    data_dict = {'original': df_peaks}  # peaks that you passed in to start

    # crops to first valid peak (based on high correlation to template)
    data_dict['original'].iloc[start_peak_idx:, :].reset_index(drop=True, inplace=True)

    # change in beat-to-beat HR. value at i is difference in HR[i+1] - HR[i]
    data_dict['original']['delta_hr'] = calculate_delta_hr(hr=list(data_dict['original']['hr']), absolute_hr=True)

    df_last = 'original'  # key for last processed df

    # optional: remove peaks based on SNR bouts ------------------------
    df, df_rem = remove_peaks_during_bouts(df_peaks=data_dict['original'], stage_name='low_quality',
                                           dfs_events_to_remove=(df_snr_ignore))

    data_dict['quality_screen'] = df
    data_dict['low_quality'] = df_rem

    df_last = "quality_screen"  # key for last processed df

    # optional: remove peaks during nonwear ---------------------------
    df, df_rem = remove_peaks_during_bouts(df_peaks=data_dict[df_last], stage_name='nonwear',
                                           dfs_events_to_remove=(df_nw))

    data_dict['nonwear_screen'] = df
    data_dict['nonwear'] = df_rem  # not needed for anything else

    df_last = 'nonwear_screen'  # key for last processed df

    # beat-to-beat heart rate and change in HR ---------------------------

    # re-calculates values since they would change after low-quality data/nonwear peaks removed
    data_dict[df_last]['hr'] = calculate_inst_hr(sample_rate=sample_rate,
                                                 peak_colname='idx',
                                                 df_peaks=data_dict[df_last],
                                                 min_quality=min_snr_quality,
                                                 max_break=3)

    data_dict[df_last]['delta_hr'] = calculate_delta_hr(hr=list(data_dict[df_last]['hr']),
                                                        absolute_hr=True)

    # Logical part of algorithm =====================================================

    # number of samples to include in general correlation checks (peak +- window_samples)
    window_samples = int(sample_rate * correl_window_size)

    # error margin when checking expected location of peak (peak +- location_samples)
    location_samples = int(sample_rate * location_margin)

    # ensures given QRS template is the correct length by cropping to window_samples and the peak is centered
    if hq_qrs_template is not None:
        hq_qrs_template = hq_qrs_template[np.argmax(hq_qrs_template) - window_samples:
                                          np.argmax(hq_qrs_template) + window_samples]

        # template for use in correct_premature_beat() since this is designed to use a smaller window
        crop_samples = int(premature_beat_correl_window_size * sample_rate)
        qrs_crop = hq_qrs_template[int(len(hq_qrs_template)/2 - crop_samples):
                                   int(len(hq_qrs_template)/2 + crop_samples)]

    valid_idx = [start_peak_idx]  # indexes in df of valid peaks (pass all screening)
    invalid_idx = []  # indexes in df of invalid peaks (fail screening)

    df = data_dict[df_last].copy()  # copies last processed df so it can be edited

    # processing_values = []  # processing values for each row
    processing_values = [[df.iloc[0].start_time, df.iloc[0].idx, False, 'Not checked', [], None,
                         None, ecg_signal[df.iloc[0].idx], ecg_signal[df.iloc[0].idx] > amplitude_thresh,
                         False, None, None, True]]

    # loops through all beats ----------------
    # skips first beat; this one has already been classified as valid
    for row in tqdm(df.iloc[1:].itertuples(), total=df.shape[0]-1):

        premature_checked, premature_str, prem_check_peaks = False, "Not checked", []
        r_next_beat, valid_r_next = None, None
        loc_checked, loc_err, valid_loc = False, None, None
        valid_amp = None

        try:
            peak_amp = ecg_signal[df.loc[row.Index + 1]['idx']]
        except KeyError:
            peak_amp = ecg_signal[df.loc[row.Index]['idx']]

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
                curr_data = ecg_signal[curr_beat['idx'] - window_samples:curr_beat['idx'] + window_samples]
                # ecg data window for next beat
                next_data = ecg_signal[next_beat['idx'] - window_samples:next_beat['idx'] + window_samples]

                # checks for premature beat and adjusts peak if appropriate
                if hq_qrs_template is not None:

                    premature_checked = True

                    new_peak, new_peak_r, prem_check_peaks = correct_premature_beat(ecg_signal=ecg_signal,
                                                                                    sample_rate=sample_rate,
                                                                                    peak_idx=next_beat.idx,
                                                                                    next_peak_idx=df.loc[next_beat.name+1]['idx'],
                                                                                    qrs_template=qrs_crop,
                                                                                    search_window=premature_search_window,
                                                                                    correl_window=premature_beat_correl_window_size,
                                                                                    volt_thresh=amplitude_thresh)

                    # recalculates some values using new-found peak
                    if new_peak != next_beat.idx:

                        # HR between idx+1 and idx+2 with new idx+1 value
                        new_hr_next = 60 / (df.loc[next_beat.name + 1]['start_time'] -
                                            raw_timestamps[new_peak]).total_seconds()

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
                            df.loc[next_beat.name, 'height'] = ecg_signal[new_peak]
                            df.loc[next_beat.name, "hr"] = new_hr_next
                            df.loc[next_beat.name, 'delta_hr'] = new_delta_hr_curr

                            # adjusts current beat's data
                            df.loc[curr_beat.name, 'delta_hr'] = new_delta_hr_curr
                            df.loc[curr_beat.name, 'hr'] = new_hr_curr

                            premature_str = "Changed"

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
                        r_next_beat = pearsonr(hq_qrs_template, next_data)[0]  # correlation with template
                        valid_r_next = (r_next_beat >= correl_thresh)

                    valid_amp = True

                    # if valid correlation, checks voltage amplitude
                    if valid_r_next:

                        # peak_amp = ecg_signal[row.idx + 1]  ##
                        # peak_amp = ecg_signal[row.Index + 1]
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
                        last_beat_interval = (curr_beat.idx - last_valid_beat.idx)
                        expected_location = curr_beat.idx + last_beat_interval
                        loc_err = abs(next_beat.idx - expected_location)
                        valid_loc = loc_err < location_samples

                        # if wrong location --> reject
                        if not valid_loc:
                            invalid_idx.append(row.Index + 1)

                        # if right location --> accept
                        if valid_loc:
                            valid_idx.append(row.Index + 1)

            processing_values.append([row.start_time, row.idx,
                                      premature_checked, premature_str, r_next_beat,
                                      valid_r_next, peak_amp, valid_amp, loc_checked, loc_err, valid_loc,
                                      row.Index + 1 == valid_idx[-1]])

    data_dict['processed'] = pd.DataFrame(processing_values, columns=['start_time', 'idx', 'premature_check',
                                                                      'premature_check_peaks',
                                                                      'premature_result', 'r', 'valid_r', 'height',
                                                                      'valid_height', 'loc_check',
                                                                      'loc_err', 'valid_loc', 'valid_beat'])

    data_dict['valid'] = df.loc[sorted(set(valid_idx))]
    data_dict['valid'].reset_index(drop=True, inplace=True)
    data_dict['valid']['hr'] = calculate_inst_hr(sample_rate=sample_rate,
                                                 peak_colname='idx',
                                                 df_peaks=data_dict['valid'],
                                                 min_quality=min_snr_quality,
                                                 max_break=3)

    data_dict['invalid'] = data_dict[df_last].loc[invalid_idx]

    data_dict['epoch'] = jumping_epoch_hr(sample_rate=sample_rate, timestamps=raw_timestamps,
                                          epoch_len=epoch_len, peaks=data_dict['valid']['idx'])

    orphanidou = run_orphanidou(signal=filt, peaks=data_dict['valid']['idx'], timestamps=ecg.ts,
                                window_size=.3, sample_rate=ecg.fs, epoch_len=epoch_len,
                                volt_thresh=250, corr_thresh=.66, rr_thresh=3, rr_ratio_thresh=3, quiet=False)
    data_dict['orph_epochs'] = orphanidou['quality']
    data_dict['orph_valid'] = orphanidou['valid_peaks']
    data_dict['orph_invalid'] = orphanidou['invalid_peaks']
    data_dict['orph_bout'] = orphanidou['bout']

    data_dict['orph_valid']['hr'] = calculate_inst_hr(sample_rate=ecg.fs, df_peaks=data_dict['orph_valid'],
                                                      peak_colname='idx', min_quality=3, max_break=3)

    return data_dict


# utilities ============


def plot_results(data_dict, ecg_signal, ecg_timestamps, subj="",
                 peak_cols=('original', 'valid'),
                 hr_cols=('original', 'valid', 'epoch'),
                 orphanidou_bouts=None, smital_quality=None, smital_raw=None):

    color_dict = {"original": 'red', 'quality_screen': 'orange', 'low_quality': 'orange', 'nonwear_screen': 'grey',
                  'nonwear': 'grey', 'processed': 'mediumorchid',
                  'valid': 'dodgerblue', 'invalid': 'dodgerblue', 'epoch': 'black',
                  'orph_valid': 'limegreen', 'orph_invalid': 'limegreen', 'orph_epochs': 'pink'}
    marker_dict = {"original": 'v', 'quality_screen': 'v', 'low_quality': 'x', 'nonwear_screen': 'v',
                   'nonwear': 'x', 'processed': 'v',
                   'valid': 'v', 'invalid': 'x', 'orph_valid': 'v', 'orph_invalid': 'x',
                   'orph_epochs': None, 'epoch': None}
    pairs_dict = {'valid': 'invalid', 'invalid': 'valid',
                  'quality_screen': 'low_quality', 'low_quality': 'quality_screen',
                  'nonwear': 'nonwear_screen', 'nonwear_screen': 'nonwear',
                  'orph_valid': 'orph_invalid', 'orph_invalid': 'orph_valid'}
    plotted = []

    n_subplots = 2
    heights = [1, .66]

    if smital_quality is not None:
        n_subplots += 1
        heights.append(.33)
    if orphanidou_bouts is not None:
        n_subplots += 1
        heights.append(.33 if smital_raw is None else .5)
    if smital_raw is not None and smital_quality is None:
        n_subplots += 1
        heights.append(.5)

    fig, ax = plt.subplots(n_subplots, sharex='col', figsize=(12, 8), gridspec_kw={"height_ratios": heights})

    plt.suptitle(subj)

    ax[0].plot(ecg_timestamps[:len(ecg_signal)], ecg_signal, color='black')

    offset = 0
    for col in peak_cols:

        has_pair = col in pairs_dict
        try:
            pair_plotted = pairs_dict[col] in plotted or col in plotted
        except KeyError:
            pair_plotted = False

        if has_pair and not pair_plotted:
            offset += 200

        if not has_pair:
            offset += 200

        plotted.append(col)

        ax[0].scatter(data_dict[col]['start_time'], ecg_signal[data_dict[col]['idx']] + offset,
                      color=color_dict[col], marker=marker_dict[col],
                      label=f"{col.capitalize()} (n={data_dict[col].shape[0]})")

    ax[0].legend(loc='lower right')

    for col in hr_cols:
        if col not in ['epoch', 'orphanidou']:
            ax[1].plot(data_dict[col]['start_time'], data_dict[col]['hr'], color=color_dict[col], label=col.capitalize())

        if col in ['epoch', 'orphanidou']:
            epoch_len = int((data_dict[col].iloc[1]['start_time'] -
                             data_dict[col].iloc[0]['start_time']).total_seconds())
            df_use = data_dict[col].dropna()
            ax[1].errorbar(df_use['start_time'] + timedelta(seconds=epoch_len/2),
                           df_use['hr'], label=f'{col.capitalize()} ({epoch_len}s)',
                           marker=None, ecolor=color_dict[col], fmt='none',
                           capsize=4, xerr=timedelta(seconds=epoch_len/2)
                           )

    ax[1].grid()
    ax[1].legend(loc='lower right')
    ax[1].set_ylabel("HR")

    if orphanidou_bouts is not None:
        ax[2].set_ylabel("Orphanidou\nInvalid")

        for row in orphanidou_bouts.itertuples():
            ax[2].axvspan(xmin=row.start_time, xmax=row.end_time, ymin=0, ymax=1, alpha=.1,
                          color='red' if not row.valid_period else 'limegreen')
        ax[2].set_yticks([])

    if smital_quality is not None:
        use_ax = ax[2] if orphanidou_bouts is None else ax[3]

        c = {"3": 'red', '2': 'dodgerblue', '1': 'limegreen', '0': 'grey'}
        for row in smital_quality.itertuples():
            use_ax.axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp, ymin=0, ymax=1, alpha=.1,
                           color=c[str(row.quality_use)])

        use_ax.set_ylabel("Smital\nQuality")

        if smital_raw is None:
            use_ax.set_yticks([])

    if smital_raw is not None:
        if smital_quality is not None:
            pass
        if smital_quality is None:
            use_ax = ax[-1]

        use_ax.plot(ecg_timestamps[:len(smital_raw):25], smital_raw[::25], color='black')
        use_ax.grid()

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig


def import_snr_bout_file(filepath: str, min_idx: int = 0, max_idx: int = -1):
    """ Imports signal-to-noise ratio (SNR) bout file from csv and formats column data appropriately.

        arguments:
        -filepath: pathway to SNR bout file

        returns:
        -dataframe
    """

    dtype_cols = {"study_code": str, 'subject_id': str, 'coll_id': str,
                  'start_idx': pd.Int64Dtype(), 'end_idx': pd.Int64Dtype(), 'bout_num': pd.Int64Dtype(),
                  'quality': str, 'avg_snr': float}
    try:
        date_cols = ['start_time', 'end_time']
        df = pd.read_csv(filepath, dtype=dtype_cols, parse_dates=date_cols)
        df['duration'] = [(row.end_time - row.start_time).total_seconds() for row in df.itertuples()]

    except (ValueError, AttributeError):
        date_cols = ['start_timestamp', 'end_timestamp']
        df = pd.read_csv(filepath, dtype=dtype_cols, parse_dates=date_cols)
        df['duration'] = [(row.end_timestamp - row.start_timestamp).total_seconds() for row in df.itertuples()]

    # replaces strings with numeric equivalents for signal qualities
    df['quality_use'] = df['quality'].replace({'ignore': 3, 'full': 1, 'HR': 1})

    df = df.loc[(df['start_idx'] >= min_idx) & (df['end_idx'] <= max_idx if max_idx != -1 else df['end_idx'].max())]

    return df


""" ===================================== SAMPLE RUN ===================================== """

if __name__ == " __main__":

    # data import --------------
    full_id = 'OND09_SBH0300'
    coll_id = '01'

    # OND09
    ecg = ECG(edf_folder="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/",
              ecg_fname=f"{full_id}_{coll_id}_BF36_Chest.edf",
              bandpass=(1.5, 25), thresholds=(5, 18),
              smital_edf_fname="")

    # data cropping for faster testing
    n_hours = 48
    end_idx = int(n_hours*3600*ecg.ecg.signal_headers[ecg.ecg.get_signal_index("ECG")]['sample_rate'])
    # end_idx = len(ecg.ecg.signals[ecg.ecg.get_signal_index("ECG")])
    ecg_signal = ecg.ecg.signals[0][:end_idx]
    ecg.ecg.filt = ecg.ecg.filt[:end_idx]
    filt = ecg.ecg.filt.copy()
    ecg.snr = ecg.snr[:end_idx]

    # nonwear data import
    try:
        ecg.df_nw = import_nw_file(filepath=f"C:/Users/ksweber/Desktop/ECG_nonwear_dev/FinalBouts_SNR/{full_id}_01_BF36_Chest_NONWEAR.csv",
                                   start_timestamp=ecg.start_stamp, sample_rate=ecg.fs)
    except FileNotFoundError:
        ecg.df_nw = pd.DataFrame(columns=['start_timestamp', 'end_timestamp'])

    # initial peak detection -----------------
    df_peaks1 = detect_peaks_initial(ecg_signal=filt, sample_rate=ecg.fs, timestamps=ecg.ts,
                                     correct_locations=True, min_height=None,
                                     correction_windowsize=.3, absolute_peaks=False)

    data = run_algorithm(ecg_signal=filt, raw_timestamps=ecg.ts, epoch_len=15,
                         df_peaks=df_peaks1, sample_rate=ecg.fs, start_peak_idx=1,
                         hq_qrs_template=None, correl_window_size=.2, correl_thresh=.66, correl_method='neighbour',
                         amplitude_thresh=100,  # 250,
                         delta_hr_thresh=15,  # 20
                         location_margin=.1, premature_beat_correl_window_size=.125, premature_search_window=.4,
                         min_snr_quality=2, df_snr_ignore=pd.DataFrame(), df_nw=ecg.df_nw)

    fig = plot_results(data_dict=data, ecg_signal=filt, ecg_timestamps=ecg.ts, subj=full_id,
                       peak_cols=('original', 'orph_valid', 'orph_invalid'),
                       hr_cols=['original', 'orph_epochs'],
                       orphanidou_bouts=data['orph_bout'].loc[~data['orph_bout']['valid_period']],
                       )
