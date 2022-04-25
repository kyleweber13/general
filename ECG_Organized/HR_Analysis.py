import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.%f")
import nwdata
import pickle
import Filtering
import nwecg.ecg_quality as ecg_quality
from datetime import timedelta as td
import neurokit2 as nk
import scipy.stats
import Run_FFT
import peakutils


""" ================================================== DATA IMPORT ================================================ """


def import_ecg_file(subj, highpass_f=1,
                    filepath_fmt="W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_{}_01_BF36_Chest.edf"):
    """Imports Bittium Faros EDF file. Runs _-25Hz BP filter.

       argument:
       -subj: str for which subject's data to import
       -filepath_fmt: fullpathway with {} for subj to get passed in

       Returns ECG object, sample rate, and filtered data.
    """

    ecg = nwdata.NWData()
    ecg.import_edf(file_path=filepath_fmt.format(subj), quiet=False)
    fs = ecg.signal_headers[ecg.get_signal_index("ECG")]['sample_rate']

    print(f"-Running {highpass_f}-25Hz bandpass filter...")
    filt = Filtering.filter_signal(data=ecg.signals[ecg.get_signal_index("ECG")], sample_f=fs,
                                   low_f=highpass_f, high_f=25, filter_order=5, filter_type='bandpass')

    return ecg, fs, filt


def import_snr_pickle(subj, snr_folder="W:/NiMBaLWEAR/OND09/analytics/ecg/smital/snr_timeseries/",
                      snr_file="OND09_{}_01_BF36_Chest.pickle"):
    """Imports pickled SNR data file (pickled output from RunSmital.py)

        arguments:
        -subj: subject ID that fits with formatted requirement given in snr_file
        -snr_folder: pathway to folder containing snr_file
        -snr_file: pickle file with {} formatting for ID given in subj

        returns:
        -time series SNR data
    """

    print("\nImporting pickled SNR file...")

    snr_file = snr_folder + snr_file.format(subj)
    f = open(snr_file, 'rb')
    snr = pickle.load(f)
    f.close()

    return snr


""" =============================================== QUALITY ANALYSIS ============================================== """


def create_snr_bouts(snr_signal, sample_rate, start_stamp, thresholds=(5, 15), shortest_time=30):
    """Creates SNR bouts based on given SNR thresholds and duration requirements. Formats into DF.

        arguments:
        -snr_signal: output from import_snr_pickle() (time series data)
        -sample_rate: Hz, of ECG/SNR signals
        -start_stamp: timestamp of start of ECG collection
        -thresholds: list/tuple of length 2 for SNR values that differentiate Q3/Q2 and Q2/Q1
            -Thresholds in ascending order
        -shortest_time: shortest duration of a 'bout'

        returns:
        -df containing SNR bouts
    """

    print(f"\n-Bouting SNR with thresholds of {thresholds} dB and minimum event durations of {shortest_time}...")

    snr_bouts = ecg_quality._annotate_SNR(rolling_snr=snr_signal, signal_len=len(snr_signal), thresholds=thresholds,
                                          sample_rate=sample_rate, shortest_time=shortest_time)

    df_snr = pd.DataFrame(snr_bouts.get_annotations())
    df_snr['duration'] = [(row.end_idx - row.start_idx) / fs for row in df_snr.itertuples()]
    df_snr['start_timestamp'] = [start_stamp + td(seconds=row.start_idx / sample_rate) for row in df_snr.itertuples()]
    df_snr['end_timestamp'] = [start_stamp + td(seconds=row.end_idx / sample_rate) for row in df_snr.itertuples()]
    df_snr['quality'] = [row.quality.value for row in df_snr.itertuples()]

    return df_snr


def create_hq_ecg_signal(df_snr, ecg_signal):
    """Uses SNR data to zero-out the ECG signal during periods of poor quality data.

        argumnets:
        -df_snr: output from create_snr_bouts()
        -ecg_signal: raw or filtered signal

        returns:
        -array of zeroed-out ECG data
        -df_snr cropped to only highest quality bouts
    """

    hq_snr_bout_dur = df_snr['duration'].min()

    print(f"\nGenerating ECG timeseries data that only contains signal "
          f"during high quality >= {hq_snr_bout_dur}-second periods...")

    df_snr_use = df_snr.loc[(df_snr['quality'] == 'Q1') & (df_snr['duration'] >= hq_snr_bout_dur)]
    hq = np.zeros(len(filt))
    for row in df_snr_use.itertuples():
        hq[row.start_idx:row.end_idx] = ecg_signal[row.start_idx:row.end_idx]

    return hq, df_snr_use


""" =============================================== PEAK ANALYSIS ================================================= """


def create_beat_template(df_snr, ecg_signal, sample_rate, peaks, plot_data=False):
    """From sections of data specified in df_snr, calculates a 'beat template' of the average heartbeat by
       averaging +- 250ms windows on either side of each given peak..

        arguments:
        -df_snr: output from create_snr_bouts(). RECOMMENDED to crop to only a few high-quality segments of data.
        -ecg_signal: array
        -sample_rate: Hz, of ecg_signal
        -peaks: indexes of peaks corresponding to ecg_signal
        -plot_data: boolean --> slow function if True
            -Plots segments of data specified in df_snr with gaps between segments (time series) on top subplot
             and all heartbeats with the template on the bottom subplot

        returns:
        -QRS template
        -mean peak amplitude
        -SD of peak amplitudes
    """

    if plot_data:
        fig, ax = plt.subplots(2, figsize=(12, 8))

    win_size = int(sample_rate / 4)

    used_peaks_idx = []
    all_beats = []

    # adds all peaks to used_peaks as a list for each signal quality bout
    for row in df_snr.itertuples():
        template_peaks = [i for i in peaks if row.start_idx < i < row.end_idx]

        if plot_data:
            t = np.arange(row.start_idx, row.end_idx)
            ax[0].plot(t/sample_rate, ecg_signal[row.start_idx:row.end_idx], color='black')

        for j in template_peaks:
            used_peaks_idx.append(j)
            # all_beats.append(ecg_signal[j-win_size:j+win_size])

    mean_peak_amp = np.mean(ecg_signal[used_peaks_idx])
    std_peak_amp = np.std(ecg_signal[used_peaks_idx])

    final_peaks = [i for i in used_peaks_idx if mean_peak_amp - std_peak_amp <= ecg_signal[i] <= mean_peak_amp + std_peak_amp]

    for peak in final_peaks:
        all_beats.append(ecg_signal[peak-win_size:peak+win_size])
    mean_final = np.mean(ecg_signal[final_peaks])
    std_final = np.std(ecg_signal[final_peaks])

    c = ['dodgerblue' if mean_peak_amp - std_peak_amp <= ecg_signal[i] <= mean_peak_amp + std_peak_amp
         else 'red' for i in used_peaks_idx]

    if plot_data:
        ax[0].scatter([i/sample_rate for i in used_peaks_idx], ecg_signal[used_peaks_idx],
                      color=c)

    qrs = np.mean(np.array(all_beats), axis=0)

    if plot_data:

        for i, beat in enumerate(all_beats):
            if i > 0:
                ax[1].plot(np.linspace(-.25, .25, len(beat)), beat, color='black')
            if i == 0:
                ax[1].plot(np.linspace(-.25, .25, len(beat)), beat, color='black', label='AllBeats')

        ax[1].plot(np.linspace(-.25, .25, len(qrs)), qrs, color='red', lw=2, label='Template')
        ax[1].axvline(0, color='grey', linestyle='dashed', label='Peaks')
        ax[1].legend()
        ax[1].set_xlabel("Seconds")

        plt.tight_layout()

    return qrs, mean_final, std_final


def remove_low_snr_peaks(snr_data, sample_rate, timestamps, peaks, snr_thresh=20):
    """Function that removes peaks whose surrounding +- 250ms-window averages below given SNR threshold.

        arguments:
        -snr_data: time series SNR signal
        -sample_rate: Hz, of snr_data
        -timestamps: of snr_data
        -peaks: indexes corresponding to detected heartbeats
        -snr_thresh: threshold below which peaks are removed. Default = 20db (~Smital Q1 threshold)

        returns:
        -dataframe where each row has the index, timestamp, and mean SNR value of each peak
    """

    print(f"\nRemoving peaks below {snr_thresh}dB mean SNR in surrounding +- .25 seconds...")

    n_old = len(peaks)

    win_size = int(sample_rate/4)

    df_peaks = pd.DataFrame({"idx": peaks, 'timestamp': timestamps[peaks],
                             'snr': [np.mean(snr_data[i-win_size:i+win_size]) for i in peaks]})

    df_peaks = df_peaks.loc[df_peaks['snr'] >= snr_thresh].reset_index(drop=True)

    n_new = df_peaks.shape[0]

    print(f"-Removed {n_old - n_new} peaks.")

    return df_peaks


def remove_outlier_peak_heights(ecg_signal, df_peaks, min_amp, max_amp):
    """Function that removes peaks that fall outside given amplitude range.

        arguments:
        -ecg_signal: array
        -df_peaks: output from remove_low_snr_peaks()
        -min_amp, max_amp: values marking boundaries of included peaks

        returns:
        -dataframe of peaks that are included
        -dataframe of only removed peaks
    """

    df = df_peaks.copy()

    print(f"\nRemoving peaks outside the range of ({min_amp:.0f}, {max_amp:.0f}) uV...")

    n_original = df.shape[0]

    mask = [min_amp < ecg_signal[row.idx] < max_amp for row in df.itertuples()]

    df = df.loc[mask].reset_index(drop=True)
    df_rem = df_peaks.loc[[not i for i in mask]].reset_index(drop=True)

    n_new = df.shape[0]

    print(f"-Removed {n_original - n_new} peaks.")

    return df, df_rem


def screen_peaks(peaks, sample_rate, corr_thresh=-1.0, min_peak=None, max_peak=None):
    """Function that accepts/rejects peaks based on each peak's correlation to the QRS template

        arguments:
        -peaks: array of peak indexes
        -sample_rate: Hz, of ecg signal
        -corr_thresh: value between -1 and 1. Peaks with a lower correlation than the threshold are rejected
        -min_peak, max_peak: amplitude thresholds for included peaks

        returns:
        -list of remaining peaks
        -dataframe of calculated values for all peaks
    """

    print(f"\nAnalyzing detection peaks using a peak amplitude threshold of "
          f"{min_peak:.0f}-{max_peak:.0f} and correlation threshold of r={corr_thresh}...")

    n_og_peaks = len(peaks)
    peaks2 = [0]
    min_dur = 60/max_hr
    skip_peak = [None]
    beat_hr = [60/((peaks[1] - peaks[0])/fs)]
    win_size = int(sample_rate/4)

    peak_amps = []
    r_vals = []
    good_peaks = []
    hr_vals = []
    hr_changes = []
    time_diffs = []
    snr_vals = []
    skip_peak_vals = []

    for i, peak in enumerate(peaks[:-1]):

        if i != skip_peak[-1] and min_peak <= hq_filt[peak] <= max_peak:
            prev_peak = peaks2[-1]

            # prev_beat = hq_filt[prev_peak-win_size:prev_peak+win_size]
            curr_beat = hq_filt[peak-win_size:peak+win_size]
            snr_val = np.mean(snr[peak-win_size:peak+win_size])
            snr_vals.append(snr_val)

            peak_amp = hq_filt[peak]
            peak_amps.append(peak_amp)

            if i > 0:
                r = scipy.stats.pearsonr(curr_beat, qrs)[0]
            if i == 0:
                r = 0
            r_vals.append(r)

            dt = (peak - prev_peak) / fs
            time_diffs.append(dt)

            hr = 60/dt
            hr_vals.append(hr)

            hr_change = np.abs(100*(hr - beat_hr[-1])/beat_hr[-1])
            hr_changes.append(hr_change)

            # print(f"Peak #{i}, {t[i]}, dt={dt:.3f}s, HR={hr:.1f}bpm ({hr_change:.1f}%), r={r:.3f}")

            good_peak = False
            if dt >= min_dur:
                good_peak = True

                if r < corr_thresh:
                    good_peak = False

                # if amp_thresh is not None:
                #    if peak_amp < amp_thresh:
                #        good_peak = False

                if good_peak:
                    peaks2.append(peak)
                    beat_hr.append(hr)

            if not good_peak:
                skip_peak.append(i + 1)
                # skip_peak.append(i)
                skip_peak_vals.append(peaks[i + 1])

            good_peaks.append(good_peak)

    peaks2.pop(0)
    beat_hr.pop(0)
    skip_peak.pop(0)
    print(f"-{len(peaks2)}/{len(peaks)} remain (removed {len(peaks) - len(peaks2)} beats "
          f"[{100 * (len(peaks) - len(peaks2)) / len(peaks):.1f}%]).")

    # skip_peak_vals = peaks[skip_peak]
    # print(len(peaks2)+len(skip_peak), len(good_peaks), len(time_diffs), len(hr_vals), len(hr_changes), len(r_vals))
    df_out = pd.DataFrame({"idx": sorted(peaks2 + skip_peak_vals), 'timestamp': t[sorted(peaks2 + skip_peak_vals)],
                           'valid': good_peaks, 'time_diff': time_diffs, 'snr': snr_vals,
                           'hr': hr_vals, 'hr_change': hr_changes, 'r': r_vals})

    return peaks2, df_out


def screen_peaks_v2(ecg_signal, peaks, max_hr, sample_rate, qrs_temp, timestamps,
                    corr_thresh=-1.0, min_peak=None, max_peak=None):

    print(f"\nAnalyzing detection peaks using a peak amplitude threshold of "
          f"{min_peak:.0f}-{max_peak:.0f} and correlation threshold of r={corr_thresh}...")

    win_size = int(sample_rate/4)
    peaks2 = [0]
    min_dur = 60/max_hr
    skip_peak = [None]
    beat_hr = [60/((peaks[1] - peaks[0])/sample_rate)]

    peak_amps = []
    r_vals = []
    good_peaks = []
    hr_vals = []
    hr_changes = []
    time_diffs = []
    skip_peak_vals = []

    for i, peak in enumerate(peaks[:-1]):

        if i != skip_peak[-1] and min_peak <= ecg_signal[peak] <= max_peak:
            prev_peak = peaks2[-1]

            # prev_beat = hq_filt[prev_peak-win_size:prev_peak+win_size]
            curr_beat = ecg_signal[peak-win_size:peak+win_size]

            peak_amp = ecg_signal[peak]
            peak_amps.append(peak_amp)

            if i > 0:
                r = scipy.stats.pearsonr(curr_beat, qrs_temp)[0]
            if i == 0:
                r = 0
            r_vals.append(r)

            dt = (peak - prev_peak) / sample_rate
            time_diffs.append(dt)

            hr = 60/dt
            hr_vals.append(hr)

            hr_change = np.abs(100*(hr - beat_hr[-1])/beat_hr[-1])
            hr_changes.append(hr_change)

            # print(f"Peak #{i}, {t[i]}, dt={dt:.3f}s, HR={hr:.1f}bpm ({hr_change:.1f}%), r={r:.3f}")

            good_peak = False
            if dt >= min_dur:
                good_peak = True

                if r < corr_thresh:
                    good_peak = False

                # if amp_thresh is not None:
                #    if peak_amp < amp_thresh:
                #        good_peak = False

                if good_peak:
                    peaks2.append(peak)
                    beat_hr.append(hr)

            if not good_peak:
                skip_peak.append(i + 1)
                # skip_peak.append(i)
                skip_peak_vals.append(peaks[i + 1])

            good_peaks.append(good_peak)

    peaks2.pop(0)
    beat_hr.pop(0)
    skip_peak.pop(0)
    print(f"-{len(peaks2)}/{len(peaks)} remain (removed {len(peaks) - len(peaks2)} beats "
          f"[{100 * (len(peaks) - len(peaks2)) / len(peaks):.1f}%]).")

    # skip_peak_vals = peaks[skip_peak]
    # print(len(peaks2)+len(skip_peak), len(good_peaks), len(time_diffs), len(hr_vals), len(hr_changes), len(r_vals))
    df_out = pd.DataFrame({"idx": sorted(peaks2 + skip_peak_vals),
                           'timestamp': timestamps[sorted(peaks2 + skip_peak_vals)],
                           'valid': good_peaks, 'time_diff': time_diffs,
                           'hr': hr_vals, 'hr_change': hr_changes, 'r': r_vals})

    return peaks2, df_out


""" ============================================== HEARTRATE ANALYSIS ============================================= """


def calculate_epoch_hr(peaks, sample_rate, epoch_len=15):
    """Calculates average heart rate over given jumping intervals.

        arguments:
        -peaks: array of peaks
        -sample_rate: Hz, of ecg signal
        -epoch_len: number of seconds over which HR is averaged

        returns:
        -list of heart rates
    """

    print(f"\nCalculating HR in {epoch_len}-second intervals...")

    idx = np.array(peaks)
    idx_window = int(sample_rate*epoch_len)

    start_i = idx[0] + int(epoch_len * sample_rate / 2)
    start_peak_i = np.argwhere(idx >= start_i)[0]

    hrs = [None] * start_peak_i[0]

    for i, peak in enumerate(idx[start_peak_i[0]:]):
        peak_window = []

        if peak + idx_window >= idx[-1]:
            break

        for j in range(i, len(idx)):
            if idx[j] <= peak + idx_window:
                peak_window.append(idx[j])
            if idx[j] > peak + idx_window:
                break

        if len(peak_window) >= epoch_len/2:
            delta_t = (peak_window[-1] - peak_window[0])/sample_rate
            n_beats = len(peak_window)
            hrs.append(60 * (n_beats - 1)/delta_t)

        if len(peak_window) < epoch_len/2:
            hrs.append(None)

    for i in range(len(idx) - len(hrs)):
        hrs.append(None)

    return hrs


def calculate_inst_hr(sample_rate, peaks, max_break=3):
    """Calculates beat-to-beat HR from given peaks (RR intervals)

        arguments:
        -peaks: peak array
        -sample_rate: Hz, of ecg signal
        -max_break: number of seconds consecutive beats can be and still calculate a HR. If RR interval is above this
            value, a None will be given instead of a very low HR

        returns:
        -heart rates

    """

    print("\nCalculating beat-to-beat HR...")

    inst_hr = [60 / ((j - i) / sample_rate) if (j - i) / sample_rate < max_break else None for i, j in
               zip(peaks[:], peaks[1:])]
    inst_hr.append(None)

    return inst_hr


""" ============================================== PLOTTING FUNCTIONS ============================================= """


def plot_data(ecg_signal, ecg_fs, ecg_obj, peaks=(), bad_peaks=(), peak_amp_thresh=(),
              show_accel=False, show_inst_hr=False, epoch_hr=None,
              start_idx=0, end_idx=None, ds_ratio=1, thresholds=(5, 15)):

    n_plots = 2 + show_accel + show_inst_hr

    inst_hr = []

    if end_idx is None:
        end_idx = len(ecg_signal)

    fig, ax = plt.subplots(n_plots, sharex='col', figsize=(12, 8))

    # ECG with peaks --------
    ax[0].set_title("Voltage")
    ax[0].plot(t[start_idx:end_idx:ds_ratio], ecg_signal[start_idx:end_idx:ds_ratio], color='black', zorder=0)
    ax[0].scatter(t[peaks], ecg_signal[peaks], color='dodgerblue', marker='o', zorder=1)
    ax[0].scatter(t[bad_peaks], ecg_signal[bad_peaks], color='red', marker='x', zorder=1)

    if len(peak_amp_thresh) == 2:
        ax[0].axhline(peak_amp_thresh[0], color='red', linestyle='dashed', label='PeakAmpRange')
        ax[0].axhline(peak_amp_thresh[1], color='red', linestyle='dashed', label='PeakAmpRange')

    curr_plot = 1

    # beat-by-beat HR
    if show_inst_hr:
        inst_hr = [60/((j-i)/ecg_fs) if (j-i)/fs < 3 else None for i, j in zip(peaks[:], peaks[1:])]
        ax[curr_plot].plot(t[peaks[:-1]], inst_hr, color='red', label='Inst.', zorder=0)

        if epoch_hr is not None:
            # avg_hr = [np.mean([i for i in inst_hr[i:i+n_beats_avg] if i is not None]) for i in range(int(n_beats_avg/2), int(len(inst_hr) - n_beats_avg/2))]

            # ax[curr_plot].scatter(t[peaks[int(n_beats_avg/2):int(len(inst_hr) - n_beats_avg/2)]], avg_hr, zorder=1, color='black', label=f"{n_beats_avg}-beat avg", s=5)
            ax[curr_plot].scatter(t[peaks], epoch_hr, zorder=1, color='black', label="epoch_hr", s=5)


        ax[curr_plot].set_title("HR Data")
        ax[curr_plot].set_ylabel("HR")
        ax[curr_plot].set_ylim(0, 220)
        ax[curr_plot].grid()
        ax[curr_plot].legend()

        curr_plot += 1

    # accelerometer -------
    if show_accel:
        ax[curr_plot].set_title("Raw Acceleration")
        ax[curr_plot].plot(t[::int(2 * ecg_fs / ecg_obj.signal_headers[ecg_obj.get_signal_index('Accelerometer x')]['sample_rate'])],
                           ecg.signals[ecg_obj.get_signal_index('Accelerometer x')][::2], color='black')
        ax[curr_plot].plot(t[::int(2 * ecg_fs / ecg_obj.signal_headers[ecg_obj.get_signal_index('Accelerometer y')]['sample_rate'])],
                           ecg.signals[ecg_obj.get_signal_index('Accelerometer y')][::2], color='red')
        ax[curr_plot].plot(t[::int(2 * ecg_fs / ecg_obj.signal_headers[ecg_obj.get_signal_index('Accelerometer z')]['sample_rate'])],
                           ecg_obj.signals[ecg_obj.get_signal_index('Accelerometer z')][::2], color='dodgerblue')
        ax[curr_plot].grid()

        curr_plot += 1

    ax[-1].set_title("SNR Data (bars = bouts)")
    ax[-1].plot(t[start_idx:end_idx:ds_ratio], snr[start_idx:end_idx:ds_ratio], color='black')

    c_dict = {"Q3": 'red', 'Q2': 'dodgerblue', 'Q1': 'limegreen'}

    for row in df_snr.loc[(df_snr['start_idx'] >= start_idx) & (df_snr['end_idx'] <= end_idx)].itertuples():
        m = np.mean(snr[row.start_idx:row.end_idx])
        ax[-1].plot([row.start_timestamp, row.end_timestamp], [m, m], lw=4, color=c_dict[row.quality])

    ax[-1].axhline(thresholds[0], color='red', linestyle='dashed')
    ax[-1].axhline(thresholds[1], color='limegreen', linestyle='dashed')
    ax[-1].grid()

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()

    return fig, inst_hr


""" ============================================================================================================== """
""" ================================================= FUNCTION CALLS ============================================= """
""" ============================================================================================================== """

subj = '0113'
age = 65
max_hr = 208 - .7 * age

thresholds = (5, 20)
ecg, fs, filt = import_ecg_file(subj=subj, highpass_f=1)
t = pd.date_range(start=ecg.header['start_datetime'], periods=len(ecg.signals[0]), freq="{}ms".format(1000/fs))
snr = import_snr_pickle(subj=subj)
df_snr = create_snr_bouts(snr_signal=snr, sample_rate=fs, start_stamp=t[0], thresholds=thresholds, shortest_time=30)

hq, df_snr_use = create_hq_ecg_signal(df_snr=df_snr, ecg_signal=filt)

hq_filt = nk.ecg_clean(ecg_signal=filt[:int(1*86400*fs)], sampling_rate=fs, method='neurokit')
# hq_filt = hq_filt * -1

peaks = nk.ecg_findpeaks(ecg_cleaned=hq_filt, sampling_rate=fs, method='neurokit', show=False)['ECG_R_Peaks']

qrs, mean_peak_height, sd_peak_height = create_beat_template(df_snr=df_snr_use.iloc[:5],
                                                             ecg_signal=hq_filt, sample_rate=fs, peaks=peaks, plot_data=True)

df_peaks = remove_low_snr_peaks(snr_data=snr, sample_rate=fs, timestamps=t, peaks=peaks, snr_thresh=5)

df_peaks2, df_peaks2_rem = remove_outlier_peak_heights(ecg_signal=hq_filt, df_peaks=df_peaks,
                                                       min_amp=mean_peak_height/3, max_amp=mean_peak_height*3)

df_peaks2['inst_hr'] = calculate_inst_hr(sample_rate=fs, peaks=df_peaks2['idx'], max_break=3)

df_peaks2['epoch_hr'] = calculate_epoch_hr(peaks=df_peaks2['idx'], sample_rate=fs, epoch_len=15)
plt.plot(df_peaks2['timestamp'], df_peaks2['epoch_hr'])

"""peaks2, df_peaks = screen_peaks_v2(peaks=peaks, corr_thresh=.8, qrs_temp=qrs,
                                   ecg_signal=hq_filt, sample_rate=fs,
                                   max_hr=max_hr, timestamps=t,
                                   min_peak=mean_peak_height - 1.96*sd_peak_height,
                                   max_peak=mean_peak_height + 1.96*sd_peak_height)

use_peaks = df_peaks.loc[df_peaks['valid']]
"""

"""fig, hr = plot_data(ecg_signal=hq_filt, ecg_fs=fs, ecg_obj=ecg,
                    peaks=df_peaks2['idx'], bad_peaks=df_peaks2_rem['idx'],
                    peak_amp_thresh=[mean_peak_height/3, mean_peak_height*3],
                    # bad_peaks=df_peaks.loc[~df_peaks['valid']]['idx'],
                    show_inst_hr=True, epoch_hr=df_peaks2['epoch_hr'],
                    start_idx=0, end_idx=int(86400*fs*2), ds_ratio=1,
                    show_accel=False, thresholds=thresholds)"""

# some kind of outlier detection if inst. HR too far from rolling something

# fig, df_fft, t_fft, Zxx_fft = Run_FFT.plot_stft(data=hq_filt[:int(6*86400*fs):20], sample_rate=fs/20, nperseg_multiplier=5, plot_data=True)

filt = Filtering.filter_signal(data=ecg.signals[ecg.get_signal_index("ECG")], sample_f=fs,
                               low_f=.5, high_f=3, filter_order=4, filter_type='bandpass')

for i in np.arange(0, int(fs*45), int(fs*15)):
    d = filt[i+int(3600*fs):i+int(3600*fs)+int(15*fs)]
    fig, df_fft = Run_FFT.run_fft(data=d, sample_rate=fs, remove_dc=False, highpass=None, show_plot=False)

    df_fft['power'] *= df_fft['power']

    df_fft['roll_power'] = df_fft['power'].rolling(window=int(1/df_fft['freq'].iloc[1])).mean()

    df_fft_hr = df_fft.loc[(df_fft['freq'] >= .5) & (df_fft['freq'] <= max_hr/60 * 1.25)].reset_index(drop=True)

    dom_f = df_fft_hr.loc[df_fft_hr['roll_power'] == df_fft_hr['roll_power'].max()].iloc[0]['freq']
    dom_hr = 60 * dom_f

    fft_peaks = peakutils.indexes(y=df_fft_hr['power'], thres=25, thres_abs=True)

    fig, ax = plt.subplots(2, figsize=(10, 6))
    plt.subplots_adjust(hspace=.3)

    ax[0].plot(np.arange(len(d)) / fs, d, color='dodgerblue')
    ax[0].set_xlabel("Seconds")

    ax[1].plot(df_fft['freq'], df_fft['power'], color='red')
    ax[1].set_xlabel("Hz")
    ax[1].set_ylabel("Power")
    ax[1].scatter(df_fft_hr['freq'].iloc[fft_peaks], df_fft_hr['power'].iloc[fft_peaks], color='limegreen')

    print(dom_hr)


# TO DO
# figure out how to find fundamental freq in FFT data and ignore harmonics
