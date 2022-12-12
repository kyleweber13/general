import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm


class CheckQuality:
    """Class method that implements the Orphanidou ECG signal quality assessment algorithm on raw ECG data.

       Orphanidou, C. et al. (2015). Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
       Derivation and Applications to Wireless Monitoring. IEEE Journal of Biomedical and Health Informatics.
       19(3). 832-838.
    """

    def __init__(self, ecg_signal, sample_rate, peaks, window_samples=None,
                 voltage_thresh=250, rr_ratio_thresh=2.5, corr_thresh=.66, rr_thresh=3, epoch_len=15,
                 ):
        """Initialization method.

        :param
        -ecg_object: EcgData class instance created by ImportEDF script
        -random_data: runs algorithm on randomly-generated section of data; False by default.
                      Takes priority over start_index.
        -start_index: index for windowing data; 0 by default
        -epoch_len: window length in seconds over which algorithm is run; 15 seconds by default
        """

        self.voltage_thresh = voltage_thresh
        self.epoch_len = epoch_len
        self.fs = sample_rate
        self.peaks = peaks
        self.window_samples = window_samples

        self.ecg_signal = ecg_signal
        self.qrs_fail = False

        self.data_dict = {"valid_period": False,
                          'n_beats': 0,
                          'peak_idx': [],
                          "valid_hr": False, "hr": None,
                          "valid_max_rr": False, "max_rr": None,
                          "valid_rr_ratio": False, "rr_ratio": None,
                          "valid_voltage": False, "voltage_range": None,
                          "valid_correlation": False, "correlation": None,
                          'template': []}

        # prep_data parameters
        self.r_peaks = peaks
        self.rr_sd = None
        self.removed_peak = []
        self.enough_beats = True
        self.hr = 0
        self.delta_rr = []
        self.removal_indexes = []
        self.rr_ratio = None
        self.volt_range = 0
        self.average_qrs = []

        # apply_rules parameters
        self.rr_ratio_thresh = rr_ratio_thresh
        self.valid_hr = None
        self.valid_rr = None
        self.rr_thresh = rr_thresh
        self.valid_ratio = None
        self.valid_range = None
        self.valid_corr = None
        self.corr_thresh = corr_thresh
        self.rules_passed = None

        # adaptive_filter parameters
        self.median_rr = None
        self.ecg_windowed = []
        self.average_r = 0

        # calculate_correlation parameters
        self.beat_ppmc = []
        self.valid_period = None

        """RUNS METHODS"""
        # Peak detection and basic outcome measures
        self.prep_data()

        # Runs rules check if enough peaks found
        if self.enough_beats:
            self.adaptive_filter()
            self.calculate_correlation()
            self.apply_rules()

    def prep_data(self):
        """Function that:
        -Calculates RR intervals
        -Removes first peak if it is within median RR interval / 2 from start of window
        -Calculates average HR in the window
        -Determines if there are enough beats in the window to indicate a possible valid period
        """

        # Checks to see if there are enough potential peaks to correspond to correct HR range ------------------------
        # Requires number of beats in window that corresponds to ~40 bpm to continue
        # Prevents the math in the self.hr calculation from returning "valid" numbers with too few beats
        # i.e. 3 beats in 3 seconds (HR = 60bpm) but nothing detected for rest of epoch
        if len(self.r_peaks) >= np.floor(40 / 60 * self.epoch_len):
            self.enough_beats = True

            n_beats = len(self.r_peaks)  # number of beats in window
            delta_t = (self.r_peaks[-1] - self.r_peaks[0]) / self.fs  # time between first and last beat, seconds
            self.hr = 60 * (n_beats - 1) / delta_t  # average HR, bpm

        # Stops function if not enough peaks found to be a potential valid period
        # Threshold corresponds to number of beats in the window for a HR of 40 bpm
        if len(self.r_peaks) < np.floor(40 / 60 * self.epoch_len):
            self.enough_beats = False
            self.valid_period = False
            return

        # Calculates RR intervals in seconds -------------------------------------------------------------------------
        for peak1, peak2 in zip(self.r_peaks[:], self.r_peaks[1:]):
            rr_interval = (peak2 - peak1) / self.fs
            self.delta_rr.append(rr_interval)

        # Approach 1: median RR characteristics ----------------------------------------------------------------------
        # Calculates median RR-interval in seconds
        median_rr = np.median(self.delta_rr)

        # SD of RR intervals in ms
        self.rr_sd = np.std(self.delta_rr) * 1000

        # Converts median_rr to samples
        self.median_rr = int(median_rr * self.fs)

        # Removes any peak too close to start/end of data section: affects windowing later on ------------------------
        # Peak removed if within median_rr/2 samples of start of window
        # Peak removed if within median_rr/2 samples of end of window
        epoch_samples = int(self.epoch_len * self.fs)
        for i, peak in enumerate(self.r_peaks):

            if self.window_samples is None:
                if peak < (self.median_rr / 2 + 1) or (epoch_samples - peak) < (self.median_rr / 2 + 1):
                    self.removed_peak.append(self.r_peaks.pop(i))
                    self.removal_indexes.append(i)

            if self.window_samples is not None:
                if (peak < self.window_samples + 1) or (epoch_samples - peak < self.window_samples + 1):
                    # self.removed_peak.append(self.r_peaks.pop(i))
                    self.removed_peak.append(peak)
                    self.r_peaks.remove(peak)
                    self.removal_indexes.append(i)

        # Removes RR intervals corresponding to
        if len(self.removal_indexes) != 0:
            self.delta_rr = [self.delta_rr[i] for i in range(len(self.r_peaks)) if i not in self.removal_indexes]

        # Calculates range of ECG voltage ----------------------------------------------------------------------------
        self.volt_range = max(self.ecg_signal) - min(self.ecg_signal)

    def adaptive_filter(self):
        """Method that runs an adaptive filter that generates the "average" QRS template for the window of data.

        - Calculates the median RR interval
        - Generates a sub-window around each peak, +/- RR interval/2 in width
        - Deletes the final beat sub-window if it is too close to end of data window
        - Calculates the "average" QRS template for the window
        """

        # Approach 1: calculates median RR-interval in seconds  -------------------------------------------------------
        # See previous method

        # Approach 2: takes a window around each detected R-peak of width peak +/- median_rr/2 ------------------------
        for peak in self.r_peaks:
            if self.window_samples is None:
                window = self.ecg_signal[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            else:
                window = self.ecg_signal[peak - self.window_samples:peak + self.window_samples]

            self.ecg_windowed.append(window)  # Adds window to list of windows

        # Approach 3: determine average QRS template ------------------------------------------------------------------
        self.ecg_windowed = np.asarray(self.ecg_windowed)[1:]  # Converts list to np.array; omits first empty array

        # Calculates "correct" length (samples) for each window (median_rr number of datapoints)
        if self.window_samples is None:
            correct_window_len = 2 * int(self.median_rr / 2)
        else:
            correct_window_len = self.window_samples

        # Removes final beat's window if its peak is less than median_rr/2 samples from end of window
        # Fixes issues when calculating average_qrs waveform
        #if len(self.ecg_windowed[-1]) != correct_window_len:
        #    self.removed_peak.append(self.r_peaks.pop(-1))
        #    self.ecg_windowed = self.ecg_windowed[:-2]

        # Calculates "average" heartbeat using windows around each peak
        try:
            self.average_qrs = np.mean(self.ecg_windowed, axis=0)
        except (ValueError, TypeError):
            self.average_qrs = [0] * len(self.ecg_windowed[0])
            self.qrs_fail = True

    def calculate_correlation(self):
        """Method that runs a correlation analysis for each beat and the average QRS template.

        - Runs a Pearson correlation between each beat and the QRS template
        - Calculates the average individual beat Pearson correlation value
        - The period is deemed valid if the average correlation is >= 0.66, invalid is < 0.66
        """

        # Calculates correlation between each beat window and the average beat window --------------------------------
        for beat in self.ecg_windowed:

            if len(beat) == len(self.average_qrs):
                r = stats.pearsonr(x=beat, y=self.average_qrs)
                self.beat_ppmc.append(abs(r[0]))
            else:
                self.beat_ppmc.append(0)

        self.average_r = float(np.mean(self.beat_ppmc))
        self.average_r = round(self.average_r, 3)

    def apply_rules(self):
        """First stage of algorithm. Checks data against three rules to determine if the window is potentially valid.
        -Rule 1: HR needs to be between 40 and 180bpm
        -Rule 2: no RR interval can be more than 3 seconds
        -Rule 3: the ratio of the longest to shortest RR interval is less than 2.2
        -Rule 4: the amplitude range of the raw ECG voltage must exceed n microV (approximate range for non-wear)
        -Rule 5: the average correlation coefficient between each beat and the "average" beat must exceed 0.66
        -Verdict: all rules need to be passed
        """

        # Rule 1: "The HR extrapolated from the sample must be between 40 and 180 bpm" -------------------------------
        if 40 <= self.hr <= 180:
            self.valid_hr = True
        else:
            self.valid_hr = False

        # Rule 2: "the maximum acceptable gap between successive R-peaks is 3s ---------------------------------------
        for rr_interval in self.delta_rr:
            if rr_interval < self.rr_thresh:
                self.valid_rr = True

            if rr_interval >= self.rr_thresh:
                self.valid_rr = False
                break

        # Rule 3: "the ratio of the maximum beat-to-beat interval to the minimum beat-to-beat interval... ------------
        # should be less than 2.5"
        self.rr_ratio = max(self.delta_rr) / min(self.delta_rr)

        if self.rr_ratio >= self.rr_ratio_thresh:
            self.valid_ratio = False

        if self.rr_ratio < self.rr_ratio_thresh:
            self.valid_ratio = True

        # Rule 4: the range of the raw ECG signal needs to be >= 250 microV ------------------------------------------
        if self.volt_range <= self.voltage_thresh:
            self.valid_range = False

        if self.volt_range > self.voltage_thresh:
            self.valid_range = True

        # Rule 5: Determines if average R value is above threshold of 0.66 -------------------------------------------
        if self.average_r >= self.corr_thresh:
            self.valid_corr = True

        if self.average_r < 0.66:
            self.valid_corr = False

        # FINAL VERDICT: valid period if all rules are passed --------------------------------------------------------
        if self.valid_hr and self.valid_rr and self.valid_ratio and self.valid_range and self.valid_corr:
            self.valid_period = True
        else:
            self.valid_period = False

        self.data_dict = {"valid_period": self.valid_period,
                          'n_beats': len(self.peaks),
                          "valid_hr": self.valid_hr, "hr": round(self.hr, 1),
                          "valid_max_rr": self.valid_rr, "max_rr": round(max(self.delta_rr), 1),
                          "valid_rr_ratio": self.valid_ratio, "rr_ratio": round(self.rr_ratio, 1),
                          "valid_voltage": self.valid_range, "voltage_range": round(self.volt_range, 1),
                          "valid_correlation": self.valid_corr, "correlation": self.average_r,
                          'template': self.average_qrs}


def run_orphanidou(signal, peaks, sample_rate, epoch_len, timestamps, window_size=0.2,
                   volt_thresh=250, corr_thresh=.66, rr_ratio_thresh=5, rr_thresh=4, quiet=True):

    # prints parameters
    if not quiet:
        print("\nRunning Orphanidou et al. 2015 quality check algorithm using the following parameters:")
        print(f"-{len(signal)/sample_rate/3600:.1f}-hour long collection")
        print(f"-{len(peaks)} peaks")
        print(f"-Sample rate of {sample_rate} Hz")
        print(f"-Correlation windows of {window_size} seconds")
        print(f"-Voltage threshold of {volt_thresh}uV")
        print(f"-Correlation threshold of r >= {corr_thresh:.3f}")
        print(f"-RR ratio threshold of {rr_ratio_thresh}")
        print(f"-Max RR threshold of {rr_thresh}")

    orphanidou = []  # epoch validity
    errors = []  # epochs that run into errors for whatever reason
    rpeaks = []  # included R peaks
    removed_rpeaks = []  # R peaks removed due to epoch boundaries
    df_out = pd.DataFrame()  # epoched output
    valid_epoch_peaks = []  # peaks only in valid epochs
    invalid_epoch_peaks = []  # peaks only in invalid epochs

    window_samples = int(sample_rate * window_size)  # samples around peak for correlation check
    epoch_samples = int(sample_rate * epoch_len)  # number of samples per epoch
    epoch_idx = np.arange(0, len(signal), epoch_samples)  # epoch start indexes

    if epoch_idx[-1] >= len(signal):
        epoch_idx[-1] = len(signal) - 1

    # loops through epochs
    for i in tqdm(epoch_idx):

        try:
            # finds peaks that fall into epoch's temporal location
            epoch_peaks = list(np.where((peaks >= i) & (peaks < i + epoch_samples))[0])
            epoch_peaks = peaks[epoch_peaks]
            epoch_peaks = [idx - i for idx in epoch_peaks]  # index relative to epoch start

            # runs orphanidou algorithm on epoch
            d = CheckQuality(ecg_signal=signal[i:i+epoch_samples], sample_rate=sample_rate,
                             peaks=epoch_peaks, voltage_thresh=volt_thresh, corr_thresh=corr_thresh,
                             rr_thresh=rr_thresh, rr_ratio_thresh=rr_ratio_thresh,
                             window_samples=window_samples, epoch_len=epoch_len)
            orphanidou.append("valid" if d.valid_period else "invalid")

            epoch_peaks = [i + j for j in sorted(d.r_peaks + d.removed_peak)]
            d.data_dict['peak_idx'] =epoch_peaks

            if d.valid_period:
                valid_epoch_peaks += epoch_peaks
            if not d.valid_period:
                invalid_epoch_peaks += epoch_peaks

            df_out = pd.concat([df_out, pd.DataFrame(pd.Series(d.data_dict)).transpose()], ignore_index=True)

            rpeaks += [j + i for j in d.r_peaks]
            removed_rpeaks += [j + i for j in d.removed_peak]

        except TypeError:
            print(f"Issue with epoch starting at index {i}.")
            orphanidou.append("error")
            errors.append(i)

    if not quiet:
        print("Complete.")

    df_valid = pd.DataFrame({"start_time": timestamps[valid_epoch_peaks], 'idx': valid_epoch_peaks,
                             'quality': [1] * len(valid_epoch_peaks)})
    df_invalid = pd.DataFrame({"start_time": timestamps[invalid_epoch_peaks], 'idx': invalid_epoch_peaks,
                               'quality': [0] * len(invalid_epoch_peaks)})

    df_out.insert(loc=0, column='start_time', value=timestamps[epoch_idx])
    df_out.insert(loc=1, column='start_idx', value=epoch_idx)

    if not quiet:
        print(f"\nOrphanidou algorithm has flagged the data as "
              f"{100 * orphanidou.count('valid') / len(orphanidou):.1f}% valid")

    df_bout = bout_orphanidou(df_epoch=df_out, timestamps=timestamps)

    return {"quality": df_out, 'valid_peaks': df_valid, 'invalid_peaks': df_invalid, 'bout': df_bout}


def bout_orphanidou(df_epoch, timestamps):

    d = df_epoch.copy()

    max_i = d['start_idx'].iloc[-1]

    d['diff'] = d['valid_period'].diff()
    df_bout = d.loc[d['diff'] != 0]
    df_bout.reset_index(drop=True, inplace=True)

    df_bout = df_bout[['start_time', 'start_idx', 'valid_period']]
    df_bout['valid_period'] = [bool(i) for i in df_bout['valid_period']]

    end_times = list(df_bout['start_time'].iloc[1:])
    end_times.append(timestamps[max_i])
    end_idx = list(df_bout['start_idx'].iloc[1:])
    df_bout.insert(1, 'end_time', end_times)
    df_bout.insert(3, 'end_idx', end_idx.append(max_i))

    df_bout['duration'] = [(j - i).total_seconds() for i, j in zip(df_bout['start_time'], df_bout['end_time'])]

    return df_bout


def get_epoch(df, timestamp):

    print(f"-Getting epoch that contains {timestamp}")
    t = pd.to_datetime(timestamp)

    df_epoch = df.loc[df['start_time'] >= t].iloc[0]

    return df_epoch


"""
peaks = df_peaks8['idx']

# peaks = all_peaks['neurokit']
df, output = run_orphanidou(signal=filt, peaks=peaks, timestamps=ecg.ts, window_size=.3,
                            sample_rate=ecg.fs, epoch_len=15, volt_thresh=250, corr_thresh=.66,
                            rr_thresh=3, rr_ratio_thresh=3,
                            quiet=False)

valid_peaks = []
for row in df.loc[df['valid_period']].itertuples():
    valid_peaks += row.peak_idx

df_valid = pd.DataFrame({"start_time": ecg.ts[valid_peaks], 'idx': valid_peaks, 'quality': [1] * len(valid_peaks)})

df_valid['hr'] = calculate_inst_hr(sample_rate=ecg.fs, df_peaks=df_valid, peak_colname='idx', min_quality=3, max_break=3)

from datetime import timedelta

fig, ax = plt.subplots(3, sharex='col', figsize=(12, 8), gridspec_kw={"height_ratios": [1, .5, .33]})
ax[0].plot(ecg.ts[:len(filt)], filt, color='black', zorder=1)

ax[0].scatter(ecg.ts[df_peaks1['idx']], filt[df_peaks1['idx']], color='red', marker='o', s=30, zorder=1, label='Orig.')
ax[0].scatter(ecg.ts[peaks], filt[peaks] + 100, color='orange', marker='v', s=30, zorder=1, label='Screened')
ax[0].scatter(ecg.ts[peaks], filt[peaks] + 200, color='gold', marker='v', s=30, zorder=1, label='NK')
ax[0].scatter(ecg.ts[output['r_peaks']], filt[output['r_peaks']] + 300, color='limegreen', s=30, marker='v', zorder=1, label='ValidEpoch')
ax[0].scatter(ecg.ts[output['removed_rpeaks']], filt[output['removed_rpeaks']] + 300, color='green', marker='x', s=30, zorder=2, label='Edge')

ax[0].scatter(ecg.ts[valid_peaks], filt[valid_peaks] + 400, color='dodgerblue', marker='v', s=30, zorder=2, label='Valid')
ax[0].axvline(df_epoch['start_time'], color='red')

ax[0].legend(loc='lower right')

ax[1].plot(df_valid['start_time'], df_valid['hr'], color='red', label='B2B', lw=.5)
ax[1].plot(df['start_time'], df['hr'], color='black', label='Epoch')
ax[1].legend(loc='lower right')
ax[1].grid()
ax[1].set_ylabel("HR (bpm)")

c_dict = {'ignore': 'grey', 'HR': 'limegreen'}
df_snr_invalid = ecg.df_snr.loc[ecg.df_snr['quality'] == 'ignore']
df_snr_invalid.reset_index(drop=True, inplace=True)
for row in df_snr_invalid.itertuples():
    if row.Index == 0:
        ax[2].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp, ymin=0, ymax=1, color=c_dict[row.quality],
                      alpha=.75, label='SNR')
    if row.Index > 0:
        ax[2].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp, ymin=0, ymax=1, color=c_dict[row.quality], alpha=.25)

df_orph_invalid = df.loc[[not i for i in df['valid_period']]]
df_orph_invalid.reset_index(drop=True, inplace=True)
for row in df_orph_invalid.itertuples():
    if row.Index == 0:
        ax[2].axvspan(xmin=row.start_time, xmax=row.start_time + timedelta(seconds=15), ymin=0, ymax=1, color='red',
                      alpha=.75, label='Orphanidou')
    if row.Index > 0:
        ax[2].axvspan(xmin=row.start_time, xmax=row.start_time + timedelta(seconds=15), ymin=0, ymax=1, color='red', alpha=.25)

ax[2].errorbar(df['start_time'] + timedelta(seconds=7.5), df['valid_period'],
               xerr=timedelta(seconds=7.5), color='black', linestyle="", marker='o', label='Orphanidou')
ax[2].set_yticks([0, 1])
ax[2].set_yticklabels(['invalid', 'valid'])
ax[2].legend()

ax[2].xaxis.set_major_formatter(xfmt)
plt.tight_layout()


# df_epoch = get_epoch(df, timestamp="2021-11-10 4:16:55")
"""