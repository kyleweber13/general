import pandas as pd

from ecgdetectors.ecgdetectors import Detectors
# https://github.com/luishowell/ecg-detectors
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy
import nwdata.NWData
from Filtering import filter_signal
import random
import datetime
import pyedflib
from datetime import timedelta


class CheckQuality:
    """Class method that implements the Orphanidou ECG signal quality assessment algorithm on raw ECG data.

       Orphanidou, C. et al. (2015). Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
       Derivation and Applications to Wireless Monitoring. IEEE Journal of Biomedical and Health Informatics.
       19(3). 832-838.
    """

    def __init__(self, raw_data, sample_rate, start_index, accel_data=None, accel_fs=1,
                 template_data='filtered', voltage_thresh=250, epoch_len=15):
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
        self.start_index = start_index
        self.template_data = template_data

        self.raw_data = [i for i in raw_data[self.start_index:self.start_index+int(self.epoch_len*self.fs)]]
        self.wavelet = None
        self.filt_squared = None

        self.accel_data = accel_data
        self.accel_fs = accel_fs

        self.qrs_fail = False

        self.index_list = np.arange(0, len(self.raw_data), self.epoch_len*self.fs)

        self.rule_check_dict = {"Valid Period": False,
                                "HR Valid": False, "HR": None,
                                "Max RR Interval Valid": False, "Max RR Interval": None,
                                "RR Ratio Valid": False, "RR Ratio": None,
                                "Voltage Range Valid": False, "Voltage Range": None,
                                "Correlation Valid": False, "Correlation": None,
                                "Accel Counts": None}

        # prep_data parameters
        self.r_peaks = None
        self.r_peaks_index_all = None
        self.rr_sd = None
        self.removed_peak = []
        self.enough_beats = True
        self.hr = 0
        self.delta_rr = []
        self.removal_indexes = []
        self.rr_ratio = None
        self.volt_range = 0

        # apply_rules parameters
        self.valid_hr = None
        self.valid_rr = None
        self.valid_ratio = None
        self.valid_range = None
        self.valid_corr = None
        self.rules_passed = None

        # adaptive_filter parameters
        self.median_rr = None
        self.ecg_windowed = []
        self.average_qrs = None
        self.average_r = 0

        # calculate_correlation parameters
        self.beat_ppmc = []
        self.valid_period = None

        """RUNS METHODS"""
        # Peak detection and basic outcome measures
        self.prep_data()

        # Runs rules check if enough peaks found
        if self.enough_beats:
            self.adaptive_filter(template_data=self.template_data)
            self.calculate_correlation()
            self.apply_rules()

        if self.valid_period:
            self.r_peaks_index_all = [peak + start_index for peak in self.r_peaks]

    def prep_data(self):
        """Function that:
        -Initializes ecgdetector class instance
        -Runs stationary wavelet transform peak detection
            -Implements 0.1-10Hz bandpass filter
            -DB3 wavelet transformation
            -Pan-Tompkins peak detection thresholding
        -Calculates RR intervals
        -Removes first peak if it is within median RR interval / 2 from start of window
        -Calculates average HR in the window
        -Determines if there are enough beats in the window to indicate a possible valid period
        """

        # Initializes Detectors class instance with sample rate
        detectors = Detectors(self.fs)

        # Runs peak detection on raw data ----------------------------------------------------------------------------
        # Uses ecgdetectors package -> stationary wavelet transformation + Pan-Tompkins peak detection algorithm
        try:
            self.r_peaks, swt, squared = detectors.swt_detector(unfiltered_ecg=self.raw_data)
        except ValueError:
            self.r_peaks = detectors.swt_detector(unfiltered_ecg=self.raw_data)

        try:
            self.r_peaks = [i for i in self.r_peaks]
        except TypeError:
            self.r_peaks = list([self.r_peaks])

        # Checks to see if there are enough potential peaks to correspond to correct HR range ------------------------
        # Requires number of beats in window that corresponds to ~40 bpm to continue
        # Prevents the math in the self.hr calculation from returning "valid" numbers with too few beats
        # i.e. 3 beats in 3 seconds (HR = 60bpm) but nothing detected for rest of epoch
        if len(self.r_peaks) >= np.floor(40/60*self.epoch_len):
            self.enough_beats = True

            n_beats = len(self.r_peaks)  # number of beats in window
            delta_t = (self.r_peaks[-1] - self.r_peaks[0]) / self.fs  # time between first and last beat, seconds
            self.hr = 60 * (n_beats-1) / delta_t  # average HR, bpm

        # Stops function if not enough peaks found to be a potential valid period
        # Threshold corresponds to number of beats in the window for a HR of 40 bpm
        if len(self.r_peaks) < np.floor(40/60*self.epoch_len):
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
        for i, peak in enumerate(self.r_peaks):
            if peak < (self.median_rr / 2 + 1) or (self.epoch_len * self.fs - peak) < (self.median_rr / 2 + 1):
                self.removed_peak.append(self.r_peaks.pop(i))
                self.removal_indexes.append(i)

        # Removes RR intervals corresponding to
        if len(self.removal_indexes) != 0:
            self.delta_rr = [self.delta_rr[i] for i in range(len(self.r_peaks)) if i not in self.removal_indexes]

        # Calculates range of ECG voltage ----------------------------------------------------------------------------
        self.volt_range = max(self.raw_data) - min(self.raw_data)

    def adaptive_filter(self, template_data="filtered"):
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
            if template_data == "raw":
                window = self.raw_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            if template_data == "filtered":
                window = self.raw_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            if template_data == "wavelet":
                window = self.wavelet[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]

            self.ecg_windowed.append(window)  # Adds window to list of windows

        # Approach 3: determine average QRS template ------------------------------------------------------------------
        self.ecg_windowed = np.asarray(self.ecg_windowed)[1:]  # Converts list to np.array; omits first empty array

        # Calculates "correct" length (samples) for each window (median_rr number of datapoints)
        correct_window_len = 2*int(self.median_rr/2)

        # Removes final beat's window if its peak is less than median_rr/2 samples from end of window
        # Fixes issues when calculating average_qrs waveform
        if len(self.ecg_windowed[-1]) != correct_window_len:
            self.removed_peak.append(self.r_peaks.pop(-1))
            self.ecg_windowed = self.ecg_windowed[:-2]

        # Calculates "average" heartbeat using windows around each peak
        try:
            self.average_qrs = np.mean(self.ecg_windowed, axis=0)
        except (ValueError, TypeError):
            self.average_qrs = [0 for i in range(len(self.ecg_windowed[0]))]
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
            if rr_interval < 3:
                self.valid_rr = True

            if rr_interval >= 3:
                self.valid_rr = False
                break

        # Rule 3: "the ratio of the maximum beat-to-beat interval to the minimum beat-to-beat interval... ------------
        # should be less than 2.5"
        self.rr_ratio = max(self.delta_rr) / min(self.delta_rr)

        if self.rr_ratio >= 2.5:
            self.valid_ratio = False

        if self.rr_ratio < 2.5:
            self.valid_ratio = True

        # Rule 4: the range of the raw ECG signal needs to be >= 250 microV ------------------------------------------
        if self.volt_range <= self.voltage_thresh:
            self.valid_range = False

        if self.volt_range > self.voltage_thresh:
            self.valid_range = True

        # Rule 5: Determines if average R value is above threshold of 0.66 -------------------------------------------
        if self.average_r >= 0.66:
            self.valid_corr = True

        if self.average_r < 0.66:
            self.valid_corr = False

        # FINAL VERDICT: valid period if all rules are passed --------------------------------------------------------
        if self.valid_hr and self.valid_rr and self.valid_ratio and self.valid_range and self.valid_corr:
            self.valid_period = True
        else:
            self.valid_period = False

        self.rule_check_dict = {"Valid Period": self.valid_period,
                                "HR Valid": self.valid_hr, "HR": round(self.hr, 1),
                                "Max RR Interval Valid": self.valid_rr, "Max RR Interval": round(max(self.delta_rr), 1),
                                "RR Ratio Valid": self.valid_ratio, "RR Ratio": round(self.rr_ratio, 1),
                                "Voltage Range Valid": self.valid_range, "Voltage Range": round(self.volt_range, 1),
                                "Correlation Valid": self.valid_corr, "Correlation": self.average_r,
                                "Accel Flatline": None}

        if self.accel_data is not None:
            accel_start = int(self.start_index / (self.fs / self.accel_fs))
            accel_end = accel_start + self.accel_fs * self.epoch_len

            svm = sum([i for i in self.accel_data["VM"].iloc[accel_start:accel_end]])
            self.rule_check_dict["Accel Counts"] = round(svm, 2)

            flatline = True if max(self.accel_data["VM"].iloc[accel_start:accel_end]) - \
                               min(self.accel_data["VM"].iloc[accel_start:accel_end]) <= .05 else False
            self.rule_check_dict["Accel Flatline"] = flatline

            sd = np.std(self.accel_data["VM"].iloc[accel_start:accel_end])
            self.rule_check_dict["Accel SD"] = sd


def run_orphanidou(signal, sample_rate, epoch_len, volt_thresh=250, quiet=True):

    if not quiet:
        print("\nRunning Orphanidou et al. 2015 quality check algorithm...")

    percent_markers = np.arange(0, len(signal)*1.1, len(signal)/20)
    index = 0

    orphanidou = []
    errors = []
    rpeaks = []
    removed_rpeaks = []

    for i in range(0, len(signal), int(sample_rate * epoch_len)):
        if not quiet:
            if i >= percent_markers[index]:
                print(f"-{index*5}% complete")
                index += 1

        try:
            d = CheckQuality(raw_data=signal, sample_rate=sample_rate, start_index=i,
                             template_data='raw', voltage_thresh=volt_thresh, epoch_len=epoch_len)
            orphanidou.append("Valid" if d.valid_period else "Invalid")

            for peak in d.r_peaks:
                rpeaks.append(peak+i)
            for peak in d.removed_peak:
                removed_rpeaks.append(peak+i)

        except TypeError:
            print(f"Issue with epoch starting at index {i}.")
            orphanidou.append("Error")
            errors.append(i)

    if not quiet:
        print("Complete.")

    output = {"Validity": orphanidou, "ErrorEpochs": errors, "R_Peaks": rpeaks, "Removed_RPeaks": removed_rpeaks}

    # Combines all peaks (kept and removed)
    output["AllPeaks"] = all_peaks = sorted(output["R_Peaks"] + output["Removed_RPeaks"])

    # beat-to-beat RR intervals
    output["RR_ints"] = [(p2-p1)/sample_rate for p1, p2 in zip(all_peaks[:], all_peaks[1:])]

    # 10-beat rolling mean HR
    output["RollMeanHR"] = [60/np.mean(output["RR_ints"][i:i+10]) for i in range(len(output["RR_ints"])-10)]

    if not quiet:
        print("\nData is {}% valid".format(round(100*output["Validity"].count("Valid")/len(output["Validity"]), 2)))

    df = pd.DataFrame({"Index": np.arange(0, len(signal), int(sample_rate * epoch_len)),
                       "Orphanidou": output["Validity"]})

    return df, output


def format_nonwear():
    df_nw = pd.read_excel("C:/Users/ksweber/Desktop/OmegaSnap_Nonwear.xlsx")
    df_nw["Subj"] = [subj in i for i in df_nw["File"]]
    df_nw = df_nw.loc[df_nw["Subj"]]

    epoch_nw = np.zeros(int(len(f)/fs/epoch_len))

    # Assigns moderate intensity (2) to all gait bouts
    for row in df_nw.itertuples():
        start = int((row.Start - start_stamp).total_seconds() / epoch_len)
        stop = int((row.Stop - start_stamp).total_seconds() / epoch_len)
        epoch_nw[start:stop] = 1

    df_epoch_nw = pd.DataFrame({"Timestamp": pd.date_range(start=start_stamp, periods=len(epoch_nw), freq=f"{epoch_len}S"),
                                "Nonwear": epoch_nw})

    return df_epoch_nw


def calculate_validity_summary(nonwear_data=None, show_plot=False):
    start_hour = pd.to_datetime(start_stamp).round("1H")
    end_hour = pd.to_datetime((start_stamp + timedelta(seconds=len(ecg)/fs))).round("1H") + timedelta(hours=1)
    hour_stamps = pd.date_range(start=start_hour, end=end_hour, freq='1H')

    orph["Timestamp"] = pd.date_range(start=start_stamp, periods=orph.shape[0], freq=f"{epoch_len}S")

    if nonwear_data is not None:
        perc_valid = []
        perc_nw = []
        for hour1, hour2 in zip(hour_stamps[:], hour_stamps[1:]):
            # Locates correct sections of dataframes based on hour
            qc = orph.loc[(orph["Timestamp"] >= hour1) & (orph['Timestamp'] <= hour2)]
            nw = df_nw.loc[(df_nw["Timestamp"] >= hour1) & (df_nw['Timestamp'] <= hour2)]

            df_both = pd.DataFrame({"Orphanidou": qc["Orphanidou"], "NW": nw["Nonwear"]})

            df_wear = df_both.loc[df_both["NW"] == 0]
            df_valid = df_wear.loc[df_wear['Orphanidou'] == "Valid"]

            try:
                p_valid = round(df_valid.shape[0]*100/df_wear.shape[0], 2)
            except ZeroDivisionError:
                p_valid = None

            perc_valid.append(p_valid)

            try:
                perc_nw.append(round(100*(nw.shape[0]-df_wear.shape[0])/nw.shape[0], 1))
            except ZeroDivisionError:
                perc_nw.append(None)

        data_len = min([len(hour_stamps), len(perc_valid), len(perc_nw)])

        df_hourly = pd.DataFrame({"Timestamp": hour_stamps[:data_len],
                                  "PercValid": perc_valid[:data_len],
                                  "PercNW": perc_nw[:data_len]})

        df_both_all = pd.DataFrame({"Orphanidou": orph["Orphanidou"], "NW": nonwear_data["Nonwear"]})

        # Calculates total validity
        valid_all = df_both_all.loc[(df_both_all["Orphanidou"] == "Valid") & (df_both_all["NW"] == 0)].shape[0]
        perc_valid_all = round((100*valid_all/df_both_all.shape[0]), 2)
        print(f"Valid data (NW excluded) = {perc_valid_all}%")

    if nonwear_data is None:
        perc_valid = []

        for hour1, hour2 in zip(hour_stamps[:], hour_stamps[1:]):
            # Locates correct sections of dataframes based on hour
            qc = orph.loc[(orph["Timestamp"] >= hour1) & (orph['Timestamp'] <= hour2)]

            df_valid = qc.loc[qc['Orphanidou'] == "Valid"]
            p_valid = round(df_valid.shape[0] * 100 / qc.shape[0], 2)
            perc_valid.append(p_valid)

        data_len = min([len(hour_stamps), len(perc_valid)])

        df_hourly = pd.DataFrame({"Timestamp": hour_stamps[:data_len],
                                  "PercValid": perc_valid[:data_len],
                                  "PercNW": [None for i in range(data_len)]})

        # Calculates total validity
        valid_all = orph.loc[orph["Orphanidou"] == "Valid"].shape[0]
        perc_valid_all = round((100*valid_all/orph.shape[0]), 2)
        print(f"Valid data (NW included) = {perc_valid_all}%")

    if show_plot:
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(df_hourly["Timestamp"], df_hourly["PercValid"], color='red', label='Hourly QC')
        plt.suptitle(f"OND_{subj}: Orphanidou Quality + Nonwear" if nonwear_data is not None else
                     f"OND_{subj}: Orphanidou Quality")
        ax.set_ylabel("%")
        ax.set_ylim(-1, 101)
        ax.set_yticks(np.arange(0, 105, 20))
        ax.axhline(y=perc_valid_all, color='red', linestyle='dashed', label="Total % Valid")

        axx = ax.twiny()
        axx.plot(np.arange(df_hourly.shape[0]), df_hourly["PercValid"], linestyle="")
        axx.set_xlabel("Hours")

        if nonwear_data is not None:
            ax.plot(df_hourly["Timestamp"], df_hourly["PercNW"], color='black', label='Nonwear')
        ax.legend()

    return df_hourly, fig if show_plot else None


def plot_everything(signal, output=None, start_time=None, ds_ratio=5, fs=250, epoch_len=15, include_epoch_lines=False):

    x_raw = np.arange(len(signal))/fs

    if output is not None:
        x_epoch = [i*epoch_len+epoch_len/2 for i in np.arange(len(output["Validity"]))]
    if output is None:
        x_epoch = [i*epoch_len+epoch_len/2 for i in np.arange(int(len(x_raw)/fs/epoch_len))]

    if start_time is not None:
        x_raw = pd.date_range(start=start_time, periods=len(signal), freq="{}ms".format(1000/fs))
        x_epoch = pd.date_range(start=start_time, periods=len(signal[::int(fs*epoch_len)]), freq="{}S".format(epoch_len))

    fig, axes = plt.subplots(4, sharex='col', figsize=(14, 8))
    axes[0].plot(x_raw[::ds_ratio], signal[::ds_ratio], color='black', label='Filtered', zorder=0)

    if output is not None:
        axes[0].scatter([x_raw[i] for i in output["R_Peaks"]], [signal[i] for i in output["R_Peaks"]],
                        color='limegreen', marker="o", zorder=1)

        axes[0].scatter([x_raw[i] for i in output["Removed_RPeaks"]], [signal[i] for i in output["Removed_RPeaks"]],
                        color='red', marker="x", zorder=1)

        axes[1].plot([i / fs for i in output["AllPeaks"][:len(output["RR_ints"])]] if start_time is None else
                     [start_time + timedelta(seconds=i / fs) for i in output["AllPeaks"][:len(output["RR_ints"])]],
                     output["RR_ints"],
                     color='black', zorder=0, label="Raw RR")

        axes[1].plot([i / fs for i in output["AllPeaks"][:len(output["RollMeanHR"])]] if start_time is None else
                     [start_time + timedelta(seconds=i / fs) for i in output["AllPeaks"][:len(output["RollMeanHR"])]],
                     output["RollMeanHR"], color='red', zorder=0, label="10-beat avg")
        axes[1].legend(loc='lower left')
        axes[1].set_ylim(0, 2.5)
        axes[1].set_ylabel("RR-int (seconds)")

        axes[2].plot([i / fs for i in output["AllPeaks"][:len(output["RollMeanHR"])]] if start_time is None else
                     [start_time + timedelta(seconds=i / fs) for i in output["AllPeaks"][:len(output["RollMeanHR"])]],
                     output["RollMeanHR"], color='red', label='10-beat avg')
        axes[2].legend(loc='lower left')
        axes[2].set_ylabel("BPM")

        axes[3].plot(x_epoch, output["Validity"], color='dodgerblue', label="Orphanidou")
        axes[3].legend(loc='lower left')

        axes[3].set_xlabel("Seconds")

    axes[0].set_ylabel("Voltage")
    axes[0].legend(loc='lower left')

    if include_epoch_lines:
        for i in np.arange(0, len(signal), int(fs*epoch_len)):
            axes[0].axvline(x=i/fs, color='purple')

    return fig


# fig = plot_everything(signal=f, start_time=start_stamp, output=None, ds_ratio=4, fs=fs, epoch_len=epoch_len, include_epoch_lines=False)


def gen_random_beat(peak_indexes, avg_qrs, pad_beats=1, peak_ind=None, fs=250, plot_r=False):

    if peak_ind is not None:
        rando = peak_ind
    if peak_ind is None:
        rando = random.randint(pad_beats, len(peak_indexes)-pad_beats)

    a = f[peak_indexes[rando-pad_beats] - int(fs/4):peak_indexes[rando+pad_beats] + int(fs/4)]
    b = f[peak_indexes[rando] - int(fs/4):peak_indexes[rando] + int(fs/4)]

    xa = np.arange(peak_indexes[rando-pad_beats] - int(fs/4), peak_indexes[rando+pad_beats] + int(fs/4))/fs
    xb = np.arange(peak_indexes[rando] - int(fs/4), peak_indexes[rando] + int(fs/4))/fs
    r = round(scipy.stats.pearsonr(avg_qrs, b)[0], 3)

    ax.plot(xa, a, color='green' if r >= .66 else 'red', zorder=0)
    ax.plot(xb, avg_qrs, color='black', zorder=1)

    ax.set_title(f"Peak = {rando}, r = {r}")

    ax.set_ylabel("Voltage")
    ax.set_xlabel("Seconds")

    if plot_r:
        ax2.bar(x=peak_indexes[rando]/fs, height=r, color='green' if r >= 0 else 'red', width=.5, alpha=.35)

    return r

"""
peaks = [i for i in output["AllPeaks"] if i < int(1800*60)]

r = []
fig, ax = plt.subplots(1, figsize=(10, 6))
ax2 = ax.twinx()
ax2.set_ylabel("Pearson R", color='green')
ax2.axhline(y=0, linestyle='dashed', color='limegreen')

for i in range(len(peaks)-1):
    r.append(gen_random_beat(peak_ind=i, peak_indexes=peaks, avg_qrs=avg_qrs, fs=fs, pad_beats=1))
"""


def template_data(signal, validity_data, peak_indexes, sample_f=250, epoch_len=15, show_plot=True):

    print("\nCalculating QRS template...")

    # Creates array of each peak ± 125ms for all peaks in valid epochs

    # Output data
    valid_qrs = []
    invalid_qrs = []
    valid_peak_vals = []
    invalid_peak_vals = []

    min_peak_ind = 0  # Crops peak index list with each iteration to speed things up

    validity_data = list(validity_data)  # Epoched Orphanidou values
    peak_indexes = np.array(peak_indexes)  # QRS peak values

    for i, validity in enumerate(validity_data):
        if i % 100 == 0:
            print(f"-{int(100*i/len(validity_data))}%")
        # Index of signal
        raw_i = int(i*sample_f*epoch_len)

        # Only includes peaks in correct epoch
        peaks = [p for p in peak_indexes[min_peak_ind:] if raw_i <= p < int(raw_i + sample_f*epoch_len)]
        min_peak_ind = np.argmax(peak_indexes >= raw_i)

        # Loops through peaks
        for p in peaks:
            # Crops data to peak +- 1/8th of second
            d = signal[p - int(sample_f/8):p + int(sample_f/8)]

            # Adds data to correct array based on validity
            if validity == "Valid":
                valid_qrs.append(d)  # window of data
                valid_peak_vals.append(signal[p])  # voltage of peak
            if validity == "Invalid":
                invalid_qrs.append(d)  # window of data
                invalid_peak_vals.append(signal[p])  # voltage of peak

    def plot_templates():
        # Average QRS shape
        avg_valid_qrs = np.mean(valid_qrs, axis=0)
        valid_r = []
        invalid_r = []

        fig, axes = plt.subplots(2, 2, sharey='row', figsize=(10, 6))

        # Plots each window of data during valid epochs
        for beat in valid_qrs:

            # Pearson R correlation between beat and template
            pearson_r = scipy.stats.pearsonr(beat, avg_valid_qrs)[0]
            valid_r.append(round(pearson_r, 3))
            axes[0][0].plot([i/fs - .125 for i in np.arange(0, int(fs/4))], beat,
                            color='green' if pearson_r >= .66 else 'black', zorder=0)

        axes[0][0].plot([i/sample_f - .125 for i in np.arange(0, int(sample_f/4))], avg_valid_qrs,
                        color='dodgerblue', lw=3, label='Avg')
        axes[0][0].legend()
        axes[0][0].set_xlabel("Seconds")
        axes[0][0].axvline(x=0, linestyle='dotted', color='red')
        axes[0][0].set_title("Valid Epochs")
        axes[0][0].set_ylabel("Voltage")

        for beat in invalid_qrs:
            pearson_r = scipy.stats.pearsonr(beat, avg_valid_qrs)[0]
            invalid_r.append(round(pearson_r, 3))

            axes[0][1].plot([i/sample_f - .125 for i in np.arange(0, int(sample_f/4))], beat,
                            color='green' if pearson_r >= .66 else 'black', zorder=0)

        axes[0][1].plot([i/sample_f - .125 for i in np.arange(0, int(sample_f/4))], avg_valid_qrs,
                        color='dodgerblue', lw=3, label="Avg")
        axes[0][1].legend()
        axes[0][1].axvline(x=0, linestyle='dotted', color='red')
        axes[0][1].set_title("Invalid Epochs")
        axes[0][1].set_xlabel("Seconds")

        axes[1][0].hist(valid_r, bins=np.arange(0, 1, .05), color='grey', edgecolor='black')
        axes[1][0].set_xlabel("Pearson r (beat~avg)")
        axes[1][0].set_ylabel("% of epochs")
        axes[1][1].hist(invalid_r, bins=np.arange(0, 1, .05), color='grey', edgecolor='black')
        axes[1][1].set_xlabel("Pearson r (beat~avg)")

    if show_plot:
        plot_templates()

    return valid_qrs, invalid_qrs, valid_peak_vals, invalid_peak_vals


"""df, output = run_orphanidou(signal=ecg, sample_rate=250, epoch_len=15, volt_thresh=250, quiet=False)

valid_qrs, invalid_qrs, valid_peak_vals, invalid_peak_vals = template_data(signal=f2,
                                                                           sample_f=fs, epoch_len=epoch_len,
                                                                           show_plot=True,
                                                                           validity_data=df['Orphanidou'],
                                                                           peak_indexes=output["AllPeaks"])

template = np.mean(valid_qrs, axis=0)"""
