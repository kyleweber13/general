import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
import scipy.fft
from datetime import datetime
from datetime import timedelta as td


""" ===========================================  FUNCTION DEFINITIONS =============================================="""


class BittiumNonwear:

    def __init__(self, signal=None, timestamps=None, power_percent_thresh=90, freq_thresh=60,
                 start_ind=0, stop_ind=-1, sample_rate=250, epoch_len=15,
                 min_nw_dur=5, min_nw_break=5,
                 rolling_mean=40, reference_nonwear=None):

        self.fs = sample_rate
        self.epoch_len = epoch_len
        self.signal = signal
        self.timestamps = timestamps
        self.power_percent_thresh = power_percent_thresh
        self.start_ind = start_ind
        self.stop_ind = stop_ind
        self.rolling_mean = rolling_mean
        self.nw_gs = reference_nonwear
        self.freq_thresh = freq_thresh
        self.min_nw_dur = min_nw_dur
        self.min_nw_break = min_nw_break

        self.df_freq = None
        self.nw_timeseries = []

        self.df_freq = self.process_frequency()
        self.df_nw, self.nw_timeseries = self.run_power_nonwear()

    def process_frequency(self):
        """Runs preliminary frequency-domain processing required for non-wear algorithm. Returns epoched dataframe.

        :argument
        -raw_data: array-like ECG data
        -timestamps: option array of timestamps; uses seconds into collection if None
        -power_percent_thresh: threshold percentage for cumulative power
            -Example: if power_freq_thresh = 90, will calculate lowest frequency below which there is 90%
                      of the power density spectrum
        -rolling_mean: if is int, will calculate rolling mean of rolling_mean epochs for
                       Power data using that many epochs
        -start_ind, stop_ind: indexes corresponding to raw_data. Defaults to entire dataset (0, -1)
        -fs: sample rate of raw_data, Hz
        -epoch_len: interval over which frequency is analyzed, seconds.
        """

        # Data length for performance speed, hours
        data_dur = len(self.signal) / self.fs / 3600

        # Loops through epochs --------------
        raw_f = []
        raw_cutoff_freqs = []

        percent_indexes = np.linspace(0, len(self.signal), 51)
        p = 0

        t0 = datetime.now()

        print(f"\nProcessing data for frequency content in {self.epoch_len}-second epochs...")

        for i in range(0, len(self.signal), int(self.fs*self.epoch_len)):
            if i >= percent_indexes[p]:
                print(f"-{int(2*p)}%")  # Progress tracker
                p += 1

            # current epoch of data
            r = self.signal[i:int(i+self.fs*self.epoch_len)]

            # Runs FFT on data ------
            fft_raw = scipy.fft.fft(r)

            L = len(r)
            xf = np.linspace(0.0, 1.0 / (2.0 * (1 / self.fs)), L // 2)

            # Calculates frequency that encompasses power_freq_thresh% of power spectrum and dominant frequnecy
            df_fft = pd.DataFrame({"Freq": xf,
                                   "Power": 2.0 / L / 2 * np.abs(fft_raw[0:L // 2])})
            df_fft["Cumulative"] = 100 * df_fft["Power"].cumsum() / sum(df_fft["Power"])

            raw_dom_f = df_fft.loc[df_fft["Power"] == df_fft["Power"].max()]["Freq"].iloc[0]

            raw_cutoff_freq = df_fft.loc[df_fft["Cumulative"] >= self.power_percent_thresh].iloc[0]["Freq"]

            raw_f.append(raw_dom_f)
            raw_cutoff_freqs.append(raw_cutoff_freq)

        t1 = datetime.now()
        dt = (t1-t0).total_seconds()

        print(f"Complete. Processing time: {round(dt, 1)} seconds ({round(dt/data_dur, 3)} sec/hour data).\n")

        df_f = pd.DataFrame({"Index": np.arange(0, len(self.signal), int(self.fs*self.epoch_len)),
                             "DomF": raw_f, "Power": raw_cutoff_freqs})

        if type(self.rolling_mean) is int:
            df_f["RollPower"] = df_f.rolling(window=self.rolling_mean).mean()["Power"]

        return df_f

    def plot_process_frequency(self):
        fig, axes = plt.subplots(3, sharex='col', figsize=(10, 6))
        axes[0].plot(self.df_freq["Index"]/self.fs if self.timestamps is None else
                     self.timestamps[::int(self.fs*self.epoch_len)],
                     self.df_freq["DomF"], color='red')
        axes[0].set_title("Dominant Freq")
        axes[0].set_ylabel("Hz")

        if self.nw_gs is not None and self.timestamps is not None:
            prefix = "" if type(self.rolling_mean) is not int else "Roll"
            for row in self.nw_gs.itertuples():
                if row.Start < self.timestamps[-1]:
                    axes[0].fill_between(x=[row.Start, row.Stop], y1=0, y2=self.df_freq["DomF"].max()*1.1,
                                         color='dodgerblue', alpha=.5)
                    axes[1].fill_between(x=[row.Start, row.Stop],
                                         y1=self.df_freq[f"{prefix}Power"].min()*1.1,
                                         y2=self.df_freq[f"{prefix}Power"].max()*1.1,
                                         color='dodgerblue', alpha=.5)
                    axes[2].fill_between(x=[row.Start, row.Stop], y1=min(self.signal)*1.1, y2=max(self.signal)*1.1,
                                         color='dodgerblue', alpha=.5)

        axes[1].plot(self.df_freq["Index"]/self.fs if self.timestamps is None else
                     self.timestamps[::int(self.fs*self.epoch_len)],
                     self.df_freq["Power"], color='purple', label="'Raw' power")

        if type(self.rolling_mean) is int:
            axes[1].plot(self.df_freq["Index"]/self.fs if self.timestamps is None else
                         self.timestamps[::int(self.fs*self.epoch_len)],
                         self.df_freq["RollPower"], color='black', label=f"{self.rolling_mean} epochs power")

        axes[1].set_title(f"{self.power_percent_thresh}% Cumulative Power Frequencies")
        axes[1].set_ylabel("Hz")
        axes[1].legend(loc='upper right')

        axes[2].plot(self.timestamps[::5] if self.timestamps is not None else
                     np.arange(0, len(self.signal), self.fs)[::5],
                     self.signal[::5], color='blue')
        axes[2].set_title("Raw Data")
        axes[2].set_xlabel("Seconds" if self.timestamps is None else "")

    def run_power_nonwear(self):
        """Runs analysis on output of process_frequency() to flag non-wear periods.

        Algorithm details:
        -Uses power density spectrum data generated from process_frequency()'s power_freq_thresh argument.
        -During non-wear periods, the cumulative power at a given low (<60Hz) frequency will decrease since there is more
         high-frequency content during non-wear periods.
        -The algorithm looks for continuous periods where the cumulative power is above a given threshold
        -Short periods surrounded by longer periods are collapsed

        :argument
        -df_freq: df output from process_frequency()
        -use_rolling: boolean for whether to use "raw" epoched data or rolling mean data in df_freq
        -min_dur: minimum duration of nonwear bout, minutes
        -min_nw_break: minimum time between consecutive nonwear bouts to treat as separate, i.e. maximum time between
                       bouts to not combine into one large bout
        -raw_data: ECG signal
        -fs: sample rate of raw_data, Hz
        -timestamps: timestamps for raw_data
        -freq_thresh: frequency threshold.
                     - Wear periods are marked if the df_freq["Power"] frequency value is below threshold
        -show_plot: boolean
        """

        print("\nRunning algorithm to find non-wear periods...")

        if type(self.rolling_mean) is int:
            prefix = "Roll"
        if not type(self.rolling_mean) is int:
            prefix = ""

        # Start/stop 1-s epoch index numbers
        invalid_starts = []
        invalid_stops = []

        if self.df_freq.iloc[0][f"{prefix}Power"] > self.freq_thresh:
            invalid_starts.append(0)

        for i in range(self.df_freq.shape[0] - 1):
            if i % 1000 == 0:
                print(f"-{round(100*i/self.df_freq.shape[0], 2)}%")

            if self.df_freq[f"{prefix}Power"].iloc[i] < self.freq_thresh and \
                    self.df_freq[f"{prefix}Power"].iloc[i+1] >= self.freq_thresh:
                invalid_starts.append(int(self.epoch_len*i))
            if self.df_freq[f"{prefix}Power"].iloc[i] >= self.freq_thresh and \
                    self.df_freq[f"{prefix}Power"].iloc[i+1] < self.freq_thresh:
                invalid_stops.append(int(self.epoch_len*(i+1)))

        # Corrects data if collection ends on nonwear bout
        if len(invalid_starts) > len(invalid_stops):
            invalid_stops.append(int(self.df_freq.shape[0]*self.epoch_len))

        # Removes invalid bouts less than min_dur long
        long_start = []
        long_stop = []
        for start, stop in zip(invalid_starts, invalid_stops):
            if (stop - start) >= (self.min_nw_dur*60):
                long_start.append(start)
                long_stop.append(stop)

        for start, stop in zip(long_start[1:], long_stop[:]):
            if start - stop < int(self.min_nw_break*60):
                long_start.remove(start)
                long_stop.remove(stop)

        freq_data = np.zeros(int(self.df_freq.shape[0]*self.epoch_len))
        for start, stop in zip(long_start, long_stop):
            freq_data[start:stop] = 1

        print("Complete.")

        df_bound = pd.DataFrame({"Start_Index": long_start, "Stop_Index": long_stop})
        df_bound["Dur"] = df_bound["Stop_Index"] - df_bound["Start_Index"]

        if self.timestamps is not None:
            df_bound["Start"] = [self.timestamps[0] + td(seconds=i) for i in df_bound["Start_Index"]]
            df_bound["Start"] = df_bound["Start"].round("1S")
            df_bound["End"] = [self.timestamps[0] + td(seconds=i) for i in df_bound["Stop_Index"]]
            df_bound["End"] = df_bound["End"].round("1S")

        print(f"\nTotal NW time = {df_bound['Dur'].sum()/60} minutes ({df_bound.shape[0]} periods found)")
        if self.nw_gs is not None:
            print(f"    -Reference data contains {round(self.nw_gs.DurMin.sum(), 1)} minutes of nonwear.")

        return df_bound, freq_data

    def plot_algorithm_output(self):

        if type(self.rolling_mean) is int:
            prefix = "Roll"
        if not type(self.rolling_mean) is int:
            prefix = ""

        fig, axes = plt.subplots(3, sharex='col', figsize=(10, 6))
        plt.subplots_adjust(top=.925, bottom=.08, right=.975, left=.1, hspace=.35)

        ts = self.timestamps[::int(self.epoch_len*self.fs)] if self.timestamps is not None else \
            np.arange(self.df_freq.shape[0])*self.epoch_len
        axes[0].plot(ts[:min(len(ts), self.df_freq.shape[0])],
                     self.df_freq[f"{prefix}Power"].iloc[:min(len(ts), self.df_freq.shape[0])], color='dodgerblue')
        axes[0].axhline(y=self.freq_thresh, color='red', linestyle='dashed', label=f"{self.freq_thresh}Hz")
        axes[0].set_title("Frequency at Cutoff Power")
        axes[0].set_ylabel("Hz")
        axes[0].legend(loc='upper left')
        axes[0].set_yticks([0, 30, 60, 90, 100, 120])

        ts = self.timestamps[::int(self.fs)] if self.timestamps is not None else \
             np.arange(len(self.nw_timeseries))
        axes[1].fill_between(x=ts[:min(len(ts), len(self.nw_timeseries))],
                             y1=0, y2=self.nw_timeseries[:min(len(ts), len(self.nw_timeseries))],
                             color='purple', alpha=.5)
        axes[1].set_title("Long, High-Power Bouts")
        axes[1].set_yticks([0, 1])

        ts = self.timestamps[::10] if self.timestamps is not None else \
             np.arange(0, len(self.signal))[::10]/self.fs
        axes[2].plot(ts[:min(len(ts), len(self.signal[::10]))], self.signal[::10], color='red', label="Signal")

        if self.nw_gs is not None and self.timestamps is not None:
            y1 = min(self.signal)
            y2 = max(self.signal)
            for row in self.nw_gs.itertuples():
                if row.Index < self.nw_gs.shape[0]:
                    axes[2].fill_between(x=[row.Start, row.Stop], y1=y1, y2=y2, color='green', alpha=.5)
                if row.Index == self.nw_gs.shape[0]:
                    axes[2].fill_between(x=[row.Start, row.Stop], y1=y1, y2=y2, color='green', alpha=.5,
                                         label='Reference NW')
        axes[2].legend(loc='upper right')


a = BittiumNonwear(signal=f, timestamps=timestamp, power_percent_thresh=90, freq_thresh=60,
                   start_ind=0, stop_ind=-1, sample_rate=250, epoch_len=15,
                   min_nw_dur=5, min_nw_break=5,
                   rolling_mean=False, reference_nonwear=df_vis)

# a.plot_process_frequency()
# a.plot_algorithm_output()
