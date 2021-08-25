import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
import scipy.fft
from datetime import datetime
from datetime import timedelta as td
from Filtering import filter_signal
from OrphanidouQC import run_orphanidou


""" ===========================================  FUNCTION DEFINITIONS =============================================="""


class BittiumNonwear:

    def __init__(self, signal=None, temperature=None, tri_accel=None,
                 start_time=None, power_percent_thresh=90, freq_thresh=60,
                 sample_rate=250, temp_sample_rate=1, accel_sample_rate=25, epoch_len=15,
                 min_nw_dur=5, min_nw_break=5,
                 rolling_mean=40, reference_nonwear=None):
        """Class to handle multiple steps of Bittium Faros/ECG nonwear detection.

            :parameters
            -signal: ECG signal
                     - Recommended .25Hz, 3rd order highpass filter)
            -temperature: temperature signal from device
            -start_time: timestamp of data collection start
            -power_percent_thresh: which % of cumulative power spectrum density to use, multiple of 5, int.
                                   - Recommended = 80%
            -freq_thresh: threshold of frequency at given power_percent_thresh to differentiate wear from nonwear, int/float
                                   - Recommended = 57Hz
            -sample_rate: sample frequency of signal, Hz
            -temp_sample_rate: sample frequency of temperature, Hz
            -epoch_len: length of jumping window used to process data in seconds, int
            -min_nw_dur: minimum required duration of a nonwear period in minutes, int
            -min_nw_break: minimum gap between consecutive nonwear periods to treat at separate periods in minutes, int
            -rolling_mean: if not None, algorithm will be run on a rolling mean of values of rolling_mean number of epochs, int
            -reference_nonwear: dataframe containing start/stop timestamps for some reference nonwear (or event) file
        """

        # Parameters and input data
        self.fs = sample_rate
        self.temp_fs = temp_sample_rate
        self.acc_fs = accel_sample_rate
        self.epoch_len = epoch_len
        self.signal = signal
        self.temperature = temperature
        self.accel = tri_accel
        self.start_time = pd.to_datetime(start_time)
        self.timestamps = pd.date_range(start=start_time, periods=len(self.signal), freq="{}ms".format(1000/self.fs))
        self.proc_time = 0

        if temperature is not None:
            self.temp_timestamps = pd.date_range(start=start_time, periods=len(self.temperature),
                                                 freq="{}ms".format(1000/self.temp_fs))
        if tri_accel is not None:
            self.acc_timestamps = pd.date_range(start=start_time, periods=len(self.accel[0]),
                                                freq="{}ms".format(1000/self.acc_fs))
            # Gravity-subtracted vector magnitude
            vm = np.array([np.sqrt(x**2 + y**2 + z**2) - 1000 for
                           x, y, z in zip(self.accel[0], self.accel[1], self.accel[2])])
            vm[vm<0] = 0
            self.accel = np.append(self.accel, [vm], axis=0)

        self.power_percent_thresh = power_percent_thresh
        self.freq_thresh = freq_thresh
        self.rolling_mean = rolling_mean
        self.min_nw_dur = min_nw_dur
        self.min_nw_break = min_nw_break

        # Nonwear tabular data
        self.nw_gs = reference_nonwear
        self.df_epochs = None

        # 1-sec epoch data
        self.nw_timeseries = []

        # Runs methods -----------------------
        # self.df_epochs = self.process_epochs()
        # self.df_nw, self.nw_timeseries = self.run_power_nonwear()

    def process_epochs(self, use_orphanidou=True):
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
        validity = []

        percent_indexes = np.linspace(0, len(self.signal), 51)
        p = 0

        t0 = datetime.now()

        # ECG FREQUENCY CONTENT --------------------------------------------------------------------------------------
        print(f"\nProcessing data for ECG frequency content in {self.epoch_len}-second epochs...")

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

            # Orphanidou quality check -----------
            if raw_cutoff_freq >= self.freq_thresh and use_orphanidou:
                # Runs QC on .67-30Hz bandpass filter in one-epoch increments
                qc = run_orphanidou(signal=filter_signal(data=self.signal[i:i+int(self.fs*self.epoch_len)],
                                                         sample_f=self.fs, filter_type='bandpass',
                                                         low_f=.67, high_f=30, filter_order=3),
                                    sample_rate=self.fs, epoch_len=self.epoch_len, volt_thresh=50, quiet=True)
                validity.append(qc["Validity"][0])

            if raw_cutoff_freq < self.freq_thresh and use_orphanidou:
                validity.append("Valid")

        # ACCELEROMETER CONTENT ---------------------------------------------------------------------------------------
        vm_mean = []
        vm_sd = []

        if self.accel is not None:
            print(f"\nProcessing data for acceleration content in {self.epoch_len}-second epochs...")
            percent_indexes = np.linspace(0, len(self.accel[0]), 51)
            p = 0

            for i in range(0, len(self.accel[0]), int(self.acc_fs*self.epoch_len)):
                if i >= percent_indexes[p]:
                    print(f"-{int(2*p)}%")  # Progress tracker
                    p += 1

                acc = self.accel[-1][i:int(i+self.acc_fs*self.epoch_len)]
                vm_mean.append(np.mean(acc))
                vm_sd.append(np.std(acc))

        # TEMPERATURE CONTENT -----------------------------------------------------------------------------------------
        temp_mean = []
        temp_min = []

        if self.temperature is not None:
            print(f"\nProcessing data for temperature content in {self.epoch_len}-second epochs...")
            percent_indexes = np.linspace(0, len(self.temperature), 51)
            p = 0

            for i in range(0, len(self.temperature), int(self.temp_fs*self.epoch_len)):
                if i >= percent_indexes[p]:
                    print(f"-{int(2*p)}%")  # Progress tracker
                    p += 1

                t = self.temperature[i:int(i+self.temp_fs*self.epoch_len)]
                temp_mean.append(np.mean(t))
                temp_min.append(min(t))

        t1 = datetime.now()
        dt = (t1-t0).total_seconds()

        print(f"Complete. Processing time: {round(dt, 1)} seconds ({round(dt/data_dur, 3)} sec/hour data).\n")
        self.proc_time = round(dt, 1)

        df_out = pd.DataFrame({"Index": np.arange(0, len(self.signal), int(self.fs*self.epoch_len)),
                               "Timestamp": [self.timestamps[i] for i in
                                            range(0, len(self.signal), int(self.fs*self.epoch_len))] if
                                            self.timestamps is not None else [None for i in range(len(raw_f))],
                               "DomF": raw_f, "Power": raw_cutoff_freqs,
                               "ECG_Valid": validity if use_orphanidou else [None for i in range(len(raw_f))],
                               "VM_Mean": vm_mean if self.accel is not None else [None for i in range(len(raw_f))],
                               "VM_SD": vm_sd if self.accel is not None else [None for i in range(len(raw_f))],
                               "Temp_Mean": temp_mean if self.temperature is not None else
                                            [None for i in range(len(raw_f))],
                               "Temp_Min": temp_min if self.temperature is not None else
                                           [None for i in range(len(raw_f))]})

        if type(self.rolling_mean) is int:
            df_out["RollPower"] = df_out.rolling(window=self.rolling_mean).mean()["Power"]

        return df_out

    def plot_process_frequency(self):
        fig, axes = plt.subplots(3, sharex='col', figsize=(10, 6))
        axes[0].plot(self.df_epochs["Index"]/self.fs if self.timestamps is None else
                     self.timestamps[::int(self.fs*self.epoch_len)],
                     self.df_epochs["DomF"], color='red')
        axes[0].set_title("Dominant Freq")
        axes[0].set_ylabel("Hz")

        if self.nw_gs is not None and self.timestamps is not None:
            prefix = "" if type(self.rolling_mean) is not int else "Roll"
            for row in self.nw_gs.itertuples():
                if row.Start < self.timestamps[-1]:
                    axes[0].fill_between(x=[row.Start, row.Stop], y1=0, y2=self.df_epochs["DomF"].max()*1.1,
                                         color='dodgerblue', alpha=.5)
                    axes[1].fill_between(x=[row.Start, row.Stop],
                                         y1=self.df_epochs[f"{prefix}Power"].min()*1.1,
                                         y2=self.df_epochs[f"{prefix}Power"].max()*1.1,
                                         color='dodgerblue', alpha=.5)
                    axes[2].fill_between(x=[row.Start, row.Stop], y1=min(self.signal)*1.1, y2=max(self.signal)*1.1,
                                         color='dodgerblue', alpha=.5)

        axes[1].plot(self.df_epochs["Index"]/self.fs if self.timestamps is None else
                     self.timestamps[::int(self.fs*self.epoch_len)],
                     self.df_epochs["Power"], color='purple', label="'Raw' power")

        if type(self.rolling_mean) is int:
            axes[1].plot(self.df_epochs["Index"]/self.fs if self.timestamps is None else
                         self.timestamps[::int(self.fs*self.epoch_len)],
                         self.df_epochs["RollPower"], color='black', label=f"{self.rolling_mean} epochs power")

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
        -df_epochs: df output from process_epochs()
        -use_rolling: boolean for whether to use "raw" epoched data or rolling mean data in df_epochs
        -min_dur: minimum duration of nonwear bout, minutes
        -min_nw_break: minimum time between consecutive nonwear bouts to treat as separate, i.e. maximum time between
                       bouts to not combine into one large bout
        -raw_data: ECG signal
        -fs: sample rate of raw_data, Hz
        -timestamps: timestamps for raw_data
        -freq_thresh: frequency threshold.
                     - Wear periods are marked if the df_epochs["Power"] frequency value is below threshold
        -show_plot: boolean
        """

        print("\nRunning algorithm to find non-wear periods...")

        # Sets column name prefix to use/not use rolling mean data
        if type(self.rolling_mean) is int:
            prefix = "Roll"
        if not type(self.rolling_mean) is int:
            prefix = ""

        # Start/stop 1-s epoch index numbers
        invalid_starts = []
        invalid_stops = []

        # Needed if data collection begins with nonwear period
        if self.df_epochs.iloc[0][f"{prefix}Power"] > self.freq_thresh:
            invalid_starts.append(0)

        # Loops epochs looking for start and stop of potential nonwear bouts
        for i in range(self.df_epochs.shape[0] - 1):

            # Progress printout
            if i % 1000 == 0:
                print(f"-{round(100*i/self.df_epochs.shape[0], 2)}%")

            # Start of potential nonwear period
            if self.df_epochs[f"{prefix}Power"].iloc[i] < self.freq_thresh and \
               self.df_epochs[f"{prefix}Power"].iloc[i+1] >= self.freq_thresh:
                invalid_starts.append(int(self.epoch_len*i))

            # End of potential nonwear period
            if self.df_epochs[f"{prefix}Power"].iloc[i] >= self.freq_thresh and \
                    self.df_epochs[f"{prefix}Power"].iloc[i+1] < self.freq_thresh:
                invalid_stops.append(int(self.epoch_len*(i+1)))

        # Corrects data if collection ends on nonwear bout
        if len(invalid_starts) > len(invalid_stops):
            invalid_stops.append(int(self.df_epochs.shape[0]*self.epoch_len))

        # calculates what % of each potential bout is valid
        perc_valids = []
        for start, stop in zip(invalid_starts, invalid_stops):
            d = self.df_epochs.iloc[int(start/self.epoch_len):int(stop/self.epoch_len)]["ECG_Valid"]
            x = [i for i in d]
            perc = 100 * x.count("Valid")/len(x)
            perc_valids.append(perc)

        # Removes invalid bouts less than min_dur long
        long_start = []
        long_stop = []
        """for start, stop, validity in zip(invalid_starts, invalid_stops, perc_valids):
            if (stop - start) >= (self.min_nw_dur*60) and validity <= 50:
                long_start.append(start)
                long_stop.append(stop)"""

        for start, stop, validity in zip(invalid_starts, invalid_stops, perc_valids):

            # Indexes if potential bout is <50% valid
            if (stop - start) >= (self.min_nw_dur*60) and validity <= 50:
                long_start.append(start)
                long_stop.append(stop)

            # Rare instance where nonwear followed by good quality but high frequency noise data
            # Finds longest invalid sub-bout within potential nonwear bout --> becomes nonwear bout
            if (stop - start) >= (self.min_nw_dur*60) and validity > 50:

                # self.df_epochs data for potential nonwear period
                data = self.df_epochs.iloc[int(start/self.epoch_len):int(stop/self.epoch_len)]

                # Calculates streak length of consecutive "Invalids"...apparently
                df = data[data['ECG_Valid'] == "Invalid"].groupby((data["ECG_Valid"] != "Invalid").cumsum())

                consec_lens = []
                for k, v in df["ECG_Valid"]:
                    consec_lens.append(v.shape[0])

                """THIS WILL MISBEHAVE IF THERE IS A TIE FOR LONGEST STREAK"""
                if max(consec_lens) * self.epoch_len >= self.min_nw_dur * 60:

                    # Needed if multiple streaks tied for longest duration
                    for i in np.where(np.array(consec_lens) == max(consec_lens))[0]:
                        redo_start_ind = start + i
                        redo_end_ind = redo_start_ind + int(self.epoch_len * max(consec_lens))

                    # redo_start_ind = start + np.where(np.array(consec_lens) == max(consec_lens))[0][0]
                    # redo_end_ind = redo_start_ind + int(self.epoch_len * max(consec_lens))

                    long_start.append(redo_start_ind)
                    long_stop.append(redo_end_ind)

        # Combines consecutive bouts if the gap between them is less than min_nw_break long
        for start, stop in zip(long_start[1:], long_stop[:]):
            if start - stop < int(self.min_nw_break*60):
                long_start.remove(start)
                long_stop.remove(stop)

        # Creates binary list for 1-sec epochs of wear (0) and nonwear (1) status
        freq_data = np.zeros(int(self.df_epochs.shape[0]*self.epoch_len))
        for start, stop in zip(long_start, long_stop):
            freq_data[start:stop] = 1

        print("Complete.")

        # dataframe if start time not given - uses indexes that correspond to 1-sec epochs
        if self.start_time is None:
            df_bound = pd.DataFrame({"bout_num": np.arange(0, len(long_start)),
                                     "start_index": long_start, "stop_index": long_stop})

            # Duration in seconds since indexes reference to 1-s epoch data
            df_bound["duration"] = df_bound["end_index"] - df_bound["start_index"]

        # dataframe if start time given
        if self.start_time is not None:
            if len(long_start) > 0:
                df_bound = pd.DataFrame({"bout_num": np.arange(0, len(long_start)),
                                         "bout_start": [self.start_time + td(seconds=int(i)) for i in long_start],
                                         "bout_end": [self.start_time + td(seconds=int(i)) for i in long_stop]})

                df_bound["bout_start"] = df_bound["bout_start"].round("1S")
                df_bound["bout_end"] = df_bound["bout_end"].round("1S")
                df_bound["duration"] = [(j-i).total_seconds() for i, j in
                                        zip(df_bound["bout_start"], df_bound["bout_end"])]
            if len(long_start) == 0:
                df_bound = pd.DataFrame({"bout_start": [], "bout_end": [], "duration": []})

        print(f"\nTotal nonwear time = {round(df_bound['duration'].sum()/60, 1)} "
              f"minutes ({df_bound.shape[0]} periods found)")

        if self.nw_gs is not None:
            df = self.nw_gs.loc[(self.nw_gs["Start"] >= self.timestamps[0]) &
                                (self.nw_gs["Stop"] <= self.timestamps[-1])]
            print(f"    -Reference data contains {round(df.DurMin.sum(), 1)} minutes of nonwear.")

        df_bound["start_dp"] = [int((row.bout_start - self.start_time).total_seconds() * self.fs) for
                                row in df_bound.itertuples()]
        df_bound["end_dp"] = [int((row.bout_end - self.start_time).total_seconds() * self.fs) for
                              row in df_bound.itertuples()]

        return df_bound, freq_data

    def plot_algorithm_output(self, downsample_ratio=1, overlay_invalid=False, show_ecg_bp=False):

        if type(self.rolling_mean) is int:
            prefix = "Roll"
        if not type(self.rolling_mean) is int:
            prefix = ""

        subplot_dict = {"CP": 0, "ECG": 1}
        n_plots = 2

        if self.temperature is not None:
            n_plots += 1
            subplot_dict["Temp"] = 2

        if self.accel is not None:
            n_plots += 1
            if self.temperature is None:
                subplot_dict["Accel"] = 2
            if self.temperature is not None:
                subplot_dict["Accel"] = 3

        fig, axes = plt.subplots(n_plots, sharex='col', figsize=(11, 7))
        plt.subplots_adjust(top=.925, bottom=.08, right=.925, left=.1, hspace=.25)

        # Power timestamps
        ts = self.timestamps[::int(self.epoch_len*self.fs)] if self.timestamps is not None else \
            np.arange(self.df_epochs.shape[0])*self.epoch_len

        axes[subplot_dict["CP"]].plot(ts[:min(len(ts), self.df_epochs.shape[0])],
                                      self.df_epochs[f"{prefix}Power"].iloc[:min(len(ts), self.df_epochs.shape[0])],
                                      color='dodgerblue')
        axes[subplot_dict["CP"]].axhline(y=self.freq_thresh,
                                         color='red', linestyle='dashed', label=f"{self.freq_thresh}Hz")
        axes[subplot_dict["CP"]].set_title(f"Frequency at {self.power_percent_thresh}% Cumulative Power")
        axes[subplot_dict["CP"]].set_ylabel("Hz")
        axes[subplot_dict["CP"]].legend(loc='upper left')
        axes[subplot_dict["CP"]].set_yticks([0, 30, 60, 90, 120])

        # ECG timestamps
        ts = self.timestamps[::downsample_ratio] if self.timestamps is not None else \
            np.arange(0, len(self.signal))[::downsample_ratio]/self.fs
        axes[subplot_dict["ECG"]].plot(ts[:min(len(ts), len(self.signal[::downsample_ratio]))],
                                       self.signal[::downsample_ratio],
                                       color='red')

        # .67-30Hz BP
        if show_ecg_bp:
            axes[subplot_dict["ECG"]].plot(ts[:min(len(ts), len(self.signal[::downsample_ratio]))],
                                           filter_signal(data=self.signal, sample_f=self.fs,
                                                         filter_type='bandpass', low_f=.67, high_f=30,
                                                         filter_order=3)[::downsample_ratio],
                                           color='black', zorder=0)

        axes[subplot_dict["ECG"]].set_title("ECG Signal")

        nw_fill = [min(self.signal), max(self.signal)]
        for bout in self.df_nw.itertuples():
            try:
                axes[subplot_dict["ECG"]].fill_between(x=[self.timestamps[bout.start_dp], self.timestamps[bout.end_dp]],
                                                       y1=0, y2=nw_fill[1]*1.05, color='purple', alpha=.3, zorder=1)
            except IndexError:
                axes[subplot_dict["ECG"]].fill_between(x=[self.timestamps[bout.start_dp], self.timestamps[-1]],
                                                       y1=0, y2=nw_fill[1]*1.05, color='purple', alpha=.3, zorder=1)

        if self.nw_gs is not None and self.timestamps is not None:
            for row in self.nw_gs.itertuples():
                axes[subplot_dict["ECG"]].fill_between(x=[row.Start, row.Stop], y1=nw_fill[0]*1.05, y2=0,
                                                       color='green', alpha=.3)
        axes[subplot_dict["ECG"]].set_yticks([])
        axes[subplot_dict["ECG"]].set_ylabel("Voltage")

        # TEMPERATURE ------
        if self.temperature is not None:
            axes[subplot_dict["Temp"]].plot(self.temp_timestamps, self.temperature, color='darkorange')
            axes[subplot_dict["Temp"]].axhline(y=30, color='darkorange', linestyle='dotted')
            axes[subplot_dict["Temp"]].set_ylabel("Deg. C")

        if self.accel is not None:
            axes[subplot_dict["Accel"]].plot(self.df_epochs["Timestamp"], self.df_epochs["VM_SD"], color='grey')
            axes[subplot_dict["Accel"]].set_ylabel("VM SD (mG)")

            """TEMPORARY"""
            if overlay_invalid:
                ax3 = axes[subplot_dict["Accel"]].twinx()
                ax3.plot(self.timestamps[::int(self.fs*self.epoch_len)], self.df_epochs["ECG_Valid"], color='green')

            axes[subplot_dict["Accel"]].set_ylabel("mG")


a = BittiumNonwear(signal=f,
                   # temperature=ecg.signals[5],
                   # tri_accel=np.array([ecg.signals[1], ecg.signals[2], ecg.signals[3]]),
                   start_time=timestamp[0], power_percent_thresh=80, freq_thresh=55,
                   sample_rate=250, temp_sample_rate=1, accel_sample_rate=25,
                   epoch_len=15, min_nw_dur=3, min_nw_break=5,
                   rolling_mean=False, reference_nonwear=df_nw)
# del ecg

a.df_epochs = a.process_epochs(use_orphanidou=True)
# a.df_epochs = process_epochs(a, use_orphanidou=True)
a.df_nw, a.nw_timeseries = a.run_power_nonwear()
# a.df_nw, a.nw_timeseries = run_power_nonwear(a)
a.plot_algorithm_output(downsample_ratio=5, overlay_invalid=False, show_ecg_bp=False)
# plot_algorithm_output(a, downsample_ratio=5, overlay_invalid=False, show_ecg_bp=False)

# a.df_nw.to_excel(f"/Users/kyleweber/Desktop/{subj}_FinalNW.xlsx", index=False)
# plt.savefig(f"{subj}_NW_Output.tiff", dpi=125)
