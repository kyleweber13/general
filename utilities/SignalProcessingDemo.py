import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft
from scipy.signal import butter, filtfilt, iirnotch


class FilterTest:
    """ Class used to demonstrate filtering and Fast Fourier Transforms """

    def __init__(self,
                 freqs: list or tuple = (),
                 filter_type: str = "lowpass",
                 notch_f: float or int = 60,
                 low_f: float or int = 1,
                 high_f: float or int = 10,
                 n_seconds: int or float = 10,
                 filter_order: int = 3,
                 plot_data: bool = True):
        """Class that is designed to demonstrate how sine wave addition, filtering, and FFTs work.
            Plots raw data, filtered data, raw data FFT, and filtered data FFT on single plot.

        :arguments
        -freqs: list of frequencies to include in the raw data. e.g. freqs=[1, 3, 5].
                A sine wave for each frequency is generated and added together.
                Waves are generated at 200Hz so select input frequencies accordingly.
        -filter_type: type of filter; "lowpass", "highpass", or "bandpass"
        -low_f: frequency used if filter_type=="lowpass" or low-end cutoff frequency for "bandpass"
        -high_f: frequency used if filter_type=="highpass" or high-end cutoff frequency for "bandpass"
        -n_seconds: signal length in seconds
        -filter_order: integer or float
        -plot_data: shows raw and filtered data and their respective FFT if True
        """

        self.fs = 200
        self.length = int(n_seconds * self.fs)
        self.x = np.arange(self.length)
        self.n_seconds = n_seconds

        self.freqs = freqs
        self.filter_type = filter_type
        self.low_f = low_f
        self.high_f = high_f
        self.notch_f = notch_f
        self.filter_order = filter_order

        self.raw_data = []
        self.filtered_data = []

        self.raw_fft = None
        self.filtered_fft = None

        self.data_freq = []
        self.filter_details = None

        # RUNS METHODS
        self.create_raw_wave()
        self.filter_signal()

        self.df_fft = pd.DataFrame({"freq": np.linspace(0.0, 1.0 / (2.0 * (1 / self.fs)), self.length // 2),
                                    "raw_power": 2.0 / self.length / 2 * np.abs(self.raw_fft[0:self.length // 2]),
                                    'filt_power': 2.0 / self.length / 2 * np.abs(self.filtered_fft[0:self.length // 2])})

        if plot_data:
            self.plot_data()

    def create_raw_wave(self):
        """Creates a sine wave that is the addition of all frequencies specified in self.freqs"""

        print("Generating signal from sine wave(s) with {} Hz frequency.".format(self.freqs))
        data = []

        for f in self.freqs:
            y = np.sin(2 * np.pi * f * self.x / self.fs)
            data.append(y)

        net_wave = np.sum(data, axis=0)

        self.raw_data = net_wave
        self.data_freq = self.freqs

        self.raw_fft = scipy.fft.fft(self.raw_data)

    def filter_signal(self):
        """Filters data using details specified by self.filter_type, self.low_f, self.high_f, and self.filter_order.
        """

        nyquist_freq = 0.5 * self.fs

        if self.filter_type == "lowpass":
            print("Running a {}Hz lowpass filter.".format(self.low_f))
            self.filter_details = "{}Hz lowpass".format(self.low_f)

            low = self.low_f / nyquist_freq
            b, a = butter(N=self.filter_order, Wn=low, btype="lowpass")
            filtered_data = filtfilt(b, a, x=self.raw_data)

        if self.filter_type == "highpass":
            print("Running a {}Hz highpass filter.".format(self.high_f))
            self.filter_details = "{}Hz highpass".format(self.high_f)

            high = self.high_f / nyquist_freq
            b, a = butter(N=self.filter_order, Wn=high, btype="highpass")
            filtered_data = filtfilt(b, a, x=self.raw_data)

        if self.filter_type == "bandpass":
            print("Running a {}-{}Hz bandpass filter.".format(self.low_f, self.high_f))
            self.filter_details = "{}-{}Hz bandpass".format(self.low_f, self.high_f)

            low = self.low_f / nyquist_freq
            high = self.high_f / nyquist_freq

            b, a = butter(N=self.filter_order, Wn=[low, high], btype="bandpass")
            filtered_data = filtfilt(b, a, x=self.raw_data)

        if self.filter_type == 'notch':

            notch_f = self.notch_f / nyquist_freq

            b, a = iirnotch(w0=notch_f, Q=30, fs=self.fs)
            filtered_data = filtfilt(b, a, x=self.raw_data)

        self.filtered_data = filtered_data
        self.filtered_fft = scipy.fft.fft(filtered_data)

    def plot_data(self):
        """ Plots raw and filtered signals (left column) and their respective FFT (right column)"""

        fig, ax = plt.subplots(2, 2, sharex='col', sharey='col', figsize=(12, 8))

        t = self.x / self.fs

        plt.suptitle(f"Filtering test with {self.freqs} Hz sine wave{'s' if len(self.freqs) > 1 else ''}")

        # Raw + filtered data ----------------------------------------------------------------------------------------
        ax[0][0].set_title("Raw and Filtered Data")
        ax[0][0].plot(t, self.raw_data,
                      label=f"{self.data_freq} sine wave{'s' if len(self.freqs) > 1 else ''}", color='red')
        ax[0][0].legend()
        ax[0][0].set_ylabel("Amplitude")
        ax[0][0].grid()

        ax[1][0].plot(t, self.filtered_data, label=self.filter_details, color='black')
        ax[1][0].legend()
        ax[1][0].set_ylabel("Amplitude")
        ax[1][0].set_xlabel("Seconds")
        ax[1][0].grid()
        ax[1][0].set_xlim(0, t[-1] + .01)
        ax[1][0].set_yticks(np.arange(-len(self.freqs), len(self.freqs)+.01))
        ax[1][0].set_ylim(-len(self.freqs) - .05, len(self.freqs)+.05)

        # FFT data ---------------------------------------------------------------------------------------------------

        ax[0][1].set_title("Fast Fourier Transform Data")
        ax[0][1].plot(self.df_fft['freq'], self.df_fft['raw_power'],
                      color='red', label="{} sine wave(s)".format(self.data_freq))
        ax[0][1].set_ylabel("Power")
        ax[0][1].legend()

        ax[1][1].plot(self.df_fft['freq'], self.df_fft['filt_power'], color='black', label=self.filter_details)

        ax[1][1].set_ylabel("Power")
        ax[1][1].set_xlabel("Frequency (Hz)")
        ax[1][1].legend()
        ax[1][1].set_xlim(0, self.fs / 2)
        ax[1][1].set_ylim(0, )

        ax[1][1].set_xlim(0, max(self.freqs) + 5)

        plt.tight_layout()


class SampleRateTest:
    """ Class used to demonstrate how sampling rate affects how well signals of varying frequencies are
        representing and aliasing error"""

    def __init__(self,
                 freqs: tuple or list = (),
                 n_seconds: int or float = 10,
                 sample_f: int = 250,
                 show_plot: bool = True):
        """ Creates a sine wave that is the addition of sine waves of given frequencies and specified duration
            as a 2000Hz 'analog' signal which is resampled at the given sample rate

            :arguments
            -freqs: list/tuple of frequencies used to generate the signal
            -n_seconds: length of the signal to generate in seconds
            -sample_f: sample rate used to resample the 'analog' signal
            -show_plot: boolean to show results
        """

        self.n_seconds = n_seconds
        self.length = int(self.n_seconds * 2000)  # Number of samples
        self.x = np.arange(self.length)
        self.fs = sample_f  # sampling rate
        self.ds_ratio = 1
        self.freqs = freqs

        self.sine_waves = []
        self.analog_data = []
        self.digital_data = []

        # RUNS METHODS
        self.create_analog_wave()
        self.resample()

        if show_plot:
            self.plot_data()

    def create_analog_wave(self):
        """Creates a sine wave that is the addition of all frequencies specified in self.freqs"""

        print("Generating signal from sine wave(s) with {} Hz frequency.".format(self.freqs))
        self.sine_waves = []

        for f in self.freqs:
            y = np.sin(2 * np.pi * f * self.x / 2000)
            self.sine_waves.append(y)

        net_wave = np.sum(self.sine_waves, axis=0)

        self.analog_data = net_wave

    def resample(self):
        """Resamples the 'analog' signal by taking every nth datapoint to correspond to self.sample_rate """

        if self.fs >= 2000:
            print("Sample rate too high. Try again with sample rate < 2000 Hz")

        if self.fs < 2000:
            self.ds_ratio = int(2000 / self.fs)
            self.fs = round(2000 / self.ds_ratio, 1)
            self.digital_data = self.analog_data[::self.ds_ratio]

    def plot_data(self):
        """ Plots results:
            -Top len(self.freqs) plots show each individual 'analog' sine wave (blue) and its
             resampled ('digital') signal (red). Red markers indicate where the digital samples are.
            -Bottom subplot: summed 'analog' sine wave signal (black) and its resampled ('digital') signal (red). Red
             markers indicate where the digital samples are.
        """

        if len(self.freqs) == 1:
            fig, ax = plt.subplots(1, figsize=(10, 8), sharex='col')

        if len(self.freqs) > 1:
            h = [.66] * len(self.freqs)
            h.append(1)

            fig, ax = plt.subplots((len(self.freqs) + 1),  figsize=(10, 8),
                                   sharex='col', gridspec_kw={'height_ratios': h})

        plt.suptitle(f"{self.freqs} Hz sine wave{'s' if len(self.freqs) > 1 else ''} resampled to {self.fs} Hz")

        ds_t = np.arange(len(self.digital_data)) / self.fs

        for ax_i, freq in enumerate(self.sine_waves):
            curr_ax = ax if len(self.freqs) == 1 else ax[ax_i]
            curr_ax.plot(self.x / 2000, freq, color='dodgerblue', lw=3, label='analog')
            curr_ax.plot(ds_t, freq[::self.ds_ratio], color='red', marker='o', markersize=4, label=f"fs={self.fs}Hz")
            curr_ax.legend()

            if len(self.freqs) > 1:
                curr_ax.set_title(f"{self.freqs[ax_i]} Hz sine wave")

            curr_ax.set_ylabel("Amplitude")
            curr_ax.set_yticks([-1, 0, 1])
            curr_ax.grid()

        if len(self.freqs) > 1:
            curr_ax = ax[-1]
        if len(self.freqs) == 1:
            curr_ax = ax

        curr_ax.plot(self.x / 2000, self.analog_data, color='black', label='analog', lw=3)
        curr_ax.plot(ds_t, self.digital_data,
                     color='red', label=f'fs={self.fs} Hz', marker='o', markersize=4)
        curr_ax.legend()
        curr_ax.set_xlim(0, self.n_seconds)
        curr_ax.set_xlabel("Seconds")
        curr_ax.set_ylabel("Amplitude")
        curr_ax.set_title(f"{self.freqs} Hz sine wave{'s' if len(self.freqs) > 1 else ''}")
        curr_ax.grid()
        curr_ax.set_yticks(np.arange(-len(self.freqs), len(self.freqs)+.01))
        plt.tight_layout()


class STFTTest:
    """ Class used to demonstrate Short Time Fourier Transform on signal with sequential
        sine waves of different frequencies"""

    def __init__(self,
                 freqs: tuple or list = (),
                 n_seconds: int = 10):
        """Class that generates sequential sine waves of specified frequencies, runs STFT and plots results.

            arguments:
            -freqs: list/tuple of frequencies used to generate sine waves
            -n_seconds: duration of each frequency's sine wave in seconds"""

        self.fs = 200
        self.length = int(n_seconds * self.fs)  # Number of samples
        self.x = np.arange(self.length)
        self.n_seconds = n_seconds

        self.freqs = freqs
        self.freq_res = 1

        self.data = []

        self.stft = None

        # RUNS METHODS
        self.create_wave()

    def create_wave(self):
        """Creates a single wave with varying frequencies using those in self.freqs.
           Sine waves are all equal-duration and appended to one another"""

        print("Generating signal from sine wave(s) with {} Hz frequency.".format(self.freqs))
        data = []

        for f in self.freqs:
            y = np.sin(2 * np.pi * f * self.x / self.fs)
            data = np.append(data, y)

        self.data = data

        self.stft = scipy.fft.fft(self.data)

    def plot_stft(self,
                  ylim: tuple or list or None = None,
                  nperseg_multiplier: int = 5,
                  plot_data: bool = True):
        """ Plots input signal with Short Time Fourier Transform heatmap

            arguments:
            -ylim: if not None, sets y-axis limit on STFT subplot
            -nperseg_multiplier: segment duration multiplier for STFT analysis. Affects temporal/frequency resolution
            -plot_data: plots input signal and STFT heatmap if True
        """

        self.freq_res = 1 / (self.fs * nperseg_multiplier / self.fs)

        f, t, Zxx = scipy.signal.stft(x=self.data, fs=self.fs,
                                      nperseg=self.fs * nperseg_multiplier, window='hamming')

        if plot_data:
            fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 8))
            ax1.set_title(f"{self.freqs}Hz sine wave{'s' if len(self.freqs) != 1 else ''}")
            ax2.set_title(f"STFT: {nperseg_multiplier}-second and {self.freq_res:.5f} Hz resolution")

            ax1.plot(np.arange(0, len(self.data)) / self.fs, self.data, color='black')
            ax1.set_yticks([-1, 0, 1])
            ax1.set_ylim(-1.025, 1.025)
            ax1.grid()

            pcm = ax2.pcolormesh(t, f, np.abs(Zxx), cmap='turbo', shading='auto')

            cbaxes = fig.add_axes([.91, .11, .03, .35])
            cb = fig.colorbar(pcm, ax=ax2, cax=cbaxes)

            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Seconds')

            if ylim is not None:
                ax2.set_ylim(ylim)

            ax2.set_xlim(0, len(self.data)/self.fs)

            plt.subplots_adjust(top=0.95, bottom=0.075, left=0.075, right=0.9, hspace=0.2, wspace=0.2)

        return fig, f, t, Zxx


sample_filter = FilterTest(freqs=[1], filter_type="bandpass", low_f=.1, high_f=8, filter_order=5, n_seconds=2, plot_data=True)

# frequencies and signal duration linked to FilterTest class instance
# sample_rate = SampleRateTest(freqs=sample_filter.freqs, n_seconds=sample_filter.n_seconds, sample_f=10, show_plot=True)

# sample_stft = STFTTest(freqs=(.5, 5, 10), n_seconds=10)
# fig, f, t, Zxx = sample_stft.plot_stft(nperseg_multiplier=5, plot_data=True, ylim=(0, 12))

