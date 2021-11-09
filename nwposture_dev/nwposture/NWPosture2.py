# Author: Arslan Salikhov
# Date: July 20th 2021
# Edits: Kyle Weber, November 2021

# Algorithm adapted from Fortune, Lugade, & Kaufman (2014). Posture and movement classification:
# the comparison of tri-axial accelerometer numbers and anatomical placement. J Biomech Eng. 136(5).

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
from tqdm.auto import tqdm
import pywt
from scipy.signal import butter, lfilter, filtfilt, iirnotch
import peakutils
import datetime
from datetime import timedelta as td


class NWPosture:

    def __init__(self, chest_dict, ankle_dict, gait_bouts=None, epoch_length=1,
                 subject_id="", study_code="", coll_id=""):
        """Class that processes chest- and ankle-worn accelerometer data to determine postures.

        Algorithm description:
            -Posture output relies on the combination of chest/lower leg postures, detected gait bouts,
            a sit-to-stand detector, and a whackload of logic.

        arguments:
        -subject_id, study_code, coll_id: strings, details of collection
        -chest_dict, ankle_dict: dictionary used to specify axis orientations and input data.
            -Required format:
                -{"Anterior": array, "Up": array, "Left": array, "start_stamp": timestamp, "sample_rate": integer}
        -gait_bouts: output from nwgait, either as pandas df or csv/excel output
        -epoch_length: interval over which postures are calculated
        """

        print("\n========================== Processing posture data ==========================")

        self.chest = chest_dict
        self.ankle = ankle_dict
        self.epoch_length = epoch_length
        self.study_code = study_code
        self.subject_id = subject_id
        self.coll_id = coll_id

        # RUNS METHODS
        self.crop_data()
        self.df_gait = self.load_gait_data(gait_bouts)
        self.gait_mask = self.create_gait_mask()

    @ staticmethod
    def load_gait_data(object):
        """Loads required output from nwgait either as an object or from csv/excel file.
        :return
        -df: dataframe object
        """

        if type(object) is str:
            if "xlsx" in object:
                df = pd.read_excel(object)
            if "csv" in object:
                df = pd.read_csv(object)
        if type(object) is pd.core.frame.DataFrame:
            df = object
        if object is None:
            df = None

        return df

    @ staticmethod
    def epochinator3000(data, sample_rate, epoch_length=1):
        """Epochs data into given epoch length using an averaging function.

        argument:
        -data: array of raw data (1D)
        -sample_rate: data's sample rate in Hz
        -epoch_length: integer, epoch length in seconds

        return:
        -epoched_data: array
        """

        epoched_data = []

        for i in tqdm(range(0, len(data), int(sample_rate) * epoch_length)):
            stepList = data[i:i + 1]
            epoched_data.append(np.mean(stepList))

        return epoched_data

    def orientator3000(self, anterior_angle, anterior, up, left):
        """Method to determine Bittium Faros' orientation (vertical or horizontal). Not currently used since
           code was re-written and requires explicit input of each axes' orientation.
        """

        # Only includes periods of time when torso is upright (anterior axis ~ horizontal = ~90 degrees)
        indexes = []
        for i, val in enumerate(anterior_angle):
            if 80 <= val <= 100:
                indexes.append(i)

        ant_test = anterior[indexes]
        up_test = up[indexes]
        left_test = left[indexes]
        angle_ant = np.arccos(ant_test / np.sqrt(np.square(ant_test) + np.square(up_test) + np.square(left_test)))
        angle_ant *= 180 / np.pi

        angle_up = np.arccos(up_test / np.sqrt(np.square(ant_test) + np.square(up_test) + np.square(left_test)))
        angle_up *= 180 / np.pi

        angle_left = np.arccos(left_test / np.sqrt(np.square(ant_test) + np.square(up_test) + np.square(left_test)))
        angle_left *= 180 / np.pi

        left_med = np.median(angle_left)

        # If median 'left' axis angle is > 60 degrees during upright torso periods, device is horizontal
        if left_med >= 60:
            orient = 1  # Horizontal
        if left_med <= 15:
            orient = 0  # Vertical

        return orient

    def create_gait_mask(self):
        """Converts gait bout data to binary list of gait (1) or no gait (0) that corresponds to given epoch length."""

        duration = int(len(self.chest['Anterior']) / self.chest['sample_rate'] / self.epoch_length)

        # Binary list of gait (1) or no gait (0) in 1-sec increments
        gait_mask = np.zeros(duration)

        if self.df_gait is None:
            return gait_mask

        # NWGait format --------------------------------------------
        try:
            if self.df_gait is not None:
                for row in self.df_gait.itertuples():
                    start = int((row.start_timestamp - self.chest['start_stamp']).total_seconds())
                    stop = int((row.end_timestamp - self.chest['start_stamp']).total_seconds())
                    gait_mask[int(start):int(stop)] = 1

        # Posture GS format -----------------------------------------
        except (KeyError, AttributeError):
            if self.df_gait is not None:
                for row in self.df_gait.itertuples():
                    start = int((row.Start - self.chest['start_stamp']).total_seconds())
                    stop = int((row.Stop - self.chest['start_stamp']).total_seconds())
                    gait_mask[int(start):int(stop)] = 1

        return gait_mask

    def crop_data(self):
        """Crops chest and ankle data so they start and end at the same time."""

        print("\nChecking data files to ensure same start time...")

        if self.ankle['start_stamp'] == self.chest['start_stamp']:
            print("Files already start at same time. No cropping performed.")

        # Crops chest data if chest file started first
        if self.ankle['start_stamp'] > self.chest['start_stamp']:
            crop_ind = int((self.ankle['start_stamp'] - self.chest['start_stamp']).total_seconds() * \
                           self.chest['sample_rate'])

            self.chest['Anterior'] = self.chest['Anterior'][crop_ind:]
            self.chest['Up'] = self.chest['Up'][crop_ind:]
            self.chest['Left'] = self.chest['Left'][crop_ind:]
            self.chest['start_stamp'] = self.ankle['start_stamp']
            print(f"Cropped chest data by {crop_ind} datapoints.")

        # Crops chest data if chest file started first
        if self.ankle['start_stamp'] < self.chest['start_stamp']:
            crop_ind = int((self.chest['start_stamp'] - self.ankle['start_stamp']).total_seconds() * \
                           self.ankle['sample_rate'])

            self.ankle['Anterior'] = self.ankle['Anterior'][crop_ind:]
            self.ankle['Up'] = self.ankle['Up'][crop_ind:]
            self.ankle['Left'] = self.ankle['Left'][crop_ind:]
            self.ankle['start_stamp'] = self.chest['start_stamp']

            print(f"Cropped ankle data by {crop_ind} datapoints.")

    def ankle_posture(self, anterior, up, left, epoch_ts, plot_results=False):
        """Ankle posture detection. Classifies shank orientation as horizontal, vertical, angled, or gait.
           Based on trig function used and axis orientations, here is the angle interpretation:
                0 degrees: positive axis is pointing skywards (opposite gravity)
                90 degrees: axis is perpendicular to gravity
                180 degrees: positive axis is pointing towards ground
        """

        # Median filter
        ant_medfilt = signal.medfilt(anterior)
        up_medfilt = signal.medfilt(up)
        left_medfilt = signal.medfilt(left)

        # Defining a Vector Magnitude(vm) array to classify data points as either dynamic or static
        vm = np.sqrt(np.square(np.array([ant_medfilt, up_medfilt, left_medfilt])).sum(axis=0)) - 1
        vm[vm < 0.6135] = 0
        vm[vm >= 0.6135] = 1

        # Low pass eliptical filter
        sos = signal.ellip(3, 0.01, 100, 0.25, 'low', output='sos')
        ant_ellip = signal.sosfilt(sos, ant_medfilt)
        up_ellip = signal.sosfilt(sos, up_medfilt)
        left_ellip = signal.sosfilt(sos, left_medfilt)

        # Gravity to angle conversion
        ant_angle_rad = np.arccos(ant_ellip /
                                  np.sqrt(np.square(ant_ellip) + np.square(left_ellip) + np.square(up_ellip)))
        up_angle_rad = np.arccos(up_ellip /
                                 np.sqrt(np.square(ant_ellip) + np.square(left_ellip) + np.square(up_ellip)))
        left_angle_rad = np.arccos(left_ellip /
                                   np.sqrt(np.square(ant_ellip) + np.square(left_ellip) + np.square(up_ellip)))

        up_angle = np.array(up_angle_rad * 180 / np.pi)
        ant_angle = np.array(ant_angle_rad * 180 / np.pi)
        left_angle = np.array(left_angle_rad * 180 / np.pi)

        time = epoch_ts
        posture = pd.DataFrame()
        posture['timestamp'] = time
        posture['ankle_posture'] = ['other' for i in range(len(ant_angle))]
        posture['ankle_anterior'] = ant_angle
        posture['ankle_up'] = up_angle
        posture['ankle_left'] = left_angle
        posture['ankle_vm'] = pd.Series(vm)
        for i in range(posture.shape[0] - len(self.gait_mask)):
            self.gait_mask = np.append(self.gait_mask, 0)
        posture['ankle_gait_mask'] = self.gait_mask
        posture['ankle_dynamic'] = posture[['ankle_gait_mask', 'ankle_vm']].sum(axis=1)

        post_dict = {0: "other", 1: "horizontal", 2: "angled", 3: 'sit/stand', 4: 'sit', 5: 'dynamic', 6: 'gait'}

        # Static with ankle tilt 0-15 degrees (sit/stand) --> vertical shank while standing or sitting upright
        posture.loc[(posture['ankle_dynamic'] == 0) & (posture['ankle_up'] < 20), 'ankle_posture'] = "sitstand"

        # Static with ankle tilt 15-45 degrees (sit) --> too angled to be standing
        posture.loc[(posture['ankle_dynamic'] == 0) &
                    (posture['ankle_up'] < 45) & (posture['ankle_up'] >= 20),
                    'ankle_posture'] = "sit"

        # Static with shank tilt 45-110 degrees --> horizontal
        posture.loc[(posture['ankle_dynamic'] == 0)
                    & (posture['ankle_up'] < 110) & (posture['ankle_up'] >= 45), 'ankle_posture'] = "horizontal"

        # Flags gait
        posture.loc[(posture['ankle_gait_mask'] > 0), 'ankle_posture'] = "gait"

        fig = None

        if plot_results:
            fig, ax = plt.subplots(2, sharex='col', figsize=(10, 6))
            ax[0].plot(epoch_ts, posture['ankle_anterior'], color='black', label='anterior')
            ax[0].plot(epoch_ts, posture['ankle_up'], color='red', label='up')
            ax[0].plot(epoch_ts, posture['ankle_left'], color='dodgerblue', label='left')
            ax[0].legend()
            ax[0].set_title("Angles (0 = skywards)")

            ax[1].plot(epoch_ts, [post_dict[i] for i in posture['ankle_posture']], color='grey')
            ax[1].grid()

            ax[-1].xaxis.set_major_formatter(xfmt)
            plt.show()

        return posture, fig

    def trunk_posture(self, anterior, up, left, epoch_ts, plot_results=False):
        """Chest posture detection. Classifies torso orientation as sitstand (unsure), 'supine', 'prone',
           'rightside', 'leftover' or 'other'.

           Based on trig function used and axis orientations, here is the angle interpretation:
                0 degrees: positive axis is pointing skywards (opposite gravity)
                90 degrees: axis is perpendicular to gravity
                180 degrees: positive axis is pointing towards ground
        """

        # Median filter for each axis
        ant_medfilt = signal.medfilt(anterior)
        up_medfilt = signal.medfilt(up)
        left_medfilt = signal.medfilt(left)

        # Defining a Vector Magnitude(vm) array to classify data points as either dynamic or static
        vm = np.sqrt(np.square(np.array([ant_medfilt, up_medfilt, left_medfilt])).sum(axis=0)) - 1
        vm[vm < .1364] = 0
        vm[vm >= .1364] = 1

        # Low pass eliptical filter for each axis
        sos = signal.ellip(3, 0.01, 100, 0.25, 'low', output='sos')
        ant_ellip = signal.sosfilt(sos, ant_medfilt)
        up_ellip = signal.sosfilt(sos, up_medfilt)
        left_ellip = signal.sosfilt(sos, left_medfilt)

        # Gravity to angle (radians) conversion
        ant_angle_rad = np.arccos(ant_ellip /
                                  np.sqrt(np.square(ant_ellip) + np.square(left_ellip) + np.square(up_ellip)))

        up_angle_rad = np.arccos(up_ellip /
                                 np.sqrt(np.square(ant_ellip) + np.square(left_ellip) + np.square(up_ellip)))

        left_angle_rad = np.arccos(left_ellip /
                                   np.sqrt(np.square(ant_ellip) + np.square(left_ellip) + np.square(up_ellip)))

        # Converts radians to degrees cause pi is cool
        up_angle = np.array(up_angle_rad * 180 / np.pi)
        ant_angle = np.array(ant_angle_rad * 180 / np.pi)
        left_angle = np.array(left_angle_rad * 180 / np.pi)

        posture = pd.DataFrame()
        posture['timestamp'] = epoch_ts
        posture['chest_posture'] = ['other' for i in range(posture.shape[0])]
        posture['chest_left'] = left_angle
        posture['chest_anterior'] = ant_angle
        posture['chest_up'] = up_angle

        posture['chest_gait_mask'] = pd.Series(self.gait_mask)
        posture['chest_vm'] = pd.Series(vm)
        posture['chest_dynamic'] = posture[['chest_gait_mask', 'chest_vm']].sum(axis=1)

        # Sit/stand: static, 'up' axis pointing up, 'anterior' axis near perpendicular
        # anterior between 60-110 = <30 deg forward flexion/<20 deg extension
        posture.loc[(posture['chest_dynamic'] == 0) & (posture['chest_up'] < 45) |
                    (60 < posture['chest_anterior']) & (posture['chest_anterior'] < 110), 'chest_posture'] = "sitstand"

        # Prone: static with anterior axis facing ground
        # Ant. of > 145 degrees = face down
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] > 145) &
                    (60 < posture['chest_up']) & (posture['chest_up'] < 100), 'chest_posture'] = "prone"

        # Supine: static with anterior axis facing sky
        # Ant. of 60 degrees = face up with 60 deg lean back
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] < 30) &
                    (60 < posture['chest_up']) & (posture['chest_up'] < 110), 'chest_posture'] = "supine"

        # Lying on left side: static, "up" axis horizontal, "left" axis towards ground
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (60 < posture['chest_up']) & (posture['chest_up'] < 120) &
                    (posture['chest_left'] > 150), 'chest_posture'] = "leftside"

        # Lying on right side: static, "up" axis horizontal, "left" axis towards sky
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (60 < posture['chest_up']) & (posture['chest_up'] < 120) &
                    (posture['chest_left'] < 30), 'chest_posture'] = "rightside"

        # Anything dynamic (transition, gait, excess movement)
        posture.loc[(posture['chest_gait_mask'] > 0), 'chest_posture'] = "gait"

        fig = None

        if plot_results:

            fig, ax = plt.subplots(2, sharex='col', figsize=(10, 6))
            ax[0].plot(epoch_ts, posture['chest_anterior'], color='black', label='anterior')
            ax[0].plot(epoch_ts, posture['chest_up'], color='red', label='up')
            ax[0].plot(epoch_ts, posture['chest_left'], color='dodgerblue', label='left')
            ax[0].legend()
            ax[0].set_title("Angles (0 = skywards)")

            ax[1].plot(epoch_ts, posture['chest_posture'], color='grey')
            ax[1].grid()

            ax[-1].xaxis.set_major_formatter(xfmt)
            plt.show()

        return posture, fig

    def calculate_postures(self, goldstandard_df=None, show_plots=False):
        """Method that calls other methods to process data."""

        def initial_processing():
            # Data epoching ----------------------------------------------------------------------------
            ankle_ant = self.epochinator3000(data=self.ankle['Anterior'],
                                             sample_rate=self.ankle['sample_rate'],
                                             epoch_length=self.epoch_length)
            ankle_up = self.epochinator3000(data=self.ankle['Up'],
                                            sample_rate=self.ankle['sample_rate'],
                                            epoch_length=self.epoch_length)
            ankle_left = self.epochinator3000(data=self.ankle['Left'],
                                              sample_rate=self.ankle['sample_rate'],
                                              epoch_length=self.epoch_length)

            chest_ant = self.epochinator3000(data=self.chest['Anterior'],
                                             sample_rate=self.chest['sample_rate'],
                                             epoch_length=self.epoch_length)
            chest_up = self.epochinator3000(data=self.chest['Up'],
                                            sample_rate=self.chest['sample_rate'],
                                            epoch_length=self.epoch_length)
            chest_left = self.epochinator3000(data=self.chest['Left'],
                                              sample_rate=self.chest['sample_rate'],
                                              epoch_length=self.epoch_length)

            # Crops epoched data to shortest dataset --------------------------------------------------
            # Only need to crop end since data's cropped to start at same time already using self.crop_data()
            length = min([len(chest_ant), len(chest_up), len(chest_left),
                         len(ankle_ant), len(ankle_up), len(ankle_left)])

            ankle_ant = ankle_ant[:length]
            ankle_up = ankle_up[:length]
            ankle_left = ankle_left[:length]

            chest_ant = chest_ant[:length]
            chest_up = chest_up[:length]
            chest_left = chest_left[:length]

            self.gait_mask = self.gait_mask[:length]

            epoch_ts = np.asarray(pd.date_range(start=self.chest['start_stamp'],
                                                periods=length,
                                                freq=f"{self.epoch_length}S"))

            # Posture processing based on angles ------------------------------------------------------
            print("-Running individual posture detection algorithm...")
            ankle_post, fig = self.ankle_posture(anterior=ankle_ant, up=ankle_up, left=ankle_left,
                                                 epoch_ts=epoch_ts, plot_results=show_plots)

            trunk_post, fig = self.trunk_posture(anterior=chest_ant, up=chest_up, left=chest_left,
                                                 epoch_ts=epoch_ts, plot_results=show_plots)

            df_posture = trunk_post.copy()
            df_posture["ankle_posture"] = ankle_post['ankle_posture']
            df_posture["ankle_anterior"] = ankle_post['ankle_anterior']
            df_posture["ankle_up"] = ankle_post['ankle_up']
            df_posture["ankle_left"] = ankle_post['ankle_left']
            df_posture["ankle_dynamic"] = ankle_post['ankle_dynamic']
            df_posture = df_posture.drop("chest_vm", axis=1)

            # Combining ankle and chest postures ------------------------------------------------------
            print("-Combining chest and ankle data...")

            df_posture['posture'] = df_posture['chest_posture']

            # chest sit/stand + ankle sit = sit
            df_posture.loc[(df_posture['chest_posture'] == 'sitstand') &
                           (df_posture['ankle_posture'] == 'sit'), 'posture'] = "sit"

            # Chest sit/stand + ankle horizontal = sitting reclined
            df_posture.loc[(df_posture['chest_posture'] == 'sitstand') &
                           (df_posture['ankle_posture'] == 'horizontal'), 'posture'] = "sit"

            # Chest other + ankle horizontal = sitting reclined
            df_posture.loc[(df_posture['chest_posture'] == 'other') &
                           (df_posture['ankle_posture'] == 'horizontal'), 'posture'] = "sit"

            df_posture.loc[(df_posture['chest_posture'] == 'other') &
                           (df_posture['ankle_posture'] == 'dynamic'), 'posture'] = "other"

            gs = np.array(['other' for i in range(df_posture.shape[0])])

            if goldstandard_df is not None:
                df_event_use = goldstandard_df.loc[goldstandard_df['EventShort'] != "Transition"].reset_index(drop=True)

                df_start = df_posture.iloc[0]['timestamp']
                for row in df_event_use.itertuples():
                    start = int((row.Start - pd.to_datetime(df_start)).total_seconds())
                    stop = int((row.Stop - pd.to_datetime(df_start)).total_seconds())

                    gs[start:stop] = row.EventShort

            df_posture['GS'] = gs
            df_posture['GS'] = df_posture['GS'].replace({"Sitti": 'sit', "Walki": "stand", "SitRe": "sit",
                                                         "Supin": "supine", "LL": 'leftside', 'Stand': 'stand',
                                                         "LR": 'rightside', 'Prone': 'prone'})

            return df_posture

        def format_nwposture_output(df):

            print("-Flagging gait bouts as 'standing'...")
            df = df.reset_index()

            df['timestamp'] = [i.round(freq='1S') for i in df['timestamp']]

            # flags gait bouts at "Standing"
            df.loc[df["posture"] == 'gait', 'posture'] = 'stand'

            return df

        def find_posture_changes(df, colname='posture'):

            def get_index(data, start_index=0):

                data = list(data)
                current_value = data[start_index]

                for i in range(start_index, len(data) - 1):
                    if data[i] != current_value:
                        return i
                    if data[i] == current_value:
                        pass

            indexes = [0]

            for i in range(df.shape[0]):
                if i >= indexes[-1]:
                    index = get_index(data=df[colname], start_index=i)
                    indexes.append(index)
                try:
                    if i <= indexes[-1]:
                        pass
                except TypeError:
                    if df.shape[0] not in indexes:
                        indexes.append(df.shape[0] - 1)

            indexes = [i for i in indexes if i is not None]

            transitions = np.zeros(df.shape[0])
            transitions[indexes] = 1

            df["transition"] = transitions

            return indexes

        def process_for_peak_detection(obj):
            """Sit-to-stand/stand-to-sit transition detection algorithm based on ____________"""

            # Acceleration magnitudes
            vm = np.sqrt(np.square(np.array([obj.chest['Anterior'],
                                             obj.chest['Left'],
                                             obj.chest['Up']])).sum(axis=0))
            vm = np.abs(vm)

            # 3Hz lowpass
            alpha_filt = filter_signal(data=vm, low_f=3, filter_order=4,
                                       sample_f=obj.chest['sample_rate'], filter_type='lowpass')

            # .25s rolling median
            # rm_alpha = [np.mean(alpha_filt[i:i + int(obj.chest['sample_rate'] / 4)]) for i in range(len(alpha_filt))]

            # Continuous wavelet transform; focus on <.5Hz band
            c = pywt.cwt(data=alpha_filt, scales=[1, 64], wavelet='gaus1',
                         sampling_period=1 / self.chest['sample_rate'])
            cwt_power = c[0][1]

            return cwt_power

        def detect_peaks(wavelet_data, method='raw', sample_rate=100, min_seconds=1):

            if method == 'raw':
                d = wavelet_data
            if method == 'abs' or method == 'absolute':
                d = np.abs(wavelet_data)
            if method == 'le' or method == 'linear envelop' or method == 'linear envelope':
                d = filter_signal(data=np.abs(wavelet_data), low_f=.25, filter_type='lowpass',
                                  sample_f=sample_rate, filter_order=5)
                d = np.abs(d)

            power_sd = np.std(d)
            peaks = peakutils.indexes(y=d, min_dist=int(min_seconds * sample_rate), thres_abs=True,
                                      thres=power_sd * 1.5)

            return peaks, power_sd * 1.5, d

        def create_peaks_df(start_time, stop_time, sts_peaks, raw_timestamps, sample_f, calculate_peak_values):

            ant_vals, left_vals, up_vals = [], [], []

            new_peak_inds = []

            for peak in sts_peaks:
                if calculate_peak_values:
                    ant = filter_signal(data=self.chest['Anterior'][peak - int(2 * sample_f):peak + int(2 * sample_f)],
                                        sample_f=sample_f, high_f=.1, filter_type='highpass', filter_order=5)
                    ant_peak = np.argmax(np.abs(ant))
                    ant_vals.append(ant[ant_peak])

                    up = filter_signal(data=self.chest['Up'][peak - int(2 * sample_f):peak + int(2 * sample_f)],
                                       sample_f=sample_f, high_f=.1, filter_type='highpass', filter_order=5)
                    up_peak = np.argmax(np.abs(up))
                    up_vals.append(up[up_peak])

                    left = filter_signal(data=self.chest['Left'][peak - int(2 * sample_f):peak + int(2 * sample_f)],
                                         sample_f=sample_f, high_f=.1, filter_type='highpass', filter_order=5)
                    left_peak = np.argmax(np.abs(left))
                    left_vals.append(left[left_peak])

                # Finds actual local maxima and creates new peaks list
                window = processed_data[int(peak - 2 * sample_f):int(peak + 2 * sample_f)]
                peak_val = max(window)
                peak_ind = int(peak - 2 * sample_f + np.argwhere(window == peak_val))
                new_peak_inds.append(peak_ind)

            ts = raw_timestamps[new_peak_inds]

            df_out = pd.DataFrame({"timestamp": ts})
            df_out['timestamp'] = [i.round('1S') for i in df_out['timestamp']]
            df_out['Ind'] = [int((i - ts[0]).total_seconds()) for i in df_out.timestamp]

            if calculate_peak_values:
                df_out['antpeak'] = ant_vals
                df_out['uppeak'] = up_vals
                df_out['left_peak'] = left_vals

            df_out = df_out.loc[(df_out["timestamp"] >= start_time) & (df_out['timestamp'] <= stop_time)]

            return df_out.reset_index(drop=True)

        def format_posture_change_dfs(df1s, transition_indexes, df_peaks, incl_peak_vals=False):

            # Index (1s epochs) of first standing (gait) bout: used as starting point for algorithm
            try:
                first_standing = df1s.loc[df1s['posture'] == 'stand'].index[0]
            except IndexError:
                first_standing = 0

            # indexes of transitions that occur at/after first standing
            use_indexes = [i for i in transition_indexes if i >= first_standing]

            # First row in df1s for each posture --> bouts
            df_index = pd.DataFrame({"timestamp": [df1s.iloc[i]['timestamp'] for i in use_indexes],
                                     "Ind": use_indexes,
                                     "Posture": [df1s.iloc[i]['posture'] for i in use_indexes],
                                     'Type': ["transition" for i in range(len(use_indexes))]})

            # df for postures at each potential STS peak
            df_sts_peaks = pd.DataFrame({"timestamp": [df1s.iloc[i]['timestamp'] for i in df_peaks['Ind']],
                                         "Ind": df_peaks['Ind'],
                                         "Posture": [df1s.iloc[i]['posture'] for i in df_peaks['Ind']],
                                         "Type": ["Peak" for i in range(df_peaks.shape[0])]})

            if incl_peak_vals:
                df_index['antpeak'] = [None for i in range(len(use_indexes))]
                df_index['uppeak'] = [None for i in range(len(use_indexes))]
                df_index['leftpeak'] = [None for i in range(len(use_indexes))]

                df_sts_peaks['antpeak'] = df_peaks['antpeak'],
                df_sts_peaks['uppeak'] = df_peaks['uppeak'],
                df_sts_peaks['leftpeak'] = df_peaks['leftpeak'],

            # Combines two previous dataframes and formats indexes
            df_index = df_index.append(df_sts_peaks)
            df_index = df_index.sort_values("timestamp")  # chronological order
            df_index = df_index.reset_index(drop=True)

            df_use = df_index.loc[(df_index['timestamp'] >= df1s.iloc[first_standing]['timestamp']) &
                                  (df_index['Type'] == "Peak")].reset_index(drop=True)

            return df_index, df_use, first_standing

        def pass1(win_size, df_peak, df1s):

            print("-Using known standing/gait periods to adjust sit/stand classifications...")

            input = np.array(df1s['posture'])
            ankle = np.array(df1s["ankle_posture"])

            # Removes one-second transients surrounded by same posture
            for i in range(1, len(input) -1):
                if input[i-1] != input[i] and input[i-1] == input[i+1]:
                    input[i] = input[i-1]

            output_postures = input.copy()

            # Loops through each potenetial STS peak and generates window around peak
            for row in range(df_peak.shape[0]):
                curr_row = df_peak.iloc[row]
                prev_row = df_peak.iloc[row - 1] if row >= 1 else df_peak.iloc[0]
                prev_row_ind = prev_row.Ind if row >= 1 else 0

                # Windows +/- win_size seconds from potential STS peak
                window = list(input[curr_row['Ind'] - win_size:curr_row['Ind'] + win_size])
                ankle_window = list(ankle[curr_row['Ind'] - win_size:curr_row['Ind'] + win_size])

                # If standing + transition = sit/stand --> sit/stand period becomes sitting
                if "stand" in window[:win_size] and "sitstand" in window[-win_size:]:
                    output_postures[curr_row.Ind:df_peak.iloc[row + 1]['Ind']] = "sit"

                # If current row is after first standing/gait bout:
                # If sit/stand + transition = standing --> sit/stand period becomes sitting
                if "sitstand" in window[:win_size] and "stand" in window[-win_size:]:
                    output_postures[prev_row_ind:curr_row.Ind] = "sit"

                if "sit" in window[:win_size] and "sitstand" in window[-win_size:]:
                    output_postures[curr_row.Ind - win_size:curr_row.Ind] = \
                        pd.Series(window[:win_size]).replace({"sitstand": "sit"})
                    if "horizontal" not in ankle_window[:-win_size]:
                        output_postures[curr_row.Ind:df_peak.iloc[row+1]['Ind']] = 'stand'
                    if "horizontal" in ankle_window[-win_size:]:
                        output_postures[curr_row.Ind:df_peak.iloc[row+1]['Ind']] = 'sit'

            return output_postures

        def reclassify_pre_firststand(df, input, win_size, df_peaks):

            print("-Reclassifying postures detected before first standing period...")

            v2 = np.array(input)

            for i in range(df.shape[0] - 1):
                curr_row = df.iloc[i]

                curr_peak = df_peaks.loc[(df_peaks['timestamp'] >= curr_row['timestamp'] + td(seconds=-win_size)) &
                                         (df_peaks['timestamp'] <= curr_row['timestamp'] + td(seconds=win_size))]

                # If standing and no transition when standing starts --> flags previous peak to current as standing
                if v2[curr_row.Ind] == 'stand' and curr_peak.shape[0] == 0:
                    prev_peak = df_peaks.loc[df_peaks['timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
                    v2[prev_peak:curr_row.Ind] = 'stand'

                if v2[curr_row.Ind] == 'sit' and curr_peak.shape[0] == 0:
                    prev_peak = df_peaks.loc[df_peaks['timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
                    v2[prev_peak:curr_row.Ind] = 'sit'

                if v2[curr_row.Ind] == 'sit' and curr_peak.shape[0] > 0:
                    try:
                        prev_peak = df_peaks.loc[df_peaks['timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
                        v2[prev_peak:curr_row.Ind] = 'stand'
                    except IndexError:
                        pass

                if v2[curr_row.Ind] == 'stand' and curr_peak.shape[0] > 0:
                    try:
                        prev_peak = df_peaks.loc[df_peaks['timestamp'] < curr_row.timestamp].iloc[-1]['Ind']
                        v2[prev_peak:curr_row.Ind] = 'sit'
                    except IndexError:
                        pass

            return v2

        def pass2(postures):

            print("-Applying final layer of logic...")

            data = np.array(postures)

            lying = ['supine', 'prone', 'rightside', 'leftside']
            curr_ind = 0
            for i in range(1, len(data) - 1):
                if i > curr_ind:
                    # Removes spurious single-epoch 'other' classifications (likely transitions)
                    if data[i - 1] != 'other' and data[i] == 'other' and data[i + 1] != 'other':
                        data[i] = data[i - 1]
                        curr_ind = i

                    # changes sit/stand to correct lying position if same event occurs pre/post sitstand bout
                    if data[i] == 'sitstand' and data[i - 1] in lying:
                        for j in range(i + 1, len(data) - 1):
                            if data[j] == data[i]:
                                pass
                            if data[j] in lying:
                                data[i:j] = data[i - 1]
                                curr_ind = j
                                break

                    # Last value carry forward if sitstand preceeded by sit or stand
                    if data[i] == 'sitstand' and data[i - 1] in ['sit', 'stand']:
                        for k in range(i + 1, len(data) - 1):
                            if data[k] == 'sitstand':
                                pass
                            if data[k] in ['sit', 'stand']:
                                data[i:k] = data[i - 1]
                                curr_ind = k
                                break

            return data

        def format_output(df):
            df.columns = ["start_timestamp", 'posture']

            df['study_code'] = [self.study_code for i in range(df.shape[0])]
            df['subject_id'] = [self.subject_id for i in range(df.shape[0])]
            df['coll_id'] = [self.coll_id for i in range(df.shape[0])]

            end = list(df['start_timestamp'].iloc[1:])
            end.append(ts[-1].round("1S"))
            df['end_timestamp'] = end
            df['posture_bout_num'] = np.arange(1, df.shape[0] + 1)
            bouts = df[['study_code', 'subject_id', 'coll_id', 'posture_bout_num',
                        'start_timestamp', 'end_timestamp', 'posture']]

            for row in bouts.itertuples():
                remaining = bouts.iloc[row.Index:]['posture'].unique()
                if len(remaining) == 1 and row.posture == remaining[0]:
                    bouts.iloc[row.Index]['end_timestamp'] = bouts.iloc[-1]['end_timestamp']
                    break
            try:
                bouts = bouts.iloc[:row.Index + 1]
            except IndexError:
                bouts = bouts.iloc[:row.Index]

            return bouts

        t0 = datetime.datetime.now()

        ts = pd.date_range(start=self.chest['start_stamp'], periods=len(self.chest['Anterior']),
                           freq="{}ms".format(1000 / self.chest['sample_rate']))

        print("-Initial combination...")
        df_posture = initial_processing()
        df1s = format_nwposture_output(df=df_posture)
        transition_indexes = find_posture_changes(df=df_posture, colname='posture')

        print("-Finding sit-to-stand transitions...")
        cwt_power = process_for_peak_detection(obj=self)
        peaks, thresh, processed_data = detect_peaks(wavelet_data=cwt_power, method='le',
                                                     sample_rate=self.chest['sample_rate'],
                                                     min_seconds=5)

        df_peaks = create_peaks_df(start_time=df_posture.iloc[0]['timestamp'],
                                   stop_time=df_posture.iloc[-1]['timestamp'],
                                   sts_peaks=peaks, raw_timestamps=ts, sample_f=self.chest['sample_rate'],
                                   calculate_peak_values=False)

        print("-First pass on sit/stand differentiation...")
        df_index, df_use, first_stand_index = format_posture_change_dfs(df1s=df1s, df_peaks=df_peaks,
                                                                        incl_peak_vals=False,
                                                                        transition_indexes=transition_indexes)
        df1s["v1"] = pass1(win_size=8, df1s=df1s, df_peak=df_use)
        transitions_indexes2 = find_posture_changes(df=df1s, colname='v1')
        df_index2 = df1s.iloc[transitions_indexes2][["timestamp", 'posture', 'transition', 'GS', "v1"]].reset_index()

        print("-Dealing with data before first standing period...")
        prestand_index = df_index2.loc[df_index2['timestamp'] <= \
                         df1s.iloc[first_stand_index]['timestamp']].sort_values("timestamp", ascending=False).reset_index(drop=True)
        try:
            prestand_index.columns = ['Ind', 'timestamp', 'posture', 'transition', 'GS', 'v1']
        except ValueError:
            prestand_index.columns = ['Ind', 'timestamp', 'posture', 'transition', 'v1']

        df1s['v2'] = reclassify_pre_firststand(df=prestand_index, win_size=8, input=df1s['v1'],
                                               df_peaks=df_peaks.loc[df_peaks['timestamp'] <=
                                                                     df1s.iloc[first_stand_index]['timestamp']])

        print("-Second sit/stand differentiation...")
        df1s['v3'] = pass2(postures=df1s['v2'])  # final posture data

        print("Algorithm complete.")

        transitions_indexes3 = find_posture_changes(df=df1s, colname='v3')
        df_index3 = df1s.iloc[transitions_indexes3][["timestamp", "v3"]].reset_index(drop=True)
        bouts = format_output(df=df_index3)

        t1 = datetime.datetime.now()
        t_total = round((t1-t0).total_seconds(), 1)
        coll_dur = (df1s.iloc[-1]['timestamp'] - df1s.iloc[0]['timestamp']).total_seconds() / 3600

        print("\n=================================================")
        print(f"Processing time: {t_total} seconds ({round(t_total/coll_dur, 1)} sec/hr)")

        return df1s, bouts, df_peaks


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


def plot_all(df1s, df_peaks=None, show_v0=True, show_v1=True, show_v2=True, show_v3=True, collapse_lying=True):
    plt.close('all')

    fig, ax = plt.subplots(6, sharex='col', figsize=(14, 9), gridspec_kw={'height_ratios': [.75, .75, .75, .75, 1, .25]})
    ax[0].plot(df1s['timestamp'], df1s['chest_anterior'], color='black', label='Chest_ant')
    ax[0].plot(df1s['timestamp'], df1s['chest_up'], color='red', label='Chest_up')
    ax[0].plot(df1s['timestamp'], df1s['chest_left'], color='dodgerblue', label='Chest_left')
    ax[0].axhline(y=90, linestyle='dashed', color='grey', label='Perp.')
    ax[0].legend()
    ax[0].set_yticks([0, 45, 90, 135, 180])
    ax[0].set_ylabel("deg")
    ax[0].grid()

    ax[1].plot(df1s['timestamp'], df1s['chest_posture'], color='black', label='ChestPosture', zorder=1)

    if df_peaks is not None:
        for row in df_peaks.itertuples():
            if row.index == 0:
                ax[1].axvline(x=row.timestamp, color='limegreen', zorder=0, label='STS_peak',
                              linestyle='dashed' if df1s.iloc[row.Ind]['chest_gait_mask'] == 1 else "solid")
            else:
                ax[1].axvline(x=row.timestamp, color='limegreen', zorder=0,
                              linestyle='dashed' if df1s.iloc[row.Ind]['chest_gait_mask'] == 1 else "solid")
            ax[1].fill_between(x=[row.timestamp + td(seconds=-8), row.timestamp + td(seconds=8)],
                               y1=0, y2=7, color='grey', alpha=.3)

    ax[1].legend()
    ax[1].grid()

    ax[2].plot(df1s['timestamp'], df1s['ankle_anterior'], color='black', label='Ankle_ant')
    ax[2].plot(df1s['timestamp'], df1s['ankle_up'], color='red', label='Ankle_up')
    ax[2].plot(df1s['timestamp'], df1s['ankle_left'], color='dodgerblue', label='Ankle_left')

    ax[2].axhline(y=90, linestyle='dashed', color='grey', label='Perp.')
    ax[2].legend()
    ax[2].set_yticks([0, 45, 90, 135, 180])
    ax[2].set_ylabel("deg")
    ax[2].grid()

    ax[3].plot(df1s['timestamp'], df1s['ankle_posture'], color='black', label='AnklePosture')
    ax[3].legend()
    ax[3].grid()

    if collapse_lying:
        gs = df1s['GS'].replace({"supine": "lying", "prone": "lying", "rightside": 'lying',
                                 'leftside': 'lying', 'standing': 'stand'})
        v0 = df1s['posture'].replace({"supine": "lying", "prone": "lying", "rightside": 'lying',
                                      'leftside': 'lying', 'standing': 'stand'})
        v1 = df1s['v1'].replace({"supine": "lying", "prone": "lying", "rightside": 'lying',
                                 'leftside': 'lying', 'standing': 'stand'})
        v2 = df1s['v2'].replace({"supine": "lying", "prone": "lying", "rightside": 'lying',
                                 'leftside': 'lying', 'standing': 'stand'})
        v3 = df1s['v3'].replace({"supine": "lying", "prone": "lying", "rightside": 'lying',
                                 'leftside': 'lying', 'standing': 'stand'})

    if not collapse_lying:
        gs = df1s['GS'].replace({"standing": 'stand'})
        v0 = df1s['posture']
        v1 = df1s['v1']
        v2 = df1s['v2']
        v3 = df1s['v3']

    ax[4].plot(df1s['timestamp'], gs, color='green', label='GS')

    if show_v0:
        ax[4].plot(df1s['timestamp'], v0, color='red', label='v0')
    if show_v1:
        ax[4].plot(df1s['timestamp'], v1, color='dodgerblue', label='v1')
    if 'v2' in df1s.columns and show_v2:
        ax[4].plot(df1s['timestamp'], v2, color='orange', label='v2')
    if 'v3' in df1s.columns and show_v3:
        ax[4].plot(df1s['timestamp'], v3, color='fuchsia', label='v3')
    ax[4].legend()
    ax[4].grid()

    ax[5].plot(df1s['timestamp'], df1s['chest_gait_mask'], color='purple', label='Gait')
    ax[5].legend()

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()
