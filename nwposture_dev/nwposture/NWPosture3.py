# Author: Arslan Salikhov
# Date: July 20th 2021
# Edits: Kyle Weber

# Algorithm adapted from Fortune, Lugade, & Kaufman (2014). Posture and movement classification:
# the comparison of tri-axial accelerometer numbers and anatomical placement. J Biomech Eng. 136(5).

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
from tqdm.auto import tqdm
from scipy.signal import butter, lfilter, filtfilt, iirnotch
import datetime
from datetime import timedelta as td
import sit2standpy as s2s

lying = ['supine', 'prone', 'leftside', 'rightside']


# TODO
# Seated reclined vs. supine: find trunk angle threshold (30 deg?)


class NWPosture:

    def __init__(self, chest_dict, ankle_dict, epoch_length=1, chest_acc_units='g',
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
        self.gait_mask = []
        self.chest_acc_units = chest_acc_units

        self.df_gait = None

    def load_gait_data(self, object):
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

        df = df.loc[df['start_timestamp'] >= max(self.chest['start_stamp'], self.ankle['start_stamp'])]

        return df

    def convert_accel_units(self, to_g=False, to_mss=True):
        """Function to convert chest accelerometer data between G's and m/s/s.

            arguments:
            -to_g: boolean; divides accel values by 9.81 if max(abs(accel)) >= 17 (assumes m/s/s)
            -to_mss: boolean; multiplies accel values by 9.81 if max(abs(accel)) <= 17 (assumes G's)
        """

        if to_mss:
            if max(np.abs(self.chest['Anterior'])) <= 17:
                print("\nConverting chest accelerometer data from G to m/s^2...")

                self.chest['Anterior'] *= 9.81
                self.chest['Up'] *= 9.81
                self.chest['Left'] *= 9.81
                self.chest_acc_units = "mss"
            else:
                print("\nChest accelerometer data already in correct units. Doing nothing.")

        if to_g:
            if max(np.abs(self.chest['Anterior'])) >= 17:
                print("\nConverting chest accelerometer data from m/s^2 to g...")

                self.chest['Anterior'] /= 9.81
                self.chest['Up'] /= 9.81
                self.chest['Left'] /= 9.81
                self.chest_acc_units = 'g'

            else:
                print("\nChest accelerometer data already in correct units. Doing nothing.")

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
                    start = int(np.ceil((row.start_timestamp - self.chest['start_stamp']).total_seconds()))
                    stop = int(np.floor((row.end_timestamp - self.chest['start_stamp']).total_seconds()))
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

        # Start times -----------------------------------------------------
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
            print(f"Cropped chest data by {crop_ind} datapoints at the start.")

        # Crops chest data if chest file started first
        if self.ankle['start_stamp'] < self.chest['start_stamp']:
            crop_ind = int((self.chest['start_stamp'] - self.ankle['start_stamp']).total_seconds() * \
                           self.ankle['sample_rate'])

            self.ankle['Anterior'] = self.ankle['Anterior'][crop_ind:]
            self.ankle['Up'] = self.ankle['Up'][crop_ind:]
            self.ankle['Left'] = self.ankle['Left'][crop_ind:]
            self.ankle['start_stamp'] = self.chest['start_stamp']

            print(f"Cropped ankle data by {crop_ind} datapoints at the start.")

        # Stop times ------------------------------------------------------
        ankle_end = self.ankle['start_stamp'] + td(seconds=len(self.ankle['Anterior'])/self.ankle['sample_rate'])
        chest_end = self.chest['start_stamp'] + td(seconds=len(self.chest['Anterior'])/self.chest['sample_rate'])

        if ankle_end > chest_end:
            # crop ankle
            diff = (ankle_end - chest_end).total_seconds()
            samples = int(diff * self.ankle['sample_rate'])
            self.ankle['Anterior'] = self.ankle['Anterior'][:-samples]
            self.ankle['Up'] = self.ankle['Up'][:-samples]
            self.ankle['Left'] = self.ankle['Left'][:-samples]
            print(f"Cropped ankle by {samples} datapoints at the end.")
        if ankle_end < chest_end:
            # crop chest
            diff = (chest_end - ankle_end).total_seconds()
            samples = int(diff * self.chest['sample_rate'])
            self.chest['Anterior'] = self.chest['Anterior'][:-samples]
            self.chest['Up'] = self.chest['Up'][:-samples]
            self.chest['Left'] = self.chest['Left'][:-samples]
            print(f"Cropped chest by {samples} datapoints at the end.")

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
        posture.loc[(posture['ankle_dynamic'] == 0) &
                    (posture['ankle_up'] < 20),
                    'ankle_posture'] = "sitstand"

        # Static with ankle tilt 20-45 degrees (sit) --> too angled to be standing
        posture.loc[(posture['ankle_dynamic'] == 0) &
                    (posture['ankle_up'] < 45) & (posture['ankle_up'] >= 25),
                    'ankle_posture'] = "sit"

        # Static with shank tilt 45-110 degrees --> horizontal
        posture.loc[(posture['ankle_dynamic'] == 0) &
                    (posture['ankle_up'] < 135) & (posture['ankle_up'] >= 45),
                    'ankle_posture'] = "horizontal"

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
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_up'] < 45) &
                    (posture['chest_anterior'] > 45) & (posture['chest_anterior'] < 135) &
                    (posture['chest_left'] > 45) & (posture['chest_left'] < 135), 'chest_posture'] = "sitstand"

        # Prone: static with anterior axis facing ground
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] > 135) &
                    (posture['chest_up'] > 45) & (posture['chest_up'] < 135) &
                    (posture['chest_left'] > 45) & (posture['chest_left'] < 135),
                    'chest_posture'] = "prone"

        # Supine: static with anterior axis facing sky
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] < 45) &
                    (posture['chest_up'] > 45) & (posture['chest_up'] < 135) &
                    (posture['chest_left'] > 45) & (posture['chest_left'] < 135),
                    'chest_posture'] = "supine"

        # Lying on left side
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] < 135) & (posture['chest_anterior'] > 45) &
                    (posture['chest_up'] > 45) & (posture['chest_up'] < 135) &
                    (posture['chest_left'] > 135),
                    'chest_posture'] = "leftside"

        # Lying on right side
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] < 135) & (posture['chest_anterior'] > 45) &
                    (posture['chest_up'] > 45) & (posture['chest_up'] < 135) &
                    (posture['chest_left'] < 45),
                    'chest_posture'] = "rightside"

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

            print("TRUNK POSTURE ALGORITHM #2")
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

            """# chest sit/stand + ankle sit = sit
            df_posture.loc[(df_posture['chest_posture'] == 'sitstand') &
                           (df_posture['ankle_posture'] == 'sit'),
                           'posture'] = "sit"

            # Chest sit/stand + ankle horizontal = sitting reclined
            df_posture.loc[(df_posture['chest_posture'] == 'sitstand') &
                           (df_posture['ankle_posture'] == 'horizontal'),
                           'posture'] = "sit"
            """

            # Chest other + ankle horizontal = sitting reclined
            # df_posture.loc[(df_posture['chest_posture'] == 'other') &
            #                (df_posture['ankle_posture'] == 'horizontal'), 'posture'] = "sit"

            # Chest supine + ankle horizontal = supine
            # df_posture.loc[(df_posture['chest_posture'] == 'supine') &
            #                (df_posture['ankle_posture'] == 'horizontal'), 'posture'] = "supine"

            # Chest supine + ankle NOT horizontal = sitting (reclined?)
            df_posture.loc[(df_posture['chest_posture'] == 'supine') &
                           (df_posture['ankle_posture'] == 'sitstand'),
                           'posture'] = "supine"
            df_posture.loc[(df_posture['chest_posture'] == 'supine') &
                           (df_posture['ankle_posture'] == 'sit'),
                           'posture'] = "supine"

            # Chest prone + ankle not horizontal = other (standing bent over?)
            df_posture.loc[(df_posture['chest_posture'] == 'prone') &
                           (df_posture['ankle_posture'] != 'horizontal'),
                           'posture'] = 'other'

            # df_posture.loc[(df_posture['chest_posture'] == 'other') &
            #               (df_posture['ankle_posture'] == 'dynamic'), 'posture'] = "other"

            gs = np.array(['other' for i in range(df_posture.shape[0])])

            if goldstandard_df is not None:
                df_event_use = goldstandard_df.loc[goldstandard_df['EventShort'] != "Transition"].reset_index(drop=True)

                df_start = df_posture.iloc[0]['timestamp']
                for row in df_event_use.itertuples():
                    start = int((row.start_timestamp - pd.to_datetime(df_start)).total_seconds())
                    stop = int((row.end_timestamp - pd.to_datetime(df_start)).total_seconds())

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

        t0 = datetime.datetime.now()

        print("-Initial combination...")
        df_posture = initial_processing()
        df1s = format_nwposture_output(df=df_posture)

        t1 = datetime.datetime.now()
        t_total = round((t1-t0).total_seconds(), 1)
        coll_dur = (df1s.iloc[-1]['timestamp'] - df1s.iloc[0]['timestamp']).total_seconds() / 3600

        print("\n=================================================")
        print(f"Processing time: {t_total} seconds ({round(t_total/coll_dur, 1)} sec/hr)")

        return df1s

    def process_sts(self, data, start_stamp, sample_rate, stand_sit=False):

        print("\nRunning sit-stand/stand-sit transition detection using sis2standpy...")

        # for unix time
        start_time = pd.to_datetime("1970-01-01 00:00:00")

        chest_ts = pd.date_range(start=start_stamp, periods=len(data), freq="{}ms".format(1000 / sample_rate))
        ts = np.array([(i - start_time).total_seconds() for i in chest_ts])

        ths = {'stand displacement': 0.125, 'transition velocity': 0.3, 'accel moving avg': 0.15,
               'accel moving std': 0.1, 'jerk moving avg': 2.5, 'jerk moving std': 3}

        if not stand_sit:
            # All default values except power_peak_kwargs
            sts = s2s.Sit2Stand(method='stillness',
                                gravity=9.81, thresholds=ths, long_still=0.3, still_window=0.3,
                                duration_factor=10,  # 6
                                displacement_factor=0.6,
                                lmin_kwargs={'height': -9.5},
                                power_band=[0, 0.5],
                                window=False,
                                # power_peak_kwargs={'distance': 128},
                                power_peak_kwargs={'distance': self.ankle['sample_rate']},
                                power_stdev_height=True, gravity_cut=.25)

        if stand_sit:
            sts = s2s.Sit2Stand(method='displacement',
                                gravity=9.81, thresholds=ths, long_still=0, still_window=0,
                                duration_factor=10,
                                displacement_factor=0.4,  # .6
                                lmin_kwargs={'height': -9.5},
                                power_band=[0, 0.5],
                                window=False,
                                # power_peak_kwargs={'distance': 128},
                                power_peak_kwargs={'distance': self.ankle['sample_rate']},
                                power_stdev_height=True, gravity_cut=.25)

        if not stand_sit:
            print("-Running data for sit-stand detection...")
        if stand_sit:
            print("-Reversing data for stand-sit detection...")
            data = np.flip(data)

        SiSt = sts.apply(accel=data, time=ts, time_units='s')

        if not stand_sit:
            t = SiSt.keys()
            s = [int((pd.to_datetime(i) - start_stamp).total_seconds()) for i in SiSt.keys()]
        if stand_sit:
            t = [chest_ts[0] + td(seconds=(chest_ts[-1] - pd.to_datetime(i)).total_seconds()) for i in SiSt.keys()]
            s = [int((i - chest_ts[0]).total_seconds()) for i in t]

        df_s2spy = pd.DataFrame({"RawIndex": [int((pd.to_datetime(i) - start_stamp).total_seconds() *
                                                  sample_rate) for i in SiSt.keys()],
                                 "Seconds": s,
                                 "Type": ['Sit-stand' if not stand_sit else 'Stand-sit'] * len(SiSt),
                                 'Timestamp': t})

        print("Complete. Found {} {} transitions.".format(df_s2spy.shape[0],
                                                          'sit-to-stand' if not stand_sit else 'stand-to-sit'))

        return df_s2spy


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


def detect_sts(nwposture_obj, remove_lessthan=2, pad_len=1):

    # Convert chest data to ms2 for STS detection
    nwposture_obj.convert_accel_units(to_g=False, to_mss=True)

    df_sitst = nwposture_obj.process_sts(data=np.array([nwposture_obj.chest['Anterior'],
                                                        nwposture_obj.chest['Up'],
                                                        nwposture_obj.chest['Left']]).transpose(),
                                         sample_rate=nwposture_obj.chest['sample_rate'],
                                         start_stamp=nwposture_obj.chest['start_stamp'],
                                         stand_sit=False)

    df_stsit = nwposture_obj.process_sts(data=np.array([nwposture_obj.chest['Anterior'],
                                                        nwposture_obj.chest['Up'],
                                                        nwposture_obj.chest['Left']]).transpose(),
                                         sample_rate=nwposture_obj.chest['sample_rate'],
                                         start_stamp=nwposture_obj.chest['start_stamp'],
                                         stand_sit=True)

    # Conversion of chest data back to G's
    nwposture_obj.convert_accel_units(to_g=True, to_mss=False)

    df = df_stsit.append(df_sitst)
    df = df.sort_values("Seconds").reset_index(drop=True)

    """------------------ Removes STS transitions that occur too close in time -----------"""
    # Boolean array for transitions to keep
    keep = np.array([True]*df.shape[0])

    for i in range(df.shape[0]-1):
        curr_time = df.iloc[i]['Seconds']  # current transition time
        next_time = df.iloc[i+1]['Seconds']  # next transition time

        if next_time - curr_time < remove_lessthan:
            keep[i+1] = False

    df['Keep'] = keep
    df = df.loc[df['Keep']]
    df = df.reset_index(drop=True)

    df = df[["RawIndex", "Seconds", "Type", "Timestamp"]]

    """ ---------------------- Removes transitions during gait ----------------- """
    gait_mask = np.array(nwposture_obj.gait_mask)

    idx = []
    for row in df.itertuples():
        epoch = gait_mask[row.Seconds - pad_len:row.Seconds + pad_len + 1]

        # Accepts transition if whole epoch (transitions +- pad_len) is not gait
        if row.Type == 'Stand-sit':
            if sum(epoch) < pad_len * 2:
                idx.append(row.Index)

        # Accepts transition if whole epoch (transitions +- pad_len) is not gait
        if row.Type == 'Sit-stand':
            if sum(epoch) == 0:
                idx.append(row.Index)

    return df.iloc[idx]


def fill_between_walks(postures, gait_mask, df_transitions, max_break=5, pad_walks=1):
    """Re-does gait_mask by 'filling' small breaks in walking with the 'walking' designation and reflags postures
       using this new gait mask for 'stand' designation.

    arguments:
    -postures: array of 1s postures
    -gait_mask: array of 1s gait classification
    -max_break: maximum number of seconds between consecutive bouts that get combined into one bout
    -pad_walks: padding on start/end of gait bout to include STS transitions

    returns:
    -edited postured array
    """

    print(f"\nFilling in postures between gait bouts with no STS transitions as 'stand' (max gap = {max_break} seconds)...")

    postures = np.array(postures)
    gait_mask = np.array(gait_mask)

    n_affected = 0

    curr_ind = 0
    for i in range(1, len(gait_mask)):
        if i > curr_ind:
            # Start of gait bout
            if gait_mask[i] == 1:

                # End of current bout
                for j in range(i+1, len(gait_mask)):
                    if gait_mask[j] == 0:
                        gait_end = j

                        # Start of next bout
                        next_start = len(gait_mask)
                        for k in range(j+1, len(gait_mask)):
                            if gait_mask[k] == 1:
                                next_start = k
                                break
                        break

                if next_start - gait_end <= max_break:
                    # Transitions that occur 1 second before end of current bout and pad_walks second(s) after start of next
                    t = df_transitions.loc[(df_transitions['Seconds'] >= gait_end - pad_walks) &
                                           (df_transitions['Seconds'] <= next_start + pad_walks)]
                    # Only includes stand-sit transitions
                    t = t.loc[t['Type'] == 'Stand-sit']

                    # if no stand-sits in window, postures become 'stand'
                    if t.shape[0] == 0:
                        postures[gait_end:next_start] = 'stand'
                        n_affected += 1

                curr_ind = j

    print(f"-Affected {n_affected} bouts.")

    return postures


def calculate_bouts(df1s, colname):
    inds = [0]
    posture = np.array(df1s[colname])
    curr_ind = 0

    for i in range(len(posture)):
        if i >= curr_ind:
            curr_post = posture[i]

            for j in range(i + 1, len(posture)):
                if posture[j] != curr_post:
                    inds.append(j)
                    curr_ind = j
                    break
    df_bout = df1s.iloc[inds][["index", 'timestamp', 'posture', colname]].reset_index(drop=True)

    return df_bout


def apply_logic(df_transitions, gait_mask, postures, first_walk_index, first_pass, quiet=False):
    """Reclassifies postures based on context relating unknown sitstand periods to lying or known standing periods."""

    curr_postures = []
    prev_postures = []
    next_postures = []
    next_diff_postures = []
    curr_sts = []
    next_sts = []

    gait_mask = np.array(gait_mask)
    postures = np.array(postures)

    # First pass of data runs it chronologically
    # Starts at first_walk_index
    if first_pass:
        print("\nRunning first pass of context logic (post-first gait bout)...")
        df_transitions = df_transitions.loc[df_transitions['Seconds'] >= first_walk_index]
        prev_post_end = 0

    # Second pass on data runs it reverse chronologically
    # Starts at first_walk_index and runs backwards
    if not first_pass:

        # Deals with posture that occurs right before first walk if no STS found right at start of walk ------
        last_sts = df_transitions.loc[(df_transitions['Seconds'] <= first_walk_index) &
                                      (df_transitions['Type'] == 'Sit-stand')].iloc[-1]

        prev_post = postures[first_walk_index - 1]  # posture before walk
        prev_post_ind = 0  # default if collection starts with prev_post
        for i in range(first_walk_index):
            if postures[first_walk_index - i] != prev_post:
                prev_post_ind = i
                break

        ind = max([last_sts['Seconds'], prev_post_ind])

        postures[ind:first_walk_index] = 'stand'

        print("\nRunning second pass of context logic to fix pre-first gait bout data...")
        # Reverses order of dataframe
        df_transitions = df_transitions.loc[df_transitions['Seconds'] <
                                            first_walk_index].sort_values('Seconds',
                                                                          ascending=False).reset_index(drop=True)
        prev_post_end = df_transitions.iloc[0]['Seconds']
        prev_post = postures[prev_post_end]

    for row in df_transitions.itertuples():

        curr_sts.append(row.Type)

        # Posture at time of STS transition
        curr_post = postures[row.Seconds]

        # index of next STS transition ----------------------------
        try:
            next_transition = df_transitions.loc[df_transitions['Seconds'] > row.Seconds].iloc[0]['Seconds']
            next_t_type = df_transitions.loc[df_transitions['Seconds'] > row.Seconds].iloc[0]['Type']
        except IndexError:
            next_transition = len(postures)
            next_t_type = 'Sit-stand'
        next_sts.append(f"{next_t_type} ({next_transition})")

        # Finds where current posture ends -----------------
        for i in range(row.Seconds, len(postures)):
            if postures[i] != curr_post and postures[i] != 'other':
                curr_post_ends = i
                curr_postures.append(f"{curr_post} ({row.Seconds} - {curr_post_ends})")
                break

        # Determines start/end indexes of next posture ---------------
        for i in range(row.Seconds, len(postures)):
            if postures[i] != curr_post and postures[i] != 'other':
                next_post = postures[i]
                next_post_start = i
                break

        for j in range(i, len(postures)):
            if postures[j] != next_post:
                next_post_end = j
                next_postures.append(f"{next_post} ({next_post_start} - {next_post_end})")
                break

        # Determines previous posture and indexes ------------
        for i in range(row.Seconds):
            if postures[row.Seconds - i] != curr_post:
                prev_post = postures[row.Seconds - i]
                prev_post_end = row.Seconds - i
                break

        prev_post_start = None
        for j in range(prev_post_end):
            if postures[prev_post_end - j] != prev_post:
                prev_post_start = prev_post_end - j
                prev_postures.append(f"{prev_post} ({prev_post_start} - {prev_post_end})")
                break
        if prev_post_start is None:
            prev_post_start = 0

        # Finds next posture that is not next_post or 'sitstand' -------
        for i in range(row.Seconds+1, len(postures)):
            if postures[i] != curr_post and postures[i] != 'sitstand':
                next_diff = postures[i]
                next_diff_post_start = i
                next_diff_postures.append(f"{next_diff} ({next_diff_post_start})")
                break

        # Index of next transition or posture end
        end_ind = min([next_transition, next_post_end])

        if not quiet:
            try:
                pp = prev_post_start
            except:
                pp = None
            print(
                f"Curr={curr_post}({row.Seconds}), prev={prev_post}({prev_post_start}-{prev_post_end}), "
                f"next={next_post}({next_post_start}-{next_post_end}), "
                f"next diff={next_diff}({next_diff_post_start}), STS={row.Type}, "
                f"nextSTS={next_t_type}({next_transition})")

        # Apply some logic to dis =================================================================================

        # If currently standing ------------------------------
        if curr_post == 'stand':
            # Stand-to-sit transitions -------
            if row.Type == 'Stand-sit':
                if next_post == 'sit' or next_post == 'sitstand':
                    # postures[row.Seconds:min([next_transition, next_diff_post_start])] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for
                         x in range(row.Seconds, min([next_transition, next_diff_post_start]))]
                    postures[row.Seconds:min([next_transition, next_diff_post_start])] = r

                if next_post == 'sitstand' and next_diff == 'sit':
                    # postures[row.Seconds:end_ind] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, end_ind)]
                    postures[row.Seconds:end_ind] = r

            # Sit-to-stand transitions ------
            if row.Type == 'Sit-stand':
                if prev_post == 'sitstand':
                    # postures[prev_post_start:row.Seconds] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_start, row.Seconds)]
                    postures[prev_post_start:row.Seconds] = r

        # If currently unsure if sitstand ----------------------
        if curr_post == 'sitstand':
            # Stand-to-sit transitions ------
            if row.Type == 'Stand-sit':
                # if next_post == 'sit':
                #    postures[prev_post_end:next_post_start] = 'stand'

                if next_post == 'sit':
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_start)]
                    postures[row.Seconds:next_post_start] = r
                    postures[prev_post_end:row.Seconds] = 'stand'

                if prev_post == 'sit':
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_end, row.Seconds)]
                    postures[prev_post_end:row.Seconds] = r

                    postures[row.Seconds:next_post_start] = 'stand'

                if prev_post == 'stand':
                    if next_post in ['other', 'sitstand']:
                        r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_start)]
                        postures[row.Seconds:next_post_start] = r

                if next_post == 'stand':
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_start)]
                    postures[row.Seconds:next_post_start] = r

            # Sit-to-stand transitions -------
            if row.Type == 'Sit-stand':
                if next_post == 'sit' and prev_post == 'sit':
                    postures[row.Seconds:curr_post_ends] = 'stand'

                if prev_post == 'sit':
                    # postures[prev_post_end:row.Seconds] = 'sit'
                    r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_end, row.Seconds)]
                    postures[prev_post_end:row.Seconds] = r

                    postures[row.Seconds:next_post_start] = 'stand'

        # If currently sitting ------------------------------
        if curr_post == 'sit':
            # Sit-to-stand transitions -------
            if row.Type == 'Sit-stand':
                if next_post == 'sitstand':
                    postures[row.Seconds:next_post_end] = 'stand'

            if row.Type == 'Stand-sit':
                if prev_post == 'stand' and prev_post_end - row.Seconds <= 3:
                    if next_post == 'sitstand':
                        # postures[row.Seconds:next_post_end] = 'sit'
                        r = ['sit' if gait_mask[x] == 0 else postures[x] for x in
                             range(row.Seconds, min([next_transition, next_post_end]))]
                        postures[row.Seconds:min([next_transition, next_post_end])] = r

        if curr_post in lying:
            if next_post == 'stand':
                postures[curr_post_ends:next_post_start] = 'sit' if row.Type == 'Stand-sit' else 'stand'
            if row.Type == 'Stand-sit' and next_post == 'sitstand':
                # postures[row.Seconds:next_post_end] = 'sit'
                r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(row.Seconds, next_post_end)]
                postures[row.Seconds:next_post_end] = r

        if curr_post == 'other':
            if row.Type == 'Sit-stand' and next_post in ['sitstand', 'stand']:
                # postures[prev_post_end:row.Seconds] = 'sit'
                r = ['sit' if gait_mask[x] == 0 else postures[x] for x in range(prev_post_start, row.Seconds)]
                postures[prev_post_end:row.Seconds] = r

                postures[row.Seconds:next_post_start] = 'stand'

                if next_post == 'sitstand' and next_post_start < next_transition:
                    postures[next_post_start:next_post_end] = 'stand'

    # Removes single-epoch events
    for i in range(len(postures)-2):
        if postures[i] != postures[i+1] and postures[i+1] != postures[i+2]:
            postures[i+1] = postures[i]

    l = min([len(df_transitions['Seconds']), len(prev_postures), len(curr_postures), len(next_postures),
            len(curr_sts), len(next_sts), len(next_diff_postures)])
    df_out = pd.DataFrame({"Seconds": df_transitions['Seconds'][:l],
                           "Previous": prev_postures[:l], "Current": curr_postures[:l], "Next": next_postures[:l],
                           "STS": curr_sts[:l], "Next STS": next_sts[:l], "NextDiff": next_diff_postures[:l]})
    return postures, df_out


def final_logic(postures, gait_mask, df_transitions, split_difference=False):
    """Another layer of logic."""

    postures = postures.copy()
    gait_mask = np.array(gait_mask)
    n_sitstand = len([i for i in postures if i == 'sitstand'])

    curr_ind = 0
    for i in range(len(postures)):
        if i > curr_ind:

            # Dealing with sitstand postures -------
            if postures[i] == 'sitstand':
                curr_post = 'sitstand'

                for j in range(i+1, len(postures)):
                    if postures[j] != 'sitstand':
                        # Crops df_transitions to span from curr_post to j
                        t = df_transitions.loc[(df_transitions['Seconds'] >= i) & (df_transitions['Seconds'] <= j)]

                        print(i, j, postures[j], list(t['Seconds']), list(t['Type']))
                        # Logic if transitions are found before posture change
                        start_idx = i

                        for row in t.itertuples():
                            # Deals with sit-stand transitions -------
                            if row.Type == 'Sit-stand':
                                # Flags i to transition as sitting if not known standing
                                for x in range(start_idx, row.Seconds+1):
                                    # Makes sure gait isn't flagged as sitting
                                    postures[x] = 'sit' if gait_mask[x] == 0 else postures[x]

                            # Deals with stand-sit transitions
                            if row.Type == 'Stand-sit':
                                # postures[row.Seconds:j] = 'sit'
                                postures[i:j] = 'sit'

                            start_idx = row.Seconds

                        # Deals with time period between final transition in window and j ---------
                        if t.shape[0] > 0:
                            postures[row.Seconds:j] = 'sit' if row.Type == 'Sit-stand' else 'stand'

                        # Logic if no transitions found -------------------------------------------
                        if t.shape[0] == 0:
                            prev_post = postures[i-1]
                            next_post = postures[j]

                            if curr_post == 'sitstand':
                                # postures are sit, flags all non-gait as sitting
                                if prev_post == 'sit' and next_post == 'sit':
                                    for x in range(i, j+1):
                                        postures[x] = 'sit' if gait_mask[x] == 0 else 'stand'

                                # postures are standing, all become standing
                                if prev_post == 'stand' and next_post == 'stand':
                                    postures[i:j] = 'stand'

                                # if previous is sitting and next is standing:
                                if prev_post == 'sit' and next_post == 'stand':
                                    # Splits sitstand event in half if split_difference
                                    if split_difference:
                                        postures[i:i+int((j-i)/2)] = 'sit'
                                        postures[i+int((j-i)/2):j] = 'stand'
                                    # if not split difference, sitstand becomes previous posture (sit)
                                    if not split_difference:
                                        postures[i:j] = 'sit'

                                # if previous is stand and next is sit:
                                if prev_post == 'stand' and next_post == 'sit':
                                    # Splits sitstand event in half
                                    if split_difference:
                                        postures[i:i+int((j-i)/2)] = 'stand'
                                        postures[i+int((j-i)/2):j] = 'sit'
                                    if not split_difference:
                                        postures[i:j] = 'stand'

                                # if sitstand transitions to lying with no STS detected
                                if prev_post == 'sit' and next_post in lying:
                                    postures[i:j] = 'sit'

                                # if sitstand transitions from lying to standing/sitting with no STS detected
                                if prev_post in lying and (next_post == 'sit' or next_post == 'stand'):
                                    postures[i:j] = prev_post

                                # if standing to lying with no STS transition detected
                                if prev_post == 'stand' and next_post in lying:
                                    postures[i:j] = 'stand'

                                # Flags 'other' as next posture if next_post is sit or stand
                                if prev_post == 'other' and next_post in ['sit', 'stand']:
                                    postures[i:j] = next_post

                            # If no transitions found and 'other' surrounded by same posture, coded as said posture
                            if curr_post == 'other':
                                if prev_post == next_post:
                                    postures[i:j] = next_post

                        curr_ind = j
                        break

    print(f"-Went from {n_sitstand} ({round(100*n_sitstand/len(postures), 2)}%) to "
          f"{len([i for i in postures if i == 'sitstand'])} 'sitstands' "
          f"({round(100*len([i for i in postures if i == 'sitstand'])/len(postures), 2)}%)")

    return postures


def fill_other(postures):
    """If 'other' posture remains and is surrounded by same posture, gets flagged as that other posture."""

    postures = postures.copy()

    curr_ind = 0
    for i in range(len(postures)):
        if i > curr_ind:

            if postures[i] == 'other':

                for j in range(i + 1, len(postures)):
                    if postures[j] != 'other':
                        next_post = postures[j]
                        curr_ind = j
                        break
                for k in range(i):
                    if postures[i-k] != 'other':
                        prev_post = postures[i-k]
                        break

                postures[i:j] = next_post

    return postures


def plot_posture_comparison(df1s, df_sts, show_transitions=True, show_v0=True, show_v1=True, show_v2=True, collapse_lying=True,
                            show_v3=True, show_v4=True, show_gs=True, first_walk_index=0, use_timestamps=True):

    fig, ax = plt.subplots(7, sharex='col', figsize=(10, 9),
                           gridspec_kw={"height_ratios": [1, .67, .67, .33, .2, .67, .67]})

    if show_transitions:
        plt.suptitle("Red = stand-to-sit; green = sit-to-stand")

    start_ts = df1s.iloc[0]['timestamp']
    lying = ['leftside', 'rightside', 'prone', 'supine']

    if show_gs:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['GS'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['GS']],
                   color='black', label="GS", zorder=1, linestyle='dashed')
    if show_v0:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['posture'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['posture']],
                   color='dodgerblue', label='Original', zorder=0)

    if show_v1:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v1'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v1']],
                   color='limegreen', label='V1', zorder=0)

    if show_v2:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v2'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v2']],
                   color='fuchsia', label='v2', zorder=0)
    if show_v3:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v3'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v3']],
                   color='purple', label='v3', zorder=0)
    if show_v4:
        ax[0].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]),
                   df1s['v4'] if not collapse_lying else [i if i not in lying else 'lying' for i in df1s['v4']],
                   color='orange', label='v4', zorder=0)

    ax[0].scatter(df1s.loc[df1s['v4'] == 'sitstand']['timestamp'] if use_timestamps else
                  df1s.loc[df1s['v4'] == 'sitstand'].index,
                  df1s.loc[df1s['v4'] == 'sitstand']['v4'], marker='o', color='red')

    ylim = ax[0].get_ylim()
    ax[0].fill_between(x=[0 if not use_timestamps else start_ts,
                          first_walk_index if not use_timestamps else start_ts + td(seconds=first_walk_index)],
                       y1=-1, y2=np.ceil(ylim[1]), color='grey', alpha=.35, label='Pre-gait')
    ax[0].set_ylim(ylim)

    ax[0].legend(loc='lower right')

    if show_transitions:
        for row in df_sts.itertuples():
            ax[0].axvline(row.Seconds if not use_timestamps else start_ts + td(seconds=row.Seconds),
                          color='limegreen' if row.Type == 'Sit-stand' else 'red',
                          linestyle='dashed', lw=1.5)

    ax[0].grid()

    ax[1].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['chest_anterior'],
               color='black', label='Chest_Ant')
    ax[1].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['chest_up'],
               color='red', label='Chest_Up')
    ax[1].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['chest_left'],
               color='dodgerblue', label='Chest_Left')
    ax[1].set_yticks([0, 90, 180])
    ax[1].set_ylabel("Deg.")
    ax[1].grid()
    ax[1].legend(loc='lower right')

    ax[2].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['ankle_anterior'], color='black', label='ankle_anterior')
    ax[2].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['ankle_up'], color='red', label='ankle_up')
    ax[2].plot(df1s['index'] if not use_timestamps else df1s['timestamp'], df1s['ankle_left'], color='dodgerblue', label='ankle_left')
    ax[2].set_yticks([0, 90, 180])
    ax[2].set_ylabel("Deg.")
    ax[2].grid()
    ax[2].legend(loc='lower right')

    try:
        ax[3].plot(chest_temp_ts if use_timestamps else np.arange(0, len(chest_acc.signals[chest_acc_indexes['Temperature']]))/chest_temp_fs,
                   chest_acc.signals[chest_acc_indexes['Temperature']], color='black', label='Chest')
        ax[3].plot(ank_temp_ts if use_timestamps else np.arange(0, len(ank_acc.signals[ank_acc_indexes['Temperature']]))/ank_temp_fs,
                   ank_acc.signals[ank_acc_indexes['Temperature']], color='red', label='Ankle')
        ax[3].legend(loc='lower right')
        ax[3].grid()
    except:
        pass

    ax[3].set_ylabel("Deg. C")

    ax[4].plot(df1s['timestamp'] if use_timestamps else np.arange(df1s.shape[0]), df1s['chest_gait_mask'], color='orange', label='Gait')
    ax[4].legend(loc='lower right')

    x = post.chest['Anterior'] if not converted_to_mss else post.chest['Anterior']/9.81
    y = post.chest['Up'] if not converted_to_mss else post.chest['Up']/9.81
    z = post.chest['Left'] if not converted_to_mss else post.chest['Left']/9.81
    ts = pd.date_range(start=post.chest['start_stamp'], periods=len(x), freq="{}ms".format(1000/post.chest['sample_rate']))

    ax[5].plot(ts if use_timestamps else np.arange(len(post.chest['Anterior'])) / chest['sample_rate'],
               x, color='black', label='chest_anterior')
    ax[5].plot(ts if use_timestamps else np.arange(len(post.chest['Anterior']))/chest['sample_rate'],
               y, color='red', label='chest_up')
    ax[5].plot(ts if use_timestamps else np.arange(len(post.chest['Anterior']))/chest['sample_rate'],
               z, color='dodgerblue', label='chest_left')
    ax[5].legend(loc='lower right')
    ax[5].set_ylabel("G")
    ax[5].grid()

    ts = pd.date_range(start=post.ankle['start_stamp'], periods=len(post.ankle['Anterior']),
                       freq="{}ms".format(1000/post.ankle['sample_rate']))
    ax[6].plot(ts if use_timestamps else np.arange(len(post.ankle['Anterior']))/ankle['sample_rate'],
               post.ankle['Anterior'], color='black', label='ankle_anterior')
    ax[6].plot(ts if use_timestamps else np.arange(len(post.ankle['Up']))/ankle['sample_rate'],
               post.ankle['Up'], color='red', label='ankle_up')
    ax[6].plot(ts if use_timestamps else np.arange(len(post.ankle['Left']))/ankle['sample_rate'],
               post.ankle['Left'], color='dodgerblue', label='ankle_left')
    ax[6].legend(loc='lower right')
    ax[6].set_ylabel("G")
    ax[6].grid()

    if not use_timestamps:
        ax[-1].set_xlabel("Seconds")
        ax[-1].set_xlim(0, df1s.shape[0])
    if use_timestamps:
        ax[-1].xaxis.set_major_formatter(xfmt)
        ax[-1].set_xlim(df1s['timestamp'].iloc[0], df1s['timestamp'].iloc[-1])

    plt.tight_layout()
    plt.subplots_adjust(hspace=.15)

    return fig, ax
