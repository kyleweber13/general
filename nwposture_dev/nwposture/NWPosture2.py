# Author: Arslan Salikhov
# Date: July 20th 2021
# Edits: Kyle Weber, November 2021

import os
import pandas as pd
import numpy as np
from numpy.core.numeric import NaN
import datetime as dt
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter("%H:%M:%S")
from tqdm.auto import tqdm

chest = {"Anterior": chest_acc.signals[2]*-1, "Up": chest_acc.signals[0], "Left": chest_acc.signals[1],
         "start_stamp": chest_acc.header['startdate'], "sample_rate": chest_fs}

ankle = {"Anterior": la_acc.signals[1], "Up": la_acc.signals[0], "Left": chest_acc.signals[2],
         "start_stamp": la_acc.header['startdate'], "sample_rate": la_fs}


class NWPosture:

    def __init__(self, chest_dict, ankle_dict, gait_bouts=None, epoch_length=1):
        print("\n========================== Processing posture data ==========================")

        self.chest = chest_dict
        self.ankle = ankle_dict
        self.epoch_length = epoch_length
        self.df_gait = self.load_gait_data(gait_bouts)
        self.gait_mask = self.create_gait_mask()

    def load_gait_data(self, object):

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

    def epochinator3000(self, data, sample_rate, epoch_length=1):
        """Epochinator3000 epochs your unepoched data.
                Args:
                data (array): unepoched data that needs to be epoched
                data_freq (float): frequency of the unepoched data in Hz
                epoch_length (int): duration of a single epoch in seconds
                Returns:
                epoched_data (array): epoched data thas has been epochinatored
        """

        epoched_data = []

        for i in tqdm(range(0, len(data), int(sample_rate) * epoch_length)):
            stepList = data[i:i + 1]
            epoched_data.append(np.mean(stepList))

        return epoched_data

    def orientator3000(self, anterior_angle, anterior, up, left):
        """Method to determine Bittium Faros' orientation (vertical or horizontal)"""

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
        """This is in place of the nwgait output --> will need reformatting"""

        print("Creating gait mask - REFORMAT FOR NWGAIT OUTPUT!")

        duration = int(len(self.chest['Anterior']) / self.chest['sample_rate'] / self.epoch_length)

        # Binary list of gait (1) or no gait (0) in 1-sec increments
        gait_mask = np.zeros(duration)

        if self.df_gait is None:
            return gait_mask

        if self.df_gait is not None:
            for row in self.df_gait.itertuples():
                start = int((row.Start - self.chest['start_stamp']).total_seconds())
                stop = int((row.Stop - self.chest['start_stamp']).total_seconds())
                gait_mask[int(start):int(stop)] = 1

        return gait_mask

    def crop_data(self):

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
        """Ankle posture detection. Classifies shank orientation as horizontal, vertical, angled, or gait."""

        # Median filter
        ant_medfilt = signal.medfilt(anterior)
        up_medfilt = signal.medfilt(up)
        left_medfilt = signal.medfilt(left)

        # Defining a Vector Magnitude(vm) array to classify data points as either dynamic or static
        vm = np.sqrt(np.square(np.array([ant_medfilt, up_medfilt, left_medfilt])).sum(axis=0)) - 1
        vm[vm < 0.06135] = 0
        vm[vm >= 0.06135] = 1

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
        posture['ankle_gait_mask'] = self.gait_mask
        posture['ankle_dynamic'] = posture[['ankle_gait_mask', 'ankle_vm']].sum(axis=1)

        post_dict = {0: "other", 1: "horizontal", 2: "angled", 3: 'sit/stand', 4: 'sit', 5: 'dynamic', 6: 'gait'}

        # Static with ankle tilt 0-15 degrees (sit/stand) --> vertical shank while standing or sitting upright
        posture.loc[(posture['ankle_dynamic'] == 0) & (posture['ankle_up'] < 20), 'ankle_posture'] = "sitstand"

        # Static with ankle tilt 15-45 degrees (sit) --> too angled to be standing
        posture.loc[(posture['ankle_dynamic'] == 0) &
                    (posture['ankle_up'] < 45) & (posture['ankle_up'] > 20), 'ankle_posture'] = "sit"

        # Static with shank tilt 45-110 degrees --> horizontal
        posture.loc[(posture['ankle_dynamic'] == 0)
                    & (posture['ankle_up'] < 110) & (posture['ankle_up'] > 45), 'ankle_posture'] = "horizontal"

        # Flags gait
        posture.loc[(posture['ankle_dynamic'] > 0), 'ankle_posture'] = "dynamic"
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

        # Median filter for each axis
        ant_medfilt = signal.medfilt(anterior)
        up_medfilt = signal.medfilt(up)
        left_medfilt = signal.medfilt(left)

        # Defining a Vector Magnitude(vm) array to classify data points as either dynamic or static
        vm = np.sqrt(np.square(np.array([ant_medfilt, up_medfilt, left_medfilt])).sum(axis=0)) - 1
        vm[vm < 13.64] = 0
        vm[vm >= 13.64] = 1

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

        """WHAT DIS"""
        # angleXY = np.sqrt(np.square(up_angle) + np.square(left_angle))

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
        # Ant. of > 150 degrees = face down
        posture.loc[(posture['chest_dynamic'] == 0) &
                    (posture['chest_anterior'] > 150) &
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
        posture.loc[(posture['chest_dynamic'] > 0), 'chest_posture'] = "dynamic"
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

    def processing(self, show_plots=False):

        self.crop_data()

        # Data epoching -------------------------------------------------------
        print("\nEpoching data...")
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

        # Crops epoched data to shortest dataset ----------------------------------
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

        print("Epoching complete.")

        # Posture processing based on angles ----------------------------------------
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

        return df_posture


df_gait = pd.read_excel("C:/Users/ksweber/Desktop/PosturePilot001_EventLog.xlsx")
df_gait = df_gait.loc[df_gait["EventShort"] == "Walking"]

nwp = NWPosture(chest_dict=chest, ankle_dict=ankle, epoch_length=1,
                gait_bouts=df_gait)
df_posture = nwp.processing(show_plots=False)
df_posture = df_posture.loc[df_posture['timestamp'] < pd.to_datetime("2021-06-07 16:00:00")]

df_posture['posture'] = df_posture['chest_posture']

# chest sit/stand + ankle sit = sit
df_posture.loc[(df_posture['chest_posture'] == 'sitstand') &
               (df_posture['ankle_posture'] == 'sit'), 'posture'] = "sit"

df_posture.loc[(df_posture['chest_posture'] == 'sitstand') &
               (df_posture['ankle_posture'] == 'horizontal'), 'posture'] = "sit"

df_posture.loc[(df_posture['chest_posture'] == 'other') &
               (df_posture['ankle_posture'] == 'dynamic'), 'posture'] = "other"


def plot_all():
    gs = np.array(['Other' for i in range(df_posture.shape[0])])

    df_event_use = df_event.loc[df_event['EventShort'] != "Transition"].reset_index(drop=True)
    print(df_event_use[["Start", "Stop", "EventShort"]])
    for row in df_event_use.itertuples():
        start = int((row.Start - chest['start_stamp']).total_seconds())
        stop = int((row.Stop - chest["start_stamp"]).total_seconds())

        gs[start:stop] = row.EventShort

    fig, ax = plt.subplots(5, sharex='col', figsize=(14, 9))
    ax[0].plot(df_posture['timestamp'], df_posture['chest_anterior'], color='black', label='Chest_ant')
    ax[0].plot(df_posture['timestamp'], df_posture['chest_up'], color='red', label='Chest_up')
    ax[0].plot(df_posture['timestamp'], df_posture['chest_left'], color='dodgerblue', label='Chest_left')
    ax[0].axhline(y=90, linestyle='dashed', color='grey', label='Perp.')
    ax[0].legend()
    ax[0].set_yticks([0, 45, 90, 135, 180])
    ax[0].set_ylabel("deg")
    ax[0].grid()

    ax[1].plot(df_posture['timestamp'], df_posture['chest_posture'], color='black', label='ChestPosture')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(df_posture['timestamp'], df_posture['ankle_anterior'], color='black', label='Ankle_ant')
    ax[2].plot(df_posture['timestamp'], df_posture['ankle_up'], color='red', label='Ankle_up')
    ax[2].plot(df_posture['timestamp'], df_posture['ankle_left'], color='dodgerblue', label='Ankle_left')
    ax[2].axhline(y=90, linestyle='dashed', color='grey', label='Perp.')
    ax[2].legend()
    ax[2].set_yticks([0, 45, 90, 135, 180])
    ax[2].set_ylabel("deg")
    ax[2].grid()

    ax[3].plot(df_posture['timestamp'], df_posture['ankle_posture'], color='black', label='AnklePosture')
    ax[3].legend()
    ax[3].grid()

    ax[4].plot(df_posture['timestamp'], gs, color='green', label='GS')
    ax[4].legend()
    ax[4].grid()

    x = df_posture["posture"].replace({"sit": "Sitti", "gait": "Walki", "other": "Other",
                                   "supine": "Supin", "prone": "Prone", "rightside": "Side", 'leftside': 'Side'})
    ax[4].plot(df_posture['timestamp'], x, color='orange', label='Alg')
    # ax[5].grid()
    # ax[5].legend()

    ax[-1].xaxis.set_major_formatter(xfmt)
    plt.tight_layout()


plot_all()
