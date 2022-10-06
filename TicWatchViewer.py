import os  # gives access to file pathways, etc.
import pandas as pd  # dataframes, csv/excel reader
import matplotlib.pyplot as plt  # plotting
from matplotlib import dates as mdates  # timestamp formatting on graphs
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
import numpy as np  # general math things
from tqdm import tqdm  # progress bar; totally superfluous
import datetime  # timestamp things
from datetime import timedelta  # timestamp calculations


def get_file_dates(files):
    # printing: \n is new line. f"" formatting allows variables to be passed in
    # len(files) gives number of elements in list files
    print(f"\nRetrieving dates from filenames of {len(files)} files...")

    dates = []  # empty list

    for file in files:  # runs block below for each file
        split_filename = file.split("_")  # splits string by "_"
        split_filename2 = split_filename[-1]  # gets last split element ({date}.csv part)
        date = split_filename2.split(".")[0]  # splits by ".", gets first part (date)

        dates.append(date)  # adds current file's date to dates list

    dates = set(dates)  # set function finds unique elements in list
    dates = sorted(dates)  # sorts by numerical order...probably.

    return dates


def combine_similar_files(folder, analysis_type, extension='.csv'):

    print(f"\nCombining {analysis_type} files in {folder}...")

    all_files = os.listdir(folder)  # all files in folder

    # only files with specified type (analysis_type) and extension type (extension)
    files = [i for i in all_files if analysis_type in i and extension in i]
    print(f"-Found {len(files)} files")

    # data dictionary set-up -------------------------------------

    # need to get types of data contained in csv files and data
    colnames = list(pd.read_csv(files[0], nrows=2, dtype=object).columns)

    df = pd.read_csv(files[0], skiprows=1)  # reads first file of specified type
    df = df.iloc[:, :len(colnames)]
    df.columns = colnames

    data_dict = {}  # empty data dictionary to store data

    # for each column in dataframe (first file), creates dictionary key + adds its data
    for column in df.columns:
        data_dict[column] = np.array(df[column])

    data_dict['file_index'] = np.zeros(df.shape[0])

    # looping through subsequent files
    file_i = 1
    for file in tqdm(files[1:]):  # skips first file; already read
        df = pd.read_csv(file, skiprows=1, dtype=object)  # reads first file of specified type
        df = df.iloc[:, :len(colnames)]
        df.columns = colnames

        for column in df.columns:
            data_dict[column] = np.append(data_dict[column], df[column])

        data_dict['file_index'] = np.append(data_dict['file_index'], np.array([file_i] * df.shape[0]))
        file_i += 1

    return data_dict


def get_sample_rate(unix_stamps):

    d = [j-i for i, j in zip(unix_stamps[:1000], unix_stamps[1:1001])]
    mean_d = np.mean(d)

    fs = 1000/round(mean_d, 0)

    return fs


def unix_to_timestamp(unix_list, sample_rate=None):

    if sample_rate is not None:
        ts = pd.date_range(start=pd.to_datetime("1970-01-01 00:00:00") + timedelta(seconds=unix_list[0]/1000),
                           periods=len(unix_list),
                           freq=f"{1000/sample_rate}ms")

    if sample_rate is None:
        start_stamp = pd.to_datetime("1970-01-01 00:00:00")
        ts = [start_stamp + timedelta(seconds=i/1000) for i in unix_list]

    return ts


def unix_to_timestamp_quick(unix_list):

    ts = pd.to_datetime([datetime.datetime.fromtimestamp(int(i) / 1000) for i in unix_list])

    return ts


def calculate_hrr(hr_data, age, resting_hr=60):

    max_hr = 207 - .7 * age
    reserve = max_hr - resting_hr
    sed_lim = [0, 30]
    light_lim = [30, 40]
    mod_lim = [40, 60]
    vig_lim = [60, 100]

    hrr = [(i - resting_hr) / reserve * 100 for i in hr_data]
    hrr = np.array([i if i >= 0 else 0 for i in hrr])

    hrr_cat = np.array(['sed'] * len(hrr))
    hrr_cat[(hrr >= light_lim[0]) & (hrr < sed_lim[1])] = 'light'
    hrr_cat[(hrr >= mod_lim[0]) & (hrr < mod_lim[1])] = 'mod'
    hrr_cat[(hrr >= vig_lim[0]) & (hrr < vig_lim[1])] = 'vig'
    hrr_cat[hrr >= vig_lim[1]] = 'max'

    return hrr, hrr_cat


def plot_things(ds_ratio=1, abs_hr=False):

    fig, ax = plt.subplots(5, sharex='col', figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1, .5, .5]})

    c_dict = {"WALKING": 'purple', 'STILL': 'grey', 'ON_BICYCLE': 'limegreen', 'NONWEAR': 'black'}

    ax[0].plot(accel['ts'][::ds_ratio], accel['X'][::ds_ratio], color='black', label='X')
    ax[0].plot(accel['ts'][::ds_ratio], accel['Y'][::ds_ratio], color='red', label='Y')
    ax[0].plot(accel['ts'][::ds_ratio], accel['Z'][::ds_ratio], color='dodgerblue', label='Z')
    ax[0].set_ylabel("Wrist\nAcc (G)")
    ax[0].set_title(c_dict)

    ax[1].plot(gyro['ts'][::ds_ratio], gyro['X'][::ds_ratio], color='black', label='X')
    ax[1].plot(gyro['ts'][::ds_ratio], gyro['Y'][::ds_ratio], color='red', label='Y')
    ax[1].plot(gyro['ts'][::ds_ratio], gyro['Z'][::ds_ratio], color='dodgerblue', label='Z')
    ax[1].set_ylabel("Wrist\nGyro (rpm)")

    ax[1].scatter(steps['ts'], [10]*len(steps['ts']), marker='v', color='purple', s=15, label='steps')

    hr_key = 'Heart Rate' if abs_hr else 'hrr'
    hr_c_dict = {'sed': 'grey', 'light': 'limegreen', 'mod': 'orange', 'vig': 'red', 'max': 'black'}
    for i in range(len(hr['ts'])-1):
        ax[2].plot([hr['ts'][i], hr['ts'][i]+timedelta(seconds=1)], [hr[hr_key][i], hr[hr_key][i+1]],
                   color='red' if abs_hr else hr_c_dict[hr['hrr_intensity'][i]])

    ax[2].set_ylabel(f"HR ({'%HRR' if not abs_hr else 'bpm'})")
    ax[2].grid()
    if abs_hr:
        ax[2].set_yticks(np.arange(25, 226, 50))
        ax[2].set_ylim(40, 225)

    labelled = []
    for i in range(0, len(act_transition['Activity'])-1, 2):
        for ax_i in range(4):
            if act_transition['Activity'][i] in labelled:
                ax[ax_i].axvspan(xmin=act_transition['ts'][i], xmax=act_transition['ts'][i + 1],
                                 ymin=.5, ymax=1, color=c_dict[act_transition['Activity'][i]], alpha=.25)
            if act_transition['Activity'][i] not in labelled:
                ax[ax_i].axvspan(xmin=act_transition['ts'][i], xmax=act_transition['ts'][i + 1],
                                 ymin=.5, ymax=1, color=c_dict[act_transition['Activity'][i]],
                                 alpha=.25, label=act_transition['Activity'][i])
                labelled.append(act_transition['Activity'][i])

    for i in range(0, len(wear['status'])-1):
        if wear['status'][i] == 'nonwear':
            for ax_i in range(4):
                if 'nonwear' in labelled:
                    ax[ax_i].axvspan(xmin=wear['ts'][i], xmax=wear['ts'][i+1], ymin=.5, ymax=1, color='black', alpha=.5)
                if 'nonwear' not in labelled:
                    ax[ax_i].axvspan(xmin=wear['ts'][i], xmax=wear['ts'][i + 1], ymin=.5, ymax=1, color='black',
                                     alpha=.5, label='nonwear')
                    labelled.append("nonwear")

    for row in df_event_log.itertuples():
        for ax_i in range(4):
            ax[ax_i].axvspan(xmin=row.start_timestamp, xmax=row.end_timestamp, ymin=0, ymax=.5, color='pink', alpha=.25)

    ax[3].plot(light['ts'], light['Light'], color='gold', label='light')
    ax[3].axhline(y=1050, label='overcast', color='grey', linestyle='dashed', lw=.75)
    ax[3].axhline(y=10000, label='sunny', color='darkorange', linestyle='dashed', lw=.75)
    ax[3].set_ylabel("Light")

    ax[4].plot(step_counter['ts'], step_counter['Steps'], color='purple')
    ax[4].set_ylabel("Cumulative\nStep Count")
    ax[4].grid()

    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[2].legend(loc='lower right')
    ax[3].legend(loc='lower right')

    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S"))
    plt.tight_layout()
    plt.subplots_adjust(top=.95, hspace=.075)

    ax[-1].set_xlim(accel['ts'][0] + timedelta(minutes=-5), accel['ts'][-1] + timedelta(minutes=5))

    return fig


os.chdir("W:/Ticwatch Test/PHIL_Test/PHIL_Ticwatch/A101X1C16PU03_20221004_1533/")  # changes working directory
files = os.listdir()  # lists all files in specified directory (not specified --> working directory)

subject_id = 'PHIL'

# Accelerometer ======================================================================================================
accel = combine_similar_files(folder=os.getcwd(), analysis_type='Accelerometer')
accel["ts"] = unix_to_timestamp_quick(unix_list=accel["Timestamp"])
accel['X'] /= 9.81  # convert m/s2 to G
accel['Y'] /= 9.81
accel['Z'] /= 9.81

# epoching
vm = np.sqrt(np.square(np.array([accel['X'], accel['Y'], accel['Z']])).sum(axis=0)) - 9.81
vm[vm < 0] = 0
accel['vm'] = vm
del vm

# Gyroscope ===========================================================================================================
gyro = combine_similar_files(folder=os.getcwd(), analysis_type='Gyroscope')
gyro['ts'] = unix_to_timestamp_quick(unix_list=gyro['Timestamp'])

# Heart Rate ==========================================================================================================
# average HR every ~1 second. unsure of their math
hr = combine_similar_files(folder=os.getcwd(), analysis_type='HeartRate')
hr['ts'] = unix_to_timestamp_quick(unix_list=hr['Timestamp'])
hr['hrr'], hr['hrr_intensity'] = calculate_hrr(hr_data=hr['Heart Rate'], age=23, resting_hr=60)

# Steps ==============================================================================================================
# event file for each step taken
steps = combine_similar_files(folder=os.getcwd(), analysis_type='StepDetector')
steps['ts'] = unix_to_timestamp_quick(unix_list=steps['Timestamp'])

# Step Counting =======================================================================================================
# cumulative step count. Unsure of sample interval
step_counter = combine_similar_files(folder=os.getcwd(), analysis_type='StepCounter')
step_counter['ts'] = unix_to_timestamp_quick(unix_list=step_counter['Timestamp'])

# Ambient Light =======================================================================================================
light = combine_similar_files(folder=os.getcwd(), analysis_type='AmbientLight')
light['ts'] = unix_to_timestamp_quick(unix_list=light['Timestamp'])

# Device Wear =========================================================================================================
wear = combine_similar_files(folder=os.getcwd(), analysis_type='onBody')
wear['ts'] = unix_to_timestamp_quick(unix_list=wear['Time'])
wear['status'] = wear['State (1 [ON]: 0 [Off])']
wear['status'] = ['wear' if i == 1 else 'nonwear' for i in wear['status']]

# Activity ============================================================================================================
activity = combine_similar_files(folder=os.getcwd(), analysis_type='Activity')
activity['ts'] = unix_to_timestamp_quick(unix_list=activity['Timestamp'])

act_transition = combine_similar_files(folder=os.getcwd(), analysis_type='ActivityTransition')
act_transition['ts'] = unix_to_timestamp_quick(unix_list=act_transition['Timestamp'])

# DEVICE PROPERTIES ===================================================================================================

# power_states = combine_similar_files(folder=os.getcwd(), analysis_type='PowerStates')
# power_states['ts'] = unix_to_timestamp_quick(unix_list=power_states['Timestamp'])

# battery = combine_similar_files(folder=os.getcwd(), analysis_type='ChargeLevel')
# battery['ts'] = unix_to_timestamp_quick(unix_list=battery['Timestamp'])

# PLOTTING ============================================================================================================

df_event_log = pd.read_excel("TEST Activity Log.xlsx")

plt.close('all')
fig = plot_things(abs_hr=False)
