import nimbalwear
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import peakutils
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S.%f")
from matplotlib.dates import num2date as n2d
from matplotlib.backend_bases import MouseButton


""" ========================================== FUNCTIONS ============================ """


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


def import_edf(la_file, ra_file):

    la = nimbalwear.Device()
    la.import_edf(la_file)

    ra = nimbalwear.Device()
    ra.import_edf(ra_file)

    la.ts = pd.date_range(start=la.header['start_datetime'], periods=len(la.signals[0]),
                          freq=f"{1000 / la.signal_headers[0]['sample_rate']:.6f}ms")
    ra.ts = pd.date_range(start=la.header['start_datetime'], periods=len(ra.signals[0]),
                          freq=f"{1000 / ra.signal_headers[0]['sample_rate']:.6f}ms")

    return la, ra


def import_clinical_timestamps(file="O:/OBI/Personal Folders/Namiko Huynh/steps_assessment_timestamps_new.xlsx"):
    df_events = pd.read_excel(file)
    df_events.columns = ['subject_id', 'start_time', 'end_time', 'comments',
                         'walk1_6m_start', 'walk1_6m_end', 'walk2_6m_start', 'walk2_6m_end',
                         'walk3_6m_start', 'walk3_6m_end', 'walk1_6mwt_start', 'walk1_6mwt_end']

    return df_events


def kyles_step_detection(la_timestamps, ra_timestamps, axis='y', use_abs=True,
                         lowpass_cut=1, threshold=.1, sample_rate=75):

    if axis == 'vertical':
        axis = 'y'
    if axis == 'ap':
        axis = 'x'
    if axis == 'ml':
        axis = 'z'

    print(f"\nRunning Kyle's peak detection on {axis} axis with {lowpass_cut}Hz lowpass filter "
          f"with{'out' if not use_abs else ''} absolute value data...")

    # gravity removal
    la_filt = filter_signal(data=la.signals[la.get_signal_index(f'Accelerometer {axis}')], sample_f=75, high_f=.05,
                            filter_type='highpass')
    ra_filt = filter_signal(data=ra.signals[ra.get_signal_index(f'Accelerometer {axis}')], sample_f=75, high_f=.05,
                            filter_type='highpass')

    if use_abs:
        la_filt = abs(la_filt)
        ra_filt = abs(ra_filt)

    la_filt = abs(filter_signal(data=la_filt, sample_f=75, low_f=lowpass_cut, filter_type='lowpass'))
    ra_filt = abs(filter_signal(data=ra_filt, sample_f=75, low_f=lowpass_cut, filter_type='lowpass'))

    la_peaks_all = peakutils.indexes(y=la_filt, thres=threshold,
                                     min_dist=sample_rate * 2 / 3, thres_abs=True)
    la_peaks_all = pd.DataFrame({"step_time": la_timestamps[la_peaks_all], 'idx': la_peaks_all,
                                 'foot': ['left'] * len(la_peaks_all)})

    ra_peaks_all = peakutils.indexes(y=ra_filt, thres=threshold,
                                     min_dist=sample_rate * 2 / 3, thres_abs=True)
    ra_peaks_all = pd.DataFrame({"step_time": ra_timestamps[ra_peaks_all], 'idx': ra_peaks_all,
                                 'foot': ['right'] * len(ra_peaks_all)})

    df_steps_all = pd.concat([la_peaks_all, ra_peaks_all])
    df_steps_all.sort_values("step_time", inplace=True)
    df_steps_all.reset_index(drop=True, inplace=True)

    return df_steps_all, la_filt, ra_filt


def mouse_click(event):
    if event.button is MouseButton.LEFT:
        x, y = event.xdata, event.ydata
        xf = n2d(x).strftime("%Y-%m-%d %H:%M:%S.%f")
        print(xf)
        return xf


def key_press(event):
    if event.key == 'a':
        x, y = event.xdata, event.ydata
        xf = n2d(x).strftime("%Y-%m-%d %H:%M:%S.%f")
        print(xf)
        manual_steps.append(xf)

        return xf

    if event.key == 'x':
        rem = manual_steps.pop()
        print(f"Removed {rem}")


def plot_data(subj, kyle_steps, axis='y', downsample=1):

    left_sig = la.signals[la.get_signal_index(f"Accelerometer {axis}")]
    right_sig = ra.signals[ra.get_signal_index(f"Accelerometer {axis}")]

    kyle_l = kyle_steps.loc[kyle_steps['foot'] == 'left']
    kyle_r = kyle_steps.loc[kyle_steps['foot'] == 'right']

    fig, ax = plt.subplots(2, figsize=(12, 8), sharex='col')
    ax[0].plot(la.ts[::downsample], left_sig[::downsample], color='red', zorder=0, label='Left')
    ax[0].plot(ra.ts[::downsample], right_sig[::downsample], color='dodgerblue', zorder=0, label='Right')
    ax[0].scatter(la.ts[kyle_l['idx']], [1.1] * kyle_l.shape[0], marker='v', color='red', label='Left')
    ax[0].scatter(ra.ts[kyle_r['idx']], [1.1] * kyle_r.shape[0], marker='v', color='dodgerblue', label='Right')

    ax[0].set_ylabel("Raw acceleration")

    ax[1].plot(la.ts[::downsample], la_filt[::downsample], color='red', zorder=0, label='Left')
    ax[1].plot(ra.ts[::downsample], ra_filt[::downsample], color='dodgerblue', zorder=0, label='Right')
    ax[1].scatter(la.ts[kyle_l['idx']], [i * 1.05 for i in la_filt[kyle_l['idx']]], marker='v', color='red', label='kyle_left')
    ax[1].scatter(ra.ts[kyle_r['idx']], [i * 1.05 for i in ra_filt[kyle_r['idx']]], marker='v', color='dodgerblue', label='kyle_right')
    ax[1].set_ylabel("Kyle's filtered data")
    ax[1].set_ylim(0, )

    event_6m1 = df_events.loc[df_events['subject_id'] == subj]

    for i in range(2):
        try:
            ax[i].axvspan(event_6m1.walk1_6m_start, event_6m1.walk1_6m_end, 0, 1, color='red', alpha=.25, label='6m #1')
        except:
            print("Can't plot first 6m walk")

        try:
            ax[i].axvspan(event_6m1.walk2_6m_start, event_6m1.walk2_6m_end, 0, 1, color='#F49800', alpha=.25, label='6m #2')
        except:
            print("Can't plot second 6m walk")

        try:
            ax[i].axvspan(event_6m1.walk3_6m_start, event_6m1.walk3_6m_end, 0, 1, color='#F4E900', alpha=.25, label='6m #3')
        except:
            print("Can't plot third 6m walk")

        try:
            ax[i].axvspan(event_6m1.walk1_6mwt_start, event_6m1.walk1_6mwt_end, 0, 1, color='limegreen', alpha=.25, label='6mwt')
        except:
            print("Can't plot 6MWT")

    ax[0].legend(loc='lower right')

    ax[-1].xaxis.set_major_formatter(xfmt)

    # plt.connect('button_press_event', mouse_click)
    plt.connect('key_press_event', key_press)

    plt.suptitle(f"Participant {subj}")
    plt.tight_layout()

    return fig


""" ======================================= FUNCTION CALLS ============================ """

all_subjs = ['STEPS_1611', 'STEPS_2938', 'STEPS_6707', 'STEPS_7914', 'STEPS_8856']

subj = 'STEPS_1611'
acc_axis = 'y'

# clinical event file
df_events = import_clinical_timestamps()

# import raw EDF data
la, ra = import_edf(la_file=f"Z:/NiMBaLWEAR/STEPS/wearables/device_edf_cropped/{subj}_01_GNOR_LAnkle.edf",
                    ra_file=f"Z:/NiMBaLWEAR/STEPS/wearables/device_edf_cropped/{subj}_01_GNOR_RAnkle.edf")

# kyle's filtering and step detection
kyle_steps, la_filt, ra_filt = kyles_step_detection(la_timestamps=la.ts, ra_timestamps=ra.ts,
                                                    axis=acc_axis, sample_rate=75,
                                                    use_abs=True, lowpass_cut=1, threshold=.1)

# plotting
manual_steps = []
# every time you push the 'a' key, cursor's timestamp will be printed. 'x' to delete
fig = plot_data(subj=subj, axis=acc_axis, downsample=1, kyle_steps=kyle_steps)
