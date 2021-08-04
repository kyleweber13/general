import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
import nwdata.NWData
from Filtering import filter_signal
import pywt
import neurokit2 as nk
import nwnonwear
import math
import scipy.signal as signal
from OrphanidouQC import run_orphanidou
from OrphanidouQC import calculate_template
from zncc import get_zncc
import peakutils

fs = 250
epoch_len = 15
subj = "008"

template_indexes = {"007": [None, None], "008": [100000, 150000], "009": [1000000, 1100000], "010": [None, None]}

ecg = nwdata.NWData()
ecg.import_bitf(file_path=f"/Users/kyleweber/Desktop/ECG Things/{subj}_OmegaSnap.EDF", quiet=False)
ind = ecg.get_signal_index("ECG")
ds_ratio = int(ecg.signal_headers[ind]["sample_rate"] / fs)
ecg.signal_headers[ind]["sample_rate"] = fs

# Bandpass filter
f = filter_signal(data=ecg.signals[ind][::ds_ratio], sample_f=ecg.signal_headers[ind]["sample_rate"],
                  filter_type='bandpass', low_f=.67, high_f=25, filter_order=3)

# Data for QRS template
if template_indexes[subj][0] is not None and template_indexes[subj][1] is not None:
    template_data = f[template_indexes[subj][0]:template_indexes[subj][1]]
else:
    template_data = f

timestamp = pd.date_range(start=ecg.header["startdate"], periods=len(f),
                          freq="{}ms".format(1000/ecg.signal_headers[ind]["sample_rate"]))
acc_ts = pd.date_range(start=ecg.header["startdate"], periods=len(ecg.signals[1]),
                       freq="{}ms".format(1000/ecg.signal_headers[1]["sample_rate"]))
temp_ts = pd.date_range(start=ecg.header["startdate"], periods=len(ecg.signals[5]),
                        freq="{}ms".format(1000/ecg.signal_headers[5]["sample_rate"]))

# output = run_orphanidou(signal=f, sample_rate=fs, epoch_len=epoch_len)
peaks = pd.read_csv(f"/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/{subj}_Peaks_HR.csv")
orph = pd.read_csv(f"/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/{subj}_Orphanidou.csv")
nonwear_file = "/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/OmegaSnap_Nonwear.xlsx"

if ds_ratio != 1:
    if round(len(f)/peaks["Peaks"].max(), 3) < 1:
        peaks["Peaks"] = [int(i/ds_ratio) for i in peaks['Peaks']]
        orph["Index"] = [int(i/ds_ratio) for i in orph['Index']]


def run_nonwear(nonwear_file):

    nw = pd.read_excel(nonwear_file)
    nw["ID"] = [i.split("/")[-1].split("_")[0] for i in nw["File"]]
    nw = nw.loc[nw['ID'] == subj]
    nw = nw.reset_index()
    nw["DurMin"] = [(i-j).total_seconds()/60 for i, j in zip(nw["Stop"], nw["Start"])]

    df_vert, nw_array = nwnonwear.vert_nonwear(x_values=ecg.signals[1]/1000,
                                               y_values=ecg.signals[2]/1000,
                                               z_values=ecg.signals[3]/1000,
                                               temperature_freq=ecg.signal_headers[5]["sample_rate"],
                                               temperature_values=ecg.signals[5],
                                               accel_freq=ecg.signal_headers[1]['sample_rate'],
                                               std_thresh_mg=8.0, num_axes=2,
                                               low_temperature_cutoff=28,
                                               high_temperature_cutoff=32,
                                               quiet=False)
    df_vert["Start"] = [timestamp[i*int(fs/ecg.signal_headers[1]['sample_rate'])] for i in df_vert["start_datapoint"]]
    df_vert["Stop"] = [timestamp[i*int(fs/ecg.signal_headers[1]['sample_rate'])] for i in df_vert["end_datapoint"]]
    df_vert["DurMin"] = [(stop - start).total_seconds()/60 for start, stop in zip(df_vert["Start"], df_vert["Stop"])]

    return nw, df_vert


def calculate_template_data(signal, sample_rate, show_template_plot=True):

    print("\nCalculating QRS template data...")

    # Peak indexes
    peaks = nk.ecg_peaks(ecg_cleaned=signal, sampling_rate=sample_rate)[1]["ECG_R_Peaks"]

    print(f"Found {len(peaks)} peaks")

    # Template and each heartbeat
    qrs, avg_qrs = calculate_template(signal=signal, peak_indexes=peaks, fs=sample_rate, show_plot=show_template_plot)

    return avg_qrs, peaks


def run_zncc_analysis(input_data, template, peaks, sample_rate=250, downsample=2, zncc_thresh=.7,
                      show_plot=True, overlay_template=False):

    print("\nRunning ZNCC analysis -------------------------------------")
    # Runs zero-normalized cross correlation
    correl = get_zncc(x=template, y=input_data)

    print("\nDetecting peaks in ZNCC signal...")
    c_peaks = peakutils.indexes(y=correl, thres_abs=True, thres=zncc_thresh)
    print(f"\nFound {len(c_peaks)} heartbeats")
    print(f"-{len(peaks)-len(c_peaks)} ({round(100*(len(peaks) - len(c_peaks)) / len(peaks), 2)}%) rejected).")

    if show_plot:
        print("\nGenerating plot...")

        fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))

        axes[0].plot(np.arange(len(input_data))[::downsample]/sample_rate, input_data[::downsample],
                     color='black', label="Filtered")

        # Overlays template on each peak
        if overlay_template:
            for peak in peaks:
                if peak == peaks[-1]:
                    axes[0].plot(np.arange(peak-len(template)/2, peak+len(template)/2)/sample_rate, template,
                                 color='red', label="Template")
                else:
                    axes[0].plot(np.arange(peak-len(template)/2, peak+len(template)/2)/sample_rate, template,
                                 color='red')

        axes[0].legend(loc='lower left')
        axes[0].set_title("Filtered data with overlayed template on peaks")
        axes[0].set_ylabel("Voltage")

        x = np.arange(len(template)/2, len(correl) + len(template)/2)/250
        axes[1].plot(x[::downsample], correl[::downsample], color='dodgerblue', label="ZNCC")
        axes[1].scatter(x[c_peaks], [correl[i]*1.1 for i in c_peaks],
                        marker="v", color='limegreen', label="ZNCCPeaks")
        axes[1].axhline(y=zncc_thresh, color='red', linestyle='dotted', label="ZNCC_Thresh")
        axes[1].legend(loc='lower left')
        axes[1].set_ylabel("ZNCC")
        axes[1].set_xlabel("Seconds")

    return correl, c_peaks


def calculate_valid_epoch_hr():
    """Calculates HR in valid epochs only"""
    hr = []
    for i in np.arange(0, orph.shape[0]):
        print(f"{round(100*i/orph.shape[0], 2)} %")

        raw_i = int(i * fs * epoch_len)
        raw_i_end = raw_i + int(fs*epoch_len)

        validity = orph.iloc[i]['Orphanidou'] == "Valid"

        if validity:
            epoch = peaks.loc[(peaks["Peaks"] >= raw_i) & (peaks["Peaks"] < raw_i_end)]

            try:
                n_beats = epoch.shape[0]-1
                delta_t = (epoch["Peaks"].iloc[-1]-epoch["Peaks"].iloc[0])/fs
                h = round(60*n_beats/delta_t, 1)
                hr.append(h)
            except IndexError:
                hr.append(None)
        if not validity:
            hr.append(None)

    return hr


def plot_hr(log=None, ds=1):
    fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))
    plt.suptitle(f"{subj}_OmegaSnap.edf")
    plt.subplots_adjust(left=.09, right=.98)
    axes[0].plot(np.arange(0, len(f))[::ds]/fs/86400, f[::ds], color='black')
    axes[0].scatter([i/86400/fs for i in peaks["Peaks"]], [f[int(i)] for i in peaks["Peaks"]], marker='o', color='green')
    axes[0].set_title("Filtered ECG")

    axes[1].plot([i/fs/86400 for i in peaks["Peaks"]], [i for i in peaks["Roll10HR"]],
                 color='red', label="10beatroll", zorder=0)
    axes[1].plot(np.arange(0, len(hr))*epoch_len/86400+(epoch_len/2)/86400, hr, color='black', label="ValidAvg", zorder=1)
    axes[1].set_ylabel("HR")

    axes[1].set_xticks(np.arange(0, 10))
    axes[1].set_xlabel("Days")
    axes[1].set_title("QC = Orphanidou "
                      "({}% valid)".format(round(100*orph["Orphanidou"].value_counts().loc["Valid"]/orph.shape[0], 2)))

    # 008 event
    # axes[1].axvline(x=6.752, label="NewElectrode", color='limegreen')

    ax2_ylim = axes[1].get_ylim()
    if log is not None:
        c = ['pink', 'orange', 'dodgerblue', 'green', 'purple']

        df_log = pd.read_excel(log)
        df_log["Date"] = [pd.to_datetime(df_log["Start"].iloc[i]).date() for i in range(df_log.shape[0])]

        for row in df_log.itertuples():
            axes[1].fill_between(x=[(row.Start - timestamp[0]).total_seconds()/86400,
                                    (row.Start - timestamp[0]).total_seconds()/86400 + row.Duration/86400],
                                 y1=ax2_ylim[0], y2=ax2_ylim[1],
                                 color=c[row.Index % len(c)], alpha=.5, zorder=2, label=row.Date)
        axes[1].legend(loc='upper right')


def process_nonwear(combine_qc=True, min_nw_dur=5, min_nw_break=10, plot_data=True):

    if combine_qc:
        print("\nCombining Vert algorithm output with ECG signal quality output...")

        # Flags 1-sec epochs that are non-wear as 1
        vert = np.zeros(int((timestamp[-1] - timestamp[0]).total_seconds()))
        for row in df_vert.itertuples():
            start_dp = int((row.Start - timestamp[0]).total_seconds())
            stop_dp = int((row.Stop - timestamp[0]).total_seconds())

            vert[start_dp:stop_dp] = 1

        # Flags 1-sec epochs as valid/invalid
        qc = []
        for i in range(orph.shape[0]):
            for j in range(15):
                qc.append(orph["Orphanidou"].iloc[i])

        final_nw = np.zeros(int((timestamp[-1] - timestamp[0]).total_seconds()))

        for i in range(len(vert)):
            if vert[i] == 1 and qc[i] == "Valid":
                final_nw[i] = 0
            else:
                final_nw[i] = vert[i]

        starts = []
        stops = []

        for i in range(len(final_nw) - 1):
            if final_nw[i] == 0 and final_nw[i+1] == 1:
                starts.append(i+1)
            if final_nw[i] == 1 and final_nw[i+1] == 0:
                stops.append(i)

        if len(starts) > len(stops):
            stops.append(len(final_nw))

        for start, stop in zip(starts, stops):
            if stop - start < int(min_nw_dur*60):
                starts.remove(start)
                stops.remove(stop)

        for start, stop in zip(starts[1:], stops[:]):
            if start - stop < int(min_nw_break*60):
                starts.remove(start)
                stops.remove(stop)

        df_final = pd.DataFrame({"nonwear_bout_id": np.arange(1, len(starts)+1),
                                 "Start": [temp_ts[i] for i in starts], "Stop": [temp_ts[i] for i in stops],
                                 "DurMin": [(stop - start)/60 for stop, start in zip(stops, starts)]})

        print("Complete.")

    if combine_qc:
        df_use = df_final
    if not combine_qc:
        df_use = df_vert

    if plot_data:
        print("\nGenerating non-wear plot...")
        fig, ax = plt.subplots(3, figsize=(10, 7), sharex='col')

        plt.subplots_adjust(left=.07, top=.90, right=.975, hspace=.125, bottom=.08)
        plt.suptitle(f"{subj}_OmegaSnap.EDF")

        ax[0].plot(timestamp[::3], f[::3], color='black')
        ax[0].set_title("Filtered ECG")

        ax[1].plot(acc_ts[::2], ecg.signals[1][::2]/1000, color='black')
        ax[1].plot(acc_ts[::2], ecg.signals[2][::2]/1000, color='red')
        ax[1].plot(acc_ts[::2], ecg.signals[3][::2]/1000, color='dodgerblue')
        ax[1].set_ylabel("G")

        ax[2].plot(temp_ts, ecg.signals[5][:len(temp_ts)], color='red', zorder=0)
        ax[2].set_ylabel("Temperature")

        min_temp = min(ecg.signals[5])
        max_temp = max(ecg.signals[5])

        nw_durs = []
        df_nw = nw.loc[nw["DurMin"] >= min_nw_dur].reset_index()

        for row in df_nw.itertuples():
            if row.Index != df_nw.shape[0] - 1:
                ax[2].fill_between([row.Start, row.Stop], y1=min_temp, y2=min_temp + (max_temp - min_temp)/2,
                                   color='grey', alpha=.5, zorder=1)
            if row.Index == df_nw.shape[0] - 1:
                ax[2].fill_between([row.Start, row.Stop], y1=min_temp, y2=min_temp + (max_temp - min_temp)/2,
                                   color='grey', alpha=.5, label="Visual", zorder=1)
            nw_durs.append(row.DurMin)

        for row in df_use.itertuples():
            if row.Index == df_vert.shape[0] - 1:
                ax[2].fill_between([row.Start, row.Stop], y1=min_temp + (max_temp - min_temp)/2, y2=max_temp,
                                   color='limegreen', alpha=.5, label="Vert + QC" if combine_qc else "Vert", zorder=1)
            if row.Index != df_vert.shape[0] - 1:
                ax[2].fill_between([row.Start, row.Stop], y1=min_temp + (max_temp - min_temp)/2, y2=max_temp,
                                   color='limegreen', alpha=.5, zorder=1)

        ax[2].legend()

    nw_hours = sum(nw_durs)/60
    total_hours = orph.shape[0]/(3600/epoch_len)
    invalid_hours = orph["Orphanidou"].value_counts()["Invalid"]/(3600/epoch_len)

    print("\nUPDATED VALIDITY DATA:")
    print(f"-Duration: {round(total_hours, 2)} hours")
    print(f"-Nonwear duration: {round(nw_hours, 2)} hours")
    print(f"-Valid data: {round(100 - 100*(invalid_hours - nw_hours)/total_hours, 1)} %")

    return df_nw, df_use


def calculate_hourly_quality():

    hourly_quality = []

    for i in range(0, orph.shape[0], int(3600/epoch_len)):
        epoch = orph.loc[i:i+int(3600/epoch_len)]
        try:
            hourly_quality.append(round(100*epoch["Orphanidou"].value_counts()["Valid"] / epoch.shape[0], 2))
        except KeyError:
            hourly_quality.append(0)

    # 1-sec resolution nonwear data from visual inspection
    nw_flag = np.zeros(int((timestamp[-1] - timestamp[0]).total_seconds()))
    for row in nw.itertuples():
        start_dp = int((row.Start - timestamp[0]).total_seconds())
        stop_dp = int((row.Stop - timestamp[0]).total_seconds())

        nw_flag[start_dp:stop_dp] = 1

    fig, axes = plt.subplots(1, figsize=(10, 6))
    axes.plot(np.arange(len(hourly_quality)), hourly_quality, color='dodgerblue', lw=1.5)
    axes.set_xlabel("Hours")
    axes.set_ylabel("% Valid")
    axes.set_ylim(0, 100)
    axes.set_title(f"{subj}_OmegaSnap: Orphanidou Hourly Averages")

    ax2 = axes.twinx()
    ax2.fill_between(x=np.arange(len(nw_flag))/3600, y1=0, y2=nw_flag, color='grey', alpha=.35)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Wear", "NonWear"])


def determine_device_orientation():
    # Low pass eliptical filter
    sos = signal.ellip(3, 0.01, 100, 0.25, 'low', output='sos')
    ellipX = signal.sosfilt(sos, ecg.signals[1])
    ellipY = signal.sosfilt(sos, ecg.signals[2])
    ellipZ = signal.sosfilt(sos, ecg.signals[3])

    angleZ = np.arccos(ellipZ / np.sqrt(np.square(ellipX) + np.square(ellipY) + np.square(ellipZ)))
    angleZ *= 180 / math.pi

    indexes = []

    for i, val in enumerate(angleZ):
        if 80 <= val <= 100:
            indexes.append(i)

    x_test = ellipX[indexes]
    y_test = ellipY[indexes]
    z_test = ellipZ[indexes]

    angle_x = np.arccos(x_test / np.sqrt(np.square(x_test) + np.square(y_test) + np.square(z_test)))
    angle_x *= 180 / math.pi
    angle_y = np.arccos(y_test / np.sqrt(np.square(x_test) + np.square(y_test) + np.square(z_test)))
    angle_y *= 180 / math.pi
    angle_z = np.arccos(z_test / np.sqrt(np.square(x_test) + np.square(y_test) + np.square(z_test)))
    angle_z *= 180 / math.pi

    x_mean = np.mean(angle_x)
    x_med = np.median(angle_x)
    y_mean = np.mean(angle_y)
    y_med = np.median(angle_y)
    z_mean = np.mean(angle_z)
    z_med = np.median(angle_z)

    print(f"X median = {round(x_med, 1)}º; Y median = {round(y_med, 1)}º; Z median = {round(z_med, 1)}º")

    if x_med >= 60:
        orient = "Horizontal"
    if x_med <= 15:
        orient = 'Vertical'

    print(f"Orientation = {orient}")


# nw, df_vert = run_nonwear(nonwear_file=nonwear_file)

avg_qrs, template_peaks = calculate_template_data(signal=template_data, sample_rate=fs, show_template_plot=True)
test_peaks = nk.ecg_peaks(ecg_cleaned=ecg.signals[0][:int(len(ecg.signals[0]/2))], sampling_rate=fs)[1]["ECG_R_Peaks"]
zncc, zncc_peaks = run_zncc_analysis(input_data=ecg.signals[0][:int(len(ecg.signals[0]/2))],
                                     template=avg_qrs, peaks=test_peaks,
                                     sample_rate=fs, downsample=5, zncc_thresh=.72,
                                     show_plot=True, overlay_template=False)

# hr = calculate_valid_epoch_hr()
# plot_hr()
# nw_crop, df_final = process_nonwear(min_nw_dur=5, min_nw_break=10, combine_qc=False, plot_data=True)
# calculate_hourly_quality()

# TODO
# Remove nonwear from hourly calculations
