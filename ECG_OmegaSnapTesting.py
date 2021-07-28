import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import nwdata.NWData
from Filtering import filter_signal

fs = 125
epoch_len = 15
subj = "007"

ecg = nwdata.NWData()
ecg.import_bitf(file_path=f"/Users/kyleweber/Desktop/{subj}_OmegaSnap.EDF", quiet=False)
ind = ecg.get_signal_index("ECG")
ds_ratio = int(ecg.signal_headers[ind]["sample_rate"] / fs)
ecg.signal_headers[ind]["sample_rate"] = fs

f = filter_signal(data=ecg.signals[ind][::ds_ratio], sample_f=ecg.signal_headers[ind]["sample_rate"],
                  filter_type='bandpass', low_f=.67, high_f=25, filter_order=3)
timestamp = pd.date_range(start=ecg.header["startdate"], periods=len(f),
                          freq="{}ms".format(1000/ecg.signal_headers[ind]["sample_rate"]))

nw = pd.read_excel("/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/OmegaSnap_Nonwear.xlsx")

# output = run_orphanidou(signal=f, sample_rate=fs, epoch_len=epoch_len)
peaks = pd.read_csv(f"/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/{subj}_Peaks_HR.csv")
orph = pd.read_csv(f"/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/{subj}_Orphanidou.csv")

if ds_ratio != 1:
    peaks["Peaks"] = [int(i/ds_ratio) for i in peaks['Peaks']]
    orph["Index"] = [int(i/ds_ratio) for i in orph['Index']]


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
    axes[1].axvline(x=6.752, label="NewElectrode", color='limegreen')

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


def plot_nw(subj_id):
    fig, ax = plt.subplots(2, figsize=(10, 6), sharex='col')
    ax[0].plot(timestamp[::3], f[::3], color='black')
    ax[1].plot(pd.date_range(start=timestamp[0], periods=len(ecg.signals[5]), freq="1S"), ecg.signals[5], color='red')

    nw_durs = []
    for row in nw.itertuples():
        if subj_id in row.File:
            ax[1].fill_between([row.Start, row.Stop], y1=20, y2=35, color='grey', alpha=.5)
            nw_durs.append(row.DurMin)

    nw_hours = sum(nw_durs)/60
    total_hours = orph.shape[0]/(3600/epoch_len)
    invalid_hours = orph["Orphanidou"].value_counts()["Invalid"]/(3600/epoch_len)

    print("\nUPDATED VALIDITY DATA:")
    print(f"-Duration: {round(total_hours, 2)} hours")
    print(f"-Nonwear duration: {round(nw_hours, 2)} hours")
    print(f"-Valid data: {round(100 - 100*(invalid_hours - nw_hours)/total_hours, 1)} %")

    return nw


def calculate_hourly_quality():

    hourly_quality = []

    for i in range(0, orph.shape[0], int(3600/epoch_len)):
        epoch = orph.loc[i:i+int(3600/epoch_len)]
        try:
            hourly_quality.append(round(100*epoch["Orphanidou"].value_counts()["Valid"] / epoch.shape[0], 2))
        except KeyError:
            hourly_quality.append(0)

    plt.plot(np.arange(len(hourly_quality)), hourly_quality, color='dodgerblue')
    plt.xlabel("Hours")
    plt.ylabel("% Valid")
    plt.ylim(0, 100)
    plt.title(f"{subj}_OmegaSnap: Orphanidou Hourly Averages")


# hr = calculate_valid_epoch_hr()
# plot_hr(log="/Users/kyleweber/Desktop/008_StravaLog.xlsx")
# plot_hr()
# nw = plot_nw(subj_id=subj)
# calculate_hourly_quality()

# TODO
# Remove nonwear from hourly calculations
