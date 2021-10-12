# Michael Eden, 2021
import bottleneck
import numpy as np
import matplotlib.pyplot as plt
import peakutils


def get_zncc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Get zero-normalized cross-correlation (ZNCC) of the given vectors.
    Adapted from: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    To improve performance, the formula given by Wikipedia is rearranged as follows::
        1 / (n * std(x) * std(y)) * (sum[x(i) * y(i)] - n * mean(x) * mean(y))
    """

    # TODO: also return whether peak meets statistical significance criteria.
    # TODO: return array of lags.
    # Ensure x is the longer signal.
    x, y = sorted((x, y), key=len, reverse=True)

    # Calculate rolling mean and standard deviation. Discard first few NaN values.
    x_mean = bottleneck.move_mean(x, len(y))[len(y) - 1:]
    x_std = bottleneck.move_std(x, len(y))[len(y) - 1:]

    # Avoid division by zero and numerical errors caused by zero or very small standard deviations.
    x_std_reciprocal = np.reciprocal(x_std, where=np.abs(x_std) > 0.0000001)

    y_mean = np.mean(y)
    y_std_reciprocal = 1 / np.std(y)

    n = len(y)

    # Calculate correlation and normalize.
    correlation = np.correlate(x, y, mode="valid")
    return (1 / n) * x_std_reciprocal * y_std_reciprocal * (correlation - n * x_mean * y_mean)


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
        axes[1].scatter(x[c_peaks], [correl[i]*1.1 for i in c_peaks], marker="v", color='red', label="ZNCCPeaks")
        axes[1].axhline(y=zncc_thresh, color='limegreen', linestyle='dotted', label="ZNCC_Thresh")
        axes[1].legend(loc='lower left')
        axes[1].set_ylabel("ZNCC")
        axes[1].set_xlabel("Seconds")

    return correl, c_peaks


"""zncc, z_peaks = run_zncc_analysis(input_data=ecg, peaks=output["AllPeaks"], template=template,
                                  zncc_thresh=.725, sample_rate=250, downsample=1,
                                  show_plot=True, overlay_template=False)"""