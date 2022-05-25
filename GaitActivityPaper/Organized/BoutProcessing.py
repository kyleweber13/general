import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta


def find_bouts(peaks_inds, fs, start_time, min_steps=3, min_duration=180, max_break=5, show_plot=False, quiet=True):

    print(f"\nFinding bouts of minimum {min_steps} steps, miniumum duration of {min_duration} seconds,"
          f" and maximum break of {max_break} seconds...")

    starts = []
    stops = []
    n_steps_list = []

    peaks_inds = list(peaks_inds)

    curr_ind = 0
    for i in range(len(peaks_inds)):
        if i >= curr_ind:

            prev_step = peaks_inds[i]
            n_steps = 1

            for j in range(i + 1, len(peaks_inds)):
                if (peaks_inds[j] - prev_step) / fs <= max_break:
                    n_steps += 1

                    if not quiet:
                        print(f"Peak = {peaks_inds[j]}, prev peak = {prev_step}, "
                              f"gap = {round((peaks_inds[j] - prev_step) / fs, 1)}, steps = {n_steps}")
                    prev_step = peaks_inds[j]

                if (peaks_inds[j] - prev_step) / fs > max_break:
                    if n_steps >= min_steps:
                        starts.append(peaks_inds[i])
                        curr_ind = j
                        stops.append(peaks_inds[j - 1])
                        n_steps_list.append(n_steps)

                    if n_steps < min_steps:
                        if not quiet:
                            print("Not enough steps in bout.")
                    break

    df_out = pd.DataFrame({"start": starts, "end": stops, "number_steps": n_steps_list})
    df_out['duration'] = [(j - i) / fs for i, j in zip(starts, stops)]

    df_out = df_out.loc[df_out['duration'] >= min_duration]

    df_out['cadence'] = 60 * df_out['number_steps'] / df_out['duration']

    df_out['start_timestamp'] = [start_time + timedelta(seconds=row.start / fs) for row in df_out.itertuples()]
    df_out['end_timestamp'] = [start_time + timedelta(seconds=row.end / fs) for row in df_out.itertuples()]

    if show_plot:
        fig, ax = plt.subplots(1, sharey='col', sharex='col')
        ax.hist(df_out['cadence'], bins=np.arange(0, 150, 5), edgecolor='black')
        ax.set_ylabel("N_walks")
        ax.set_xlabel("cadence")
        ax.set_title(f"New ({min_steps} step min., {max_break}s max break, {min_duration}s min. duration)")

    print(f"-Found {df_out.shape[0]} bouts.")

    return df_out.reset_index(drop=True)
