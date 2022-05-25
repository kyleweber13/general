import os
import pandas as pd
import nwgait
import pyedflib
import nwgait
from GaitActivityPaper.Organized.DataImport import import_edf


def create_ankle_edf_filenames(df):
    """Creates ankle filenames given df which contains wrist_edf column. Uses ankle on dominant wrist side if available.

        returns:
        -list of file names
    """

    ankle_fnames = []

    for row in df.itertuples():
        fname = row.wrist_edf.split("/")[-1]
        filepath = row.wrist_edf[:-len(fname)]

        ankle_fname = "{}_01_{}_{}Ankle.edf"

        if os.path.exists(filepath + ankle_fname.format(row.full_id, fname.split("_")[3], row.Hand)):
            ankle_fnames.append(filepath + ankle_fname.format(row.full_id, fname.split("_")[3], row.Hand))
        else:
            if os.path.exists(filepath + ankle_fname.format(row.full_id, fname.split("_")[3], "L" if row.Hand == "R" else "R")):
                ankle_fnames.append(filepath + ankle_fname.format(row.full_id, fname.split("_")[3], "L" if row.Hand == "R" else "R"))
            else:
                ankle_fnames.append(None)

    return ankle_fnames


def format_gaitbout_files(df, write_file=False, save_dir=""):
    """Formats OND06 step files into current format"""

    for row in df.itertuples():
        print(row.full_id, row.steps_file)

        if "OND09" in row.steps_file:
            pass

        if "OND06" in row.steps_file:
            if "xlsx" in row.steps_file:
                df = pd.read_excel(row.steps_file)
            if "csv" in row.steps_file:
                df = pd.read_csv(row.steps_file)

                df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
                df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])
                df['duration'] = [(j-i).total_seconds() for i, j in zip(df['start_timestamp'], df['end_timestamp'])]
                df['cadence'] = 60 * df['number_steps'] / df['duration']

                df = df[['study_code', 'subject_id', 'start_timestamp', 'end_timestamp', 'number_steps', 'duration', 'cadence']]

            if write_file:
                df.to_csv(f"{save_dir}{row.full_id}_AllBouts.csv", index=False)


def reformat_ond09_steps_file(file, outdir):
    """Takes nwgait output and reformats for this code."""

    print(f"\nReformatting {file}...")
    df = pd.read_csv(file)
    df = df[["step_time", 'step_index']]
    df.columns = ['timestamp', 'idx']

    df.to_csv(outdir + file.split("/")[-1], index=False)
    print(f"File saved to {outdir}.")

    return df


def run_nwgait(df, subjs=(), save_file=True, save_dir=""):
    """Runs nwgait and spits out df_steps for given filename. Ability to write steps file to csv."""

    print(f"\nRunning step detection for {subjs}...")
    steps_output_files = []
    bout_output_files = []

    n_proc = 1
    for row in df.itertuples():
        print(row.ankle_edf)

        if row.full_id in subjs:
            print(f"\nProcessing {row.full_id} ({n_proc}/{len(list(subjs))})...")
            n_proc += 1

            ankle = import_edf(row.ankle_edf)

            # convert inputs to objects as inputs
            accel_x_sig = ankle.get_signal_index('Accelerometer x')
            accel_y_sig = ankle.get_signal_index('Accelerometer y')
            accel_z_sig = ankle.get_signal_index('Accelerometer z')

            obj = nwgait.AccelReader.sig_init(raw_x=ankle.signals[accel_x_sig],
                                              raw_y=ankle.signals[accel_y_sig],
                                              raw_z=ankle.signals[accel_z_sig],
                                              startdate=ankle.header['startdate'] if 'startdate' in ankle.header.keys() else ankle.header['start_datetime'],
                                              freq=ankle.signal_headers[accel_x_sig]['sample_rate'])

            # run gait algorithm to find bouts
            wb = nwgait.WalkingBouts(obj, obj, left_kwargs={'axis': None}, right_kwargs={'axis': None})

            # save bout times
            bouts = wb.export_bouts()

            # save step times
            steps = wb.export_steps()

            steps['full_id'] = [row.full_id] * steps.shape[0]

            steps_out = "{}{}_Steps.csv".format(save_dir, os.path.basename(row.ankle_edf).split(".")[0])
            bout_out = "{}{}_Bouts.csv".format(save_dir, os.path.basename(row.ankle_edf).split(".")[0])

            if save_file:
                steps.to_csv(steps_out, index=False)
                print("-File saved to {}".format(steps_out))

                bouts.to_csv(bout_out, index=False)
                print("-File saved to {}".format(bout_out))

                steps_output_files.append(steps_out)
                bout_output_files.append(bout_out)

            if not save_file:
                steps_output_files.append(None)
                bout_output_files.append(None)

        if row.full_id not in subjs:
            steps_output_files.append(None)
            bout_output_files.append(None)

    return steps_output_files, bout_output_files
