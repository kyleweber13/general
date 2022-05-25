import nwdata
import pandas as pd
import pyedflib
from nwdata import EDF


def import_edf(filename):

    data = nwdata.NWData()
    data.import_edf(file_path=filename, quiet=False)

    data.ts = pd.date_range(start=data.header["start_datetime"] if 'start_datetime' in data.header.keys() else data.header['startdate'],
                            periods=len(data.signals[0]),
                            freq="{}ms".format(1000 / data.signal_headers[0]["sample_rate"]))

    return data


def import_steps_file(filename):

    df = pd.read_csv(filename)
    df = df.loc[df['step_state'] == 'success']

    df = df[['full_id', 'foot', 'step_index', 'step_time']]

    df['step_time'] = pd.to_datetime(df['step_time'])
    df = df.sort_values('step_time')

    return df


def import_bouts_file(filename):

    df = pd.read_csv(filename)

    df = df[['gait_bout_num', 'start_timestamp', 'end_timestamp', 'start_dp', 'end_dp', 'bout_length_sec', 'number_steps']]
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])

    return df


def get_sample_rates(filenames):

    print(f"\n-Checking sample rate for {len(filenames)} files...")

    sample_rates = []
    for i, file in enumerate(list(filenames)):
        a = EDF.EDFFile(file)
        a.read_header()

        for h in a.signal_headers:
            if h['label'] == 'Accelerometer x':
                sample_rates.append(h['sample_rate'])
                break

    return sample_rates


def get_starttimes(filenames):

    print(f"\n-Checking start times for {len(filenames)} files...")

    start_times = []
    for i, file in enumerate(list(filenames)):
        a = EDF.EDFFile(file)
        a.read_header()

        try:
            start_times.append(a.header['startdate'])
        except KeyError:
            start_times.append(a.header['start_datetime'])

    return start_times