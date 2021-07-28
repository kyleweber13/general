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

import nwnonwear as nw

df, nw_array = nw.vert_nonwear(x_values=ecg.signals[1], y_values=ecg.signals[2], z_values=ecg.signals[3],
                               temperature_values=ecg.signals[5], accel_freq=25, temperature_freq=1)
