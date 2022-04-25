import pandas as pd
import os
import nwdata
import matplotlib.pyplot as plt
import numpy as np

edf_folder = "O:/OBI/ONDRI@Home/Data Processing/Algorithms/Non-Wear/Non-Wear Data/"
edf_files = [i for i in os.listdir(edf_folder) if 'Accelerometer' in i]

data = []
for i, file in enumerate(edf_files):
    print(file)
    edf_file = nwdata.NWData()

    if file.split("_")[0] != 'Test8':
        edf_file.import_edf(edf_folder + file)
        h = edf_file.header['duration'].total_seconds()
    if file.split("_")[0] == 'Test8':
        edf_file.import_gnac(edf_folder + 'Test8.bin')

    nw = pd.read_excel("{}{}.xlsx".format(edf_folder, file.split("_")[0]))
    nw['Off'] = pd.to_datetime(nw['Off'])
    nw['On'] = pd.to_datetime(nw['On'])
    nw['duration'] = [(row.On - row.Off).total_seconds() for row in nw.itertuples()]

    total_nw = nw['duration'].sum()
    min_nw = nw['duration'].min()
    max_nw = nw['duration'].max()

    data.append([file.split("_")[0], h, total_nw, min_nw, max_nw])


df = pd.DataFrame(data, columns=['subj', 'duration', 'total_nw', 'min_nw', "max_nw"])
df['nw%'] = 100 * df['total_nw'] / df['duration']
