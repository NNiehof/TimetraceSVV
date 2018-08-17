import pandas as pd
import numpy as np
import scipy.io
import os.path
import pickle

"""
Read in txt files with StepGVS data, convert to
dataframe (parameters) and numpy arrays (measurements)
and save to Matlab file and to pickle.
"""

# read in data
data_folder = "D:/OneDrive/Code/TimetraceSVV/Data/001/"
data_file = data_folder + "001_stepGVS_exp_20180817_120206.txt"
df = pd.read_csv(data_file, sep="; ", engine="python")
sj = os.path.basename(os.path.normpath(data_folder))

# recast data:
# outer list comprehension turns string into list
# nested: turn strings inside the list into float
df["line_drift"] = [[float(y) for y in x.strip("[]").split(",")] for x in df["line_drift"]]
df["line_ori"] = [[float(y) for y in x.strip("[]").split(",")] for x in df["line_ori"]]
df["frame_time"] = [[float(y) for y in x.strip("[]").split(",")] for x in df["frame_time"]]
df["frame_ori"] = df["frame_ori"].apply(pd.to_numeric, errors="coerce")

# get array size from first trial
n_samples = len(df["line_ori"][0])
n_trials = df.shape[0]
drift = np.empty((n_trials, (n_samples + 1)))
line_ori = np.empty((n_trials, n_samples))
timestamp = np.empty((n_trials, n_samples))

# 2D numpy arrays for data per frame
for index, row in df.iterrows():
    drift[index, :] = row["line_drift"]
    line_ori[index, :] = row["line_ori"]
    # downsample data that accidentally had two timestamps per frame
    if len(row["frame_time"]) == (2 * n_samples):
        timestamp[index, :] = row["frame_time"][::2]
    else:
        timestamp[index, :] = row["frame_time"]

# drop columns that are now in numpy arrays
df = df.drop(columns=["line_drift", "line_ori", "frame_time"])

# save to .mat file
f_name = "{0}sj{1}_data.mat".format(data_folder, sj)
scipy.io.savemat(f_name,
                 {"df": df.to_dict("list"), "drift": drift,
                  "line_ori": line_ori, "timestamp": timestamp},
                 oned_as="column")

# save to pickle
p_name = "{0}sj{1}_data.pickle".format(data_folder, sj)
with open(p_name, "wb") as f:
    pickle.dump(df, f)
    pickle.dump(drift, f)
    pickle.dump(line_ori, f)
    pickle.dump(timestamp, f)
