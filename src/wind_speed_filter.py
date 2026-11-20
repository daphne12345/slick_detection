"""
This script takes a file with wind speed and images which have the time when they were taken in
their name (according to the format DATE_FORMAT in src.utils). Every image, that
can be assigned to a wind speed which is higher than 5 m/s at 10 cm above the sea surface is
excluded. At the end of the script the given images are split into a test and a training subset.
"""

import random
import pandas as pd
import numpy as np

from src.utils import datenum_to_datetime, list_all_files, parse_times_from_names, RANDOM_STATE

path_to_images = "AlteWeser/Camera"  # 31 m
path_to_file = "images/ws_ts_1221_0622.csv"  # 17 m
choose_from = [["True", "False"]]
z = 0.1  # height right above sea surface
threshold_on_z = 5.0  # u(z) --> wind speed at height z
reference_height = 31  # 17   # zr --> height where wind was measured
roughness_length = 1e-4  # z0
threshold = (
    threshold_on_z
    * np.log(reference_height / roughness_length)
    / np.log(z / roughness_length)
)
threshold = np.ceil(threshold)

if path_to_file.endswith(".dat"):
    data = pd.read_csv(path_to_file, sep="\t", header=4, index_col="Time")
    data = data.drop(index="# Data:")
    relevant_column = "Value"
else:
    data = pd.read_csv(path_to_file)
    dates = [datenum_to_datetime(datenum) for datenum in data.iloc[:, 0]]
    data.index = dates
    relevant_column = "wind speed"

# "round" datetimes (ignore milliseconds)
data.index = pd.to_datetime(data.index).round("1s")
data = (data.loc[:, relevant_column]).squeeze()
# drop all nans:
data = data.dropna()


files = list_all_files(path_to_images)
# only take jpgs:
files = [
    file for file in files if file.endswith(".jpg")
]
# parse dates from titles:
files = [
    file.replace("_AlteWeser", "") if "AlteWeser" in file else file for file in files
]
dates_ims = parse_times_from_names(files)
dates_ims = pd.DataFrame(index=dates_ims, columns=["file_name"], data=files)

# merge DataSeries:
data = pd.concat([data, dates_ims])
data = data.sort_index()

# interpolation:
data_new = data.interpolate(method="time", limit_area="inside")
# do not assume that the values are equally spaced but assume that the index gives a clue about the
# spacing also do only interpolate, do not extrapolate todo: how many values are lost here?

interpolated_values = data_new.loc[dates_ims.index.values, :]
interpolated_values.dropna(inplace=True)  # from 26 000 to 23 000

# create a new training and test-dataset based only on images where the wind speed was above a
# certain threshold
subset = interpolated_values.iloc[(interpolated_values.iloc[:, 0] < threshold).values]

# now filter further:
inds_to_keep = []
for idx, sub in enumerate(subset.iloc[:, 1]):
    if np.any([choose in sub for choose in choose_from[0]]):
        inds_to_keep.append(idx)

subset = subset.iloc[inds_to_keep, :]
subset = subset.drop_duplicates()

# make two subsets for training and testing
random.seed(RANDOM_STATE)
points = subset.index.values
points = random.sample(list(points), k=len(points))

length = len(points)
split70 = int(length * 0.7)
train_inds = points[:split70]
test_inds = points[split70:length]

train_dirs = subset.loc[train_inds, "file_name"].values
test_dirs = subset.loc[test_inds, "file_name"].values

if "AlteWeser" in path_to_images:
    train_dirs = [train.replace(".jpg", "_AlteWeser.jpg") for train in train_dirs]
    test_dirs = [test.replace(".jpg", "_AlteWeser.jpg") for test in test_dirs]

# save dataset-splits:
with open(f"{path_to_images}/train_{int(threshold)}.csv", "w") as file:
    file.writelines(", ".join(train_dirs))

with open(f"{path_to_images}/test_{int(threshold)}.csv", "w") as file:
    file.writelines(", ".join(test_dirs))

