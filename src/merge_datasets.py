"""

"""
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import pyplot as plt
from src.utils import parse_times_from_names

plt.ioff()

suffix = "_6_subsets"
train = True

files_to_merge = [
    f"images/classified/view_{'train' if train else 'test'}.csv",
    f"Spiekeroog/events/{'train' if train else 'test'}.csv",
    f"AlteWeser/Camera/{'train' if train else 'test'}_10.csv",
    f"Fino/Deck/{'train' if train else 'test'}.csv",
    f"Fino2/cam02/{'train' if train else 'test'}_10.csv",
    f"Fino2/cam08/{'train' if train else 'test'}_10.csv",
]
labels_for_files = [
    "images/classified/list_of_labels.csv",
    "Spiekeroog/events/list_of_labels.csv",
    "AlteWeser/list_of_labels.csv",
    "Fino/Deck/list_of_labels.csv",
    "Fino2/cam02/list_of_labels.csv",
    "Fino2/cam08/list_of_labels.csv",
]


train_images = []
train_labels = []

year = 2022

for file_to_merge, label_for_file in zip(files_to_merge, labels_for_files):
    with open(file_to_merge, "r") as file:
        lines = file.readlines()[0]
        lines = lines.split(", ")
    im_names = [line.split("/")[-1].split(".")[0] for line in lines]
    label_file = pd.read_csv(label_for_file, index_col=0)
    labels = label_file.loc[im_names].values
    # convert labels to bools
    if labels.dtype == bool:
        labels = ["True" if label else "False" for label in labels]
    else:
        labels = list(np.array(labels).T)[0]
    train_labels.extend(labels)
    train_images.extend(lines)

with open(f"{'train' if train else 'test'}{suffix}.csv", "w") as file:
    file.writelines(", ".join(train_images))

with open(f"{'train' if train else 'test'}{suffix}_labels.csv", "w") as file:
    file.writelines(", ".join(train_labels))

# analyse dataset:
dts = parse_times_from_names(train_images)
year_filter = [True if dt.year == year else False for dt in dts]
dts_filtered = np.array(dts)[year_filter]
begin = min(dts_filtered)
begin = datetime(year=begin.year, month=begin.month, day=begin.day)
end = max(dts_filtered)
week = begin
weeks = []
while week < end:
    weeks.append(week)
    week += timedelta(days=7)

week_ticks = [
    ""
    if week.day > 7
    else datetime(year=week.year, month=week.month, day=week.day).strftime("%Y/%m")
    for week in weeks[:-1]
]

hist_dts = np.histogram(dts_filtered, bins=weeks)

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(211)
h = ax1.bar(np.arange(len(hist_dts[0])), hist_dts[0])
plt.xticks(rotation=30)
ax1.set_xticks(np.arange(len(hist_dts[0])))
ax1.set_xticklabels(week_ticks)
ax1.set_title(f"Number of images per week (Total: {len(dts_filtered)})")
ax1.set_xlabel("Weeks")
ax1.set_ylabel("Absolute frequency")

labels_filtered = np.array(train_labels)[year_filter]
hist = []
for ind in np.arange(len(weeks) - 1):
    start = weeks[ind]
    stop = weeks[ind + 1]
    hist.append(
        labels_filtered[np.all([dts_filtered >= start, dts_filtered < stop], axis=0)]
    )

label_freqs = np.ones(shape=(len(hist), 2)) * np.nan
for idx, week_hist in enumerate(hist):
    label_freqs[idx, 0] = sum(week_hist == "True")
    label_freqs[idx, 1] = sum(week_hist == "False")

sum_row = np.sum(label_freqs, axis=1)
rel_freqs = label_freqs.astype(float) / np.array([sum_row, sum_row]).T
rel_freqs[np.isnan(rel_freqs)] = 0.0

ax2 = fig.add_subplot(212)
ax2.bar(np.arange(rel_freqs.shape[0]), rel_freqs[:, 0], label="True")
ax2.bar(
    np.arange(rel_freqs.shape[0]),
    rel_freqs[:, 1],
    bottom=rel_freqs[:, 0],
    label="False",
)
plt.xticks(rotation=30)
ax2.set_xticks(np.arange(rel_freqs.shape[0]))
ax2.set_xticklabels(week_ticks)
total_portion = sum(labels_filtered == "True") / len(train_labels) * 100
ax2.set_title("Portion of slick-images per week (Total: %.2f %%)" % total_portion)
ax2.set_xlabel("Weeks")
ax2.set_ylabel("Relative frequency")
ax2.legend()

fig.subplots_adjust(hspace=0.5)
fig.savefig(f"results/{'train' if train else 'test'}{suffix}_{year}.png")


train = False

files_to_merge = [
    f"images/classified/view_{'train' if train else 'test'}.csv",
    f"Spiekeroog/events/{'train' if train else 'test'}.csv",
    f"AlteWeser/Camera/{'train' if train else 'test'}_10.csv",
    f"Fino/Deck/{'train' if train else 'test'}.csv",  # f"Fino/AlphaVentus/{'train' if train else 'test'}.csv",
    f"Fino2/cam02/{'train' if train else 'test'}_10.csv",
    f"Fino2/cam08/{'train' if train else 'test'}_10.csv",
]
# f"Fino3/HDeck/{'train' if train else 'test'}_10.csv"]

labels_for_files = [
    "images/classified/list_of_labels.csv",
    "Spiekeroog/events/list_of_labels.csv",
    "AlteWeser/list_of_labels.csv",
    "Fino/Deck/list_of_labels.csv",  # "Fino/AlphaVentus/classified/list_of_labels.csv",
    "Fino2/cam02/list_of_labels.csv",
    "Fino2/cam08/list_of_labels.csv",
]
# "Fino3/HDeck/list_of_labels.csv"]

Falses = np.ones(np.shape(labels_for_files))
Trues = Falses.copy()
for idx, (file_to_merge, label_for_file) in enumerate(
    zip(files_to_merge, labels_for_files)
):
    label_df = pd.read_csv(label_for_file, index_col=0)
    # pick im_names:
    with open(file_to_merge, "r") as f:
        ims = f.readlines()[0]
    ims = ims.split(", ")
    ims = [im.split("/")[-1].split(".")[0] for im in ims]
    label_df = label_df.loc[ims]

    # convert labels to bools
    if label_df.values.dtype == bool:
        labels = np.array(["True" if label else "False" for label in label_df.values])
    else:
        labels = label_df.values

    Falses[idx] = sum(labels == "False")
    Trues[idx] = sum(labels == "True")

Aliases = [
    "Nordergründe",
    "Spiekeroog",
    "Alte Weser",
    "Fino1 - Deck",  # "Fino1 - Alpha Ventus",
    "Fino2 - cam02",
    "Fino2 - cam08",
]

fig = plt.figure(figsize=(17 / 2.5, 12 / 2.5))
ax = fig.add_subplot(111)
inds = np.arange(len(labels_for_files))
ax.bar(inds, Trues)
ax.bar(inds, Falses, bottom=Trues)
plt.xticks(rotation=30, ha="right")
ax.set_xticks(inds)
ax.set_xticklabels(Aliases)
portions = Trues / (Trues + Falses) * 100
for idx in np.arange(len(labels_for_files)):
    ax.text(
        x=inds[idx],
        y=Trues[idx] + Falses[idx] + 5,
        s="%.2f %%" % portions[idx],
        ha="center",
        va="bottom",
    )
ax.legend(["True", "False"])
ax.set_ylim([0, (max(Trues + Falses) + 5) * 1.1])
ax.set_title(
    f"Number of images/labels per dataset for {'training' if train else 'testing'}"
)
ax.set_ylabel("Absolute frequencies")
plt.subplots_adjust(bottom=0.2, top=0.9)
fig.savefig(f"results/{'train' if train else 'test'}_dataset.png")
