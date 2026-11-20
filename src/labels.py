"""
Image lists, labels and inter-rater reliability (Sec. III-B and III-C of the paper):

* list-files:  write all image paths below a directory into contents.txt (input format of the jobs),
* list-labels: create list_of_labels.csv from a folder structure where images are sorted into
               folders named after their label,
* irr-subset:  choose a random subset of 1000 images to be labelled by a second person,
* irr:         compute the agreement and Cohen's kappa between both labellers.
"""

import random
import shutil
from os import makedirs
from shutil import copy

import click
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils import RANDOM_STATE, list_all_files


def read_image_list(filename):
    with open(filename, "r") as f:
        return f.readlines()[0].split(", ")


@click.group()
def cli():
    pass


@cli.command("list-files")
@click.option("--directory", type=str, default="Fino/Deck")
def list_files(directory):
    """Write all files below directory into directory/contents.txt."""
    with open(f"{directory}/contents.txt", "w") as file:
        file.writelines(", ".join(list_all_files(directory)))


@cli.command("list-labels")
@click.option("--directory", type=str, default="subset_sortiert/train_test")
@click.option("--saveto", type=str, default="None")
def list_labels(directory, saveto):
    """Map image names to the name of the folder they are in."""
    saveto = directory if saveto == "None" else saveto
    save_labels = ["False", "True", "marginal", "unsure"]

    files = [file for file in list_all_files(directory) if file.endswith(".jpg")]
    names = np.array([file.split("/")[-1].replace(".jpg", "") for file in files])
    labels = np.array([file.split("/")[-2] for file in files])
    keep_inds = [label in save_labels for label in labels]

    dataframe = pd.DataFrame(index=names[keep_inds], data=labels[keep_inds])
    dataframe.to_csv(f"{saveto}/list_of_labels.csv")


@cli.command("irr-subset")
@click.option("--total_subset", type=int, default=1000)
@click.option("--image_lists", type=str, multiple=True,
              default=["test_6_subsets.csv", "train_6_subsets.csv"])
@click.option("--alphaventus", type=str, default="Fino/AlphaVentus/classified/train_test.csv")
@click.option("--saveto", type=str, default="subset")
def irr_subset(total_subset, image_lists, alphaventus, saveto):
    """Choose a subset of images for the determination of the inter-rater-reliability."""
    ims_list = []
    for csv_file in image_lists:
        ims_list.extend(read_image_list(csv_file))

    ims_alphaventus = read_image_list(alphaventus)
    # only take classes marginal, True and False:
    labels = [im.split("/")[-2] for im in ims_alphaventus]
    choose = [label in ["True", "False", "marginal"] for label in labels]
    ims_alpha = list(np.array(ims_alphaventus)[choose])

    total_whole = len(ims_alpha) + len(ims_list)
    alpha = int(len(ims_alpha) / total_whole * total_subset)
    rest = total_subset - alpha

    random.seed(RANDOM_STATE)
    subset_alpha = random.sample(ims_alpha, alpha)
    subset_rest = random.sample(ims_list, rest)

    for folder, subset in [("train_test", subset_rest), ("AlphaVentus", subset_alpha)]:
        makedirs(f"{saveto}/{folder}", exist_ok=True)
        # save original image directories, the copies are numbered in this order:
        with open(f"{saveto}/{folder}/im_dir.csv", "w") as f:
            f.writelines(", ".join(subset))
        for idx, image in enumerate(subset):
            im_name = image.split("/")[-1]
            try:
                copy(image, f"{saveto}/{folder}/{idx}_{im_name}")
            except FileNotFoundError:
                print(f"{image} was not found.")
    shutil.make_archive(saveto, "zip")


@cli.command()
@click.option("--mylabels_dir", type=str, default="all_6_subsets_labels.csv")
@click.option("--otherlabels_dir", type=str,
              default="subset_sortiert/train_test/list_of_labels.csv")
def irr(mylabels_dir, otherlabels_dir):
    """
    Calculate the inter-rater-reliability (percent of agreement and Cohen's kappa) between the
    subset created with irr-subset and the previously annotated images.
    """
    mylabels = pd.read_csv(mylabels_dir, index_col=0)
    otherlabels = pd.read_csv(otherlabels_dir, index_col=0)

    if "AlphaVentus" in mylabels_dir and "AlphaVentus" in otherlabels_dir:
        # remove random integer from beginning of labels in otherlabel:
        otherlabels.index = ["_".join(ind.split("_")[1:])
                             if len(ind.split("_")[0]) < 8 else ind for ind in otherlabels.index]
        # if mylabels contains directories, extract image names:
        mylabels.index = [ind.split(".")[0].split("/")[-1] for ind in mylabels.index]
    else:
        # im_dir is ordered like the integers at the beginning of the image-name:
        im_dirs = np.array(read_image_list("/".join(otherlabels_dir.split("/")[:-1]) + "/im_dir.csv"))
        otherlabels.index = [im_dirs[int(ind.split("_")[0])] for ind in otherlabels.index]

    common_images = list(set(otherlabels.index).intersection(set(mylabels.index)))
    mine = mylabels.loc[common_images, "0"].values
    other = otherlabels.loc[common_images, "0"].values

    agreement = np.sum(mine == other) / len(common_images)
    kappa = cohen_kappa_score(mine, other)
    # for AlphaVentus: kappa = 0.747; agreement = 0.916
    # for train_test: kappa = 0.803; agreement = 0.96
    print(f"agreement: {agreement:.3f}, Cohen's kappa: {kappa:.3f}")


if __name__ == "__main__":
    cli()
