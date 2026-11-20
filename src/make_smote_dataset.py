"""
This job creates an oversampled dataset to use in src.run_autosklearn.
"""

# pylint: disable=no-value-for-parameter
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

from os.path import isdir

import click
import pandas as pd
import numpy as np

from sklearn.feature_selection import f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE

from src.utils import (
    standardize_features,
    RANDOM_STATE,
    get_best_from_each_feature_class,
)
from src.features import calc_or_load_features, FEATURES

TRAINING_SET = "images/classified/train.csv"
TEST_SET = "images/classified/test.csv"
LOAD_FROM = "images/classified/slicks_features.csv"
SAVETO = "images/classified"
TOP = 1


@click.command()
@click.option("--im_dir_train", type=str, default=TRAINING_SET)
@click.option("--im_dir_test", type=str, default=TEST_SET)
@click.option("--load_from", type=str, default=LOAD_FROM)
@click.option("--features", type=str, multiple=True, default=FEATURES)
@click.option("--saveto", type=str, default=SAVETO)
@click.option("--filter_method", type=click.Choice(["r", "f", "mi"]))
def routine(im_dir_train, im_dir_test, load_from, features, saveto, filter_method):
    """
    :param im_dir_train: Directory to csv-file where a list of image directories used for
    training is stored.
    :param im_dir_test: Directory to csv-file where a list of image directories used for
    testing is stored.
    :param load_from: Directory to csv-file where the features for the images are stored.
    :param features: List of features to consider.
    :param saveto: Directory where the new csv is saved to.
    :param filter_method: For the filtering of the features.
    :return: A new csv-file is saved. This file contains all relevant features in the columns and
    in the rows all training and test images as well as the additional SMOTE-samples. Also, a file
    with the new training instances is saved.
    """

    # load file with image directories:
    with open(im_dir_train, "r") as f:
        ims_train = f.readlines()[0]
    ims_train = ims_train.split(", ")

    with open(im_dir_test, "r") as f:
        ims_test = f.readlines()[0]
    ims_test = ims_test.split(", ")

    # load features and/or calculate additional features if they are missing
    X_train, y_train = calc_or_load_features(
        features=features,
        image_dirs=ims_train,
        labeled=True,
        save_append_dir=load_from,
        load=True,
    )
    X_test, y_test = calc_or_load_features(
        features=features,
        image_dirs=ims_test,
        labeled=True,
        save_append_dir=load_from,
        load=True,
    )

    # standardize the dataset
    X_train, data_norm = standardize_features(
        features=pd.DataFrame(data=X_train, index=ims_train, columns=features)
    )
    X_test = standardize_features(
        features=pd.DataFrame(data=X_test, index=ims_test, columns=features),
        data_normalization=data_norm,
    )

    # filter the features
    stats = ["f", "mi"]
    funcs = [f_classif, mutual_info_classif]
    filter_method_scores = pd.DataFrame(index=features, columns=stats)
    # todo: dangerous line of code (breaks in non-binary case)
    y_train_numeric = np.array([1.0 if y == "True" else 0.0 for y in y_train])
    for stat, func in zip(stats, funcs):
        values = func(X_train.values.astype(float), y_train_numeric)
        if "f" in stat:
            values = values[1]
        filter_method_scores.loc[:, stat] = values

    # filter out the best feature from every class:
    best_features = get_best_from_each_feature_class(
        features=filter_method_scores,
        column_to_use=filter_method,
        maximize=filter_method != "f",
        top=TOP,
    )
    X_train = X_train.loc[:, best_features]
    X_test = X_test.loc[:, best_features]
    # oversampling with SMOTE algorithm:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_train, y_train.astype(int))
    # todo: dangerous line of code (breaks in non-binary case)
    X_smote.loc[:, "label"] = ["False" if y == 0 else True for y in y_smote]

    # new im_dir_train file:
    im_dir_train_file = ", ".join([str(ind) for ind in X_smote.index.values])

    # concatenate this dataframe with test-set:
    y_test = ["False" if y == 0 else True for y in y_test]
    X_test.loc[:, "label"] = y_test
    X_smote = pd.concat([X_smote, X_test])

    saved = False
    while saved is False:
        try:
            with open(saveto, "w") as file:
                X_smote.to_csv(file)
            with open(saveto.replace(".csv", "_im_dir_train.csv"), "w") as file:
                file.writelines(im_dir_train_file)

        except (FileNotFoundError, IsADirectoryError, PermissionError):
            if isdir(saveto):
                saveto = f"{saveto}/smote_filter_{filter_method}.csv"
            else:
                saveto = input(f"Path {saveto} was not found. Suggest another path: ")
        else:
            saved = True


if __name__ == "__main__":
    routine()
