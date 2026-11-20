"""
Classify images by quality (clear, partly blurred, blurred, dark) to discard blurred
images before labelling (Sec. III-A of the paper).
"""

import pickle
import pandas as pd
from skimage import io as skio
from os import path, mkdir
from shutil import move
import click
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features import calc_or_load_features
from src.utils import (
    standardize_features,
    list_all_files,
    CHOSEN_FEATURES_FINAL_SUBSET as features,
    CLASSES,
)


SAVE_CLF_TO = "images/clf_train_large_medium_weights.pickle"
LOAD_FEATURES_FROM = "images/chosen_features.csv"
LOAD_TRAINING_SET_FROM = "images/labeled/train_large.csv"
SORT_INTO = "images/classified"


class CustomClassifier:
    """
    This custom class saves the classifier used for the quality classification along with additional
    neccessary information.
    """

    def __init__(self, classifier, training_samples, image_size, data_normalization):
        self.classifier = classifier
        self.training_samples = training_samples
        self.image_size = image_size
        self.data_normalization = data_normalization


def train_classifier(training_file, features, features_file, classifier, saveto=None):
    """

    :param training_file: Directory to the file containing the image directories for training. The
    file should be created with or like in src/labels.py (list-files).
    :param features: List of features to include in the training.
    :param features_file: Directory to the file containing the calculated features or directory
    where a new file should be saved.
    :param classifier: Instance of LogisticRegression (from sklearn).
    :param saveto: Where to save the trained classifier to.
    :return: Classifier if saveto is None.
    """

    with open(training_file, "r") as f:
        im_dirs = f.readlines()[0]
        im_dirs = im_dirs.split(", ")

    # try image:
    sample = skio.imread(im_dirs[0])
    if sample.size < 1e6:
        # make factor 1:
        features = [
            "_".join(feat.split("_")[:-1]) + "_1.0" if "." in feat.split("_")[-1] else feat
            for feat in features
        ]
    feats, labels = calc_or_load_features(
        features=features,
        image_dirs=im_dirs,
        labeled=True,
        save_append_dir=features_file,
        load=True,
    )

    feats, data_norm = standardize_features(
        features=pd.DataFrame(data=feats, index=im_dirs, columns=features),
        data_normalization=None,
    )

    classifier.fit(feats, labels)

    clf = CustomClassifier(
        classifier=classifier,
        data_normalization=data_norm,
        training_samples=im_dirs,
        image_size=sample.shape,
    )

    if saveto is not None:
        with open(saveto, "wb") as f:
            pickle.dump(clf, f)
    else:
        return clf


def classify_images(directory, classifier, features_file, saveto=None):

    """
    Classify the images in directory with the given classifier and save the classified labels to a
    file at saveto.
    :param directory: Directory from which to take the images to classify from.
    :param classifier: Instance of CustomClassifier or directory to the location where it is
    pickled.
    :param features_file: Name of file where features to the images in directory are saved or should
    be saved.
    :param saveto: Directory where to save the results file with the classified labels to.
    :return: If saveto is None, the results are returned.g
    """

    if isinstance(classifier, str):
        with open(classifier, "rb") as f:
            classifier = pickle.load(f)

    im_dirs = list_all_files(directory)
    im_dirs = [im_dir for im_dir in im_dirs if im_dir.endswith(".jpg")]

    data_norm = classifier.data_normalization
    features = calc_or_load_features(
        features=data_norm.columns.values,
        image_dirs=im_dirs,
        labeled=False,
        save_append_dir=features_file,
        load=True,
        resize_shape=classifier.image_shape,
    )

    features = standardize_features(
        pd.DataFrame(data=features, index=im_dirs, columns=data_norm.columns.values),
        data_normalization=data_norm,
    )

    # classify:
    pred = classifier.classifier.predict(features)

    output = pd.DataFrame(index=im_dirs, data=pred, columns=["quality"])

    if saveto is not None:
        with open(saveto, "w") as f:
            output.to_csv(f)
    else:
        return output


# pylint: disable=no-value-for-parameter
# pylint: disable=too-many-arguments


@click.command()
@click.option("--load_features_from", type=str, default=LOAD_FEATURES_FROM)
@click.option("--load_training_set_from", type=str, default=LOAD_TRAINING_SET_FROM)
@click.option("--save_clf_to", type=str, default=SAVE_CLF_TO)
@click.option("--directory", type=str, default="None")
@click.option("--sort_into", type=str, default=SORT_INTO)
@click.option("--class_weight", type=dict, default="None")
def routine(
    load_features_from,
    load_training_set_from,
    save_clf_to,
    directory,
    sort_into,
    class_weight,
):

    """
    Classify images according to their quality and sort them into folders named like the respective
     labels

    :param load_features_from: Path to csv-file containing the features needed for the
     classification. If it does not
    exist, the features will be calculated and the table will be saved to this directory.
    :param load_training_set_from: Path to csv-file containing the image-directories used for
     training.
    :param save_clf_to: File name to save model to (as pickle).
    :param directory: Directory to load images from. ALL jpg-images below this level are considered.
    :param sort_into: Directory to sort images into.
    :param class_weight: Dictionary mapping class names to class weights for the prediction.
    :return:
    """

    if not path.exists(save_clf_to):
        # if the directory to the classifier is not valid, train a new classifier with the
        # given images
        with open(load_training_set_from, "r") as f:
            im_dirs = f.readlines()[0]
            im_dirs = im_dirs.split(", ")

        if class_weight is None:
            labels = np.array([im_dir.split("/")[-2] for im_dir in im_dirs])
            class_weight = {CLASS: 1 / sum(labels == CLASS) for CLASS in CLASSES}

        # then train a new classifier
        clf = LogisticRegression(class_weight=class_weight)
        train_classifier(
            training_file=load_training_set_from,
            features=features,
            features_file=load_features_from,
            classifier=clf,
            saveto=save_clf_to,
        )

    # todo: do something about image shape (add shape to CustomClassifier?)
    # classify image quality
    out = classify_images(
        directory=directory,  # ToDo: do something about duplicates
        classifier=save_clf_to,
        features_file=load_features_from,
    )

    if sort_into != "None":
        # sort the images into folders:
        for quality_class in np.unique(out.values):
            if not path.exists(f"{sort_into}/{quality_class}"):
                # make new subfolder for every label
                mkdir(f"{sort_into}/{quality_class}")
        for im in out.index.values:
            splits = im.split("/")
            new_dest = f"{sort_into}/{out.loc[im].values[0]}/{splits[-1]}"
            move(src=im, dst=new_dest)


if __name__ == "__main__":
    routine()
