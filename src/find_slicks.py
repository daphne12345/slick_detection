"""
This script can be called to classify images. The output is a folder where the images are sorted
according to their classified labels and a file mapping the image directories to their labels.
"""

import click
from os import listdir, mkdir, remove
from os.path import isdir
from shutil import copy
import numpy as np
import pandas as pd
from keras.models import load_model
from autokeras import CUSTOM_OBJECTS
import tensorflow as tf
from tensorflow.data import Dataset as tf_Dataset
from src.prepare_datasets import Dataset

BATCH_SIZE = 16

@click.command()
@click.option("-d", "--directory", type=str, help="Directory where the images which are supposed "
                                                  "to be classified are stored. Only files which "
                                                  "end on jpg are considered and no files from "
                                                  "subdirectories are considered.")
@click.option("--saveto", type=str, help="Directory to which to copy the classified images. "
                                         "Into this directory also the classification.csv is saved"
                                         " where the image names are mapped to the labels. The "
                                         "default directory will be the same as  directory.",
              default="None")
@click.option("--load_from", type=str, help="Directory to the model to use if the model is not at"
                                            " src/model/final_classifier.h5.",
              default="src/model/final_classifier.h5")
@click.option("--threshold", type=float,
              help="The probability above which an image is classified as True. "
                   "The default is 0.636.",
              default=0.636)
def routine(directory, saveto, load_from, threshold):

    assert isdir(directory), f"{directory} (argument '--directory'/'-d') is not a directory!"
    if saveto != "None":
        assert isdir(saveto), f"{saveto} (argument '--saveto') is not a directory!"
    else:
        saveto = directory
    im_dirs = listdir(directory)
    # only take jpgs and prepend directory to the file names:
    im_dirs = [f"{directory}/{im_dir}" for im_dir in im_dirs if im_dir.endswith(".jpg")]
    assert len(im_dirs) > 0, f"No images were found in {directory}!"
    # make comma separated file to provide to Dataset:
    filename = f"{directory}/im_dir.csv"
    with open(filename, "w") as f:
        f.writelines(", ".join(im_dirs))
    dataset = Dataset(im_dir=filename, resize_shape=(224, 224))
    dataset.create_dataset()
    im_dirs = np.array(dataset.im_dirs)
    dataset = tf_Dataset.from_tensor_slices(tf.convert_to_tensor(dataset.dataset))
    # load model:
    classifier = load_model(load_from, custom_objects=CUSTOM_OBJECTS)
    # classify:
    prediction = classifier.predict(x=dataset.batch(BATCH_SIZE))
    prediction = prediction.T[0]
    classification = prediction > threshold
    print(f"{classification.sum()}/{len(classification)} images were classified as True. "
          f"Threshold is {threshold}.")
    # write predictions to file:
    df = pd.DataFrame(index=im_dirs, data=np.array([prediction, classification]).T,
                 columns=["probability", "classification"])
    with open(f"{saveto}/prediction.csv", "w") as f:
        df.to_csv(f)
    # sort images into subfolders according to their label:
    for subdir in ["True", "False"]:
        mkdir(f"{saveto}/{subdir}")
    trues = im_dirs[classification]
    for image in trues:
        im_name = image.split("/")[-1]
        copy(image, f"{saveto}/True/{im_name}")
    falses = im_dirs[~classification]
    for image in falses:
        im_name = image.split("/")[-1]
        copy(image, f"{saveto}/False/{im_name}")
    # remove im_dir.csv:
    remove(filename)


if __name__ == "__main__":
    routine()


