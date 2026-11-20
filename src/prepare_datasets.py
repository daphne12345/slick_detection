"""
This module contains a class which is needed to create the dataset used in src.run_autokeras and
src.train_keras_model.
"""

from os import path
import pickle
from zipfile import ZipFile

import numpy as np
from skimage import io as skio
from skimage.transform import resize



class Dataset:
    """
    This class has the methods create_dataset and load_dataset_into_directory. When using
    create_dataset, a dataset is saved at the given directory. To load this dataset, the static
    method load_dataset_into_directory can be used. This method can be accessed without having to
    initialize a Dataset instance.
    """
    def __init__(self, im_dir: str, resize_shape: tuple = None):
        """

        :param im_dir: Directory to file containing the directories to all the relevant images.
        :param resize_shape: Shape in which the images should be saved. Format is (nrows, ncols,
         nchannels).
        If the dimension is 2 (nchannels is missing), the grayscale image is taken.
        """
        with open(im_dir, "r") as f:
            ims = f.readlines()[0]
        ims = ims.split(", ")
        # check existence of image-directories:
        remove = []
        for im in ims:
            if not path.exists(im):
                print(
                    f"{im} was not found. Removing this image from the list of images to load."
                )
                remove.append(im)
        ims = list(set(ims).difference(set(remove)))

        self.im_dirs = ims

        if resize_shape is None:
            self.resize_shape = None
        else:
            assert (
                len(resize_shape) == 3 or len(resize_shape) == 2
            ), f"resize_shape needs 2 or 3 elements, but {len(resize_shape)} were given" \
               f" ({resize_shape})."
            assert np.all(
                [isinstance(r, int) for r in resize_shape]
            ), f"All elements of resize_shape need to be integers, {resize_shape} was given."
            self.resize_shape = resize_shape
        self.dataset = None

    def create_dataset(self, output_file=None):
        """

        :param output_file: Name of the output-file (ends with .zip).
        :return:
        """
        # todo: Note that when images are resized, the scale is changed to 0...1. Watch out that
        #  0...1 is not mixed
        # up with uint8! ==> Only for 3d images (because as_gray converts images to 0...1)
        # create a numpy array from the images.
        for idx, im in enumerate(self.im_dirs):
            image = skio.imread(im, as_gray=len(self.resize_shape) == 2)
            if image.size != self.resize_shape:
                image = resize(image, self.resize_shape)
            if idx < 1:
                # AutoKeras expects a nested tensorflow object (see tensorflow.nest for explanation)
                # X = [np.ones(SIZE)*np.nan for _ in range(len(ims_train))]
                X = np.ones([len(self.im_dirs), *self.resize_shape]) * np.nan
                X = X.astype("float32")
            if len(self.resize_shape) == 2:
                X[idx, :] = image.astype("float32")
            else:
                X[idx, :, :] = image.astype("float32")
            if idx % 100 == 0:
                print(f"Training image {idx + 1} of {len(self.im_dirs)} loaded.")
        self.dataset = X

        if output_file is not None:
            zip_ref = ZipFile(output_file, "w")
            zip_ref.writestr(zinfo_or_arcname=output_file, data=pickle.dumps(self))
            zip_ref.close()

    @staticmethod
    def load_dataset_into_directory(filename, directory="/tmp"):
        """

        :param filename: Name of the zip-file to load content from.
        :param directory: Name of the directory to unpack the zip-file to.
        :return:
        """
        assert path.exists(filename), f"File {filename} was not found!"
        # see: https://towardsdatascience.com/an-informative-colab-guide-to-load-image-datasets
        # -from-github-kaggle-and-local-machine-75cae89ffa1e
        zip_ref = ZipFile(filename, "r")
        zip_ref.extractall(path=f"{directory}")
        zip_ref.close()

        # load pickle file:
        ds = pickle.load(open(f"{directory}/{filename}", "rb"))
        return ds
