"""
Constants, helper functions, metrics and plotting functions shared by all scripts.
"""

import numpy as np
from pywt import families
import pandas as pd
from skimage.transform import rescale
from skimage.color import rgb2gray
from os import listdir, path
from typing import Sequence
import random
from datetime import datetime, timedelta
from random import sample, choices
from collections import Counter
import matplotlib.pyplot as plt


def log10mean(vals):
    return np.nanmean(np.log10(vals))


def log10var(vals):
    return np.nanvar(np.log10(vals))


def calc_modus(vals, edges):
    frequencies, _ = np.histogram(vals.ravel(), bins=edges)
    modus_idx = np.argmax(frequencies)
    edges_modus = edges[modus_idx : modus_idx + 2]
    center = 10 ** (np.log10(edges_modus[0]) + np.diff(np.log10(edges_modus)) / 2)
    return center[0]


MEASURE = {
    "mean": np.nanmean,
    "var": np.nanvar,
    "median": np.nanmedian,
    "mean_log10": log10mean,
    "var_log10": log10var,
    "modus": calc_modus,
}


# also see: https://www.python-kurs.eu/python3_properties.php on getter/setter


class FeatureTypes:
    def __init__(self):
        self.depth_feature = "depth_feature"
        self.cooccurrence_feature = "cooccurrence_feature"
        self.histogram_feature = "histogram_feature"
        self.cooccurrence_integrated_feature = "cooccurrence_integrated_feature"
        self.cooccurrence_gradient_feature = "cooccurrence_gradient_feature"
        self.edge_feature = "edge_feature"
        self.tamura_feature = "tamura_feature"
        self.autocorrelation_feature = "autcorrelation_feature"
        self.fractal_psd_feature = "fractal_psd_feature"
        self.fractal_gvs_feature = "fractal_gvs_feature"
        self.fractal_ibc_feature = "fractal_ibc_feature"


class Identifier:
    def __init__(
        self,
        method: str,
        window_shape: (tuple, int) = None,
        factor: float = None,
        measure: str = None,
        pixel_distance: (int, tuple) = None,
        max_exponent: int = None,
    ):
        """
        When initializing the Identifier, the type and variables for the feature are inferred from
        the arguments. For further information which arguments have to be provided for which
        feature and what they mean, look at src/features.py.
        """
        feature_types = FeatureTypes()
        splits = method.split("_")
        if "wavelet" in splits[0]:
            assert len(splits) == 3, f"{method} is too short or too long"
            self.method = WaveletMethod(wavelet=splits[1], measure=splits[2])
            feature_type = feature_types.depth_feature

        elif "tenengrad" in splits[0]:
            assert len(splits) == 2, f"{method} is too short or too long"
            self.method = TenengradMethod(measure=splits[1])
            feature_type = feature_types.depth_feature

        elif "laplace" in splits[0]:
            assert (
                len(splits) == 2 or len(splits) == 3
            ), f"{method} is too short or too long"
            self.method = LaplaceMethod(
                measure=splits[-1], ksize=int(splits[1]) if len(splits) == 3 else None
            )
            feature_type = feature_types.depth_feature

        elif "kurtosis" in splits[0]:
            self.method = KurtosisMethod(method=splits[0])
            feature_type = feature_types.depth_feature

        elif splits[0] == "autocorrelation":
            self.method = AutoCorrelationMethod(max_distance=int(splits[-1]))
            feature_type = feature_types.autocorrelation_feature

        elif splits[-1] == "glgcm":
            self.method = CoOccurrenceGradientMethod(method=method)
            feature_type = feature_types.cooccurrence_gradient_feature

        elif splits[0] == "edge":
            self.method = EdgeMethod(max_distance=splits[-1])
            feature_type = feature_types.edge_feature

        elif splits[0] == "fractalPSD":
            self.method = FractalPSDMethod(method)
            feature_type = feature_types.fractal_psd_feature

        elif splits[0] == "fractalGVS":
            self.method = FractalGVSMethod(method)
            feature_type = feature_types.fractal_gvs_feature

        elif splits[0] == "fractalIBC":
            self.method = FractalIBCMethod(method)
            feature_type = feature_types.fractal_ibc_feature

        elif pixel_distance is not None:
            if isinstance(pixel_distance, int):
                self.method = CoOccurrenceIntegratedMethod(
                    method=method, distance=pixel_distance
                )
                feature_type = feature_types.cooccurrence_integrated_feature
            else:
                self.method = CoOccurenceMethod(
                    method=method, pixel_distance=pixel_distance
                )
                feature_type = feature_types.cooccurrence_feature

        elif max_exponent is not None and splits[0] == "tamura":
            assert isinstance(max_exponent, int), "max_exponent needs to be an integer!"
            self.method = TamuraTextureMethod(max_exponent=max_exponent)
            feature_type = feature_types.tamura_feature
        else:
            self.method = HistogramMethod(method=method)
            feature_type = feature_types.histogram_feature

        self.feature_type = feature_type

        if feature_type == feature_types.depth_feature:
            if isinstance(window_shape, int):
                window_shape = (window_shape, window_shape)
            elif isinstance(window_shape, tuple):
                assert all(
                    isinstance(w, int) for w in window_shape
                ), "At least one entry of window-shape is not an integer"
            assert all(
                w > 0 for w in window_shape
            ), "At least one entry of window-shape is negative or zero"

            assert isinstance(factor, float), "Factor must be a float"

            if "portion" in measure:
                measure_split = measure.split("_")
                used_metric = [
                    MEAS for MEAS in MEASURE.keys() if MEAS in measure_split[0]
                ]
                assert len(used_metric) > 0, f"No known metric found in {measure}"
            else:
                assert measure in MEASURE.keys(), f"No known metric found in {measure}"

            self.factor = factor
            self.window_shape = window_shape
            self.measure = measure

    @property
    def factor(self):
        if self.feature_type == FeatureTypes().depth_feature:
            return self.__factor
        else:
            return None

    @property
    def window_shape(self):
        if self.feature_type == FeatureTypes().depth_feature:
            return self.__window_shape
        else:
            return None

    @property
    def measure(self):
        if self.feature_type == FeatureTypes().depth_feature:
            return self.__measure
        else:
            return None

    def construct_string(self):
        """
        A method which builds a string with all the neccessary information to identify the feature.
        """
        if self.feature_type == FeatureTypes().depth_feature:
            win = (
                [str(self.window_shape[0])]
                if self.window_shape[0] == self.window_shape[1]
                else [str(w) for w in self.window_shape]
            )
            return f"{self.method.get_str()}_{self.measure}_win{'_'.join(win)}_{self.factor}"
        elif self.feature_type == FeatureTypes().cooccurrence_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().histogram_feature:
            return f"{self.method.get_str()}_"
        elif self.feature_type == FeatureTypes().cooccurrence_integrated_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().cooccurrence_gradient_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().tamura_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().edge_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().autocorrelation_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().fractal_psd_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().fractal_gvs_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().fractal_ibc_feature:
            return self.method.get_str()
        elif self.feature_type == FeatureTypes().edge_feature:
            return self.method.get_str()
        else:
            return None

    @staticmethod
    def construct_id(string: str):
        """
        Infer an Identifer instance from a string.
        """

        if string.endswith("_"):
            # Histogram feature
            return Identifier(method=string.split("_")[0])
        elif string.endswith("glgcm"):
            return Identifier(method=string)
        elif string.startswith("autocorrelation"):
            return Identifier(method=string)
        elif string.startswith("tamura"):
            return Identifier(method="tamura", max_exponent=int(string.split("_")[-1]))
        elif string.startswith("edge"):
            return Identifier(method=string)
        elif string.startswith("fractalPSD"):
            return Identifier(method=string)
        elif string.startswith("fractalGVS"):
            return Identifier(method=string)
        elif string.startswith("fractalIBC"):
            return Identifier(method=string)
        elif all(map(str.isdigit, string.split("_")[-2:])):
            # if the last two entries divided by an underscore are digits, assume a
            # co-occurence-matrix feature
            return Identifier(
                method=string.split("_")[0],
                pixel_distance=tuple(map(int, string.split("_")[-2:])),
            )
        elif str.isdigit(string.split("_")[-1]):
            # if the last entry is a digit but the feature is not a co-occurrence feature, the
            # feature must be a co-occurrence-integrated feature - isdigit("1.0") is False so this
            # should not be confused with depth-feature
            return Identifier(
                method=string.split("_")[0], pixel_distance=int(string.split("_")[-1])
            )
        elif "win" in string:
            # assume a feature derived from a depth-map
            split_win = string.split("_win")
            split2 = split_win[1].split("_")
            # look for acceptable metric in first split:
            measure = [MEAS for MEAS in MEASURE.keys() if MEAS in split_win[0]]
            if len(measure) == 2:
                if "var" in measure:
                    if string.startswith(
                        ("laplace_3_var", "wavelet_db6_var", "tenengrad_var")
                    ):  # what to do about other ksizes?
                        measure.pop(measure.index("var"))
            # check wether measure contains two entries and remove entry "var" in case
            assert len(measure) > 0, f"No known measure in {string}"
            measure = measure[0]
            if "portion" in split_win[0]:
                split1 = split_win[0].split("_")
                # determine position of determined metric:
                where = np.nonzero(
                    [True if measure in sp1 else False for sp1 in split1]
                )[0]
                measure = "_".join(split1[where[0] :])

            method = split_win[0].split(f"_{measure}")[0]

            if "var_var" in string:
                method = f"{method}_var"

            window_shape = tuple(map(int, split2[:-1]))
            if len(window_shape) == 1:
                window_shape = tuple([window_shape[0]] * 2)
            return Identifier(
                method=method,
                measure=measure,
                window_shape=window_shape,
                factor=float(split2[-1]),
            )

    @window_shape.setter
    def window_shape(self, value):
        self.__window_shape = value

    @factor.setter
    def factor(self, value):
        self.__factor = value

    @measure.setter
    def measure(self, value):
        self.__measure = value


class WaveletMethod:
    def __init__(self, wavelet: str, measure: str):
        # remove integer number from end of string:
        is_digit = list(map(str.isdigit, wavelet))
        assert is_digit[-1], "Mother-wavelet needs to end with an integer!"
        # assert that only the end is occupied by digits:
        assert (
            np.diff(is_digit).sum() == 1
        ), "Provide only one integer in the mother-wavelet string!"
        # assert that the given mother-wavelet exists:
        mother_wavelet = "".join([letter for letter in wavelet if not letter.isdigit()])
        assert (
            mother_wavelet in families()
        ), f"The mother-wavelet {wavelet} is not implemented in pywt"

        assert measure in [
            "var",
            "sum",
            "ratio",
        ], f" Choose another measure than {measure}"
        self.measure = measure
        self.mother_wavelet = mother_wavelet
        self.wavelet = wavelet

    def get_str(self):
        return f"wavelet_{self.wavelet}_{self.measure}"


class TenengradMethod:
    def __init__(self, measure: str):
        assert measure in ["measure", "var"], f" Choose another measure than {measure}"
        self.measure = measure

    def get_str(self):
        return f"tenengrad_{self.measure}"


class LaplaceMethod:
    def __init__(self, measure: str, ksize: (int, None) = None):
        if ksize is not None:
            assert (
                measure == "var"
            ), f"Flexible filter size for Laplace only for measure var"
            assert isinstance(ksize, int), f"ksize needs to be an integer"
            self.ksize = ksize
        else:
            self.ksize = None

        assert measure in [
            "var",
            "diagonal",
            "modified",
        ], f"Choose another measure than {measure}"
        self.measure = measure

    def get_str(self):
        return f"laplace{f'_{self.ksize}' if self.ksize is not None else ''}_{self.measure}"


class KurtosisMethod:
    def __init__(self, method):
        self.method = method

    def get_str(self):
        return "kurtosis"


class CoOccurenceMethod:
    def __init__(self, method: str, pixel_distance: tuple):
        assert (
            pixel_distance[0] >= 0
            and pixel_distance[0] <= 360
            and isinstance(pixel_distance[0], int)
        ), f"Choose an integer between (and including) 0 and 360 as angle"
        assert pixel_distance[1] > 0 and isinstance(
            pixel_distance[1], int
        ), "Choose a positive integer as distance"
        assert method in [
            "homogeneity",
            "energy",
            "entropy",
            "contrast",
            "dissimilarity",
            "correlation",
        ], f"Method {method} unknown"

        self.angle = pixel_distance[0]
        self.distance = pixel_distance[1]
        self.method = method

    def get_str(self):
        return f"{self.method}_{self.angle}_{self.distance}"


class HistogramMethod:
    def __init__(self, method: str):
        assert method in [
            "mean",
            "var",
            "energy",
            "entropy",
        ], f"Method {method} is not a histogram method!"
        self.method = method

    def get_str(self):
        return f"{self.method}"


class CoOccurrenceIntegratedMethod:
    def __init__(self, method: str, distance: int):
        assert method in [
            "ASM",
            "contrast",
            "entropy",
            "homogeneity",
        ], f"Method {method} is no known CoOccurrenceIntegratedMethod!"
        assert distance > 0 and isinstance(
            distance, int
        ), f"Choose a positive integer as distance, not {distance}"

        self.distance = distance
        self.method = method

    def get_str(self):
        return f"{self.method}_{self.distance}"


class CoOccurrenceGradientMethod:
    def __init__(self, method: str):
        assert method.split("_")[0] in [
            "energy"
        ], f"Method {method} is no known CoOccurrenceGradientMethod!"

        self.method = method

    def get_str(self):
        return self.method


class EdgeMethod:
    def __init__(self, max_distance):
        self.max_distance = max_distance

    def get_str(self):
        return f"edge_{self.max_distance}"


class TamuraTextureMethod:
    def __init__(self, max_exponent):
        self.max_exponent = max_exponent

    def get_str(self):
        return f"tamura_{self.max_exponent}"


class AutoCorrelationMethod:
    def __init__(self, max_distance):
        self.max_distance = max_distance

    def get_str(self):
        return f"autocorrelation_{self.max_distance}"


class FBMAutoCorrelationMethod:
    ...


class FractalPSDMethod:
    def __init__(self, method):
        splits = method.split("_")
        numbers = [int(r) for r in splits[-2:]]
        self.r1 = max(numbers)
        self.r2 = min(numbers)

    def get_str(self):
        return f"fractalPSD_{self.r1}_{self.r2}"


class FractalGVSMethod:
    def __init__(self, method):
        splits = method.split("_")
        self.N = int(splits[-1])

    def get_str(self):
        return f"fractalGVS_{self.N}"


class FractalIBCMethod:
    def __init__(self, method):
        splits = method.split("_")
        numbers = [int(r) for r in splits[-2:]]
        self.r1 = min(numbers)
        self.r2 = max(numbers)

    def get_str(self):
        return f"fractalIBC_{self.r1}_{self.r2}"


RANDOM_STATE = 456
DATE_FORMAT = "%Y%m%d_%H%M%S"  # for saving
HISTOGRAM_BINS = 20  # for the histogram stats (mean_, var_, energy_, entropy_)
NLEVELS_CO_OCCURENCE = 32  # for the binning of the images for the co-occurence-matrix resolution
FILE_SEP = ","
NUM = 100

TIMESTAMP_POSITION = ((4, 44), (0, 430))

CLASSES = ["clear", "partly_blurred", "blurred", "dark"]
CLASSES_NUMBERS = {class_: int_ for int_, class_ in enumerate(CLASSES)}

LOG_EDGES = {
    "laplace_3_var": np.logspace(start=-4, stop=1, num=NUM),
    "wavelet_db6_ratio": np.logspace(start=-16, stop=0, num=NUM),
    "wavelet_db6_var": np.logspace(start=-16, stop=0, num=NUM),
    "tenengrad_measure": np.logspace(start=-16, stop=0, num=NUM),
    "tenengrad_var": np.logspace(start=-16, stop=0, num=NUM),
    "laplace_diagonal": np.logspace(start=-4, stop=1, num=NUM),
    "laplace_modified": np.logspace(start=-4, stop=1, num=NUM),
    "wavelet_db6_sum": np.logspace(start=-16, stop=0, num=NUM),
}


def create_co_occurence_feature_names(feats, angles, distances):
    feats = [
        [[f"{feat}_{angle}_{distance}" for feat in feats] for distance in distances]
        for angle in angles
    ]
    # flatten list:
    feats = list(np.array(feats).flatten())
    return feats


CHOSEN_FEATURES_FINAL_SUBSET = [
    "homogeneity_0_5",
    "homogeneity_90_3",
    "energy_0_3",
    "tenengrad_var_0.07mean_portion_win5_0.25",
    "wavelet_db6_ratio_mean_win5_0.5",
    "tenengrad_measure_var_win9_0.25",
    "tenengrad_var_0.001mean_portion_win9_0.25",
    "laplace_3_var_0.1mean_portion_win17_0.25",
    "laplace_3_var_modus_win9_0.25",
    "laplace_3_var_mean_win21_0.25",
    "correlation_0_1",
    "laplace_modified_var_win21_0.25",
    "wavelet_db6_var_mean_win13_0.5",
    "laplace_diagonal_var_win5_0.25",
    "wavelet_db6_sum_mean_win13_0.5",
    "dissimilarity_90_3",
    "wavelet_db6_sum_median_win13_0.5",
    "tenengrad_var_median_win5_0.25",
    "tenengrad_var_modus_win9_0.25",
    "tenengrad_measure_modus_win13_0.25",
]


def get_portion_features(factor=1.0):
    features = [
        "laplace_modified",
        "laplace_diagonal",
        "laplace_3_var",
        "wavelet_db6_var",
        "wavelet_db6_ratio",
        "wavelet_db6_sum",
        "tenengrad_measure",
        "tenengrad_var",
    ]

    feats = []
    measures = ["mean"]
    portions = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    for feat in features:
        for window_shape in [(5, 5), (9, 9), (13, 13)]:
            for meas in measures:
                for portion in portions:
                    feats.append(
                        Identifier(
                            method=feat,
                            measure=f"{portion}{meas}_portion",
                            window_shape=window_shape,
                            factor=factor,
                        )
                    )

    return feats


INTEGRATED_FREQUENCIES = pd.DataFrame(
    index=["ASM", "contrast", "entropy", "homogeneity"],
    columns=[0, 45, 90, 135],
    data=[
        [0.21, 0.26, 0.26, 0.27],
        [0.14, 0.28, 0.25, 0.33],
        [0.15, 0.29, 0.27, 0.29],
        [0.22, 0.26, 0.26, 0.26],
    ],
)


DIRECTORY_IMAGES = "C:/Users/Me/Documents/Python Scripts/SlicksRecognition/images"

EXCLUDE_LABELS = ["blurred", "dark"]

# also see:
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes


# on grey images
def rescale_image(image, factor):
    if image.ndim == 3:
        # convert to grey picture
        image = rgb2gray(image)
    return rescale(image, factor, anti_aliasing=False)


def shift_half_step(array):
    return array[0:-1] + np.diff(array) / 2


def bin_image(
    image: np.ndarray,
    nlevels: int = None,
    range: tuple = (0, 255),
    edges=None,
    assign_rank=False,
):
    """
    Bin the grayscales in the given image into nlevels equidistant classes and assign the center of
     each class back to the image.
    :param image: ndarray. Grayscale image (2 dimensions).
    :param nlevels: int. Number of classes of grayscales.
    :param range: tuple. Range of the histogram.
    :return:
    """
    image = image.copy()
    assert image.ndim < 3, "No RGB-image; max 2 dimensions!"
    assert edges is not None or nlevels is not None, "Either provide edges or nlevels!"
    if edges is None:
        freq, edges = np.histogram(image, nlevels, range=range)
    else:
        freq, edges = np.histogram(image, edges)
    centers = shift_half_step(edges).astype(image.dtype)
    for edge_tuple_no in np.arange(len(edges) - 1):
        lower = edges[edge_tuple_no]
        upper = edges[edge_tuple_no + 1]
        if assign_rank:
            image[(image > lower) & (image <= upper)] = edge_tuple_no
        else:  # assign center
            image[(image > lower) & (image <= upper)] = centers[edge_tuple_no]

    return image


def divide_dataset(directory, classes, test_prop=0.2, validation_prop=0.2):
    """
    Function that makes a divison of the images into training, validation and test-dataset and
    stores this information. The function expects a directory tree
    :param directory:
    :param classes:
    :param test_prop: Portion for the test dataset.
    :param validation_prop: Portion for the validation dataset.
    :return:
    """

    dirs = []
    if path.isdir(directory):
        for label in classes:
            sub_dir = f"{directory}/{label}"
            if path.isdir(sub_dir):
                dirs.extend([f"{sub_dir}/{dir_}" for dir_ in listdir(sub_dir)])
            else:
                print(f"{sub_dir} is not a valid directory!")
                return
    else:
        print(f"{directory} is not a valid directory!")
        return

    random.seed(a=RANDOM_STATE)
    random.shuffle(dirs)
    length = len(dirs)
    len_test = int(length * test_prop)
    len_val = int(length * validation_prop)

    test_set = dirs[0:len_test]
    val_set = dirs[len_test : (len_test + len_val)]
    train_set = dirs[(len_test + len_val) : length]

    # save the directories in a file:
    with open(f"{directory}/dataset_division_{length}.csv", "x") as file:
        file.write(
            "\ntest_set: \n"
            + ",".join(test_set)
            + "\nvalidation_set: \n"
            + ",".join(val_set)
            + "\ntrain_set: \n"
            + ",".join(train_set)
        )


def get_dataset_divisions(path_to_division: str, merge_train_and_val=False):
    """
    Returns the directories for the training, (validation) and test datasets.
    :param path_to_division:
    :param merge_train_and_val:
    :return:
    """

    with open(path_to_division, "r") as file:
        all_dirs = file.readlines()
    train_dirs = all_dirs[all_dirs.index("train_set: \n") + 1].split(",")
    test_dirs = all_dirs[all_dirs.index("test_set: \n") + 1].split(",")
    val_dirs = all_dirs[all_dirs.index("validation_set: \n") + 1].split(",")

    train_dirs[-1] = train_dirs[-1].split("\n")[0]
    test_dirs[-1] = test_dirs[-1].split("\n")[0]
    val_dirs[-1] = val_dirs[-1].split("\n")[0]

    if not merge_train_and_val:
        return {"train_dirs": train_dirs, "test_dirs": test_dirs, "val_dirs": val_dirs}
    train_dirs.extend(val_dirs)
    return {"train_dirs": train_dirs, "test_dirs": test_dirs}


def filter_nan_rows(list_of_arrays):
    """
    Function which takes a list of arrays which all must have the same number of rows and returns
     a new list of arrays where all entries only contains those rows with no nans (over all arrays).

    :param list_of_arrays: List of numpy arrays.
    :return:
    """
    # check wether all list entries have the same number of rows
    length = np.unique([np.shape(array)[0] for array in list_of_arrays])
    assert len(length) == 1, "Not all arrays have the same number of rows."
    notnan_rows = np.array([True] * length[0])
    for array in list_of_arrays:
        if array.ndim > 1:
            notnan_rows[np.any(np.isnan(array), 1)] = False
        else:
            notnan_rows[np.isnan(array)] = False

    return [array[notnan_rows] for array in list_of_arrays]


def list_all_files(directory: str, separator="/"):
    """
    Returns all file-directories in all subdirectories of this directory.
    :param directory:
    :return:
    """
    if path.exists(directory):
        if path.isdir(directory):
            # recursion
            leaves = list_all_dirs(directory, separator)
            return leaves


def list_all_dirs(subdirectory, separator):
    """
    Recursive function returning total directories of all files in this directory.
    :param subdirectory:
    :return:
    """
    leaves = []
    subsub = listdir(subdirectory)
    subsub = [f"{subdirectory}{separator}{subsub_}" for subsub_ in subsub]
    if len(subsub) > 0:
        # filter out all directories from subsub
        subsub_dirs = [subsub_ for subsub_ in subsub if path.isdir(subsub_)]
        # everything besides a directory must be a file
        subsub_files = list(set(subsub).difference(set(subsub_dirs)))
        if len(subsub_dirs) > 0:
            # as long as there are still subdirectories, recall the function
            for subsub_dir in subsub_dirs:
                leaves.extend(list_all_dirs(subsub_dir, separator))
        # add files in this subsub-directory to all files from deeper directories
        return subsub_files + leaves
    # if no directory is left, return the list of file-directories
    return leaves


def standardize_features(features: pd.DataFrame, data_normalization: pd.DataFrame = None):
    """
    Returns standardized features based on the data_normalization object or the dataset itself.
    :param features: pd.DataFrame. Each row in this dataframe is one sample and each column one
    feature. The means and standard deviations are calculated along the columns (one for each
     column).
    :param data_normalization: pd.DataFrame. If provided, the features-dataframe is standardized
    with the given means and standard deviations. Else the dataframe is standardized to have the
    mean 0 and the standard deviation 1 and the data_normalization dataframe is returned.
    :return: standardized features and data_normalization (if not provided).
    """

    if data_normalization is None:
        data_normalization = pd.DataFrame(index=["mean", "std"], columns=features.columns)
        data_normalization.loc["mean"] = features.mean()
        data_normalization.loc["std"] = features.std()
        return_both = True
    else:
        # make sure that the Dataframe is compatible:
        try:
            data_normalization = data_normalization.loc[:, features.columns.values]
        except KeyError:
            print("Not all features are included in data_normalization object")
            return None

        return_both = False

    # only when std is not zero
    not_zero = np.nonzero(data_normalization.loc["std"].values)[0]
    features.iloc[:, not_zero] = (
        features.iloc[:, not_zero]
        - data_normalization.iloc[
            np.where(data_normalization.index.values == "mean")[0][0], not_zero
        ]
    ) / data_normalization.iloc[np.where(data_normalization.index.values == "std")[0][0], not_zero]

    if return_both:
        return features, data_normalization
    return features


class ConfusionMatrix:
    def __init__(self, truth: np.ndarray, prediction: np.ndarray, classes: Sequence = None):
        self.confusion_matrix = self.calc_confusion_matrix(truth, prediction, classes)
        self.epsilon = 1e-8  # small number to add to the denominator

    def calc_confusion_matrix(
        self, truth: np.ndarray, prediction: np.ndarray, classes: Sequence = None
    ):
        """
        # confusion-matrix of the form:

                        prediction
                    FALSE            TRUE
        truth
          FALSE    true negative     false positive

          TRUE     false negative    true positive


        in multinomial case:

                                    prediction
                    class0            class1            class2          ...
        truth
          class0

          class1

          class2

          ...

        :param truth:
        :param prediction:
        :return: normalized confusion matrix.
        """
        if classes is None:
            classes = np.unique(np.concatenate([truth, prediction]))

        confusion_matrix = np.ones(shape=(len(classes), len(classes))) * np.nan

        for truth_ind, truth_class in enumerate(classes):
            for pred_ind, pred_class in enumerate(classes):
                confusion_matrix[truth_ind, pred_ind] = (
                    np.all([prediction == pred_class, truth == truth_class], axis=0)
                ).sum()

        self.info = [[(truth, pred) for pred in classes] for truth in classes]

        return confusion_matrix / len(truth)

    def get_info(self):
        return self.info

    def get_matrix(self):
        return self.confusion_matrix

    def calculate_accuracy(self, balanced=False):

        if balanced:
            acc = [
                self.confusion_matrix[row, row] / self.confusion_matrix[row, :].sum()
                if self.confusion_matrix[row, :].sum() > 0
                else 0
                for row in range(self.confusion_matrix.shape[0])
            ]
            acc = np.mean(acc)
        else:
            acc = self.confusion_matrix.trace()

        return acc

    def calculate_roc(self):

        assert self.confusion_matrix.size == 4, "ROC only works for binomial case"
        # true positive/ positive (sensitivity) (y-axis)
        true_positive_rate = self.calculate_sensitivity()

        # false positive/ negative (x-axis)
        false_positive_rate = 1 - self.calculate_specificity()

        return false_positive_rate, true_positive_rate

    def calculate_specificity(self):
        # portion of correctly identified negatives in relation to all negatives
        assert self.confusion_matrix.size == 4, "No implementation for multinomial case!"
        return self.confusion_matrix[0, 0] / (self.confusion_matrix[0, :].sum() + self.epsilon)

    def calculate_sensitivity(self):
        # portion of correctly identified positives in relation to all positives
        assert self.confusion_matrix.size == 4, "No implementation for multinomial case!"
        return self.confusion_matrix[1, 1] / (self.confusion_matrix[1, :].sum() + self.epsilon)

    def calculate_precision(self):
        # portion of correctly identified positives in relation to all positive predictions
        assert self.confusion_matrix.size == 4, "No implementation for multinomial case!"
        return self.confusion_matrix[1, 1] / (self.confusion_matrix[:, 1].sum() + self.epsilon)

    def calculate_f1_score(self):
        return 2 / (
            1 / (self.calculate_precision() + self.epsilon)
            + 1 / (self.calculate_sensitivity() + self.epsilon)
        )


def check_features_and_labels(features: np.ndarray, labels: np.ndarray, true_class: (str, dict)):
    """
    Function that converts labels to boolean values and filters out labels and features with samples
    with nan-entries.

    :param features: DataFrame with rows as samples and columns as features.
    :param labels: Numpy array with the corresponding labels for each sample in features.
    :param true_class: string or dictionary determining which labels should be converted to True.
    A dictionary should be used if for some reason labels is mixed, i. e. some labels are 1 and
    others are "True".
    :return: Filtered and converted features and labels.
    """
    # change dtype of features
    if not isinstance(features, np.float_):
        features = features.astype(np.float_)

    # convert labels to boolean array or int-array in case of dictionary:
    if isinstance(true_class, str):
        labels = (labels == true_class).astype(np.float_)
    else:
        labels = np.array(
            [true_class[label] if not (str(label) == "nan") else np.nan for label in labels]
        )

    # filter nan-rows:
    features, labels = filter_nan_rows([features, labels])

    return features, labels


def join_two_dataframes(main_df: pd.DataFrame, second_df: pd.DataFrame, features: Sequence = None):

    """
    Joins the information in two dataframes based on nan-entries. When there is information in
    both frames, main_df is preferred.

    :param main_df:
    :param second_df:
    :param features: List of features (columns) that should be included in the joint dataframe.
    :return:
    """

    inds = main_df.index.values
    if features is None:
        features = main_df.columns.values

    new_dataframe = pd.DataFrame(index=main_df.index, columns=features)

    for feature in features:
        col = main_df.loc[inds, feature]
        new_dataframe.loc[inds, feature] = col
        missing = col.isna().values
        new_dataframe.loc[inds[missing], feature] = second_df.loc[inds[missing], feature]

    return new_dataframe


def create_dummy_image(shape):
    """
    Create some image for testing.
    :param shape: 2d shape of resulting image.
    :return:
    """
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    factor = 20 / min(shape)
    image = np.sin(factor * x) + np.cos(factor * y)
    image += np.random.rand(*image.shape) * (x - 2 * y) * factor
    return image


def datenum_to_datetime(datenum):
    # code taken from: https://gist.github.com/victorkristof/b9d794fe1ed12e708b9d
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) + timedelta(days=days) - timedelta(days=366)


def parse_times_from_names(file_names):
    """
    Find out when images were saved according to their filename. This works for images which have
    their filename according to the signature in DATE_FORMAT.
    :param file_names: Array of file_names. Can be directories.
    :return:
    """
    dates = [
        datetime.strptime(
            date.split("/")[-1].replace(".jpg", "").replace("_NG17", "").replace("_AlteWeser", ""),
            DATE_FORMAT,
        )
        for date in file_names
    ]
    return dates


def match_times_and_names(file_names, dates):
    """

    :param file_names:
    :param dates:
    :return:
    """
    files = (
        np.ones(
            len(dates),
        )
        * np.nan
    )
    for idx, date in enumerate(dates):
        part = datetime.strftime(date, DATE_FORMAT)
        # look for matching file name:
        file_name = [name for name in file_names if part in name]
        if len(file_name) == 1:
            files[idx] = file_name[0]

    return files


def extract_labels_from_directories(directories, saveto):
    """
    When labeled images are stored in a folder which is called like the according label, this
    function extracts the labels for a list of directories and stores them as a csv into saveto.

    :param directories:
    :param saveto:
    :return:
    """
    all_files = []

    if isinstance(directories, str):
        directories = [directories]
    for direc in directories:
        all_files.extend(list_all_files(direc))

    labels = [file.split("/")[-2] for file in all_files]
    names = [file.split("/")[-1] for file in all_files]

    df = pd.DataFrame(index=names, columns=["label"], data=labels)

    with open(saveto, "w") as write_to:
        df.to_csv(write_to)


def random_subsets(X: pd.DataFrame, y: np.ndarray, balancing_method: str, number_of_subsets: int):
    """
    # Custom function to create randomly oversampled or undersampled datasets.

    :param X: pd.DataFrame. Features in the format (n_samples, n_features).
    :param y: np.ndarray. Target in the format (n_samples, ).
    :param balancing_method: str. Choose from: "undersampling", "oversampling".
    :param number_of_subsets: int. Number of under-/oversampled subsets to generate.
    :return: Generator of length number_of_subsets. Returns a tuple containing the under-/
    oversampled X and y.
    """

    counter = Counter(y)
    minority_class = None
    minimum = np.inf
    for class_, occurence in counter.items():
        if occurence <= minimum:
            minimum = occurence
            minority_class = class_
    if balancing_method == "undersampling":
        subset_size = counter[minority_class]
        dataset_size = len(y) - subset_size
        keep_inds = np.nonzero(y == minority_class)[0]
        take_subset_inds = np.nonzero(y != minority_class)[0]
    elif balancing_method == "oversampling":
        keep_inds = np.nonzero(y != minority_class)[0]
        take_subset_inds = np.nonzero(y == minority_class)[0]
        dataset_size = counter[minority_class]
        subset_size = len(y) - dataset_size
    else:
        return

    if subset_size > dataset_size:
        select = choices  # sample with replacement
    else:
        select = sample  # sample without replacement

    population = list(np.arange(dataset_size))
    for _ in range(number_of_subsets):
        selected_inds = select(population=population, k=subset_size)
        inds = np.concatenate([keep_inds, take_subset_inds[selected_inds]])
        inds = np.sort(inds)
        yield X.iloc[inds, :], y[inds]


def get_best_from_each_feature_class(
    features: pd.DataFrame, column_to_use: str, maximize=True, top=1
):
    """
    From a given dataframe mapping features to f-value or mutual information select the best
    features. Also, when one feature is represented more often with different parameters (for
    example homogeneity_90_1 and homogeneity_90_4), only one of these features is used because it
    is assumed that these features are already very similiar.
    :param features: pd.DataFrame. Index is the features and the columns should be filled with
    the f-value or the mutual information for the respective feature.
    :param column_to_use: Name of the column to use.
    :param maximize: Wether the best features should be determined by sorting the values in the
    column to use in an ascending or descending order.
    :param top: How many features to choose.
    :return: List of the best features.
    """
    classes = {}
    for feat in features.index.values:
        splits = feat.split("_")
        if feat == "energy_glgcm":
            feat_class = "_".join(splits)
        elif feat.startswith("fractalIBC"):
            feat_class = splits[0]
        else:
            feat_class = "_".join(splits[:-1])

        # if feat_class already exists in the dictionary, add param to the list associated with
        # feat_class else create feat_class and a list with entry param
        if feat_class in classes.keys():
            classes[feat_class].append(feat)
        else:
            classes[feat_class] = [feat]

    best_features = []
    for subclass, subfeatures in classes.items():
        subdf = features.loc[subfeatures, column_to_use]
        sorted_df = abs(subdf).sort_values(ascending=not maximize)
        best_features.extend(sorted_df.index.values[:top])

    return best_features


# see also: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# #sphx-glr-auto-examples-classification-plot-classifier-comparison-py


def evaluate_results(
    confusion_matrix_train=None, confusion_matrix_test=None, name=None, df=None
):
    """
    Write metrics calculated from confusion matrices into a file.
    Confusion matrices in this case are instances of src.utils.ConfusionMatrix.
    :param confusion_matrix_train:
    :param confusion_matrix_test:
    :param name: This is what the row in the evaluation table is going to be named.
    :param df: When a dataframe already exists, it can be provided with this argument. Like this,
    evaluations for several models can be included in one table.
    :return: A dataframe with the evaluation is returned.
    """
    if df is None:
        df = pd.DataFrame()

    elif (
        confusion_matrix_train is not None
        or confusion_matrix_test is not None
        and name is not None
    ):

        if confusion_matrix_test is not None:
            if isinstance(confusion_matrix_test, (list, tuple, np.ndarray)):
                mat_test = [conf.get_matrix() for conf in confusion_matrix_test]
                mat_test = np.mean(np.array(mat_test), axis=0)
                confusion_matrix_test = confusion_matrix_test[0]
                confusion_matrix_test.confusion_matrix = mat_test
            else:
                mat_test = confusion_matrix_test.get_matrix()

        if confusion_matrix_train is not None:
            if isinstance(confusion_matrix_train, (list, tuple, np.ndarray)):
                mat_train = [conf.get_matrix() for conf in confusion_matrix_train]
                mat_train = np.mean(np.array(mat_train), axis=0)
                confusion_matrix_train = confusion_matrix_train[0]
                confusion_matrix_train.confusion_matrix = mat_train

        if confusion_matrix_test is not None:
            recall = confusion_matrix_test.calculate_sensitivity()
            precision = mat_test[1, 1] / mat_test[:, 1].sum()

        if confusion_matrix_train is not None:
            df.loc[name, "train: accuracy"] = confusion_matrix_train.calculate_accuracy(
                False
            )
            df.loc[
                name, "train: balanced accuracy"
            ] = confusion_matrix_train.calculate_accuracy(True)

        if confusion_matrix_test is not None:
            df.loc[name, "test: accuracy"] = confusion_matrix_test.calculate_accuracy(
                False
            )
            df.loc[
                name, "test: balanced accuracy"
            ] = confusion_matrix_test.calculate_accuracy(True)
            df.loc[name, "test: TP/(TP + FN) (sensitivity/recall)"] = recall
            df.loc[name, "test: TP/(TP + FP) (precision)"] = precision
            df.loc[name, "test: F-score"] = (2 * recall * precision) / (
                recall + precision
            )
            df.loc[name, "test: TN, FP, FN, TP"] = ", ".join(
                (["%.3f" % el for el in mat_test.ravel()])
            )
            df.loc[name, "test: confusion matrix"] = confusion_matrix_test
        if confusion_matrix_train is not None:
            df.loc[name, "train: confusion matrix"] = confusion_matrix_train

    return df


def plot_keras_history(
    filename,
    plot_quantity,
    ax=None,
    return_val_mean_std=False,
    running_mean=None,
    column_name=None,
):
    """
    Plot the evolution of metrics calculated on the training and the validation dataset during the
    training.
    :param filename: Directory to where the csv-file is stored which contains the evaluation
    metrics. This file should be in the format of the files which keras saves as histories
    (when using the CSVLogger-callback).
    :param plot_quantity: Specify which quantity to plot. For example, when quantity is "accuracy",
    the "accuracy" and the "val_accuracy" is plotted (if it exists).
    :param ax: Provide an axis in which the plot should be drawn. Else a new figure is created.
    :param return_val_mean_std: If True, also return the mean and the standard deviation of the
    specified metric on the validation dataset as a tuple.
    :param running_mean: Specify an integer here or None. If an integer is specified, a running mean
    for the validation metric will be calculated and plotted. The integer, which should be uneven,
    determines how many values should be used to get one datapoint for the running mean. For
    example, if running_mean is 5, a new datapoint is calculated as the average from the four values
    before and the current value.
    :param column_name: Alternative to plot_quantity. When for example only the "val_accuracy"
    should be plotted, this can be specified with this argument.
    :return: figure object when ax was not provided, mean and standard deviation on valiation set if
    the validation metric exists and return_val_mean_std is True
    """

    # Read history:
    dataframe = pd.read_csv(filename, index_col=0)
    dataframe.dropna(inplace=True)

    if ax is None:
        fig = plt.figure(figsize=(17 / 2.5, 12 / 2.5))
        ax = fig.add_subplot(1, 1, 1)

    quantity = plot_quantity
    labels = []
    plot_done = False

    if column_name is not None:
        # plot specific metric
        ax.plot(dataframe.index.values + 1, dataframe.loc[:, column_name], "k-")

        if running_mean is not None:
            assert running_mean % 2 == 1, "running_mean needs to be an uneven integer!"
            running_mean_array = np.convolve(
                dataframe.loc[:, column_name].values,
                [1 / running_mean] * running_mean,
                mode="valid",
            )
            ax.plot(
                (dataframe.index.values + 1)[(running_mean - 1) :],
                running_mean_array,
                "r--",
            )
            labels.append("running mean (%i epochs)" % running_mean)
        if return_val_mean_std:
            val_mean_std = (
                dataframe.loc[:, column_name].mean(),
                dataframe.loc[:, column_name].std(),
            )

    else:
        # plot quantity (metric on training set and validation set)
        try:
            ax.plot(dataframe.index.values + 1, dataframe.loc[:, quantity], "k-")
            labels.append("training set")
        except:
            pass
        try:
            try:
                ax.plot(
                    dataframe.index.values + 1,
                    dataframe.loc[:, f"val_{quantity}"],
                    "r-",
                )
                labels.append("validation set")
                prefix = "val"
            except:
                ax.plot(
                    dataframe.index.values + 1,
                    dataframe.loc[:, f"test_{quantity}"],
                    "r-",
                )
                labels.append("test set")
                prefix = "test"

            if running_mean is not None:
                # running mean on the validation metric
                assert (
                    running_mean % 2 == 1
                ), "running_mean needs to be an uneven integer!"
                running_mean_array = np.convolve(
                    dataframe.loc[:, f"{prefix}_{quantity}"].values,
                    [1 / running_mean] * running_mean,
                    mode="valid",
                )
                ax.plot(
                    (dataframe.index.values + 1)[(running_mean - 1) :],
                    running_mean_array,
                    "r--",
                )
                labels.append("running mean (%i epochs)" % running_mean)
            if return_val_mean_std:
                val_mean_std = (
                    dataframe.loc[:, f"{prefix}_{quantity}"].mean(),
                    dataframe.loc[:, f"{prefix}_{quantity}"].std(),
                )
        except:
            pass
    ax.set_title(f"Evolution of the {quantity} over the epochs trained")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{quantity}")
    ax.grid()
    leg = ax.figure
    leg.legend(labels)

    if "fig" in locals():
        if "val_mean_std" in locals():
            return fig, val_mean_std
        else:
            return fig
    else:
        if "val_mean_std" in locals():
            return val_mean_std
