"""
Image features (Sec. IV-B of the paper): grey level co-occurrence features, edge frequency,
auto-correlation and fractal dimensions, plus the blur measures used for image selection.
"""

from os import path
from typing import Sequence

import click
import numpy as np
import pandas as pd
import skimage.io as skio
from numpy.lib.stride_tricks import sliding_window_view
from pandas.errors import EmptyDataError
from pywt import dwt2
from scipy.ndimage import convolve
from scipy.stats import kurtosis
from skimage import filters
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.transform import resize

from src.utils import (
    Identifier,
    FeatureTypes,
    MEASURE,
    LOG_EDGES,
    HISTOGRAM_BINS,
    NLEVELS_CO_OCCURENCE,
    FILE_SEP,
    INTEGRATED_FREQUENCIES,
    shift_half_step,
    bin_image,
    rescale_image,
)


PIXEL_PAIR_DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, 0.5 * np.pi]


def histogram_statistics(image, nbins):
    """
    Create a histogram of the grayscales in an image and calculate metric from this histogram.

    # see Distante 2020, pp. 267

    :param image: np.ndarray, gray-scale image.
    :param nbins: Integer, number of bins in the histogram.
    :return: Dictionary containing the values calculated from the histogram.
    """
    freq, edges = np.histogram(image.ravel(), nbins)
    freq = freq / np.sum(freq)
    centers = shift_half_step(edges)
    mean = np.sum(freq * centers)
    var = np.sum((centers - mean) ** 2 * freq)
    return {
        "mean": mean,  # average intensity
        "var": var,  # dispersion
        "smoothness": 1 - (1 / (1 + var)),  # 0 means very smooth
        "energy": np.sum((freq) ** 2),  # homogeneity
        "entropy": -np.sum(
            np.log10(freq[np.nonzero(freq)]) * freq[np.nonzero(freq)]
        ),  # disorder
    }


def co_occurence_analysis(
    image,
    nlevels,
    mode="contrast",
    pixel_pair_distances=None,
    angles=None,
    integrated_frequencies=None,
    mask=None,
):
    """
    Calculate features based on the co-occurrence matrix.
        When angles and pixel_pair_distances are given:
            The feature in "mode" is calculated from the co-occurrence matrix where pixel-pairs are
            defined by the angle and the distance.
        When integrated_frequencies and pixel_pair_distances is given:
            The feature "mode" is calculated from several co-occurrence matrices (as an average).
            Here, only the pixel_pair_distances need to be provided.
        When neither pixel_pair_distances nor angles is given:
            Calculate feature from Gray-level gradient co-occurrence matrix (glgcm) instead.

    :param image: np.ndarray, gray-scale image.
    :param nlevels: integer specifying the number of grayscale levels to reduce the image to. Like
    this, the size of the co-occurrence matrix is decreased.
    :param mode: Name or list of names of the features to calculate.
    :param pixel_pair_distances: Distances between the pixel pairs.
    :param angles: Directions between the pixel pairs.
    :param integrated_frequencies: Weights of the directions per feature (Table 1 in Pan et al.).
    :param mask: np.ndarray, same size as image and mask only contains 0s and 1s.
    :return: List with one entry per feature in mode.
    """

    if integrated_frequencies is not None:
        assert set(angles).issubset(
            set(integrated_frequencies.columns.values)
        ), "Given angles and angles in integrated frequency table do not match"
    # check wether image is rgb-image:
    if image.ndim == 3:
        image = rgb2gray(image)
    # check wether image is made of floats:
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image *= 255
        image = image.astype("uint8", copy=False)
    # bin the image:
    image = bin_image(image, nlevels)
    step_size = int(256 / nlevels)
    image = ((image - image.min()) / step_size).astype("uint8")

    if not (pixel_pair_distances is None and angles is None):
        if pixel_pair_distances is None:
            pixel_pair_distances = PIXEL_PAIR_DISTANCES
        if angles is None:
            angles = ANGLES
        if mask is not None:
            mask = mask > 0.5
            if nlevels >= 255:
                # set masked values to nlevels:
                image = image.astype("uint16")
            image[mask.astype("bool")] = nlevels  # next value
            # (just make sure that this value has not been assigned to other fields)
            matrices = graycomatrix(
                image=image,
                distances=pixel_pair_distances,
                angles=angles,
                levels=nlevels + 1,
            )
            # remove the last column and the last row from the matrices to block out all masked
            # values (i.e. all combinations with masked values)
            matrices = matrices[:-1, :-1, :, :]
        else:
            matrices = graycomatrix(
                image=image, distances=pixel_pair_distances, angles=angles
            )
    # co-occurrence matrices are output in the form P(i, j, d, theta)
    if not isinstance(mode, (list, np.ndarray)):
        mode = [mode]
    props = []

    if integrated_frequencies is not None:
        # create a weighted mean of the cooccurrence matrices:
        for mode_ in mode:
            weights = integrated_frequencies.loc[mode_, angles].values
            # weigh matrices along last axis and sum them up along last axis
            mat_sum = sum(
                [
                    matrices[:, :, :, angle_ind] * weights[angle_ind]
                    for angle_ind in range(len(angles))
                ]
            )
            if mode_ in ["ASM", "contrast", "homogeneity"]:
                # the implementations of "contrast" and "correlation" are different from
                # Hailang 2020!
                props.append(
                    graycoprops(
                        np.reshape(mat_sum, list(mat_sum.shape) + [1]), prop=mode_
                    )
                )
            elif mode_ == "entropy":
                sums = mat_sum.sum(axis=(0, 1))
                sums[sums == 0] = 1
                mat_norm = mat_sum / sums
                mat_norm_log = mat_norm.copy()
                mat_norm_log[
                    mat_norm_log == 0
                ] = 1  # value does not matter since this will be zero when multiplied with zero
                props.append(
                    -np.sum(
                        mat_norm * np.log(mat_norm_log), axis=(0, 1), keepdims=True
                    )[0].T
                )  # Hailang 2020
                del mat_norm_log, mat_norm
    elif pixel_pair_distances is None and angles is None:
        # glgcm:
        sobel_image = sobel(image)
        # normalize sobel-gradient (Hailang 2020):
        sobel_image = (sobel_image * float(127 / sobel_image.max())).astype("uint8")
        # normalize gray-level-image (Hailang 2020):
        image = (image * float(127 / image.max())).astype("uint8")
        size = 128
        if mask is not None:
            # masked pixels get the extra grey value 128 which is removed from the matrix afterwards
            size = 129
            image[mask.astype("bool")] = 128
            sobel_image[mask.astype("bool")] = 128
        glgcm = np.bincount(
            image.ravel().astype(np.int64) * size + sobel_image.ravel(), minlength=size**2
        ).reshape(size, size)
        if mask is not None:
            glgcm = glgcm[:-1, :-1]
        glgcm = glgcm / glgcm.sum()

        for mode_ in mode:
            if mode_.startswith("energy"):
                props.append((glgcm**2).sum())
    else:
        for mode_ in mode:
            props.append(graycoprops(matrices, prop=mode_))

    return props


def autocorrelation(image, offsets_x, offsets_y, mask=None):
    """
    Calculate the coarseness of an image with its autocorrelation. Taken from Hailang 2020.
    :param image: np.ndarray, gray-scale image.
    :param offsets_x: int or seq. Every element will be treated as another maximum value.
    :param offsets_y: int or seq.
    :param mask: has the same size as image; contains only 0s and 1s.
    :return: Matrix of the second moment autocorrelation for all combinations of offsets.
    """
    # if the texture is coarse, the autocorrelation function declines faster than if the texture
    # was smooth
    if isinstance(offsets_x, int):
        offsets_x = np.array(offsets_x)
    if isinstance(offsets_y, int):
        offsets_y = np.array(offsets_y)
    measure_at_x = offsets_x.copy()
    measure_at_y = offsets_y.copy()
    coarseness = np.zeros(shape=(len(measure_at_x), len(measure_at_y)))

    offsets_x = np.arange(max(offsets_x)) + 1
    offsets_y = np.arange(max(offsets_y)) + 1

    if mask is not None:
        mask = mask > 0.5

    for offset_x in offsets_x:
        for offset_y in offsets_y:

            if mask is None:
                variance = (image[offset_x:, offset_y:] ** 2).sum()
                autocov = (image[offset_x:, offset_y:] * image[:-offset_x, :-offset_y]).sum()
                autocorr = autocov / variance

            else:
                keep_inds = ~np.any(
                    [mask[offset_x:, offset_y:], mask[:-offset_x, :-offset_y]], axis=0
                ).ravel()
                variance = (image[offset_x:, offset_y:].ravel()[keep_inds] ** 2).sum()
                autocov = (
                    image[offset_x:, offset_y:].ravel()[keep_inds]
                    * image[:-offset_x, :-offset_y].ravel()[keep_inds]
                ).sum()
                autocorr = autocov / variance

            y, x = np.meshgrid(offset_y <= measure_at_y, offset_x <= measure_at_x)
            coarseness[np.all([x, y], axis=0)] += offset_x**2 * offset_y**2 * autocorr
    return coarseness


def fractal_dim_ibc(image: np.ndarray, r1: int, r2: int, mask: (np.ndarray, None) = None):
    """
    Calculate fractal dimension (which gives a clue about the coarseness of an image) with the box
    counting method. Here the image is divided in many subimages. This is done for two widths and
    heights: r1 and r2.
    from Hailang 2020, fractals

    :param image: np.ndarray, Gray-scale image.
    :param r1: Integer, size (width and height) of first subimages.
    :param r2: Integer, size (width and height) of second subimages. r2 should be > r1.
    :param mask: np.ndarray with same size as image with only 0s and 1s.
    :return: one value representing the fractal dimension.
    """

    if mask is not None:
        image = image.astype("float")
        mask = mask > 0.5
        image[mask.astype("bool")] = np.nan
    # correct image size so that there are no partly subwindows:
    image_r1 = image[: image.shape[0] // r1 * r1, : image.shape[1] // r1 * r1]
    subimages_r1 = image_r1.reshape(image_r1.shape[0] // r1, r1, image_r1.shape[1] // r1, r1)

    image_r2 = image[: image.shape[0] // r2 * r2, : image.shape[1] // r2 * r2]
    subimages_r2 = image_r2.reshape(image_r2.shape[0] // r2, r2, image_r2.shape[1] // r2, r2)

    # subimages are of size rxr
    # extract corners:
    corners_r1 = np.array(
        [
            subimages_r1[:, 0, :, 0],
            subimages_r1[:, 0, :, r1 - 1],
            subimages_r1[:, r1 - 1, :, 0],
            subimages_r1[:, r1 - 1, :, r1 - 1],
        ]
    )
    no_nans_corners_r1 = ~np.any(np.isnan(corners_r1), axis=0)
    corners_r2 = np.array(
        [
            subimages_r2[:, 0, :, 0],
            subimages_r2[:, 0, :, r2 - 1],
            subimages_r2[:, r2 - 1, :, 0],
            subimages_r2[:, r2 - 1, :, r2 - 1],
        ]
    )
    no_nans_corners_r2 = ~np.any(np.isnan(corners_r2), axis=0)
    # only consider the subimages without nans in the corners

    nr1 = (
        np.floor(
            (
                np.max(corners_r1[:, no_nans_corners_r1], axis=0)
                - np.min(corners_r1[:, no_nans_corners_r1], axis=0)
            )
            / r1
        )
        + 1
    )
    nr2 = (
        np.floor(
            (
                np.max(corners_r2[:, no_nans_corners_r2], axis=0)
                - np.min(corners_r2[:, no_nans_corners_r2], axis=0)
            )
            / r2
        )
        + 1
    )
    epsilon = 1e-10

    not_nan_r1 = np.sum((~np.isnan(subimages_r1)).astype(int), axis=(1, 3))
    not_nan_r2 = np.sum((~np.isnan(subimages_r2)).astype(int), axis=(1, 3))

    gamma_r1 = (
        np.nansum(subimages_r1, axis=(1, 3)) / not_nan_r1 - np.nanmin(subimages_r1, axis=(1, 3))
    ) / (np.nanmax(subimages_r1, axis=(1, 3)) - np.nanmin(subimages_r1, axis=(1, 3)) + epsilon)
    gamma_r1 = gamma_r1[no_nans_corners_r1]
    gamma_r2 = (
        np.nansum(subimages_r2, axis=(1, 3)) / not_nan_r2 - np.nanmin(subimages_r2, axis=(1, 3))
    ) / (np.nanmax(subimages_r2, axis=(1, 3)) - np.nanmin(subimages_r2, axis=(1, 3)) + epsilon)
    gamma_r2 = gamma_r2[no_nans_corners_r2]
    # if gamma is larger than 0.8, set nr to 1, else use calculated value for nr
    threshold = 0.8
    nr1[gamma_r1 > threshold] = 1
    nr2[gamma_r2 > threshold] = 1

    Nr1 = np.sum(nr1) / len(nr1) * subimages_r1.shape[0] * subimages_r1.shape[2]
    Nr2 = np.sum(nr2) / len(nr2) * subimages_r2.shape[0] * subimages_r2.shape[2]
    D = -(np.log(Nr1) - np.log(Nr2)) / (np.log(r1) - np.log(r2))
    return D


def fractal_dim_gvs(image: np.ndarray, rs: Sequence = (1, 2, 3, 5), N: int = 140, mask=None):
    """
    Calculates fractal dimension with grayscale value statistic.
    from Hailang 2020, fractals

    :param image: np.ndarray, gray-scale image.
    :param rs: List of integers. Defines the image sections.
    :param N: Integer. Size of subimages.
    :param mask: np.ndarray with same size as image with only 0s and 1s.
    :return: Value for fractal dimension.
    """
    # like in the ibc method, make sections from the image
    image_N = image[: image.shape[0] // N * N, : image.shape[1] // N * N]
    if mask is not None:
        mask = mask[: image.shape[0] // N * N, : image.shape[1] // N * N]
        mask = mask > 0.5
        image_N[mask.astype(bool)] = 0
        mask = 1 - mask.astype(int)
        submask_N = mask.reshape(image_N.shape[0] // N, N, image_N.shape[1] // N, N)
    subimages_N = image_N.reshape(image_N.shape[0] // N, N, image_N.shape[1] // N, N)

    # quadratic window of size NxN
    E = np.ones([len(rs), subimages_N.shape[0], subimages_N.shape[2]]) * np.nan
    # spatial dimension:
    d = 2
    rs = np.sort(rs)
    for idx, r in enumerate(rs):
        if mask is not None:
            denominator_x = np.sum(
                np.all([submask_N[:, :-r, :, :], submask_N[:, r:, :, :]], axis=0),
                axis=(1, 3),
            )
            denominator_y = np.sum(
                np.all([submask_N[:, :, :, :-r], submask_N[:, :, :, r:]], axis=0),
                axis=(1, 3),
            )
        else:
            denominator_x = denominator_y = N * (N - r)
        Ex = (
            np.sum(abs(subimages_N[:, :-r, :, :] * subimages_N[:, r:, :, :]), axis=(1, 3))
            / denominator_x
        )
        Ey = (
            np.sum(abs(subimages_N[:, :, :, :-r] * subimages_N[:, :, :, r:]), axis=(1, 3))
            / denominator_y
        )
        E[idx, :, :] = Ex + Ey

    # regression fit this system of equations:
    # {ln(E(r1)) = H * ln(r1) + ln(C)
    #  ln(E(r2)) = H * ln(r2) + ln(C)}
    # average H from regressions:
    H = np.ones([len(rs) - 1, subimages_N.shape[0], subimages_N.shape[2]]) * np.nan
    for idx in range(len(rs) - 1):
        # the y-axis intersection is of no interest, only calculate the slope:
        H[idx] = np.log(E[idx + 1] / E[idx]) / np.log(rs[idx + 1] / rs[idx])
    H_mean = np.mean(H, axis=0)
    D = d + 1 - H_mean.mean()
    return D


def edge_frequency(image, edge_freqs, mask=None):
    """
    Calculate coarseness of an image with the edge method for several different parameters.
    From Hailang, 2020

    :param image: np.ndarray. Gray-scale image.
    :param edge_freqs: List of integers. One value corresponds to the longest edge which will be
    detected in the image. For every such length, a different coarseness is computed.
    :param mask: np.ndarray. Mask should have the same size as image and only contain 0s and 1s.
    :return: Multiple values representing the coarseness of the image for the different edge
    frequencies.
    """

    if isinstance(image[0, 0], (int, np.uint8, np.uint16)):
        image = image.astype(float) / 255.0
    # calculate gradient frequency:
    lengths = np.unique([int(edge.method.max_distance) for edge in edge_freqs])
    ds = np.arange(max(lengths)) + 1
    P = np.empty(shape=list(image.shape) + [len(ds)])
    if mask is not None:
        image = image.astype(float)
        image[mask.astype(bool)] = np.nan
    for ind_d, d in enumerate(ds):
        gradient = (
            abs(image[d:-d, d:-d] - image[: (-2 * d), d:-d])
            + abs(image[d:-d, d:-d] - image[(2 * d) :, d:-d])
            + abs(image[d:-d, d:-d] - image[d:-d, (2 * d) :])
            + abs(image[d:-d, d:-d] - image[d:-d, : (-2 * d)])
        )
        P[:, :, ind_d] = np.pad(gradient, (d,), "constant", constant_values=0)
    corrected_size = (image.shape[0] - 2 * ds) * (image.shape[1] - 2 * ds)
    if mask is not None:
        corrected_size = corrected_size - np.sum(np.isnan(P), axis=(0, 1))
    density_maximum = {}
    for length in lengths:
        density = np.nansum(P[:, :, :length], axis=(0, 1)) / np.squeeze(corrected_size)[:length]
        density_maximum[length] = density.max()

    return density_maximum



######################## blur measures used for the image quality classification (Sec. III-A)


def calc_depth_map(image: np.ndarray, features: Sequence, window_shape: (int, tuple)):
    """
    Calculate maps of blur measures (laplace, tenengrad, wavelet, kurtosis) over sliding windows.
    """
    parsed_features = {
        "laplace": [feat for feat in features if "laplace" in feat],
        "tenengrad": [feat for feat in features if "tenengrad" in feat],
        "wavelet": [feat for feat in features if "wavelet" in feat],
        "kurtosis": [feat for feat in features if "kurtosis" in feat],
    }

    if isinstance(window_shape, int):
        window_shape = (window_shape, window_shape)

    # only take full windows and only accept uneven window-sizes:
    assert all(map(lambda x: x % 2, window_shape)), "Choose uneven window"

    margin = tuple(map(lambda x: x // 2, window_shape))

    nrows, ncols = image.shape

    # prepare maps:
    depth_maps = {}
    for feat in features:
        if "wavelet" in feat:
            depth_maps[feat] = (
                np.empty(shape=(nrows // 2 - 2 * margin[0], ncols // 2 - 2 * margin[1]))
                * np.nan
            )
        else:
            depth_maps[feat] = (
                np.empty(shape=(nrows - 2 * margin[0], ncols - 2 * margin[1])) * np.nan
            )

    if len(parsed_features["wavelet"]) > 0:
        wavelets = [feat.split("_")[1] for feat in parsed_features["wavelet"]]
        for wvlt in np.unique(wavelets):
            # extract measure:
            measures = np.unique(
                [
                    feat.split("_")[-1]
                    for feat in parsed_features["wavelet"]
                    if wvlt in feat
                ]
            )
            approx, coeffs = dwt2(
                data=image, wavelet=wvlt, mode="periodization"
            )  # this mode leads to the smallest
            # output-size (height and width of original image divided by two)

            splitter = split_sliding_window(
                image_shape=approx.shape, window_shape=window_shape, margin=margin
            )
            while True:
                try:
                    ((r0, r1), (c0, c1)), ((map_r0, map_r1), (map_c0, map_c1)) = next(
                        iter(splitter)
                    )
                except StopIteration:
                    break

                approx_view = sliding_window_view(approx[r0:r1, c0:c1], window_shape)
                coeffs_view = [
                    sliding_window_view(coeff[r0:r1, c0:c1], window_shape)
                    for coeff in coeffs
                ]

                for mode in measures:
                    depth_maps[f"wavelet_{wvlt}_{mode}"][
                        map_r0:map_r1, map_c0:map_c1
                    ] = wavelet_map(
                        mode=mode, approx_sw=approx_view, coeffs_sw=tuple(coeffs_view)
                    )

        del approx, coeffs

    if len(parsed_features["tenengrad"]) > 0:
        sobel_edges_h = filters.sobel_h(image)
        sobel_edges_v = filters.sobel_v(image)
        modes = [tenengrad.split("_")[-1] for tenengrad in parsed_features["tenengrad"]]

        splitter = split_sliding_window(
            image_shape=image.shape, window_shape=window_shape, margin=margin
        )
        while True:
            try:
                ((r0, r1), (c0, c1)), ((map_r0, map_r1), (map_c0, map_c1)) = next(
                    iter(splitter)
                )
            except StopIteration:
                break

            sobel_edges_h_view = sliding_window_view(
                x=sobel_edges_h[r0:r1, c0:c1], window_shape=window_shape
            )
            sobel_edges_v_view = sliding_window_view(
                x=sobel_edges_v[r0:r1, c0:c1], window_shape=window_shape
            )
            for mode in modes:
                depth_maps[f"tenengrad_{mode}"][
                    map_r0:map_r1, map_c0:map_c1
                ] = tenengrad_map(
                    sobel_edges_h=sobel_edges_h_view,
                    sobel_edges_v=sobel_edges_v_view,
                    mode=mode,
                )

        del sobel_edges_v, sobel_edges_h

    if len(parsed_features["laplace"]) > 0:

        modes = [laplace.split("_")[-1] for laplace in parsed_features["laplace"]]
        splitter = split_sliding_window(
            image_shape=image.shape, window_shape=window_shape, margin=margin
        )
        while True:
            try:
                ((r0, r1), (c0, c1)), ((map_r0, map_r1), (map_c0, map_c1)) = next(
                    iter(splitter)
                )
            except StopIteration:
                break

            # calculate sliding_window_view:
            image_windows = sliding_window_view(
                image[r0:r1, c0:c1], window_shape=window_shape
            )

            for mode in modes:
                depth_maps[f"laplace_{f'{mode}' if mode != 'var' else '3_var'}"][
                    map_r0:map_r1, map_c0:map_c1
                ] = laplace_map(image_sw=image_windows, mode=mode)

    if len(parsed_features["kurtosis"]) > 0:

        # split the image into parts:
        splitter = split_sliding_window(
            image_shape=image.shape, window_shape=window_shape, margin=margin
        )

        while True:
            try:
                ((r0, r1), (c0, c1)), ((map_r0, map_r1), (map_c0, map_c1)) = next(
                    iter(splitter)
                )
            except StopIteration:
                break

            image_windows = sliding_window_view(
                x=image[r0:r1, c0:c1], window_shape=window_shape
            )
            image_windows = np.reshape(
                image_windows,
                newshape=list(image_windows.shape[0:2])
                + [np.multiply(*image_windows.shape[2:4])],
            )
            depth_maps["kurtosis"][map_r0:map_r1, map_c0:map_c1] = kurtosis(
                a=image_windows, axis=2
            )
    return depth_maps


def split_sliding_window(image_shape: tuple, window_shape: tuple, margin: tuple):

    split_rows = np.linspace(
        0,
        image_shape[0],
        max(min([(image_shape[0]) // margin[0], window_shape[0] // 2]), 2),
    ).astype(int)
    split_cols = np.linspace(
        0,
        image_shape[1],
        max(min([(image_shape[1]) // margin[1], window_shape[1] // 2]), 2),
    ).astype(int)
    # make sure with max(..., 2) that in case only one split is needed two numbers are returned

    for idx_row in np.arange(len(split_rows) - 1):
        r0 = split_rows[idx_row] - (0 if idx_row == 0 else margin[0])
        r1 = split_rows[idx_row + 1] + (
            0 if idx_row == (len(split_rows) - 2) else margin[0]
        )
        for idx_col in np.arange(len(split_cols) - 1):
            c0 = split_cols[idx_col] - (0 if idx_col == 0 else margin[1])
            c1 = split_cols[idx_col + 1] + (
                0 if idx_col == (len(split_cols) - 2) else margin[1]
            )

            map_r0 = split_rows[idx_row] - (margin[0] if not idx_row == 0 else 0)
            map_r1 = split_rows[idx_row + 1] - margin[0]
            map_c0 = split_cols[idx_col] - (margin[1] if not idx_col == 0 else 0)
            map_c1 = split_cols[idx_col + 1] - margin[1]

            yield ((r0, r1), (c0, c1)), ((map_r0, map_r1), (map_c0, map_c1))


def calc_measures(depth_maps: dict, features: Sequence):
    """
    Summarise depth maps with the measures given in the feature names (e.g. mean, var, modus).
    :param depth_maps: Output of calc_depth_map.
    :param features: Sequence of ids (see class Identifier) or strings corresponding to ids.
    :return: Dictionary mapping feature names to values.
    """
    features = [
        feat if isinstance(feat, Identifier) else Identifier.construct_id(feat)
        for feat in features
    ]

    feature_dict = {feat.construct_string(): None for feat in features}

    for feat in features:

        method, meas = feat.method.get_str(), feat.measure
        meas_ = meas.split("_")

        if len(meas_) > 2 or (len(meas_) > 1 and "log10" not in meas_):
            if "portion" == meas_[1]:
                meas_ = meas_[0]
                map_ = list(map(str.isdigit, meas_))
                # intermittent False is decimal point
                border = np.where(map_)[0][-1] + 1
                portion, measure = float(meas_[:border]), meas_[border:]
                if "modus" not in meas:
                    threshold = MEASURE[measure](depth_maps[method].ravel()) * portion
                else:
                    threshold = (
                        MEASURE[measure](depth_maps[method].ravel(), LOG_EDGES[method])
                        * portion
                    )
                feature_dict[feat.construct_string()] = (
                    np.sum(depth_maps[method].ravel() < threshold)
                    / depth_maps[method].size
                )
        else:
            measure = meas

            if "modus" in measure:
                feature_dict[feat.construct_string()] = MEASURE[measure](
                    depth_maps[method], LOG_EDGES[method]
                )
            else:
                feature_dict[feat.construct_string()] = MEASURE[measure](
                    depth_maps[method]
                )

    return feature_dict


def laplace_map(image_sw: np.ndarray, mode: str):

    assert (
        image_sw.ndim == 4
    ), f"image_sw needs 4 dimensions but has {image_sw.ndim} dimensions"

    if "modified" in mode or "diagonal" in mode:
        mask = np.array([[-1, 2, -1]])
        laplace = abs(convolve(image_sw, np.array([[mask]]), mode="constant", cval=0))
        laplace = laplace + abs(
            convolve(image_sw, np.array([[mask.T]]), mode="constant", cval=0)
        )

        if "diagonal" in mode:
            mask_diag = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]]) / np.sqrt(2)
            laplace = laplace + abs(
                convolve(image_sw, np.array([[mask_diag]]), mode="constant", cval=0)
            )
            laplace = laplace + abs(
                convolve(
                    image_sw,
                    np.array([[np.flipud(mask_diag)]]),
                    mode="constant",
                    cval=0,
                )
            )
        laplace = 1 / np.multiply(*image_sw.shape[2:]) * np.sum(laplace, axis=(2, 3))

    else:  # "var" in mode:
        mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian mask
        laplace = convolve(image_sw, np.array([[mask]]), mode="constant", cval=0)
        laplace = np.var(laplace, axis=(2, 3))

    return laplace


def wavelet_map(mode, approx_sw: np.ndarray, coeffs_sw: tuple):
    assert approx_sw.ndim == 4, f"approx needs 4 dimensions but has {approx_sw.ndim}"

    if "sum" in mode:
        wavelet = (
            1
            / np.multiply(*approx_sw.shape[2:])
            * np.sum(
                abs(coeffs_sw[0]) + abs(coeffs_sw[1]) + abs(coeffs_sw[2]), axis=(2, 3)
            )
        )
    elif "var" in mode:
        wavelet = (
            np.var(coeffs_sw[0], axis=(2, 3))
            + np.var(coeffs_sw[1], axis=(2, 3))
            + np.var(coeffs_sw[2], axis=(2, 3))
        )
    elif "ratio" in mode:
        wavelet = (
            1
            / np.multiply(*approx_sw.shape[2:])
            * np.sum(
                (coeffs_sw[0] ** 2 + coeffs_sw[1] ** 2 + coeffs_sw[2] ** 2), axis=(2, 3)
            )
            / np.sum(approx_sw**2, axis=(2, 3))
        )
    else:
        wavelet = None

    return wavelet


def tenengrad_map(sobel_edges_v: np.ndarray, sobel_edges_h: np.ndarray, mode: str):

    if mode == "measure":
        sobel_edges_v = sobel_edges_v**2
        sobel_edges_h = sobel_edges_h**2
        tenengrad = (
            1
            / np.multiply(*sobel_edges_v.shape[2:])
            * np.sum(sobel_edges_v + sobel_edges_h, axis=(2, 3))
        )

    elif mode == "var":
        sobel_edges_h = np.sqrt(sobel_edges_h**2 + sobel_edges_v**2)
        # overwrite sobel_edges_h to avoid too much memory usage
        tenengrad = np.var(sobel_edges_h, axis=(2, 3))
    else:
        tenengrad = None
    return tenengrad


######################################################## feature calculation


def calc_or_load_features(
    features: list,
    image_dirs: list,
    masks: (list, None) = None,
    labeled: bool = False,
    save_append_dir: str = None,
    load: bool = False,
    save_interval=100,
    resize_shape=None,
    nlevels_co_occurence=NLEVELS_CO_OCCURENCE,
    histogram_bins=HISTOGRAM_BINS,
):
    """
    Function that calculates given features of the images given by image_dirs.
    :param features: List of strings or list of Identifier-objects (class defined in src.utils).
    :param image_dirs: List of images to include (list of their directories).
    :param masks: List of arrays (same shape as the images, only 0s and 1s) where pixels with 1
     are ignored. Used for the t-test in src.feature_ttest.
    :param labeled: If True, this function will extract a label from the directory if features are
     calculated and returns a label-array as second output if load=True
    :param save_append_dir: Directory to the file which should be created or appended.
    :param load: boolean. If this parameter is True, existing values will not be overwritten (only
     missing values will be calculated). Furthermore, the dataset will be returned.
    :param save_interval: specify how often the features should be saved to the file at
     save_append_dir. If 100, every 100 images the dataframe is saved. Saving the dataframe to file
     takes some time which is why it should not be done too often.
    :param resize_shape: Specify a shape to which the images should be resized to before calculating
     features from them.
    :param nlevels_co_occurence: Number of grey levels for the co-occurrence matrices.
    :param histogram_bins: Number of bins for the histogram statistics.
    :return: Array of features (and labels if labeled) when save_append_dir is None or load is True.
    """

    feature_types = FeatureTypes()

    # check wether entries are strings or Identifier-objects:
    is_id = [isinstance(feat, Identifier) for feat in features]
    features_str, features_id = [], []
    for is_id_, feat in zip(is_id, features):
        if is_id_:
            features_str.append(feat.construct_string())
            features_id.append(feat)
        else:
            # to avoid confusion with fractalIBC for example where the given order of the radii does
            # not matter since they are resorted so that the larger one comes first (or last), the
            # given strings all need to go through construct_string
            try:
                id_loop_var = Identifier.construct_id(feat)
                features_str.append(id_loop_var.construct_string())
                features_id.append(id_loop_var)
            except Exception:
                print(f"Feature {feat} is not recognized.")
                return

    if save_append_dir is not None:
        # append dataset if it already exists:
        if path.exists(save_append_dir):
            # load pd.DataFrame:
            with open(save_append_dir, "r") as file:
                try:
                    dataframe = pd.read_csv(file, index_col=0, sep=FILE_SEP)
                    # drop rows where index is nan (artifact from reading the file)
                    new_ind = dataframe.index.dropna()
                    dataframe = dataframe.loc[new_ind, :]
                    # drop duplicates:
                    dataframe = dataframe[~dataframe.index.duplicated(keep="first")]
                except EmptyDataError:
                    # if the file is empty
                    dataframe = pd.DataFrame(index=image_dirs, columns=features_str + ["label"])
                else:
                    # check wether all features are already in the dataframe and append columns if
                    # neccessary:
                    existing_features = dataframe.columns
                    add_features = list(set(features_str).difference(set(existing_features)))
                    # check wether the column label exists or will be added; otherwise add it to
                    # add_features
                    if "label" not in existing_features and "label" not in add_features:
                        add_features.append("label")
                    # Concat also works with empty dataframes
                    dataframe = pd.concat(
                        [
                            dataframe,
                            pd.DataFrame(index=dataframe.index, columns=add_features),
                        ],
                        axis="columns",
                    )
                    # do the same with the index
                    existing_inds = dataframe.index
                    add_inds = list(set(image_dirs).difference(set(existing_inds)))
                    dataframe = pd.concat(
                        [
                            dataframe,
                            pd.DataFrame(index=add_inds, columns=dataframe.columns),
                        ],
                        axis="index",
                    )
        else:
            # create new file:
            with open(save_append_dir, "x"):
                pass
            # create new dataframe
            dataframe = pd.DataFrame(index=image_dirs, columns=features_str + ["label"])
    else:
        # create new dataframe
        dataframe = pd.DataFrame(index=image_dirs, columns=features_str + ["label"])

    for idx_dir, im_dir in enumerate(image_dirs):
        if load:
            # only calculate when there are nan-values
            nans_inds = np.nonzero(dataframe.loc[im_dir, features_str].isna().values)[0]
            if len(nans_inds) == 0:
                continue
            features_to_calc = list(np.array(features_str)[nans_inds])
            print(
                f"The following features are missing and will be calculated: "
                f"{', '.join(features_to_calc)}"
            )
        else:
            features_to_calc = features_str

        ############################################################################################
        # load image and calculate features
        try:
            image = skio.imread(im_dir)
            if resize_shape is not None:
                print(f"Resize image {im_dir} from shape {image.shape} to shape {resize_shape}.")
                image = resize(image, output_shape=resize_shape)
        except FileNotFoundError:
            print(f"{im_dir} was not found. Skipping this image.")
            dataframe.loc[im_dir, :] = np.nan
            continue
        except ValueError:
            print(f"Could not load {im_dir}.")
            continue

        mask = None if masks is None else masks[idx_dir]
        if mask is not None:
            mask = mask.astype("uint8")
            if mask.shape != image.shape[:2]:
                print(f"Mask for image {im_dir} has shape {mask.shape} instead of "
                      f"{image.shape[:2]}. Skipping this image.")
                dataframe.loc[im_dir, :] = np.nan
                continue

        if image.ndim > 2:
            image = (rgb2gray(image) * 255).astype("uint8")
        elif image.dtype != "uint8" and image.max() <= 1:
            image = (image * 255).astype("uint8")
        ######################## HISTOGRAM STATISTICS ##############################################
        if np.any((feat.feature_type == feature_types.histogram_feature for feat in features_id)):
            stats = histogram_statistics(image, nbins=histogram_bins)
            for feat, val in stats.items():
                if f"{feat}_" in features_to_calc:
                    dataframe.loc[im_dir, f"{feat}_"] = val

        ####################### CO-OCCURENCE-MATRIX ################################################
        modes = [
            feat for feat in features_id if feat.feature_type == feature_types.cooccurrence_feature
        ]
        if len(modes) > 0:
            # parse variables
            distances = np.array([mode.method.distance for mode in modes])
            mode_info = np.array([mode.method.method for mode in modes])
            angles = np.array([mode.method.angle for mode in modes])
            props = co_occurence_analysis(
                image=image,
                nlevels=nlevels_co_occurence,
                mode=np.unique(mode_info),
                pixel_pair_distances=np.unique(distances),
                angles=np.unique(angles),
                mask=mask,
            )

            for feat in modes:
                # determine the correct value of props (depending on the angle and the distance)
                dataframe.loc[im_dir, feat.construct_string()] = props[
                    np.nonzero(np.unique(mode_info) == feat.method.method)[0][0]
                ][np.nonzero(np.unique(distances) == feat.method.distance)[0][0]][
                    np.nonzero(np.unique(angles) == feat.method.angle)[0][0]
                ]

        ################################################ DEPTH MAPS ################################

        # parse features:
        depth_features = np.array(
            [feat for feat in features_id if feat.feature_type == feature_types.depth_feature]
        )
        factors = [depth.factor for depth in depth_features]

        window_sizes = []
        window_sizes_unique = []
        for depth in depth_features:
            if depth.window_shape not in window_sizes:
                window_sizes_unique.append(depth.window_shape)
            window_sizes.append(depth.window_shape)

        for factor in np.unique(factors):
            image_new = rescale_image(image, factor)
            for window_size in window_sizes_unique:
                chosen = np.all(
                    [
                        factor == factors,
                        [window_size == win_size for win_size in window_sizes],
                    ],
                    axis=0,
                )
                if chosen.sum() > 0:
                    feats = [feat.method.get_str() for feat in depth_features[chosen]]
                    output = calc_depth_map(
                        image=image_new,
                        features=np.unique(feats),
                        window_shape=window_size,
                    )
                    # calculate the measures from the depth-maps
                    output_vals = calc_measures(depth_maps=output, features=depth_features[chosen])
                    for key, val in output_vals.items():
                        dataframe.loc[im_dir, f"{key}"] = val

        ################################### HAILANG 2020: GLCM + INTEGRATED FREQUENCIES ############

        integrated_glcm_features = np.array(
            [
                feat
                for feat in features_id
                if feat.feature_type == feature_types.cooccurrence_integrated_feature
            ]
        )

        if len(integrated_glcm_features) > 0:
            distances = np.unique([integ.method.distance for integ in integrated_glcm_features])
            measures = np.unique([integ.method.method for integ in integrated_glcm_features])

            props = co_occurence_analysis(
                image=image,
                nlevels=nlevels_co_occurence,
                mode=np.unique(measures),
                pixel_pair_distances=np.unique(distances),
                angles=[0, 45, 90, 135],
                integrated_frequencies=INTEGRATED_FREQUENCIES,
                mask=mask,
            )

            for feat in integrated_glcm_features:
                # determine the correct value of props (depending on the angle and the distance)
                dataframe.loc[im_dir, feat.construct_string()] = props[
                    np.nonzero(np.unique(measures) == feat.method.method)[0][0]
                ][np.nonzero(np.unique(distances) == feat.method.distance)[0][0]][0]

        ############################ GLGCM FEATURES (HAILANG 2020) #################################

        glgcm_features = np.array(
            [
                feat
                for feat in features_id
                if feat.feature_type == feature_types.cooccurrence_gradient_feature
            ]
        )
        if len(glgcm_features) > 0:
            measures = [glgcm.method.method for glgcm in glgcm_features]
            props = co_occurence_analysis(image=image, mode=measures, nlevels=128, mask=mask)

            for glgcm_no, glgcm in enumerate(glgcm_features):
                dataframe.loc[im_dir, glgcm.construct_string()] = props[glgcm_no]

        ########################### EDGE FREQUENCY #################################################

        edge_freq = np.array(
            [feat for feat in features_id if feat.feature_type == feature_types.edge_feature]
        )

        if len(edge_freq) > 0:
            freqs = edge_frequency(image=image, edge_freqs=edge_freq, mask=mask)
            for name, density_max in freqs.items():
                dataframe.loc[im_dir, f"edge_{name}"] = density_max

        #################################### AUTOCORRELATION #######################################

        autocorr = np.array(
            [
                feat
                for feat in features_id
                if feat.feature_type == feature_types.autocorrelation_feature
            ]
        )

        if len(autocorr) > 0:
            max_distances = [corr.method.max_distance for corr in autocorr]
            second_moment_autocorr_matrix = autocorrelation(
                image=image, offsets_x=max_distances, offsets_y=max_distances, mask=mask
            )
            for idx, max_distance in enumerate(max_distances):
                dataframe.loc[
                    im_dir, f"autocorrelation_{max_distance}"
                ] = second_moment_autocorr_matrix[idx, idx]

        ############################## FRACTAL DIMENSION FROM GRAYSCALE VALUE STATISTIC ############

        fractal_gvs = np.array(
            [
                feat
                for feat in features_id
                if feat.feature_type == feature_types.fractal_gvs_feature
            ]
        )

        if len(fractal_gvs) > 0:
            for fractal_gvs_ in fractal_gvs:
                fractal_dim = fractal_dim_gvs(image=image, N=fractal_gvs_.method.N, mask=mask)
                dataframe.loc[im_dir, fractal_gvs_.construct_string()] = fractal_dim

        ######################### FRACTAL DIMENSION FROM IMPROVED BOX COUNTING METHOD ##############

        fractal_ibc = np.array(
            [
                feat
                for feat in features_id
                if feat.feature_type == feature_types.fractal_ibc_feature
            ]
        )

        if len(fractal_ibc) > 0:
            for fractal_ibc_ in fractal_ibc:
                fractal_dim = fractal_dim_ibc(
                    image=image,
                    r1=fractal_ibc_.method.r1,
                    r2=fractal_ibc_.method.r2,
                    mask=mask,
                )
                dataframe.loc[im_dir, fractal_ibc_.construct_string()] = fractal_dim

        print(f"{im_dir} checked/calculated ({idx_dir + 1}/{len(image_dirs)})")

        if save_append_dir is not None and (
            idx_dir % save_interval == 0 or (idx_dir + 1) == len(image_dirs)
        ):
            # save the dataframe:
            with open(save_append_dir, "w") as file:
                dataframe.to_csv(file, sep=FILE_SEP)
                print(f"Saved the new dataframe at iteration {idx_dir}.")
    if save_append_dir is None or load:
        # always return the dataset in case of load
        if labeled:
            labels = dataframe.loc[image_dirs, "label"].values
            if any((not isinstance(label, (str, bool, int)) for label in labels)):
                labels = [val.split("/")[-2] for val in dataframe.loc[image_dirs].index.values]
            return dataframe.loc[image_dirs, features_str].values, labels
        return dataframe.loc[image_dirs, features_str].values


# Candidate features (Sec. IV-B1). Missing variations were excluded by the t-test in
# src.feature_ttest or for performance reasons (autocorrelation > 25).
FEATURES = [
    "homogeneity_90_5",
    "energy_90_5",
    "contrast_90_5",
    "dissimilarity_90_5",
    "correlation_90_5",
    "homogeneity_90_4",
    "energy_90_4",
    "contrast_90_4",
    "dissimilarity_90_4",
    "correlation_90_4",
    "homogeneity_90_3",
    "energy_90_3",
    "contrast_90_3",
    "dissimilarity_90_3",
    "correlation_90_3",
    "homogeneity_90_2",
    "energy_90_2",
    "contrast_90_2",
    "dissimilarity_90_2",
    "homogeneity_90_1",
    "energy_90_1",
    "dissimilarity_90_1",
    "homogeneity_0_5",
    "energy_0_5",
    "dissimilarity_0_5",
    "correlation_0_5",
    "homogeneity_0_4",
    "energy_0_4",
    "contrast_0_4",
    "dissimilarity_0_4",
    "correlation_0_4",
    "homogeneity_0_3",
    "energy_0_3",
    "dissimilarity_0_3",
    "homogeneity_0_2",
    "energy_0_2",
    "contrast_0_2",
    "dissimilarity_0_2",
    "correlation_0_2",
    "homogeneity_0_1",
    "energy_0_1",
    "dissimilarity_0_1",
    "homogeneity_10",
    "homogeneity_9",
    "homogeneity_8",
    "homogeneity_7",
    "homogeneity_6",
    "homogeneity_5",
    "homogeneity_4",
    "homogeneity_3",
    "homogeneity_2",
    "homogeneity_1",
    "ASM_10",
    "ASM_9",
    "ASM_8",
    "ASM_7",
    "ASM_6",
    "ASM_5",
    "ASM_4",
    "ASM_3",
    "ASM_2",
    "ASM_1",
    "contrast_10",
    "contrast_9",
    "contrast_8",
    "contrast_7",
    "contrast_6",
    "contrast_5",
    "contrast_4",
    "contrast_3",
    "contrast_2",
    "entropy_10",
    "entropy_9",
    "entropy_8",
    "entropy_7",
    "entropy_6",
    "entropy_5",
    "entropy_4",
    "entropy_3",
    "entropy_2",
    "entropy_1",
    "energy_glgcm",
    "edge_10",
    "edge_20",
    "edge_30",
    "edge_40",
    "edge_50",
    "edge_5",
    "edge_15",
    "edge_25",
    "edge_35",
    "edge_45",
    "autocorrelation_5",
    "autocorrelation_15",
    "autocorrelation_25",
    "autocorrelation_10",
    "autocorrelation_20",
    "fractalGVS_280",
    "fractalGVS_210",
    "fractalGVS_245",
    "fractalIBC_10_15",
    "fractalIBC_40_50",
    "fractalIBC_10_30",
]


IM_DIR = "images/classified/train.csv"
SAVETO = "images/classified/slicks_features.csv"
RESIZE_SHAPE = (1080, 1920, 3)


@click.command()
@click.option("--features", default=FEATURES, type=str, multiple=True)
@click.option("--im_dir", default=IM_DIR, type=str)
@click.option("--saveto", default=SAVETO, type=str)
@click.option("--start", default=None, type=int)
@click.option("--resize_shape", default=RESIZE_SHAPE, nargs=3, type=int)
def routine(features, im_dir, saveto, start, resize_shape):
    """
    Calculate the features of the images listed in im_dir and save them to saveto.

    :param features: List of feature names (see Identifier in src.utils).
    :param im_dir: Directory of the file containing the directories to the file from which the
     features should be calculated.
    The file contains the string of directories concatenated with ", ".
    :param saveto: Directory of the file to load features from and/or save features to.
    :param start: Integer. Use when only part of the images should be considered --> im_dir =
     im_dir[start:]
    """
    with open(im_dir, "r") as file:
        ims = file.readlines()[0].split(", ")
    if start is not None:
        ims = ims[start:]

    calc_or_load_features(
        features=features,
        image_dirs=ims,
        load=True,
        save_append_dir=saveto,
        labeled=False,
        resize_shape=resize_shape,
    )


if __name__ == "__main__":
    routine()
