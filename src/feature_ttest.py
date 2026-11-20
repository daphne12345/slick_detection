"""
Feature filtering with the one-sample t-test (Sec. IV-B2 of the paper).

All features are calculated for a subset of images and again for the same images where 10% of the
pixels are ignored randomly. Features whose ratio between both versions deviates significantly from
1 (p-value < 0.05) are excluded.
"""

import click
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from src.features import calc_or_load_features, FEATURES
from src.utils import RANDOM_STATE


@click.command()
@click.option("--im_dir", type=str, default="images/classified/view_train.csv")
@click.option("--step", type=int, default=140, help="Use every step-th image of im_dir.")
@click.option("--mask_portion", type=int, default=10, help="Percentage of pixels to ignore.")
@click.option("--image_shape", type=int, nargs=2, default=(1080, 1920))
@click.option("--saveto", type=str, default="images/classified/ttest.csv")
def routine(im_dir, step, mask_portion, image_shape, saveto):
    with open(im_dir, "r") as f:
        list_of_images = f.readlines()[0].split(", ")[::step]

    rng = np.random.default_rng(RANDOM_STATE)
    mask = rng.random(image_shape) < mask_portion / 100
    masks = [mask] * len(list_of_images)

    features_with_mask = calc_or_load_features(
        features=FEATURES,
        image_dirs=list_of_images,
        masks=masks,
        labeled=False,
        load=True,
        save_append_dir=f"images/classified/features_random_mask{mask_portion}.csv",
    )
    features_wo_mask = calc_or_load_features(
        features=FEATURES,
        image_dirs=list_of_images,
        masks=None,
        labeled=False,
        load=True,
        save_append_dir="images/classified/features_wo_mask.csv",
    )

    features_with_mask = pd.DataFrame(data=features_with_mask, index=list_of_images,
                                      columns=FEATURES)
    features_wo_mask = pd.DataFrame(data=features_wo_mask, index=list_of_images, columns=FEATURES)
    ratio = features_wo_mask / features_with_mask

    # null hypothesis: the mean of the ratio is 1
    t_test = ttest_1samp(ratio.values.astype(float), popmean=1.0, axis=0)
    result = pd.DataFrame(data={"pvalue": t_test.pvalue, "mean of ratio": ratio.mean(0).values},
                          index=FEATURES)
    result.loc[:, "exclude"] = t_test.pvalue < 0.05
    result.to_csv(saveto)
    print(f"Excluded features: {', '.join(result.index[result['exclude']])}")


if __name__ == "__main__":
    routine()
