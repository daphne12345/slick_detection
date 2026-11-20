"""
Figures and evaluation of the paper:

* map:              map of the sites (Fig. 2),
* precision-recall: precision-recall curves of saved predictions (Figs. 4 and 5), the threshold
                    maximising the f1-score and the metrics at these thresholds (Tables IV, V).
"""

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.utils import ConfusionMatrix, evaluate_results

plt.ioff()

# https://jan-46106.medium.com/plotting-maps-with-european-data-in-python-part-i-decd83837de4
SITES = {
    # https://www.erneuerbare-energien.de/EE/Redaktion/DE/Standardartikel/Offshore-Windenergie/Projekte/nordergruende.html
    "Nordergründe": (8.158, 53.833),
    # https://www.fino1.de/de/standort.html
    "Fino1": (6.588, 54.015),
    # https://www.fino2.de/de/projekt/standort.html
    "Fino2": (13.154, 55.007),
    # https://www.laengengrad-breitengrad.de/gps-koordinaten-von-leuchtturm-alte-weser
    "AlteWeser": (8.128, 53.863),
    # https://uol.de/icbm/forschungsplattformen-schiffe/messstation/informationen
    "Spiekeroog": (7.671, 53.750),
}


@click.group()
def cli():
    pass


@cli.command("map")
@click.option("--directory", type=str, default="map",
              help="Directory containing the GADM shape files "
                   "(https://gadm.org/download_country_v3.html). The map is saved here.")
def site_map(directory):
    """Plot the map with the locations of the sites."""
    import geopandas as gpd

    map_de = gpd.read_file(f"{directory}/Germany/gadm36_DEU_1.shp")
    ax = map_de.plot(figsize=[40, 14], color=[0.65] * 3)
    for country, code in [("Sweden", "SWE"), ("Netherlands", "NLD"), ("Denmark", "DNK")]:
        gpd.read_file(f"{directory}/{country}/gadm36_{code}_1.shp").plot(ax=ax, color=[0.5] * 3)
    ax.set_ylim([53.35, 55.2])
    ax.set_xlim([6, 14])
    fig = ax.figure
    fig.tight_layout()
    for site, (x, y) in SITES.items():
        ax.scatter(x, y, s=500, c="r")
        if site != "Nordergründe":
            ha, va, site = "right", "bottom", site + " " * 2
        else:
            ha, va, site = "left", "top", " " * 2 + site
        ax.text(x, y, s=site, ha=ha, va=va, fontsize=50)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(f"{directory}/map.png", bbox_inches="tight")


@cli.command("precision-recall")
@click.option("--predictions", type=str, multiple=True, required=True,
              help="csv-files with the columns truth (0 or 1) and prediction (probability), as "
                   "saved by src.train_keras_model. Pass training data first, then test data.")
@click.option("--labels", type=str, multiple=True, default=["training", "test"])
@click.option("--saveto", type=str, required=True, help="Path of the png-file to save.")
def precision_recall(predictions, labels, saveto):
    """Draw precision-recall curves and evaluate the model at several thresholds."""
    epsilon = 1e-8
    colors = ["k", "r", "b", "g"]
    dfs = [pd.read_csv(filename, index_col=0) for filename in predictions]

    def f1_scores(prec, rec):
        return 2 / (1 / (prec + epsilon) + 1 / (rec + epsilon))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx, df in enumerate(dfs):
        ax.scatter(df.loc[:, "prediction"].values,
                   df.loc[:, "truth"].values + np.random.rand(df.shape[0]) * 0.2 - 0.1,
                   s=0.5, c=colors[idx], label=labels[idx])
    ax.legend()
    ax.set_xlabel("prediction"), ax.set_ylabel("truth")
    fig.savefig(saveto)

    thresholds = [0.5]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx, df in enumerate(dfs):
        prec, rec, thr_pc = precision_recall_curve(y_true=df.loc[:, "truth"].values,
                                                   probas_pred=df.loc[:, "prediction"].values)
        ax.scatter(prec, rec, s=0.5, c=colors[idx], label=labels[idx])
        # threshold corresponding to the highest f1-score:
        f1_scores_curve = f1_scores(prec=prec, rec=rec)
        where_max_f1_pc = np.argmax(f1_scores_curve)
        thresholds.append(thr_pc[where_max_f1_pc])
        if len(dfs) < 2:
            ax.set_title("Precision-recall curve; best f1-score: %.3g at threshold %.3g" % (
                f1_scores_curve[where_max_f1_pc], thr_pc[where_max_f1_pc]))
            ax.scatter(prec[where_max_f1_pc], rec[where_max_f1_pc], s=4, c="r")
        else:
            ax.set_title("Precision-recall curve")
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.grid()
    ax.legend()
    fig.savefig(saveto.replace(".png", "_precision_recall.png"))

    for idx, df in enumerate(dfs):
        res = evaluate_results(df=None)
        for threshold in thresholds:
            truth = df.loc[:, "truth"].values[:-1]
            pred = (df.loc[:, "prediction"].values > threshold).astype(int)[:-1]
            conf_test = ConfusionMatrix(truth=truth, prediction=pred)
            res = evaluate_results(confusion_matrix_test=conf_test, name="%.3g" % threshold, df=res)
        res.to_csv(predictions[idx].replace(".csv", "_results.csv"))


if __name__ == "__main__":
    cli()
