"""
This job creates a model with auto-sklearn.
Note: auto-sklearn is not compatible with Windows! This needs to be run on a Linux machine
(Google Colab works fine) and preferably in a virtual environment since auto-sklearn requires
older versions of common packages such as scikit-learn.
"""

from collections import Counter
import pickle
from os.path import isdir
import click

import autosklearn.classification
from autosklearn.metrics import (
    balanced_accuracy,
    accuracy,
    f1,
    roc_auc,
    log_loss,
    recall,
    precision,
)
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

from src.utils import (
    standardize_features,
    ConfusionMatrix,
    random_subsets,
    get_best_from_each_feature_class,
    evaluate_results,
)
from src.features import FEATURES

TRAINING_SET = "images/classified/train.csv"
TEST_SET = "images/classified/test.csv"
LOAD_FROM = "images/classified/slicks_features.csv"
SAVETO = "images/classified"
FILTER_METHOD = "f"
METRIC = "accuracy"
BALANCING_METHOD = "None"
ENSEMBLE_SIZE = 50
N_ITER = 1
TOP = 1
CV = 5

# map metric names to functions
METRICS_FUNCTIONS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "f1": f1,
    "roc_auc": roc_auc,
    "neg_log_loss": log_loss,
    "recall": recall,
    "precision": precision,
}


@click.command()
@click.option("--im_dir_train", type=str, default=TRAINING_SET)
@click.option("--im_dir_test", type=str, default=TEST_SET)
@click.option("--load_from", type=str, default=LOAD_FROM)
@click.option("--load_from_test", type=str, default="None")
@click.option("--features", type=str, multiple=True, default=FEATURES)
@click.option("--saveto", type=str, default=SAVETO)
@click.option(
    "--filter_method",
    type=click.Choice(["r", "f", "mi", "None"]),
    default=FILTER_METHOD,
    help="Choose from: f (f-value clf), mi (mutual information clf), r (Pearson's r)," "None",
)
@click.option(
    "--metric",
    type=click.Choice(
        [
            "f1",
            "balanced_accuracy",
            "accuracy",
            "roc_auc",
            "neg_log_loss",
            "recall",
            "precision",
        ]
    ),
    help="Choose one from: f1, balanced_accuracy, accuracy, roc_auc, neg_log_loss",
    default=METRIC,
)
@click.option(
    "--balancing_method",
    type=click.Choice(["None", "undersampling", "oversampling"]),
    default=BALANCING_METHOD,
)
@click.option("--ensemble_size", type=int, default=ENSEMBLE_SIZE)
@click.option("--load_model_from", type=str, default="None")
def routine(
    im_dir_train,
    im_dir_test,
    load_from,
    load_from_test,
    features,
    saveto,
    filter_method,
    metric,
    balancing_method,
    ensemble_size,
    load_model_from,
):

    """
    Note: The module autosklearn is only available for Unix-based systems!
    Function to train and/or test auto-sklearn-models.
    This function saves two files in the directory saveto: The evaluation of the model as csv-file
    (as long as data for testing is provided), and the model as a pickle-file (as long as data for
    training is provided).

    :param im_dir_train: Directory to csv-file where a list of image directories used for
    training is stored.
    :param im_dir_test: Directory to csv-file where list of image directories used for testing
    is stored.
    :param load_from: Directory to csv-file where the features for the training images are
    stored.
    :param load_from_test: Directory to csv-file where the features for the testing images are
    stored.
    :param features: List of features to consider.
    :param saveto: Directory to save the model and the evaluation results to.
    :param filter_method: For the filtering of the features.
    :param metric: Optimization objective for auto-sklearn.
    :param balancing_method: How to rebalance the training feature set.
    :param ensemble_size: Number of single classifiers that auto-sklearn uses to build an ensemble
    model.
    :param load_model_from: If a model already exists and only an evaluation is required, the path
    to this model can be provided here.
    :return:
    """

    # create name for the model depending on the parameters which were set:
    model_name = f"automl_filter_{filter_method}_{balancing_method}_{metric}_{ensemble_size}.csv"
    # load image names used for the testing:
    with open(im_dir_train, "r") as f:
        ims_train = f.readlines()[0]
    ims_train = ims_train.split(", ")

    with open(im_dir_test, "r") as f:
        ims_test = f.readlines()[0]
    ims_test = ims_test.split(", ")

    # extract features from the file saved under "load_from":
    df = pd.read_csv(load_from, index_col=0)
    if im_dir_train != "None":
        train = df.loc[ims_train, :]

    if load_from_test != "None":
        df_test = pd.read_csv(load_from_test, index_col=0)
        test = df_test.loc[ims_test, :]
    else:
        test = df.loc[ims_test, :]

    if "all" in features:
        # Take all features which are represented in the provided feature file (load_from)
        features = list(df.columns.values)
        features = list(set(features).difference({"label"}))

    # extract labels and features from dataframes and convert labels to integers 0 or 1
    y_train = train.loc[:, "label"].values
    X_train = train.loc[:, features].values
    # todo: the next line is a dangerous line of code! The conversion to 0 or 1 will give false
    #  results when other labels than "True"/"False", 0/1 or True/False are present. When more than
    #  these labels are used, this needs to be fixed
    y_train = np.array([0 if y == "False" or not y else 1 for y in y_train])

    y_test = test.loc[:, "label"].values
    X_test = test.loc[:, features].values
    y_test = np.array([0 if y == "False" or not y else 1 for y in y_test])

    # standardize the features and use the same mean and standard deviation on the test set
    X_train, data_norm = standardize_features(
        features=pd.DataFrame(data=X_train, index=ims_train, columns=features)
    )
    X_test = standardize_features(
        features=pd.DataFrame(data=X_test, index=ims_test, columns=features),
        data_normalization=data_norm,
    )
    # first filtering of the features
    if filter_method != "None":
        stats = ["f", "mi"]
        funcs = [f_classif, mutual_info_classif]
        filter_method_scores = pd.DataFrame(index=features, columns=stats)
        for stat, func in zip(stats, funcs):
            values = func(X_train.values.astype(float), y_train.astype(float))
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
        X_train, X_test = X_train.loc[:, best_features], X_test.loc[:, best_features]

    # get corresponding function to name, return accuracy if metric is unknown
    metric_func = METRICS_FUNCTIONS.get(metric, "accuracy")
    # load model if one exists with these specifications
    try:
        if load_model_from == "None":
            load_model_from = f"{saveto}/{model_name}".replace(".csv", ".pickle")
        with open(load_model_from, "rb") as f:
            cls = pickle.load(f)
            print(f"Loaded model from {saveto}/{model_name.replace('.csv', '.pickle')}")
        do_not_save_model = True
    except FileNotFoundError:
        do_not_save_model = False
        # otherwise prepare to search/train a model
        cls = autosklearn.classification.AutoSklearnClassifier(
            metric=metric_func,
            resampling_strategy_arguments={"cv": CV},
            ensemble_size=ensemble_size,
        )

    # balancing of the classes (here only None and random_undersampling is implemented because for
    # random undersampling the training and evaluation needs to be repeated. If any other balancing
    # methods should be used, use src.make_smote_dataset (for SMOTE) to create
    # a fitting csv-file with features that are already oversampled. Also, provide "None" for the
    # filter method because it is already considered in make_SMOTE_dataset_for_automl.)):
    if balancing_method == "None":
        sampler = [[X_train, y_train]]
    else:
        sampler = random_subsets(
            X=X_train,
            y=y_train,
            number_of_subsets=N_ITER,
            balancing_method=balancing_method,
        )

    conf_matrix_train, conf_matrix_test = [], []

    print(f"Original training size: {Counter(y_train)}")

    for idx, (X, y) in enumerate(sampler):
        print(f"Training size after balancing method {balancing_method}: {Counter(y)}")

        if not do_not_save_model:
            print(f"Fitting the classifier (iteration {idx + 1} of {N_ITER}): ...")
            # train model if None was loaded before
            cls.fit(X.iloc[:, :].values.astype("float"), y)

        # out-of-sample and insample predictions
        pred_test = cls.predict(X_test.values.astype("float"))
        pred_train = cls.predict(X.iloc[:, :].values.astype("float"))

        # clfs = {key: val["sklearn_classifier"] for key, val in cls.show_models().items()}

        # evaluate model performance
        conf_matrix_train.append(ConfusionMatrix(truth=y, prediction=pred_train, classes=[0, 1]))
        conf_matrix_test.append(ConfusionMatrix(truth=y_test, prediction=pred_test, classes=[0, 1]))

    df = evaluate_results(df=None)
    df = evaluate_results(
        confusion_matrix_train=conf_matrix_train,
        confusion_matrix_test=conf_matrix_test,
        name="automl",
        df=df,
    )

    saved = False
    while saved is False:
        # check wether directory exists
        if not isdir(saveto):
            saveto = input(f"Path {saveto} was not found. Suggest another path: ")
        try:
            with open(f"{saveto}/{model_name}", "w") as file:
                df.to_csv(file)
            # save model:
            if balancing_method != "undersampling" and not do_not_save_model:
                with open(f"{saveto}/{model_name}".replace(".csv", ".pickle"), "wb") as f:
                    pickle.dump(cls, f)

        except (FileNotFoundError, IsADirectoryError, PermissionError):
            pass
        else:
            saved = True


if __name__ == "__main__":
    routine()
