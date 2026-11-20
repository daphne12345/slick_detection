"""
This job executes the AutoKeras search and evaluates the performance of the best model which is
saved as a h5-file.
"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=no-value-for-parameter

from os.path import isdir

import click
import pandas as pd

import autokeras as ak
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from keras.metrics import Recall, Precision, BinaryAccuracy
from keras.callbacks import EarlyStopping, CSVLogger
from keras.models import load_model
import keras_tuner
from src.utils import RANDOM_STATE
from src.train_keras_model import load_zip_file, F1_Tuner, evaluate

SAVETO = "results/"
LOAD_LABELS_FROM = "images/classified/list_of_labels.csv"
METRIC = "f1"
EPOCHS = 0
MAX_TRIALS = 50

SIZE = (224, 224)


@click.command()
@click.option("--zip_file_train", type=str, default="None")
@click.option("--zip_file_test", type=str, default="None")
@click.option("--load_labels_from", type=str, default=LOAD_LABELS_FROM)
@click.option("--saveto", type=str, default=SAVETO)
@click.option("--balancing_method", type=click.Choice(["None", "undersampling"]))
@click.option(
    "--metric",
    type=click.Choice(["recall", "binary_accuracy", "precision", "loss", "f1"]),
    help="Choose one from: recall, binary_accuracy, precision, loss, f1",
    default=METRIC,
)
@click.option("--train_new", type=bool)
@click.option("--project_name", type=str)
@click.option("--model_dir", type=str, default="None")
def routine(
    zip_file_train,
    zip_file_test,
    load_labels_from,
    saveto,
    balancing_method,
    metric,
    train_new,
    project_name,
    model_dir,
):
    """
    Training and/or testing of Autokeras models. Outputs are saved to the directory saveto: In case
    of train_new=True the keras-model is saved as .h5 as well as a history-file as csv. In case data
    for testing is given, a csv file containing a DataFrame with the columns "truth" and
    "prediction" is saved containing the true labels as well as the predicted probability of
     belonging to the class 1.

    :param zip_file_train: Path to the zip-file containing the pickled Dataset-class
    (make all zip files with the class in src/prepare_datasets.py).
    :param zip_file_test: Path to the zip-file containing the pickled Dataset-class.
    :param load_labels_from: Path to csv-file mapping the names of the images
    (directory dir1/dir2/image_abc.jpg or image_abc) to labels "True"/"False" or 0/1 or True/False.
    :param saveto: Name of the directory to save outputs to.
    :param balancing_method: Choose from "None" or "undersampling".
    :param metric: Name of the objective of the training(val_ is automatically added as the
     prefix).
    :param train_new: If True, training is started or resumed. If train_new is False but
     zip_file_train != "None", a prediction on the training data is saved.
    :return:
    """

    ########################################### prepare images #####################################
    BATCH_SIZE = 32
    # check wether the directory saveto exists and ask for another one if not:
    valid_dir = False
    while not valid_dir:
        if isdir(saveto):
            valid_dir = True
        else:
            saveto = input(f"Path {saveto} was not found. Suggest another path: ")

    # load labels:
    label_df = pd.read_csv(load_labels_from, index_col=0)

    # if no model to load is given, the model_name and its directory to be saved to is determined
    # here
    if model_dir == "None":
        model_name = f"autokeras_{balancing_method}_{metric}"
        model_dir = f"{saveto}/{model_name}.h5"
    else:
        model_name = model_dir.split("/")[-1].split(".")[0]

    # load training data if provided:
    if zip_file_train != "None":

        (X_train, X_val), (y_train, _), (ims_train, _) = load_zip_file(
            zip_file_train, label_df=label_df, validation=True
        )

    ################################## define and train classifier #################################

    if train_new:
        metrics = [Recall(), Precision(), BinaryAccuracy()]
        if metric != "f1":
            obj = f"val_{metric}"
        else:
            # because f1 is a custom metric, keras needs to know wether to maximize or minimize it
            obj = keras_tuner.Objective("val_f1", direction="max")

        clf = ak.ImageClassifier(
            metrics=metrics,
            objective=obj,
            seed=RANDOM_STATE,
            max_trials=MAX_TRIALS,
            project_name=project_name,
        )
        print("Fitting the classifier...")
        # Train the ImageClassifier with EarlyStopping and also document its progress in a history
        # file as csv. F1 is implemented as a callback and not a metric here. If it was implemented
        # as a metric, f1 would be calculated for every training and validation epoch and then
        # averaged in the end. However, since f1 is a nonlinear metric, this leads to a different
        # result than calculating f1 on the whole dataset which is what is implemented in the
        # F1_Tuner callback.
        clf.fit(
            x=X_train,
            validation_data=X_val,
            callbacks=[
                F1_Tuner(),
                EarlyStopping(
                    monitor=f"val_{metric}",
                    min_delta=0.01,
                    patience=10,
                    verbose=1,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                ),
                CSVLogger(f"{saveto}/{model_name}_history.csv", separator=",", append=True),
            ],
            batch_size=BATCH_SIZE,
        )

        # get best performing model and save it:
        model = clf.export_model()
        model.save(f"{model_dir.replace('.h5', '')}")
        print(f"Saved model to {model_dir.replace('.h5', '')}.")

        # if no validation data is provided explicitly, AutoKeras trains the best model in the end
        # on the training and the validation data for the defined number of epochs. For better
        # control the validation data is explicitly provided and the subsequent training is done in
        # the following to get the untrained version of the best model:
        while True:
            try:
                # Do the final fit autokeras does internally (train again on the whole dataset):
                copied_fit_kwargs = {
                    "callbacks": [
                        F1_Tuner(),
                        CSVLogger(
                            f"{saveto}/{model_name}_history_final_fit.csv",
                            separator=",",
                            append=True,
                        ),
                    ],
                    "epochs": 0,  # save without training
                    "x": X_train.concatenate(X_val).batch(BATCH_SIZE),
                    "verbose": 1,
                }

                _, model, _ = clf.tuner.final_fit(**copied_fit_kwargs)
            except ResourceExhaustedError:
                BATCH_SIZE //= 2
                print(f"Try with new batch size {BATCH_SIZE}.")
            else:
                break

        # save model:
        model.save(f"{saveto}/{model_name}_{EPOCHS}_final_fit")
        print(f"Saved model to {saveto}/{model_name}_{EPOCHS}_final_fit.")

    if not train_new or zip_file_train == "None":
        # load previously saved classifier from disk
        model = load_model(f"{saveto}/{model_name}.h5", custom_objects=ak.CUSTOM_OBJECTS)
        print(f"Loaded model from {saveto}/{model_name}.h5.")

        # evaluate performance on training set
        save_file_to = f"{saveto}/{model_name}_prediction_insample.csv"
        evaluate(
            model=model,
            X_test=X_train,
            y_test=y_train,
            ims_test=ims_train,
            chosen_batch_size=BATCH_SIZE,
            save_file_to=save_file_to,
        )

    #################################### prepare test dataset ######################################
    if zip_file_test != "None":
        # load test set
        X_test, y_test, ims_test = load_zip_file(
            zip_file=zip_file_test, label_df=label_df, validation=False
        )
        # the model was either trained in this run (train_new is True) or loaded because
        # zip_file_train == "None"
        save_file_to = f"{saveto}/{model_name}_prediction.csv"
        evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ims_test=ims_test,
            chosen_batch_size=BATCH_SIZE,
            save_file_to=save_file_to,
        )


if __name__ == "__main__":
    routine()
