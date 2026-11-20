"""
This job takes an existing keras-model (h5-file; for example model from src.run_autokeras) and
trains and/or tests it under several conditions.
"""

from abc import ABC
import gc
from os import path, listdir, mkdir
from os.path import isdir
from collections import Counter
from random import seed, shuffle
import pickle
import click

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.data import Dataset as tf_Dataset
from tensorflow.compat.v1.keras import backend as K
import autokeras as ak
from keras.metrics import (
    Recall,
    Precision,
    BinaryAccuracy,
)
from keras import optimizers
from keras.callbacks import (
    CSVLogger,
    BackupAndRestore,
    Callback,
)
from keras.models import load_model
import keras_tuner

from src.utils import RANDOM_STATE, plot_keras_history
from src.prepare_datasets import Dataset
from keras.callbacks import EarlyStopping

BATCH_SIZE = 16
EPOCHS = 100
PROJECT_NAME="KerasTuner"
METRIC = "f1"
LEARNING_RATE = 2.783e-6 # 2e-5
EPOCHS_TUNER_MIN = 5
LEARNING_TYPE = "Choice"
LEARNING_RATES = {"name": "learning_rate",
                  "values": list(np.logspace(-6, -4, 10).astype(float))}
CALLBACKS = [EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5, verbose=1,
                           mode="min", baseline=None, restore_best_weights=True)]
SAVETO = "results/"
LOAD_LABELS_FROM = "images/classified/list_of_labels.csv"

SIZE = (224, 224)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


################################

# Be careful with the F1-Score!

# The f1-score was actually removed from keras metrics because it does not make any sense as a
# metric: The metrics are calculated over each batch and then all the batch-values are averaged.
# But averaging the f1-score leads to wrong results. This becomes especially clear when the batch
# size is set to 1: Everytime a true negative, false negative or false positive occurs,
# the f1-score becomes zero!

# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
# pointed to a package that implements the f1-score correctly as a callback so that it can also be
# used in EarlyStopping!
# https://github.com/PhilipMay/mltb#module-keras
# The following class F1_Tuner is my own implementation for the f1-score which simply takes the
# averaged recall and the averaged precision from the log-file and calculates the f1-score.

class F1_Tuner(Callback):
    """
    Customized keras-callback which stops at the end of an epoch and calculates the f1-score from
    averaged recall and precision. In order to use this callback, Precision and Recall need to be
    metrics given to the keras-model.
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def on_epoch_end(self, epoch, logs=None):
        # calculating the f1-score on the whole validation dataset or calculating it from the
        # averaged recall/precision yields the same result!
        print(logs)
        if logs is None:
            print(
                "The f1-score could not be calculated because the metrics were not yet calculated!"
            )
            raise ValueError
        if any(["val" in key for key in logs.keys()]):
            if "val_recall" not in logs.keys():
                print("The recall needs to be a metric!")
                raise ValueError
            if "val_precision" not in logs.keys():
                print("The precision needs to be a metric!")
                raise ValueError
            rec = logs["val_recall"]
            prec = logs["val_precision"]
            logs["val_f1"] = (2 * rec * prec) / (rec + prec + self.epsilon)
        else:
            if "recall" not in logs.keys():
                print("The recall needs to be a metric!")
                raise ValueError
            if "precision" not in logs.keys():
                print("The precision needs to be a metric!")
                raise ValueError
            rec = logs["recall"]
            prec = logs["precision"]
            logs["f1"] = (2 * rec * prec) / (rec + prec + self.epsilon)


class CustomHyperband(keras_tuner.Hyperband, ABC):
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # mind this: https://github.com/keras-team/keras-tuner/issues/635
        # This Custom class resets the initial_epoch to 0 if the checkpoint from a previous round
        # does no longer exist. Effectively, the training then starts from epoch 0 again (since each
        # checkpoint was 700 MB large the Hyperband-Algorithm caused memory problems; but
        # calculating a few epochs again does not take that long)

        hp = trial.hyperparameters
        vals = hp.values
        if vals["tuner/initial_epoch"] > 0:
            if "tuner/trial_id" in vals:
                trial_id = vals["tuner/trial_id"]
                trial_folder = self._get_checkpoint_fname(trial_id)
                files = listdir("/".join(trial_folder.split("/")[:-1]))
                if any(map(lambda f: f.startswith("checkpoint.data"), files)):
                    # when  checkpoint exists, continue as normal
                    return super().run_trial(trial, *fit_args, **fit_kwargs)
                else:
                    # begin training from epoch 0 again
                    fit_kwargs["initial_epoch"] = 0
                    vals["tuner/initial_epoch"] = 0
                    hp.values = vals
                    trial.hyperparameters = hp
                    return super().run_trial(trial, *fit_args, **fit_kwargs)
            else:
                return super().run_trial(trial, *fit_args, **fit_kwargs)
        else:
            return super().run_trial(trial, *fit_args, **fit_kwargs)

    def _build_hypermodel(self, hp):
        model = super(keras_tuner.Hyperband, self)._build_hypermodel(hp)
        print("_build_hypermodel; hp = ", hp.values)
        if "tuner/trial_id" in hp.values and hp.values["tuner/initial_epoch"] > 0:
            trial_id = hp.values["tuner/trial_id"]
            # Load best checkpoint from this trial.
            model.load_weights(self._get_checkpoint_fname(trial_id))
        return model


def getattr_custom(list_of_names):
    """
    Function to extract optimizer.
    :param list_of_names: path to optimizer.
    :return:
    """
    length = len(list_of_names)
    return getattr_recursive(length, list_of_names)


def getattr_recursive(ind, list_of_names):
    ind -= 1
    if ind == 1:
        obj = getattr(optimizers, list_of_names[ind - 1])
    else:
        obj = getattr_recursive(ind, list_of_names)
    return getattr(obj, list_of_names[ind])


def get_optimizer(model):
    """
        Function to extract optimizer.
        :param model: model from which to extract optimizer.
    """
    modules = model.optimizer.__module__ + f".{model.optimizer._name}"
    modules = modules.split(".optimizers.")[-1]
    return getattr_custom(modules.split("."))


class MyHyperModel_final(keras_tuner.HyperModel, ABC):
    """
        Custom HyperModel class to build model with provided learning rate and optimizer of the
        model.
    """
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.initial_learning_rate = float(model.optimizer.learning_rate.numpy())
        self.initial_optimizer_params = self.model.optimizer.__dict__

    def build(self, hyperparam):

        hyperparam_class = getattr(hyperparam, LEARNING_TYPE)

        learning_rate = hyperparam_class(**LEARNING_RATES)

        # https://stats.stackexchange.com/questions/499013/adam-adaptive-optimizers-learning-rate-tuning
        # do not tune betas for Adam
        # retrieve used optimizer:
        opt = get_optimizer(self.model)
        optimizer = opt(learning_rate=learning_rate)

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[Recall(), Precision(), BinaryAccuracy()],
        )

        return self.model


def load_zip_file(zip_file, label_df, validation=False):
    """
    Load train or test sets.
    :param zip_file: Directory to zip-file created with src.prepare_datasets.
    :param label_df: Dataframe which assigns labels to the image directories.
    :param validation: If True, split the dataset 80/20.
    :return:
    """
    # load custom Dataset object
    if not path.exists(f"/tmp/{zip_file}"):
        dataset_file = Dataset.load_dataset_into_directory(zip_file)
        print(f"Loaded dataset from {zip_file} and unpacked images to /tmp.")
    else:
        dataset_file = pickle.load(open(f"/tmp/{zip_file}", "rb"))
    X = dataset_file.dataset
    # match labels to image names/directories
    if "." in label_df.index.values[0]:
        im_names = dataset_file.im_dirs
    else:
        im_names = [
            im.split("/")[-1].split(".")[0].split("#")[0] for im in dataset_file.im_dirs
        ]
    del dataset_file
    gc.collect()
    y = label_df.loc[im_names].values
    y = y.T[0]
    y = np.array([0 if y_iter == "False" or not y_iter else 1 for y_iter in y])
    print("Labels: ", Counter(list(y)))
    if validation:
        with CustomObjectScope({"f1_score": f1_score}, {"val_f1_score": f1_score}):
            # shuffle X_train and y_train:
            seed(RANDOM_STATE)
            inds = list(np.arange(len(y)))
            shuffle(inds)

            split_at = int(len(inds) * 0.8)
            inds_main, inds_val = np.array(inds)[:split_at], np.array(inds)[split_at:]
            # split X_train and y_train into two subsets
            if X.ndim == 3:
                X_val = X[inds_val, :, :]
                X = X[inds_main, :, :]
            else:
                X_val = X[inds_val, :, :, :]
                X = X[inds_main, :, :, :]
            y_val = np.array(y)[inds_val]
            print(f"Validation labels: ", Counter(list(y_val)))
            y_val = np.reshape(y_val, (len(y_val), 1))
            y = np.array(y)[inds_main]
            print(f"Train labels: ", Counter(list(y)))
            y = np.reshape(y, (len(y), 1))

            print(
                f"Fitting the model on X_train with shape {X.shape} and X_val with shape"
                f" {X_val.shape}."
            )
            # create tensorflow Dataset object
            # this Dataset-object is in memory:
            # https://medium.com/@acordier/tf-data-dataset-generators-with-parallelization-the-easy-way-b5c5f7d2a18
            X_val = tf_Dataset.from_tensor_slices(
                (tf.convert_to_tensor(X_val), tf.convert_to_tensor(y_val))
            )
            X_val.cache(filename="/tmp/X_val")
            X = tf_Dataset.from_tensor_slices(
                (tf.convert_to_tensor(X), tf.convert_to_tensor(y))
            )
            X.cache(filename="/tmp/X_train")
            im_names = np.array(im_names)[inds]
            im_train, im_val = im_names[:split_at], im_names[split_at:]
            gc.collect()
        return (X, X_val), (y, y_val), (im_train, im_val)
    X = tf_Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X), tf.convert_to_tensor(y))
    )

    gc.collect()
    return X, y, im_names


def tune_hyperparams(model, X_train, X_val, saveto, model_name, batch_size):
    """

    :param model: Keras-model.
    :param X_train: tensorflow dataset for training.
    :param X_val: tensorflow dataset for validation.
    :param saveto: Directory to save the histories of the training of several batch sizes to.
    :param model_name: The name of model.
    :param batch_size: If batch size is None, four batch sizes are tested. If this is not wanted,
    specify the batch size as an integer.
    :return:
    """

    # according to Bengio, 2012:
    # once the hyperparameter tuning is completed, tune batch size - this mainly has an impact on
    # the speed of convergence. It does not affect other hyperparameters except for the learning
    # rate. This is why after fitting different batch sizes, the learning rate should be refit.

    # check wether directory exists:
    if not isinstance(batch_size, int):
        save_batches_to = f"{saveto}/batch_sizes"
        if not isdir(save_batches_to):
            mkdir(save_batches_to)

        batch_sizes = [32, 16, 8, 4]
        fig = plt.figure(figsize=(20, 12))
        # Train the given model on four different batch sizes for the number of epochs defined in
        # the constants at the top of this file.
        for idx, batch_size in enumerate(batch_sizes):
            print(f"Starting training for batch size {batch_size}")
            csv_file_name = f"{save_batches_to}/history_{batch_size}.csv"
            csvlogger_batch = CSVLogger(csv_file_name, separator=",", append=True)

            backup_batch = BackupAndRestore(
                backup_dir=f"{save_batches_to}/backup_{batch_size}"
            )
            f1_tuner = F1_Tuner()
            # model.compile reinitializes the optimizer!
            model.compile(
                loss="binary_crossentropy",
                optimizer=model.optimizer,
                run_eagerly=False,
                metrics=[Recall(), Precision(), BinaryAccuracy()],
            )
            model.fit(
                x=X_train.batch(batch_size),
                epochs=EPOCHS,
                validation_data=X_val.batch(batch_size),
                callbacks=[f1_tuner, csvlogger_batch, backup_batch],
            )
            # https://stackoverflow.com/questions/59553964/incomplete-last-batch-in-tensorflow
            # (but shuffle is actually ignored when using tf.data.Dataset)
            # f1_callback needs to be before csv_batch so that val_f1 is included in the history
            ax = fig.add_subplot(2, 2, idx + 1)
            plot_keras_history(filename=csv_file_name, plot_quantity=METRIC, ax=ax)
            ax.set_title(f"Batch size {batch_size}")
        fig.savefig(f"{save_batches_to}/histories.png")

        # choose a batch size manually:
        chosen_batch_size = int(input(f"Choose batch size from {batch_sizes}: "))
        while chosen_batch_size not in batch_sizes:
            chosen_batch_size = int(
                input(
                    f"This batch size is not available. Choose batch size from {batch_sizes}: "
                )
            )
        print(f"Setting batch size to batch size {chosen_batch_size}.")
    else:
        chosen_batch_size = BATCH_SIZE

    obj = keras_tuner.Objective(
        name=f"val_{METRIC}", direction="min" if METRIC == "loss" else "max"
    )
    # Execute hyperband optimization to tune the learning rate
    print("factor is 2!")
    new_tuner = CustomHyperband(
        MyHyperModel_final(model=model, batch_size=chosen_batch_size),
        objective=obj,
        hyperband_iterations=1,
        project_name=saveto,
        max_epochs=EPOCHS,
        factor=2,
        seed=RANDOM_STATE,
    )
    new_tuner.search_space_summary()
    cb = [
        F1_Tuner(),
        CSVLogger(
            f"{saveto}/{model_name}_history_tuner.csv",
            separator=",",
            append=True,
        ),
        BackupAndRestore(backup_dir=f"{saveto}/backup"),
    ]
    new_tuner.search(
        x=X_train.batch(chosen_batch_size),
        validation_data=X_val.batch(chosen_batch_size),
        callbacks=cb,
        epochs=EPOCHS_TUNER_MIN,
        # see: https://github.com/keras-team/keras-tuner/issues/392
    )
    # load and return best performing model:
    best_hyperparams = new_tuner.get_best_hyperparameters()
    print(f"Best hyperparameters are: {best_hyperparams[0]}")
    new_tuner.results_summary()
    model = new_tuner.get_best_models(num_models=1)[0]

    return model, chosen_batch_size


def evaluate(model, X_test, y_test, ims_test, chosen_batch_size, save_file_to):
    """
    Function that evaluates the model performance. A file with results is saved to "save_file_to".
    :param model: keras model.
    :param X_test: tensorflow dataset for X_test.
    :param y_test: list. Truth labels.
    :param ims_test: list or np.array. Image directories corresponding to X_test and y_test.
    :param chosen_batch_size: integer with the batch size.
    :param save_file_to: string. Directory where to save the file with the results of the evaluation
    save to.
    :return:
    """

    # classify images:
    y_test_pred = model.predict(x=X_test.batch(chosen_batch_size))
    y_test_pred = y_test_pred.T[0]
    print(f"Predicted outcomes for {f'X_test='.split('=')[0]}.")

    if not isinstance(y_test[0], (int, float)):
        y_test = np.array([0 if y == "False" or not y else 1 for y in y_test])

    # create dataframe to save results to:
    prediction = pd.DataFrame(index=ims_test, columns=["truth", "prediction"])
    prediction.loc[:, "truth"] = y_test
    prediction.loc[:, "prediction"] = y_test_pred

    # save dataframe:
    with open(save_file_to, "w") as file:
        prediction.to_csv(file)
        print(f"Saved prediction to {save_file_to}.")


@click.command()
@click.option("--zip_file_train", type=str, default="None")
@click.option("--zip_file_test", type=str, default="None")
@click.option("--load_labels_from", type=str, default=LOAD_LABELS_FROM)
@click.option("--saveto", type=str, default=SAVETO)
@click.option("--model_dir", type=str)
@click.option("--tune_hyperparameters", type=bool, default=False)
@click.option("--evaluate_after_x_steps", type=str, default="None")
@click.option("--without_validation", type=bool, default=False)
# difference between RAM and GPU-RAM:
# https://stackoverflow.com/questions/51005147/keras-release-memory-after-finish-training-process
def routine(
    zip_file_train,
    zip_file_test,
    load_labels_from,
    saveto,
    model_dir,
    tune_hyperparameters,
    evaluate_after_x_steps,
    without_validation,
):
    """
    :param zip_file_train: Path to the zip-file containing the pickled Dataset-class.
    :param zip_file_test: Path to the zip-file containing the pickled Dataset-class.
    :param load_labels_from: Path to csv-file mapping the names of the images
    (directory dir1/dir2/image_abc.jpg --> image_abc) to labels "True"/"False" or 0/1 or True/False.
    :param saveto: Name of the directory to save outputs to.
    :param model_dir: Path to the keras model to train.
    :param tune_hyperparameters: bool. Specify wether to enter the block where the tuning is done.
     If the batch size in the config-file (the constants at the top of this file) is None, a
     training will be done on several batch sizes. After manually specifying a batch size, the
     learning-rate will be tuned.
     If tune_hyperparameters == False, the model will only be trained and tested on the conditions
     defined in the constants at the top of this file. Depending on wether zip_file_train is None
     or not the model will be trained and if a test-file is given also tested.
    :param evaluate_after_x_steps: str. If the string can be converted to an integer, this is the
    interval of epochs in which the model is evaluated on the test-dataset.
    :param without_validation: If True, train the model on the whole dataset without validating it.
    :return:
     History file(s)
     Trained model (if training dataset is specified)
     csv-file with pandas DataFrame - index: image names and column truth and prediction where truth
     is either 0 or 1 and prediction is a probability between 0 and 1.
    """
    if evaluate_after_x_steps != "None":
        evaluate_after_x_steps = int(evaluate_after_x_steps)

    valid_dir = False
    while not valid_dir:
        if isdir(saveto):
            valid_dir = True
        else:
            saveto = input(f"Path {saveto} was not found. Suggest another path: ")

    # load labels:
    label_df = pd.read_csv(load_labels_from, index_col=0)

    model_name = model_dir.split("/")[-1].split(".")[0]

    ####################################### LOAD MODEL #############################################

    if model_dir != "None":
        # load model
        with CustomObjectScope({"f1_score": f1_score}):
            model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    ################################ LOAD TRAINING SET #############################################
    if zip_file_train != "None":
        if without_validation and not tune_hyperparameters:
            X_train, y_train, ims_train = load_zip_file(
                zip_file=zip_file_train, label_df=label_df, validation=False
            )
        else:
            (X_train, X_val), (y_train, y_val), (ims_train, ims_val) = load_zip_file(
                zip_file=zip_file_train, label_df=label_df, validation=True
            )
        ############################### HYPERPARAMETER TUNING ######################################
        if tune_hyperparameters:
            model, chosen_batch_size = tune_hyperparams(
                model=model,
                X_train=X_train,
                X_val=X_val,
                saveto=saveto,
                model_name=model_name,
                batch_size=BATCH_SIZE,
            )

        else:
            ##################### TRAIN THE MODEL WITH CHOSEN HYPERPARAMETERS ######################

            chosen_batch_size = BATCH_SIZE
            print(f"Setting learning rate = {LEARNING_RATE}")
            opt = get_optimizer(model)
            optimizer = opt(learning_rate=LEARNING_RATE)

            print(f"Setting batch size = {chosen_batch_size}")

            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=[Recall(), Precision(), BinaryAccuracy()],
            )
        cb = [
            F1_Tuner(),
            CSVLogger(f"{saveto}/{model_name}_history.csv", separator=",", append=True),
            BackupAndRestore(backup_dir=f"{saveto}/backup"),
        ] + CALLBACKS

        if evaluate_after_x_steps != "None":

            X_test, y_test, ims_test = load_zip_file(
                zip_file=zip_file_test, label_df=label_df, validation=False
            )
            epochs = np.arange(EPOCHS + 1)

            dirs = listdir(saveto)
            for dir_ in dirs:
                if dir_.startswith(model_name):
                    last_part = dir_.split("_")[-1]
                    integer, file_ending = last_part.split(".")
                    if integer.isdigit() and file_ending == "h5":
                        # load model:
                        with CustomObjectScope({"f1_score": f1_score}):
                            model = load_model(
                                model_dir, custom_objects=ak.CUSTOM_OBJECTS
                            )
                        # redefine epochs
                        epochs = epochs[epochs > int(integer)]

            section = [
                epoch
                for epoch in epochs
                if (epoch - 1) % evaluate_after_x_steps == 0 and epoch != 1
            ]
            epochs = np.array_split(epochs, indices_or_sections=section)
            epochs = [[ar[0], ar[-1]] for ar in epochs if len(ar) > 0]

            print("This is the steps how the model will be trained: ")
            print(epochs)
            for start, end in epochs:
                if without_validation:
                    model.fit(
                        x=X_train.batch(chosen_batch_size),
                        callbacks=cb,
                        validation_data=None,
                        epochs=end,
                        initial_epoch=max([start - 1, 0]),
                    )
                else:
                    model.fit(
                        x=X_train.batch(chosen_batch_size),
                        callbacks=cb,
                        validation_data=X_val.batch(chosen_batch_size),
                        epochs=end,
                        initial_epoch=max([start - 1, 0]),
                    )
                # also save insample:
                save_file_to = f"{saveto}/{model_name}_prediction_insample_{end}.csv"

                evaluate(
                    model=model,
                    X_test=X_train,
                    y_test=y_train,
                    ims_test=ims_train,
                    chosen_batch_size=chosen_batch_size,
                    save_file_to=save_file_to,
                )

                save_file_to = f"{saveto}/{model_name}_prediction_{end}.csv"
                evaluate(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    ims_test=ims_test,
                    chosen_batch_size=chosen_batch_size,
                    save_file_to=save_file_to,
                )
                model.save(f"{saveto}/{model_name}_{end}.h5")
                print(f"Saved model to {saveto}/{model_name}_{end}.h5.")

        else:
            if without_validation:
                model.fit(
                    x=X_train.batch(chosen_batch_size),
                    callbacks=cb,
                    validation_data=None,
                    epochs=EPOCHS,
                )
            else:
                model.fit(
                    x=X_train.batch(chosen_batch_size),
                    callbacks=cb,
                    validation_data=X_val.batch(chosen_batch_size),
                    epochs=EPOCHS,
                )

            model.save(f"{saveto}/{model_name}.h5")
            print(f"Saved model to {saveto}/{model_name}.h5.")

            ############################## prepare test dataset ####################################

            # insample evaluation:
            save_file_to = f"{saveto}/{model_name}_prediction_insample.csv"

            evaluate(
                model=model,
                X_test=X_train,
                y_test=y_train,
                ims_test=ims_train,
                chosen_batch_size=chosen_batch_size,
                save_file_to=save_file_to,
            )
    else:
        chosen_batch_size = BATCH_SIZE

    if zip_file_test != "None":
        # load test-dataset
        X_test, y_test, ims_test = load_zip_file(
            zip_file=zip_file_test, label_df=label_df, validation=False
        )

        save_file_to = f"{saveto}/{model_name}_prediction.csv"

        evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ims_test=ims_test,
            chosen_batch_size=chosen_batch_size,
            save_file_to=save_file_to,
        )


if __name__ == "__main__":
    routine()
