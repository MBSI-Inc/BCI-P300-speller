from doctest import Example
import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
from mat4py import loadmat
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
import seaborn as sns

# from pyriemann.estimation import Covariances
# from pyriemann.tangentspace import TangentSpace
from mne.decoding import Vectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
)
from mne import io, compute_raw_covariance, read_events, pick_types, Epochs
from mne.preprocessing import Xdawn
from mne.viz import plot_epochs_image
import mne
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

# Our stuff
from load_data import load_prepared_data, get_data_from_file, get_files
from utils import reshape_data_for_training


def training(data_x, data_y, channels, fs):
    """
    Training the data and return predictions, model.

    Args:
        data_x (_type_): signal
        data_y (_type_): a bunch of 0 and 1, signal hit or no hit
        channels (_type_):
        fs (_type_):

    Returns:
        preds, clf
    """
    # Set log level so it's less clutter
    old_log_level = mne.set_log_level(verbose="warning", return_old_level=True)

    info = mne.create_info(channels, fs, "eeg", verbose="warning")
    info.set_montage("standard_1020")
    X_mne = mne.EpochsArray(data_x, info)

    # acc, cm = xdawn_results(model, X_mne, y)

    # def xdawn_results(model, data, label):
    # xdawn_results(model=LDA(shrinkage='auto', solver='eigen'), data -> X_mne, label -> y)
    n_filter = 15  # the more filters the better, but longer
    clf = make_pipeline(
        Xdawn(n_components=n_filter), Vectorizer(), StandardScaler(), LDA(shrinkage="auto", solver="eigen")
    )  # model here

    # Cross validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Do cross-validation
    preds = np.empty(len(data_y))
    for train_ind, test_ind in cv.split(X_mne, data_y):
        clf.fit(X_mne[train_ind], data_y[train_ind])
        preds[test_ind] = clf.predict(X_mne[test_ind])

    mne.set_log_level(verbose=old_log_level)
    return preds, clf


def testing(preds, flash):
    """
    Compare the result with truth. We don't compare it with Y because it meaningless.
    We will clean the predictions, get the top 2 which should give a better accuracy

    Args:
        preds (_type_): _description_
        flash ([int, int, int, int]): arrays of 'timepoint id', 'duration', 'stimulation(row/column)', 'hit/nohit'

    Returns:
        acc: accuracy
    """
    N_TRAINING_SAMPLE = 4080  # 34*12*10
    N_FLASH_SAMPLE = 4200  # 35*12*10
    N_SIMULATION = 12  # 6 rows, 6 columns
    N_REPEAT = 10
    N_TRIAL = 34  # Actually 35 but last one not finished

    # Reshape predict hit
    preds_set_hit = np.zeros(N_TRAINING_SAMPLE // N_SIMULATION)
    for i in range(N_SIMULATION):
        rows = preds[i::N_SIMULATION]
        preds_set_hit = np.vstack((preds_set_hit, rows))
    preds_set_hit = np.transpose(preds_set_hit[1:, :])
    # preds_set_hit example:
    # [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
    # [0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
    # [1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]]

    # Get flashed row/column id sequence
    id_seq = [each[2] for each in flash]
    set_seq = np.zeros(N_FLASH_SAMPLE // N_SIMULATION)
    for i in range(N_SIMULATION):
        rows = id_seq[i::N_SIMULATION]
        set_seq = np.vstack((set_seq, rows))
    set_seq = np.transpose(set_seq[1:, :])
    # set_seq example
    # [[11.  5.  7.  6.  8.  4. 12.  1. 10.  2.  9.  3.]
    # [ 8.  5. 11.  2. 10.  4.  7.  3. 12.  6.  9.  1.]
    # [ 8.  5. 10.  2. 11.  1.  9.  6. 12.  4.  7.  3.]
    # [12.  6. 10.  1. 11.  5.  9.  4.  8.  2.  7.  3.]
    # [ 8.  1. 11.  3. 10.  5.  7.  4.  9.  2. 12.  6.]]

    # Get flashed hit sequence (truth value)
    hit_seq = [each[3] - 1 for each in flash]
    set_hit = np.zeros(N_FLASH_SAMPLE // N_SIMULATION)
    for i in range(N_SIMULATION):
        rows = hit_seq[i::N_SIMULATION]
        set_hit = np.vstack((set_hit, rows))
    set_hit = np.transpose(set_hit[1:, :])
    # set_hit example
    # [[0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0.]
    # [1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    # [1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    # [0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0.]

    # Get truth coordinates
    set_coords = [seq_i[hit_i == 1] for seq_i, hit_i in zip(set_seq, set_hit)]
    for i in range(len(set_coords)):
        # Make sure 1-6 is in [0], 7-12 is in [1] for easier comparision
        if set_coords[i][0] > 6:
            set_coords[i] = np.array([set_coords[i][1], set_coords[i][0]])
    set_coords = np.array(set_coords)
    # set_coords example
    # [[4. 8.] [4. 8.] [4. 8.] ... [4. 8.]
    # [3. 9.] [3. 9.] ... [3. 9.]]

    # Get predict coordinates
    preds_set_coords = [seq_i[hit_i == 1] for seq_i, hit_i in zip(set_seq, preds_set_hit)]
    preds_set_coords = np.array(preds_set_coords)

    # Pick the best 2 + clean up and compare
    preds_set_coords2 = np.array([0, 0])
    true_set_coords2 = np.array([0, 0])
    for letter_id in range(N_TRIAL):
        modes = (
            pd.DataFrame(
                [
                    elt
                    for lst in preds_set_coords[letter_id * N_REPEAT : letter_id * N_REPEAT + N_REPEAT]
                    for elt in lst
                ]
            )
            .value_counts()
            .index.tolist()
        )
        preds_set_coords2 = np.vstack((preds_set_coords2, [modes[0][0], modes[1][0]]))
        true_set_coords2 = np.vstack((true_set_coords2, set_coords[letter_id * N_REPEAT]))
    for i in range(len(preds_set_coords2)):
        # Make sure 1-6 is in [0], 7-12 is in [1] for easier comparision
        if preds_set_coords2[i][0] > 6:
            preds_set_coords2[i] = np.array([preds_set_coords2[i][1], preds_set_coords2[i][0]])

    final_acc = np.sum(preds_set_coords2 == true_set_coords2) / np.size(preds_set_coords2 == true_set_coords2)
    return final_acc


def main():
    start = 0
    stop = 1
    fs = 250
    data_path = "./p300dataset/*.mat"
    channels = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

    files = get_files(data_path)
    for i in range(1):  # Number of files to load
        # Load data
        print("ID", i)
        appX, appY, flash = load_prepared_data(files[i], start, stop, fs, False)
        print("AppX shape", np.shape(appX))
        print("AppY shape", np.shape(appY))
        data_x, data_y = reshape_data_for_training(appX, appY, fs, len(channels))

        # Training and test
        preds, clf = training(data_x, data_y, channels, fs)
        acc = testing(preds, flash)
        print("ACC", acc)


if __name__ == "__main__":
    main()
