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

import warnings

warnings.filterwarnings("ignore")

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

# Our stuff
from load_data import load_prepared_data


def reshape_data(appX, appY, fs, n_channel):
    # Reshaped
    data_x = np.array([subject for subject in appX])  # 1360 flashes of 2000 samples
    data_y = np.array([np.array(subject) for subject in appY])
    data_x = np.vstack(data_x)  # put
    data_x = data_x.reshape(
        data_x.shape[0], int(data_x.shape[1] / fs), int(data_x.shape[1] / n_channel)
    )  # reshape into 8 channels of 250 = 2000
    data_y = data_y.reshape(data_y.shape[0] * data_y.shape[1])

    # X.shape is 1360x8x250. 8 channel of 250 sample(1s)
    return data_x, data_y


def training(data_x, data_y, channels, fs):
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
    return preds


def histogram(preds):
    preds_set_hit = np.zeros(len(preds) // 12)
    for i in range(12):
        # rows = hit_seq[i::12]
        rows = preds[i::12]
        # set_hit = np.vstack((set_hit, rows))
        preds_set_hit = np.vstack((preds_set_hit, rows))
    plt.hist(sum(preds_set_hit[1:, :]))
    print("before transpose")
    print(np.shape(preds_set_hit))
    print(sum(preds_set_hit[1:, :]))
    preds_set_hit = np.transpose(preds_set_hit[1:, :])
    print("after transpose", np.shape(preds_set_hit))
    print(preds_set_hit[0:10])
    plt.show()


def get_flashed_column_row(flash):
    """
    _summary_

    Args:
        flash ([int, int, int, int]): arrays of 'timepoint id', 'duration', 'stimulation(row/column)', 'hit/nohit'
    """
    id_seq = [each[2] for each in flash]
    # print(len(id_seq))
    # print(id_seq[0:20])
    # print(id_seq[0:20:2])
    # print(id_seq[0::1000])
    set_seq = np.zeros(4200 // 12)
    for i in range(12):
        rows = id_seq[i::12]
        set_seq = np.vstack((set_seq, rows))
    # print(np.shape(set_seq))
    # plt.hist(sum(set_seq[1:,:]))
    set_seq = np.transpose(set_seq[1:, :])
    # print(np.shape(set_seq))
    # print(set_seq[0:5])

    hit_seq = [each[3] - 1 for each in flash]
    # print(len(id_seq))
    set_hit = np.zeros(4200 // 12)
    for i in range(12):
        rows = hit_seq[i::12]
        set_hit = np.vstack((set_hit, rows))
    # print(np.shape(set_hit))
    plt.hist(sum(set_hit[1:, :]))
    set_hit = np.transpose(set_hit[1:, :])
    # print(np.shape(set_hit))
    print(set_hit[0:10])


def main():
    start = 0
    stop = 1
    fs = 250
    data_path = "./p300dataset/*.mat"
    channels = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]

    # Load data
    # appX appy extended for each datafile
    (appX, appY) = load_prepared_data(1, data_path, start, stop, fs, False)
    # (appX_nonUS, appY_nonUS) = load_prepared_data(1, data_path, start, stop, fs, False)
    print("AppX shape", np.shape(appX))
    print("AppY shape", np.shape(appY))

    # Reshape
    data_x, data_y = reshape_data(appX, appY, fs, len(channels))
    print(data_x.shape, data_y.shape)

    # Training and test
    preds = training(data_x, data_y, channels, fs)
    print(preds[:36])

    histogram(preds)


if __name__ == "__main__":
    main()
