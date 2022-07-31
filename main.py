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

from load_data import load_prepared_data


def main():
    start = 0
    stop = 1
    fs = 250

    # Load data
    (appX, appY) = load_prepared_data(1, "./p300dataset/*.mat", start, stop, fs, True)
    (appX_nonUS, appY_nonUS) = load_prepared_data(1, "./p300dataset/*.mat", start, stop, fs, False)


if __name__ == "__main__":
    main()
