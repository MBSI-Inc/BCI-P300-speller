from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from imblearn.under_sampling import RandomUnderSampler
import mne
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer

# Our stuff
from utils import time_series
from analysis import print_predict_and_truth

# sampling frequency


def setup():
    FREQ_NUM = 8
    dir = "data/brandon" + str(FREQ_NUM) + "hz/brandon" + str(FREQ_NUM) + "hz"
    n_letter_repeats = 3
    num_markers = [1, 3, 9, 7, 2, 6, 8, 4, 5] * n_letter_repeats  # brandon139726845
    t_min = 0.2
    t_max = 0.4
    low = 1
    high = 10
    epochs, y, y_val = time_series(dir, num_markers, 5, t_min, t_max, low, high, plot=False)
    data = epochs
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    print("Data shape: ", data.shape, "Labels shape: ", y.shape, "Values shape: ", y_val.shape)
    # Set up the MNE info
    sf = 250
    ch_names = ["O1", "Oz", "O2", "Pz"]
    ch_types = ["eeg"] * 4
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sf)
    info.set_montage("standard_1020")
    print(info)
    mne.set_log_level(verbose="CRITICAL")
    return data, y, info, y_val, num_markers


def train_and_predict(data, y, info):
    # Create classification pipeline
    n_filter = 4
    clf = make_pipeline(
        Xdawn(n_components=n_filter), Vectorizer(), MinMaxScaler(), LDA(solver="eigen", shrinkage="auto")
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_repeats = 10
    # Do cross-validation
    preds = []
    final_y = []
    X_nonUS = data.reshape([data.shape[0], 4, data.shape[1] // 4])
    print("X_nonUS", X_nonUS.shape)
    X_nonUS_mne = mne.EpochsArray(X_nonUS, info)

    # Random undersample some nontargets
    for i in range(n_repeats):
        sampler = RandomUnderSampler(sampling_strategy="majority")
        X_samp, y_samp = sampler.fit_resample(data, y)
        # print("1",X_samp.shape)
        X_samp = X_samp.reshape([X_samp.shape[0], 4, X_samp.shape[1] // 4])
        # print("2", X_samp.shape)
        X_samp = mne.EpochsArray(X_samp, info)

        for train, test in cv.split(X_samp, y_samp):
            clf.fit(X_samp[train], y_samp[train])
            final_y += list(y_samp[test])
            preds += list(clf.predict(X_samp[test]))

            preds_nonUS = clf.predict(X_nonUS_mne)

        # Save model
        dump(clf, "model.joblib")
    print(type(preds_nonUS))
    print(preds_nonUS.shape)
    return y, preds_nonUS


def main():
    data, y, info, y_val, num_markers = setup()
    y, preds_nonUS = train_and_predict(data, y, info)
    print_predict_and_truth(y, y_val, preds_nonUS, num_markers)


if __name__ == "__main__":
    main()
