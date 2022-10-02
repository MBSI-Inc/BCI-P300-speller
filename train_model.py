from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from imblearn.under_sampling import RandomUnderSampler
import mne
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
import pandas as pd

# Our stuff
from utils import format_data_for_prediction, time_series
from analysis import print_predict_and_truth

sf = 250
t_min = 0.2
t_max = 0.4
low = 1
high = 10
ch_names = ["O1", "Oz", "O2", "Pz"]
ch_types = ["eeg"] * 4


def predict_for_pygame(dir, model_name):
    # Setup parameter
    mne.set_log_level(verbose="CRITICAL")
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sf)
    info.set_montage("standard_1020")

    # Format data
    data, y_val = format_data_for_prediction(dir, t_min, t_max, low, high)
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    data = data.reshape([data.shape[0], 4, data.shape[1] // 4])
    data_mne = mne.EpochsArray(data, info)

    # Load model
    clf = load(model_name)

    # Prediction
    preds = clf.predict(data_mne)
    ith = 0
    votes = y_val[ith * 45: (ith + 1) * 45][preds[ith * 45: (ith + 1) * 45] == 1]
    modes = pd.DataFrame(votes).value_counts().index.tolist()
    vote_result = modes[0][0]
    return vote_result


def setup():
    FREQ_NUM = 8
    dir = "data/brandon" + str(FREQ_NUM) + "hz/brandon" + str(FREQ_NUM) + "hz"
    n_break = 3  # Number of time of break or number of time press spacebar - 1
    n_letter_repeats = 5  # Number of time each character has to flash before a break / press spacebar
    num_markers = [1, 3, 9, 7, 2, 6, 8, 4, 5] * n_break  # brandon139726845
    epochs, y, y_val = time_series(dir, num_markers, n_letter_repeats, t_min, t_max, low, high, plot=False)
    print("Epoch shape", epochs.shape)
    # Epoch shape is (x, 4, y) where x = n_char * n_repeat * n_break (ex: 1215 = 9 * 5 * 3)
    # y is length of spelling signal
    data = epochs
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    print("Data shape: ", data.shape, "Labels shape: ", y.shape, "Values shape: ", y_val.shape)
    # Set up the MNE info

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
    preds = []  # Unused
    final_y = []  # Unused
    X_nonUS = data.reshape([data.shape[0], 4, data.shape[1] // 4])
    print("X_nonUS shape", X_nonUS.shape)
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
    # print(predict_for_pygame("data/default/default", "model.joblib"))
