import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import glob
from mat4py import loadmat


def get_files(folder):
    """
    Get a list of files from folder location. Files type is *.mat.

    Args:
        folder (str): Folder path with file regex ("./p300dataset/*.mat")

    Returns:
        str[]: list of filename
    """
    files = glob.glob(folder)
    # files.pop(0) #drop the data badly structured
    # test_file = files.pop(0) #last subject for testing the network
    return files


def get_data_from_file(filename):
    """
    Load data from a file
    - X contains extracted signal arrays
    - Y contains labels, not yet used
    - T 'trial', not yet used
    - F contains 'flash', arrays of 'point id', 'duration', 'stimulation', 'hit/nohit'

    Args:
        filename (str): filename

    Returns:
        tuple: (X, Y, T, F)
    """
    channels = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "PO8", "Oz"]
    raw_data = loadmat(filename)
    useful_data = raw_data["data"].copy()
    X = useful_data["X"]
    Y = useful_data["y"]
    T = useful_data["trial"]
    F = useful_data["flash"]

    return X, Y, T, F


def prepare_data(X, Y, flash, start, stop, fs, undersample_bool):
    """
    Load data from file, with some cleaning.

    Args:
        X (_type_): contains extracted signal arrays
        Y (_type_): contains labels, not yet used
        flash ([int, int, int, int]): arrays of 'point id', 'duration', 'stimulation', 'hit/nohit'
        start (float): start time
        stop (float): stop time
        fs (float): frequency sampling rate
        undersample_bool (bool): whether we undersample the data or not

    Returns:
        (X, Y): tuple
    """
    X = np.array(X)
    start_samples = int(start * fs)
    stop_samples = int(stop * fs)
    X_samples = np.zeros((len(flash) - 120, int(stop_samples - start_samples), 8))

    for i in range(len(flash) - 120):
        event = flash[i][0]
        X_samples[i, :, :] = X[event + start_samples : event + stop_samples :]
    label = [i[3] - 1 for i in flash]

    LIMIT = 4080  # the last trial is incomplete
    y = np.array(label[:LIMIT])
    X_samples = X_samples.reshape(X_samples.shape[0], X_samples.shape[1] * X_samples.shape[2])

    if undersample_bool:
        undersample = RandomUnderSampler(sampling_strategy="majority")
        X_samples, y = undersample.fit_resample(X_samples, y)

    return X_samples, y


def load_prepared_data(n_files, data_path, start, stop, fs, undersample_bool):
    """
    Load cleaned data from multiple files, ready to use.

    Args:
        n_files (int): number of files to load. Must less than number of files available
        data_path (str): the folder location where it contain data files ("./p300dataset/*.mat")
        start (float): start time
        stop (float): stop time
        fs (float): frequency sampling rate
        undersample_bool (bool): whether we undersample the data or not

    Returns:
        (X, Y): tuple
    """
    appX = []
    appY = []
    files = get_files(data_path)
    if len(files) < n_files:
        print("ERROR: (load_data/load_prepared_data) Not enough files to load")

    for i in range(n_files):
        file = files[i]
        X, Y, trials, flash = get_data_from_file(file)
        # paired X(rows of stacked 8x250 data), y(label) containing equal number of hits & nohits: expect to be 2*(35-1)*10=680 each? so sum to 1360
        X_clean, y_clean = prepare_data(X, Y, flash, start, stop, fs, undersample_bool)
        appX.append(X_clean)
        appY.append(np.array(y_clean))

    return appX, appY


# TODO: Filtering methods
if __name__ == "__main__":
    print("==========load_data.py unit testing==========")
    files = get_files("./p300dataset/*.mat")
    print("No. of files = " + str(len(files)))
    print("adw")
    print("Files:", files)
