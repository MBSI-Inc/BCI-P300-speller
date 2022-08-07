from tkinter import Y
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import glob
from mat4py import loadmat
from utils import butter_bandpass_filter


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


# In data.y you have all the markers that represent whether or not that time point was coincidental with the the
# stimulus that was meant to elicit a P300 response. If this value is 0, no stimulus was applied. If this value
# is 1 then a row/column stimulus was applied which do not correspond to the expected letter. If this value is
# 2, then the row/column which correspond to the target letter was present.

# In data.flash is composed of 4 fields, [sample point id, duration, stimulation, hit/nohit]. The first is the index
# time point, where the stimulation begins. The following is the duration in time points, a marker which means what
# stimulation was activated, and hit/nohit if the stimulation was of a target letter or not.
def get_data_from_file(filename):
    """
    Load data from a file
    - X contains extracted signal arrays (microVolt)
    - Y contains labels, not yet used
    - T 'trial', not yet used
    - F contains 'flash', arrays of 'point id', 'duration', 'stimulation', 'hit/nohit'

    Args:
        filename (str): filename

    Returns:
        tuple: (X, Y, T, F)
    """
    raw_data = loadmat(filename)
    useful_data = raw_data["data"].copy()
    x = useful_data["X"]
    y = useful_data["y"]
    t = useful_data["trial"]
    f = useful_data["flash"]

    return x, y, t, f


def prepare_data(X, Y, flash, start, stop, fs, undersample_bool):
    """
    Load data from file, with some cleaning.

    Args:
        X (_type_): contains extracted signal arrays
        Y (_type_): contains labels
        flash ([int, int, int, int]): arrays of 'timepoint id', 'duration', 'stimulation(row/column)', 'hit/nohit'
        start (float): start time
        stop (float): stop time
        fs (float): frequency sampling rate
        undersample_bool (bool): whether we undersample the data or not

    Returns:
        (X, Y): tuple
    """
    # Domain knowledge from the data
    N_SIMULATION = 12  # 6 rows, 6 columns (also number of flashes each round)
    N_REPEAT = 10  # Play 10 rounds of flashes each trial (meaning 120 flashes total) before choose the character
    N_CHANNEL = 8  # EEG channels
    N_TRIAL = 35  # But last trial is incompleted

    X = np.array(X)
    start_samples = int(start * fs)
    stop_samples = int(stop * fs)

    # Remove the samples of the last trial, since it's incomplete
    # Therefore we should have 34 * 120 = 4080 for length of flash (from 1 file)
    flash = flash[: len(flash) - (N_REPEAT * N_SIMULATION)]

    X_samples = np.zeros((len(flash), int(stop_samples - start_samples), N_CHANNEL))

    for i in range(len(flash)):
        event = flash[i][0]
        X_samples[i, :, :] = X[event + start_samples : event + stop_samples :]

    # The variable nohit/hit of flash[3] is 1/2, so we minus it 1 to become 0/1
    label = [i[3] - 1 for i in flash]

    y = np.array(label)
    X_samples = X_samples.reshape(X_samples.shape[0], X_samples.shape[1] * X_samples.shape[2])

    if undersample_bool:
        undersample = RandomUnderSampler(sampling_strategy="majority")
        X_samples, y = undersample.fit_resample(X_samples, y)

        # Make sure it divisible by 12, so I remove excess sample
        # Actually, may not make sense, ignore it for now
        # tmp = (len(X_samples) // 12) * 12
        # X_samples = X_samples[:tmp]
        # y = y[:tmp]

    return X_samples, y


def load_prepared_data(file, start, stop, fs, undersample_bool):
    """
    Load cleaned data from multiple files, ready to use.
    Return X, Y and flash. Flash is an array of ['timepoint id', 'duration', 'stimulation(row/column)', 'hit/nohit']

    Args:
        n_files (int): number of files to load. Must less than number of files available
        data_path (str): the folder location where it contain data files ("./p300dataset/*.mat")
        start (float): start time
        stop (float): stop time
        fs (float): frequency sampling rate
        undersample_bool (bool): whether we undersample the data or not

    Returns:
        (X, Y, flash): tuple
    """
    appX = []
    appY = []
    x, y, trials, flash = get_data_from_file(file)
    # Should be 1360 for undersampled, and 4080 for non-undersampled, for one data file.
    # paired X(rows of stacked 8x250 data), y(label) containing equal number of hits & nohits: expect to be 2*(35-1)*10=680 each? so sum to 1360
    X_clean, y_clean = prepare_data(x, y, flash, start, stop, fs, undersample_bool)
    appX.append(X_clean)
    appY.append(np.array(y_clean))

    return appX, appY, flash


# Unit testing
if __name__ == "__main__":
    print("==========load_data.py unit testing==========")
    start = 0
    stop = 1
    fs = 250
    data_path = "./p300dataset/*.mat"
    files = get_files(data_path)
    for file in files:
        x, y, t, f = get_data_from_file(file)
        print("FLASHHHHH")
        print(np.shape(f))
        print(f[:24])


# TODO: Filtering methods
