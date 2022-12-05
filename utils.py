import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import numpy as np

CH_LABELS = ["Fz", "C4", "Cz", "C3"]
sf = 250
n_ch = 4


def extract_epochs(sig, sig_times, event_times, t_min, t_max, sf):
    """Extracts epochs from signal
    Args:
        sig: EEG signal with the shape: (N_chan, N_sample)
        sig_times: Timestamp of the EEG samples with the shape (N_sample)
        event_times: Event marker times
        t_min: Starting time of the epoch relative to the event time
        t_max: End time of the epoch relative to the event time
        sf: Sampling rate
    Returns:
        (numpy ndarray): EEG epochs
    """
    offset_st = int(t_min * sf)
    offset_end = int(t_max * sf)
    epoch_list = []
    for i, event_t in enumerate(event_times):
        try:
            idx = np.argmax(sig_times > event_t)
            epoch_list.append(np.array(sig[:, idx + offset_st: idx + offset_end]))
            print("AAAAAAAAAAAAA i idx", np.array(sig[:, idx + offset_st: idx + offset_end]).shape, i, idx)
        except:
            None
    return np.array(epoch_list)


def reject_bad_epochs(epochs, p2p_max):
    """Rejects bad epochs based on a peak-to-peak amplitude criteria
    Args:
        epochs: Epochs of EEG signal
        p2p_max: maximum peak-to-peak amplitude

    Returns:
        (numpy ndarray):EEG epochs
    """
    temp = epochs.reshape((epochs.shape[0], -1))
    res = epochs[np.ptp(temp, axis=1) < p2p_max, :, :]
    print(f"{temp.shape[0] - res.shape[0]} epochs out of {temp.shape[0]} epochs have been rejected.")
    return res


def custom_filter(exg, lf, hf, sf, type):
    """
    Args:
        exg: EEG signal with the shape: (N_chan, N_sample)
        lf: Low cutoff frequency
        hf: High cutoff frequency
        sf: Sampling rate
        type: Filter type, 'bandstop' or 'bandpass'
    Returns:
        (numpy ndarray): Filtered signal (N_chan, N_sample)
    """
    N = 4
    b, a = signal.butter(N, [lf / sf, hf / sf], type)
    return signal.filtfilt(b, a, exg)


def get_markers_and_ts(temp_markers, label_nontargets, label_targets):
    # Gets the markers timestamps but IN ORDER of appearance
    y = []
    y_val = []
    for ind in range(temp_markers["Code"].shape[0]):
        if temp_markers["Code"].to_numpy()[ind] in label_targets:
            y += [1]
        else:
            y += [0]
        a = temp_markers["Code"].to_list()
        y_val += [int(a[ind][-1])]
    ts_markers = temp_markers["TimeStamp"].to_numpy()
    return y, ts_markers, y_val


def format_data_for_prediction(dir, t_min, t_max, lf, hf,):
    exg_filename = dir + "_ExG.csv"
    marker_filename = dir + "_Marker.csv"

    # Import data
    exg = pd.read_csv(exg_filename)
    markers = pd.read_csv(marker_filename)
    ts_sig = exg["TimeStamp"].to_numpy()

    sig = exg[["ch" + str(i) for i in range(1, n_ch + 1)]].to_numpy().T
    sig -= sig[0, :] / 2
    filt_sig = custom_filter(sig, 45, 55, sf, "bandstop")
    filt_sig = custom_filter(filt_sig, lf, hf, sf, "bandpass")

    all_ts_markers = markers["TimeStamp"].to_numpy()
    y_val = markers["Code"]
    # Remove the "sw_" part from marker code
    y_val = np.array(list(map(lambda marker: marker[3:], y_val)))

    epochs = extract_epochs(filt_sig, ts_sig, all_ts_markers, t_min, t_max, sf)
    return epochs, y_val


def time_series(dir, num_markers, n_letter_repeats, t_min, t_max, lf, hf, plot=False):
    exg_filename = dir + "_ExG.csv"
    marker_filename = dir + "_Marker.csv"

    # Import data
    exg = pd.read_csv(exg_filename)
    markers = pd.read_csv(marker_filename)
    ts_sig = exg["TimeStamp"].to_numpy()

    all_ts_markers = []
    all_y = []
    all_y_val = []
    for num_ind in range(len(num_markers)):
        num = num_markers[num_ind]
        label_targets = "sw_" + str(num)
        label_nontargets = ["sw_" + str(i) for i in range(1, 10) if i != num]

        p2p_max = 70  # rejection criteria, units in uV
        temp_markers = markers.iloc[num_ind * n_letter_repeats * 9: (num_ind + 1) * n_letter_repeats * 9, :]
        y, ts_markers, y_val = get_markers_and_ts(
            temp_markers=temp_markers, label_nontargets=label_nontargets, label_targets=label_targets
        )
        all_ts_markers += ts_markers.tolist()
        all_y += y
        all_y_val += y_val
    all_y = np.array(all_y).copy()
    all_y_val = np.array(all_y_val).copy()
    all_ts_markers = np.array(all_ts_markers).copy()

    sig = exg[["ch" + str(i) for i in range(1, n_ch + 1)]].to_numpy().T
    sig -= sig[0, :] / 2
    filt_sig = custom_filter(sig, 45, 55, sf, "bandstop")
    filt_sig = custom_filter(filt_sig, lf, hf, sf, "bandpass")

    epochs = extract_epochs(filt_sig, ts_sig, all_ts_markers, t_min, t_max, sf)
    #   epochs = reject_bad_epochs(epochs, p2p_max)
    if plot:
        erp_target = epochs[(all_y == 1)].mean(axis=0)
        erp_nontarget = epochs[(all_y == 0)].mean(axis=0)

        t = np.linspace(t_min, t_max, erp_target.shape[1])
        fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=2)
        for i, ax in enumerate(axes.flatten()):
            ax.plot(t, erp_nontarget[i, :], label="Non-target")
            ax.plot(t, erp_target[i, :], "tab:orange", label="Target")
            ax.plot([0, 0], [-30, 30], linestyle="dotted", color="black")
            ax.set_ylabel("\u03BCV")
            ax.set_xlabel("Time (s)")
            ax.set_title(CH_LABELS[i])
            ax.set_ylim([-10, 20])
            ax.legend()
        plt.show()

    return epochs, all_y, all_y_val
