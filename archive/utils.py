from scipy.signal import welch, freqz, butter, filtfilt
import numpy as np

# For filtering the data
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def reshape_data_for_training(appX, appY, fs, n_channel):
    # Reshaped
    data_x = np.array([subject for subject in appX])  # 1360 flashes of 2000 samples
    print("SHAPEEE", np.shape(appY))
    data_y = np.array([np.array(subject) for subject in appY])
    print("SHAPEEE2", np.shape(appY))
    data_x = np.vstack(data_x)  # put
    data_x = data_x.reshape(
        data_x.shape[0], int(data_x.shape[1] / fs), int(data_x.shape[1] / n_channel)
    )  # reshape into 8 channels of 250 = 2000
    data_y = data_y.reshape(data_y.shape[0] * data_y.shape[1])
    print("SHAPEEE3", np.shape(appY))

    # X.shape is 1360x8x250. 8 channel of 250 sample(1s)
    return data_x, data_y
