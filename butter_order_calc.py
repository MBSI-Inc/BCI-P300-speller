import matplotlib.pyplot as plt
import numpy as np
from utils import butter_bandpass
from scipy.signal import freqz

# Figure out the order for the filter (order = 5)
low = 1
high = 10
fs = 250
# Frequency response
plt.figure(1, figsize=(25, 8))
plt.clf()
for order in [1, 2, 3, 4, 5, 6]:
    b, a = butter_bandpass(low, high, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], "--", label="sqrt(0.5)")
plt.title("Butterworth filter frequency response", fontsize=30)
plt.xlabel("Frequency (Hz)", fontsize=20)
plt.xlim([0, 50])
plt.ylabel("Gain", fontsize=20)
plt.grid(True)
plt.legend(loc="best")
plt.show()

# Conclusion: order=5
