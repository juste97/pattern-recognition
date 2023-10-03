# -*- coding: utf-8 -*-
"""
@author: juste
"""

from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def load_sample(filename, duration=4 * 44100, offset=44100 // 10):
    signal = np.load(os.path.join(dir, filename))
    highest_abs_position = np.argmax(np.abs(signal))
    start = highest_abs_position + offset

    plt.figure()
    plt.plot(signal)

    end = start + duration
    signal_cropped = signal[start:end]
    return signal_cropped


def compute_frequency(signal, min_freq=20):
    fourier_mag = np.abs(np.fft.fft(signal))
    fourier_mag_cropped = fourier_mag
    min_freq = round(min_freq * ((1.0 / 44100.0) * len(signal)))
    fourier_mag_cropped[0:min_freq] = 0
    highest_abs_position = np.argmax(fourier_mag_cropped)
    freq = highest_abs_position / ((1.0 / 44100.0) * len(signal))
    return freq


if __name__ == "__main__":
    sig = load_sample("", duration=4 * 44100, offset=44100 // 10)
    x = compute_frequency(sig, min_freq=20)
    print(x)
