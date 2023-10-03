# -*- coding: utf-8 -*-
"""
@author: juste
"""

import numpy as np
import matplotlib.pyplot as plt


def ChirpSignal(
    samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool
):
    t = np.linspace(0, duration, int(samplingrate * duration))
    if linear:
        c = (freqto - freqfrom) / duration
        return np.sin(2 * np.pi * (freqfrom + (c / 2) * t) * t)
    if not linear:
        k = (freqto ** (1 / duration)) / freqfrom
        return np.sin((2 * np.pi * freqfrom) / np.log(k) * (k**t - 1))


chirp = ChirpSignal(200, 1, 1, 10, True)
plt.plot(np.linspace(0, 1, 200), chirp)
plt.show()
