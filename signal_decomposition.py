# -*- coding: utf-8 -*-
"""
@author: juste
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def SawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    results = np.zeros(samples)
    for t in range(samples):
        scaled_t = t / samples
        results[t] = amplitude / 2
        for k in range(1, k_max):
            results[t] = results[t] - (amplitude / math.pi) * (
                math.sin(2 * math.pi * k * frequency * scaled_t) / k
            )
    return results


func_values = np.transpose([SawtoothSignal(200, 2, 10000, 1)])
plt.figure()
plt.plot(np.arange(0, 1, 1 / 200), func_values)
plt.title("SawTooth")


def SquareSignal(samples: int, frequency: int, k_max: int):
    results = np.zeros(samples)
    for t in range(samples):
        scaled_t = t / samples
        for k in range(1, k_max):
            results[t] = results[t] + (4 / math.pi) * (
                math.sin(2 * math.pi * (2 * k - 1) * frequency * scaled_t)
            ) / (2 * k - 1)
    return results


func_values = np.transpose([SquareSignal(200, 2, 10000)])
plt.figure()
plt.plot(np.arange(0, 1, 1 / 200), func_values)
plt.title("Square")


def TriangleSignal(samples: int, frequency: int, k_max: int):
    results = np.zeros(samples)
    for t in range(samples):
        scaled_t = t / samples
        for k in range(0, k_max):
            results[t] = (
                results[t]
                + (8 / (math.pi) ** 2)
                * ((-1) ** k)
                * (math.sin(2 * math.pi * (2 * k + 1) * frequency * scaled_t))
                / (2 * k + 1) ** 2
            )
    return results


func_values = np.transpose([TriangleSignal(200, 2, 10000)])
plt.figure()
plt.plot(np.arange(0, 1, 1 / 200), func_values)
plt.title("Triangle")
