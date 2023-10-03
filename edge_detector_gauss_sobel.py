# -*- coding: utf-8 -*-
"""
@author: juste
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from convo import make_kernel
from PIL import Image


def gaussFilter(img_in, ksize, sigma):
    kernel = make_kernel(ksize, sigma)
    return kernel, np.int32(convolve(img_in, kernel))


def sobel(img_in):
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_flipped = np.transpose(kernel)
    return np.int32(convolve(img_in, kernel_flipped)), np.int32(
        convolve(img_in, kernel)
    )


def gradientAndDirection(gx, gy):
    return np.int32(np.sqrt(np.multiply(gx, gx) + np.multiply(gy, gy))), np.float64(
        np.arctan2(gy, gx)
    )


def convertAngle(angle):
    angle = 360.0 * angle / (2.0 * np.pi)
    if angle > 180:
        angle = angle - np.floor(angle / 180) * 180
    ret_angle = 0
    maxDist = 9999.0
    if np.abs(angle - 0.0) < maxDist or np.abs(angle - 180.0):
        maxDist = np.abs(angle - 0)
        if np.abs(angle - 180.0) < maxDist:
            maxDist = np.abs(angle - 180.0)
        ret_angle = 0
    if np.abs(angle - 135.0) < maxDist:
        maxDist = np.abs(angle - 135.0)
        ret_angle = 135
    if np.abs(angle - 90.0) < maxDist:
        maxDist = np.abs(angle - 90.0)
        ret_angle = 90
    if np.abs(angle - 45.0) < maxDist:
        maxDist = np.abs(angle - 45.0)
        ret_angle = 45
    return ret_angle


def maxSuppress(g, theta):
    islocal = np.zeros_like(g)
    for i in range(g.shape[0] - 1):
        for j in range(g.shape[1] - 1):
            if i == 0 or j == 0:
                continue

            angle = convertAngle(theta[i, j])
            maxDist = 9999
            searchdir = [0, 0]
            if np.abs(angle - 0) < maxDist:
                maxDist = np.abs(angle - 0)
                searchdir = [0, 1]
            if np.abs(angle - 90) < maxDist:
                maxDist = np.abs(angle - 90)
                searchdir = [1, 0]
            if np.abs(angle - 135) < maxDist:
                maxDist = np.abs(angle - 135)
                searchdir = [1, 1]
            if np.abs(angle - 45) < maxDist:
                maxDist = np.abs(angle - 45)
                searchdir = [-1, +1]

            if (
                g[i, j] >= g[i + searchdir[0], j + searchdir[1]]
                and g[i, j] >= g[i - searchdir[0], j - searchdir[1]]
            ):
                islocal[i, j] = g[i, j]

    return np.int32(islocal)


def hysteris(max_sup, t_low, t_high):
    treshimg = np.zeros_like(max_sup)
    treshimg[np.where(max_sup > t_low)] = 1
    treshimg[np.where(max_sup > t_high)] = 2

    max_sup = np.zeros_like(max_sup)
    for i in range(max_sup.shape[0]):
        for j in range(max_sup.shape[1]):
            if treshimg[i, j] == 2:
                max_sup[i, j] = 255
                for k in range(-1, 1 + 1):
                    for l in range(-1, 1 + 1):
                        if (
                            i + k >= max_sup.shape[0]
                            or i + k < 0
                            or j + l >= max_sup.shape[1]
                            or j + l < 0
                        ):
                            continue
                        if treshimg[i + k, j + l] == 1:
                            max_sup[i + k, j + l] = 255

    return max_sup


def canny(img):
    kernel, gauss = gaussFilter(img, 5, 2)
    gx, gy = sobel(gauss)
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gx, "gray")
    plt.title("gx")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, "gray")
    plt.title("gy")
    plt.colorbar()
    plt.show()

    g, theta = gradientAndDirection(gx, gy)
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(g, "gray")
    plt.title("gradient magnitude")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title("theta")
    plt.colorbar()
    plt.show()
    fig = plt.figure(figsize=(10, 5))

    maxS_img = maxSuppress(g, theta)
    plt.imshow(maxS_img, "gray")
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result


if __name__ == "__main__":
    img = np.float64(np.array(Image.open("")))[:, :, 0]
    canny(img)
