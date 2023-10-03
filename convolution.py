# -*- coding: utf-8 -*-
"""
@author: juste
"""

from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    half_k = (ksize - 1) / 2.0
    grid_value = np.linspace(-half_k, half_k, ksize)
    x, y = np.meshgrid(grid_value, grid_value)
    kernel = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma))
    normalized_kernel = kernel / np.sum(np.sum(kernel))
    return normalized_kernel


def slow_convolve(arr, k):
    half_usizeUp = int(np.ceil(k.shape[0] / 2)) - 1
    half_vsizeUp = int(np.ceil(k.shape[1] / 2)) - 1
    half_usizeDown = int(np.floor(k.shape[0] / 2))
    half_vsizeDown = int(np.floor(k.shape[1] / 2))
    su = half_usizeUp - half_usizeDown
    sv = half_vsizeUp - half_vsizeDown
    M = np.pad(
        arr, [(half_usizeDown, half_usizeDown), (half_vsizeDown, half_vsizeDown)]
    )
    M_new = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for u in range(-half_usizeDown, half_usizeUp + 1):
                for v in range(-half_vsizeDown, half_vsizeUp + 1):
                    M_new[i, j] += (
                        k[u + half_usizeDown, v + half_vsizeDown]
                        * M[i - u + half_usizeDown + su, j - v + half_vsizeDown + sv]
                    )
    return M_new


if __name__ == "__main__":
    k = make_kernel(9, 2)
    print(k)
    image = np.float64(np.array(Image.open("")))[:, :, 0]
    original = Image.fromarray(np.uint8(image))
    original.save("")
    image_blurred = slow_convolve(image, make_kernel(3, 1))
    image_final = np.clip(image + 1.0 * (image - image_blurred), 0, 255)
    image_final = np.uint8(image_final)
    out = Image.fromarray(image_final)
    out.save("")
