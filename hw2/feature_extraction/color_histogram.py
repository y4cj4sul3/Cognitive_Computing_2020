import cv2
import numpy as np


def color_histogram(img, n_bin, gx=1, gy=1):
    # parametrs
    height, width = img.shape[:2]

    # color
    hists = []
    for i in range(gx):
        for j in range(gy):

            grid = img[i*height//gx:(i+1)*height//gx, j*width//gy:(j+1)*width//gy]
            hist = cv2.calcHist([grid], [0, 1, 2], None, n_bin, [0, 256, 0, 256, 0, 256], accumulate=False)

            hists.append(hist)

    # normalization
    hists = np.array(hists)
    hists = hists / np.sum(hists)
    # flatten
    hists = hists.flatten()

    return hists
