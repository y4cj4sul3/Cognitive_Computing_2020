import numpy as np


def grid_color_moment(img, gx=5, gy=5):
    # parametrs
    height, width = img.shape[:2]

    # grid color moment
    gcm = []
    for i in range(gx):
        for j in range(gy):
            for k in range(3):
                grid = img[i*height//gx:(i+1)*height//gx, j*width//gy:(j+1)*width//gy, k]

                m1 = np.mean(grid)
                m2 = np.std(grid)
                m3 = np.mean((grid-m1)**3)
                m3 = np.sign(m3) * np.power(np.abs(m3), 1/3)

                gcm.append(m1)
                gcm.append(m2)
                gcm.append(m3)

    return gcm
