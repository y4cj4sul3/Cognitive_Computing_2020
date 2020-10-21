import cv2
import numpy as np


def gabor_texture(img, filters, gx=1, gy=1):
    # parametrs
    height, width = img.shape[:2]
    if gx == -1:
        gx = height
    if gy == -1:
        gy = width

    # Convolution
    feature_vec = []
    for kernel in filters:
        conv = cv2.filter2D(img, -1, kernel)

        # chunk into grids
        for i in range(gx):
            for j in range(gy):
                grid = conv[i*height//gx:(i+1)*height//gx, j*width//gy:(j+1)*width//gy]
                # Feature vector
                feature_vec.append(np.mean(grid))
                feature_vec.append(np.std(grid))

    return np.array(feature_vec)


def genGaborFilters(ksize=127, K=6, S=4, U_h=0.4, U_l=0.05):

    a = (U_h/U_l)**(1/(S-1))
    # print(a)

    # gamma = sigma_x / sigma_y = sigma_v / sigma_u
    sigma_u = ((a-1)*U_h)/((a+1)*np.sqrt(2*np.log(2)))
    #sigma_v = np.tan(np.pi/(2*K))*(U_h-2*np.log(sigma_u**2/U_h))*(2*np.log(2)-((2*np.log(2)*sigma_u)/U_h)**2)**(-1/2)
    sigma_v = np.tan(np.pi/(2*K))*np.sqrt(U_h**2/(2*np.log(2))-sigma_u**2)
    gamma = sigma_v / sigma_u

    sigma_base = 1/(2*np.pi*sigma_u)
    # wavelength = 1/freq ?
    lambd_base = 1/U_h
    # print(sigma_u, sigma_v, sigma_base)

    filters = []

    for m in range(S):
        lambd = lambd_base / a**(-m)
        sigma = sigma_base / a**(-m)
        # print(lambd, sigma, gamma)

        for n in range(K):
            theta = n * np.pi / K

            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, cv2.CV_32F)
            filters.append(kernel)

    return np.array(filters)
