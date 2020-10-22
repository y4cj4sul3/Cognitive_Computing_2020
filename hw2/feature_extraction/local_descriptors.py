import cv2
import itertools
import numpy as np
from sklearn.cluster import KMeans

import os
import pickle


def localDescriptor(img, detector, g=1):
    kps, dess = detector.detectAndCompute(img, None)

    if g == 1:
        return np.array(dess)
    else:
        height, width = img.shape[:2]
        gz_x, gz_y = height // g, width // g

        grids = [[] for i in range(g**2)]
        for kp, des in zip(kps, dess):
            i = kp.pt[0] // gz_x
            j = kp.pt[1] // gz_y
            grids[g*i + j].appned(des)

        return np.array(grids)


# def SIFT(img):
#     # make sure the version of opencv-python >= 4.4.0
#     sift = cv2.SIFT_create()
#     kps, des = sift.detectAndCompute(img, None)
#     # convert to normal list
#     kps = [kp.pt for kp in kps]

#     return np.array([kps, des])


# def ORB(img):
#     orb = cv2.ORB_create()
#     kps, des = orb.detectAndCompute(img, None)
#     # convert to normal list
#     kps = [kp.pt for kp in kps]

#     return np.array([kps, des])


def genCodeBook(folder, k, det='SIFT'):
    # load codebook
    filepath = os.path.join('codebook', '{}_{}_k{}.pickle'.format(folder, det, k))
    if os.path.isfile(filepath):
        print('load code book from file')
        with open(filepath, 'rb') as fp:
            codebook = pickle.load(fp)
        return codebook

    # load local descriptor
    des_filepath = os.path.join('codebook', '{}_{}.npy'.format(folder, det))
    if os.path.isfile(des_filepath):
        print('load local descriptor from file')
        flat_des = np.load(des_filepath, allow_pickle=True)
    else:
        # detector
        if det == 'ORB':
            detector = cv2.ORB_create()
        else:
            detector = cv2.SIFT_create()

        # load images
        descriptor = []
        for filename in glob.glob(os.path.join(folder, '/*/*.jpg')):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # detect feature
            _, des = detector.detectAndCompute(img, None)
            descriptor.append(des)

        # flatten
        flat_des = list(itertools.chain.from_iterable(descriptor))
        # save local descriptor
        np.save(des_filepath, flat_des)

    # k-means for code book
    codebook = KMeans(n_clusters=k, verbose=1).fit(flat_des)

    # save codebook
    with open(filepath, 'wb') as fp:
        pickle.dump(codebook, fp)

    return codebook


# def genCodeBook(name, des, k, path='features/', overwrite=False):
#     # load visual words
#     cb_filepath = os.path.join(path, name + '_cb.pickle')
#     if not overwrite and os.path.isfile(cb_filepath):
#         print('load code book from file')
#         with open(cb_filepath, 'rb') as fp:
#             codebook = pickle.load(fp)
#     else:
#         # flatten descriptors ( -> (# images x # keypoints) x d components)
#         flat_des = list(itertools.chain.from_iterable(des))
#         # k-means for code book
#         codebook = KMeans(n_clusters=k, verbose=1).fit(flat_des)
#         # save codebook
#         with open(cb_filepath, 'wb') as fp:
#             pickle.dump(codebook, fp)

#     return codebook


# def loadCodeBook(filepath):
#     cb_filepath = os.path.join(path, name + '_vw.pickle')
#     if os.path.isfile(cb_filepath):
#         print('load code book from file')
#         with open(cb_filepath, 'rb') as fp:
#             codebook = pickle.load(fp)
#         return codebook
#     else:
#         return None

def VLAD(name, des, k, codebook, grids=1, path='features/', overwrite=False):
    '''
        des: local descriptors (# images x # descriptors x d components)
        k: number of visual words

    '''
    # load from files
    filepath = os.path.join(path, name + '.npy')
    if not overwrite and os.path.isfile(filepath):
        print('load from file')
        return np.load(filepath, allow_pickle=True)

    # load code book
    # vw_filepath = os.path.join(path, name + '_vw.pickle')
    # if not overwrite and os.path.isfile(vw_filepath):
    #     print('load VM from file')
    #     with open(vw_filepath, 'rb') as fp:
    #         VWs = pickle.load(fp)
    # else:
    #     # flatten descriptors ( -> (# images x # keypoints) x d components)
    #     flat_des = list(itertools.chain.from_iterable(des))
    #     # k-means for visual words
    #     VWs = KMeans(n_clusters=k, verbose=1).fit(flat_des)
    #     # save VWs
    #     with open(vw_filepath, 'wb') as fp:
    #         pickle.dump(VWs, fp)

    # parameters
    d = des[0].shape[1]  # d-dimension
    # centroids
    c = codebook.cluster_centers_

    if grids == 1:
        VLADs = []
        # for each image
        for X in des:
            # X (# keypoints x d components)
            # get NN(X)
            nn_x = codebook.predict(X.tolist())

            # compute VLAD descriptors (D = k x d)
            V = np.zeros((k, d))
            for i in range(k):
                if np.sum(nn_x == i) > 0:
                    # sum of difference
                    V[i] = np.sum(X[nn_x == i, :] - c[i], axis=0)

            # PCA & ADC

            # L2 normalization
            V = V / np.linalg.norm(V, ord=2)

            VLADs.append(V)
    else:
        VLADs = []
        # for each image
        for X_g in des:
            V_g = []
            for X in X_g:
                # X (# keypoints x d components)
                # get NN(X)
                nn_x = codebook.predict(X.tolist())

                # compute VLAD descriptors (D = k x d)
                V = np.zeros((k, d))
                for i in range(k):
                    if np.sum(nn_x == i) > 0:
                        # sum of difference
                        V[i] = np.sum(X[nn_x == i, :] - c[i], axis=0)

                # PCA & ADC

                # L2 normalization
                V = V / np.linalg.norm(V, ord=2)
                V_g.append(V)

            VLADs.append(V_g)

    # save features into file
    VLADs = np.array(VLADs)
    np.save(filepath, VLADs)

    return VLADs
