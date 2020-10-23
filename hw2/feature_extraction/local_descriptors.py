import cv2
import itertools
import numpy as np
from sklearn.cluster import KMeans

import os
import glob
import pickle


def localDescriptor(img, detector, g=1):
    kps, dess = detector.detectAndCompute(img, None)
    
    if g == 1:
        return np.array(dess)
    else:
        height, width = img.shape[:2]
        gz_x, gz_y = width / g, height / g

        grids = [[] for i in range(g**2)]
        for kp, des in zip(kps, dess):
            grids[int(g*(kp.pt[0] // gz_x) + kp.pt[1] // gz_y)].append(des)

        return np.array(grids)


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
        for filename in glob.glob(os.path.join(folder, '*/*.jpg')):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # detect feature
            kp, des = detector.detectAndCompute(img, None)
            
            if len(kp) > 0:
                descriptor.append(des)

        # flatten
        flat_des = list(itertools.chain.from_iterable(descriptor))
        # save local descriptor
        np.save(des_filepath, flat_des)
        
    # print(np.shape(flat_des))

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
    vlad_filepath = os.path.join(path, 'vlad_' + name + '.npy')
    hist_filepath = os.path.join(path, 'hist_' + name + '.npy')
    if not overwrite and os.path.isfile(vlad_filepath) and os.path.isfile(hist_filepath):
        print('load pre-computed features from file')
        return np.load(vlad_filepath, allow_pickle=True), np.load(hist_filepath, allow_pickle=True)


#     if grids == 1:
#         des = np.reshape(des.shape[0])
    
#     des = np.array(des)
    
    # parameters
    # centroids
    c = codebook.cluster_centers_
    # d-dimension
#     print(np.shape(des))
#     print(np.shape(des[0][0]))
    d = 0
    for i in range(len(des)):
        for j in range(len(des[i])):
            if len(des[i][j]) > 0:
                d = np.shape(des[i][j])[-1]
                break
        if d > 0:
            break
    print(d)
    
#     if grids == 1:
#         VLADs = []
#         # for each image
#         for X in des:
#             # X (# keypoints x d components)
#             # get NN(X)
#             nn_x = codebook.predict(X.tolist())

#             # compute VLAD descriptors (D = k x d)
#             V = np.zeros((k, d))
#             for i in range(k):
#                 if np.sum(nn_x == i) > 0:
#                     # sum of difference
#                     V[i] = np.sum(X[nn_x == i, :] - c[i], axis=0)

#             # PCA & ADC

#             # L2 normalization
#             V = V / np.linalg.norm(V, ord=2)

#             VLADs.append(V)
#     else:
    VLADs = []
    hists = []
    # for each image
    for X_g in des:
        if grids == 1:
            X_g = np.array([X_g])
            
        V_g = []
        H_g = []
        for X in X_g:
            # X (# keypoints x d components)
            # V (k class x d components)
            X = np.array(X)
            V = np.zeros((k, d))
            H = np.zeros(k)
            
            if len(X) > 0:
                # get NN(X)
                nn_x = codebook.predict(X)

                # compute VLAD descriptors (D = k x d)

                for i in range(k):
                    if np.sum(nn_x == i) > 0:
                        # sum of difference
                        V[i] = np.sum(X[nn_x == i, :] - c[i], axis=0)
                        
                    # histogram
                    H[i] = np.sum(nn_x == i)

                # PCA & ADC (not implemented)

                # L2 normalization
                V = V / np.linalg.norm(V, ord=2)
                # normalization
                H = H / len(X)
                
            V_g.append(V)
            H_g.append(H)

        # flatten
        V_g = np.array(V_g)
        VLADs.append(V_g.flatten())
        H_g = np.array(H_g)
        hists.append(H_g.flatten())

    # save features into file
    VLADs = np.array(VLADs)
    np.save(vlad_filepath, VLADs)
    hists = np.array(hists)
    np.save(hist_filepath, hists)

    return VLADs, hists
