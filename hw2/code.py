import cv2
import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

import os
import glob
import pickle

from feature_extraction.color_histogram import color_histogram
from feature_extraction.grid_color_moment import grid_color_moment
from feature_extraction.gabor_texture import gabor_texture, genGaborFilters
from feature_extraction.local_descriptors import localDescriptor, VLAD, genCodeBook

# parameters
NUM_OF_CATEGORY = 35
NUM_OF_RELEVANT = 20

# categories
categories = [c.split('/')[1] for c in glob.glob('data/*')]
categories.sort()

def getCategory(index):
    return categories[index // NUM_OF_RELEVANT]

def getFilename(index):
    category = getCategory(index)
    return category, os.path.join('data', category, category + '_' + str(index % NUM_OF_RELEVANT + 1) + '.jpg')

# Feature extraction
def extract_feature(name, method, cs='RGB', path='features/', overwrite=False, **kwargs):
    
    # load from files
    filepath = os.path.join(path, name + '.npy')
    if not overwrite and os.path.isfile(filepath):
        print('load from file')
        return np.load(filepath, allow_pickle=True)
    
    features = []
    for category in categories:
        for i in range(1, 21):

            # read image
            filename = os.path.join('data', category, category + '_' + str(i) + '.jpg')
            
            # color space
            img = cv2.imread(filename)
            if cs == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif cs == 'HSV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif cs == 'YCrCb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif cs == 'GRAY':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # extract feature
            feature = method(img, **kwargs)

            # store feature
            features.append(feature)
            
    # save features into file
    features = np.array(features)
    np.save(filepath, features)
    
    return features
    
# Grid Color Moment
def genGCM(cs, grids, ow=False):
    # filename
    name = 'gcm_g{}_{}'.format(grids, cs)

    # extract features
    feature = extract_feature(name, grid_color_moment, overwrite=ow, cs=cs, gx=grids, gy=grids)
    # shape: (grids)^2 * color_channel * 3(mean & std & skewness)
    
    return name, feature

# Color Histogram
def genCH(cs, grids, n_bin, ow=False):
    # filename
    name = 'ch_g{}_{}_({})'.format(grids, cs, '_'.join(str(x) for x in n_bin))

    # extract features
    feature = extract_feature(name, color_histogram, overwrite=ow, cs=cs, gx=grids, gy=grids, n_bin=n_bin)
    # shape: (grids)^2 * n_bin
    return name, feature


# Gabor Texture
def genGabor(cs, grids, K, S, ow=False):
    name = 'gabor_g{}_k{}_s{}_{}'.format(grids, K, S, cs)

    # extract features
    gabor_filters = genGaborFilters(K=K, S=S)
    feature = extract_feature(name, gabor_texture, overwrite=ow, cs=cs, filters=gabor_filters, gx=grids, gy=grids)
    # shape: K * S * (grids)^2 * 2(mean & std)
    return name, feature

# Loal Descriptors in various numbers of grids
def genLD(cs, grids, det='SIFT', ow=False):
    '''
        grids: needs to be power of 2
    '''
    name = '{}_g{}_{}'.format(det, grids, cs)
    
    # detector
    if det == 'ORB':
        detector = cv2.ORB_create()
    else:
        detector = cv2.SIFT_create()

    # compute SIFT descriptor
    feature = extract_feature(name, localDescriptor, overwrite=ow, cs=cs, detector=detector, g=grids)
    
    # shape: (grids)^2 * n descriptor * d component
    return name, feature

# # SIFT
# def genSIFT(cs, ow=False):
#     name = 'sift_{}'.format(cs)

#     # compute SIFT descriptor
#     feature = extract_feature(name, SIFT, overwrite=ow)
#     # shape: [n_kp * 2(x, y), n_kp * d component]
#     return name, feature

# # ORB
# def genORB(cs, ow=False):
#     name = 'orb_{}'.format(cs)

#     # compute ORB descriptor
#     feature = extract_feature(name, ORB, overwrite=ow)
#     # shape: [n_kp * 2(x, y), n_kp * d component] 
#     return name, feature

# VLAD
def genVLAD(k, des, des_name, codebook, grids, ow=False):
    # parameters
    name = 'vlad_k{}_{}'.format(k, des_name)

    # compute VLAD descriptor
    feature = VLAD(name, des, k, codebook, grids, overwrite=ow)
    # (grids)^2 * k class * d components 
    return name, feature

# Pre-generate -- VLAD
for db_folder in ['101_ObjectCategories', 'data']:
    for detector in ['SIFT', 'ORB']:
        for k in [16, 32, 64, 128]:
            # generate codebook
            codebook = genCodeBook(db_folder, k, detector)
            
#             for grids in [1, 2, 4, 8]:
#                 # load local descriptor
#                 name_ld, feature_ld = genLD(cs='RGB', grids=grids, det=detector)
            
#                 # VLAD
#                 name_vlad, feature_vlad = genVLAD(k, feature_ld, name_ld, codebook, grids)
#                 print(name_vlad, feature_vlad)