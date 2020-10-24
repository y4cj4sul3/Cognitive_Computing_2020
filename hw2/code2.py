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

# Local Descriptors in various numbers of grids
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

# VLAD
def genVLAD(k, des, des_name, codebook, grids, ow=False):
    # parameters
    name = 'k{}_{}'.format(k, des_name)

    # compute VLAD descriptor
    feature_vlad, feature_hist = VLAD(name, des, k, codebook, grids, overwrite=ow)
    # (grids)^2 * k class * d components 
    return 'vlad_'+name, feature_vlad, 'hist_'+name, feature_hist

# # Pre-generate -- Color Histogram
# for cs, n_bin in zip(['RGB', 'HSV'], [[4, 4, 4], [18, 3, 3]]):
#     for grids in range(1, 10):
#         name_ch, feature_ch = genCH(cs, grids, n_bin)
        
#         print(name_ch, feature_ch.shape)

# # Pre-generate -- Gabor Texture
# for cs in ['GRAY', 'RGB', 'HSV']:
#     for K, S in [[6, 8], [8, 6], [12, 8]]:
#         for grids in range(1, 10):
#             name_gabor, feature_gabor = genGabor(cs, grids, K, S)

#             print(name_gabor, feature_gabor.shape)

# # Pre-generate -- Local Descriptors
# for det in ['SIFT', 'ORB']:
#     for cs in ['RGB', 'GRAY']:
#         for grids in [1, 2, 4, 8]:
#             name_ld, feature_ld = genLD(cs, grids, det)
            
#             print(name_ld, feature_ld.shape)
#             if grids == 1:
#                 print(feature_ld[0].shape)
#             else:
#                 print(np.shape(feature_ld[1, 0]))

# # Pre-generate -- VLAD
# for db_folder in ['101_ObjectCategories', 'data']:
#     for detector in ['SIFT', 'ORB']:
#         for k in [16, 32, 64, 128]:
#             # generate codebook
#             codebook = genCodeBook(db_folder, k, detector)
            
#             # load local descriptor
#             name_ld, feature_ld = genLD(cs='RGB', grids=grids, det=detector)
            
#             # VLAD
#             for grids in [1, 2, 4, 8]:
#                 name_vlad, feature_vlad = genVLAD(k, feature_ld, name_ld, codebook, grids)
#                 print(name_vlad, feature_vlad)

# Ranking
def ranking(name, features, dist_func, ascend=True, symmetric=True, path='results/', overwrite=False, **kargs):
    '''
        ascend: ranking in ascending (for distance) or descending order (for similarity)
        symmetric: if the distance function has commutative property, the process can be speedup
    '''
    # load from files
    rank_filepath = os.path.join(path, 'ranking', name + '.npy')
    dist_filepath = os.path.join(path, 'dist', name + '.npy')
    if not overwrite and os.path.isfile(rank_filepath) and os.path.isfile(dist_filepath):
        print('load from file')
        return np.load(rank_filepath, allow_pickle=True), np.load(dist_filepath, allow_pickle=True)
    
    # distance/similarity
    n_feature = len(features)
    dist = []
    if symmetric:
        for i in range(n_feature):
            dist_row = []
            # reduce reduntant computation
            for j in range(i):
                dist_row.append(dist[j][i])
            # computer features
            for j in range(i, n_feature):
                dist_row.append(dist_func(features[i], features[j], **kargs))
            
            dist.append(dist_row)
    else:
        for i in range(n_feature):
            dist_row = []
            # computer features
            for j in range(n_feature):
                dist_row.append(dist_func(features[i], features[j], **kargs))
            
            dist.append(dist_row)
    
    # ranking (including the target image itself)
    ranked_all = np.argsort(dist, axis=1)
    if not ascend:
        # reverse for descending order
        ranked_all = ranked_all[:, ::-1]
    
    # baseline check: the first retrived image should be the target image itself
    ranked = []
    for i in range(len(ranked_all)):
        target_rank = 0
        if ranked_all[i, 0] != i:
            target_rank = np.where(ranked_all[i] == i)
            print('Baseline check Failed: ', i, ranked_all[i, 0], target_rank)
            
        # remove the target image from the result
        ranked.append(np.delete(ranked_all[i], target_rank))
        
    # save features into file
    ranked = np.array(ranked)
    dist = np.array(dist)
    np.save(rank_filepath, ranked)
    np.save(dist_filepath, dist)
    
    return np.array(ranked), dist

# distance/similarity function
def Ln_distance(a, b, ord=1, weight=None):
    if weight is not None:
        # weighting over each feature
        return np.linalg.norm((a-b)*weight, ord=ord)
    else:
        return np.linalg.norm(a-b, ord=ord)

# local feature matching function
def matching(query_des, train_des, match_func, grids=1, rt=None):
    '''
        k: find best k match of each query descriptor
    '''
    if grids == 1:
        query_des = [query_des]
        train_des = [train_des]
    
    # run over each grid
    counts = 0
    for des_q, des_t in zip(query_des, train_des):
        # skip empty grids
        if len(des_q) == 0 or len(des_t) == 0:
            continue
        
        # matching
        if rt is None:
            matches = match_func(np.array(des_q), np.array(des_t))
            counts += len(matches)
        else:
            # knn & ratio test
            matches = match_func(np.array(des_q), np.array(des_t), k=2)
            
            good_match = 0
            for m in matches:
                if len(m) > 1 and m[0].distance < rt * m[1].distance:
                    # if the distance of best match is much smaller than the second
                    # then it is a good match
                    good_match += 1
                    
            counts += good_match
    
    return counts

def computeMAP(ranking):
    # precisiion & recall
    P = []
    R = []
    
    # AP of each image
    APs = []
    
    for i in range(len(ranking)):
        
        tp = 0 # true positive
        fp = 0 # false positive 

        precision = []
        recall = []
        AP = []
        
        cur_category = i // NUM_OF_RELEVANT
        
        for j in range(len(ranking[i])):
            if ranking[i, j] // NUM_OF_RELEVANT == cur_category:
                # matched
                tp += 1
                precision.append(tp / (tp + fp))
                recall.append(tp / NUM_OF_RELEVANT)
                # add precision in to the list
                AP.append(precision[-1])
            else:
                # mismatched
                fp += 1
                precision.append(tp / (tp + fp))
                recall.append(tp / NUM_OF_RELEVANT)
                
        AP = np.mean(AP)
        APs.append(AP)
        P.append(precision)
        R.append(recall)
        
    MMAP = np.mean(APs)
    MAP = [np.mean(APs[i*NUM_OF_RELEVANT:(i+1)*NUM_OF_RELEVANT]) for i in range(NUM_OF_CATEGORY)]
    P = np.mean(P, axis=0)
    R = np.mean(R, axis=0)
    
    return MMAP, MAP, P, R
                
# Ln distance
def evalLn(fname, feature, ord, weight=None, ow=False):
    # filename
    name = fname + '_L{}'.format(ord)
    
    # ranking
    ranked, _ = ranking(name, feature, Ln_distance, overwrite=ow, ord=ord, weight=weight)
    # MAP
    MMAP, MAP, P, R = computeMAP(ranked)
    
    return name, {'MMAP': MMAP, 'MAP': MAP, 'P': P, 'R': R}

# Evaluations
results = {}

# # Evaluations -- SIFT
# det = 'SIFT'
# matcher = cv2.BFMatcher()
# for cs in ['RGB', 'GRAY']:
#     for grids in [1, 2, 4, 8]:
#         # load features
#         name_ld, feature_ld = genLD(cs, grids, det)
#         print(name_ld, feature_ld.shape)

#         # evaluation
#         ranked, _ = ranking(name_ld, feature_ld, matching, ascend=False, symmetric=False, overwrite=False, match_func=matcher.knnMatch, grids=grids, rt=0.7)
#         # MAP
#         MMAP, MAP, P, R = computeMAP(ranked)

#         results[name_ld] = {'MMAP': MMAP, 'MAP': MAP, 'P': P, 'R': R}

# # Gabor Texture
# for cs in ['GRAY', 'RGB', 'HSV']:
#     for K, S in [[6, 8], [8, 6], [12, 8]]:
#         for grids in range(1, 10):
#             # generate (or load) features
#             name_gabor, feature_gabor = genGabor(cs, grids, K, S)
#             print(name_gabor, feature_gabor.shape)

#             # evaluation (or load result)
#             for order in [1, 2]:
#                 weight = 1/np.std(feature_gabor, axis=0)
#                 name, result = evalLn(name_gabor, feature_gabor, ord=order, weight=weight)
#                 results[name] = result
                
# VLAD & Histogram of Visual Word
for db_folder in ['101_ObjectCategories', 'data']:
    for detector in ['SIFT', 'ORB']:
        for k in [16, 32, 64, 128]:
            # generate (or load) codebook
            codebook = genCodeBook(db_folder, k, detector)
            
            for grids in [1, 2, 4, 8]:
                # generate (or load) local descriptor
                name_ld, feature_ld = genLD(cs='RGB', grids=grids, det=detector)

                # generate (or load) features
                name_vlad, feature_vlad, name_hist, feature_hist = genVLAD(k, feature_ld, name_ld, codebook, grids)
                print(name_vlad, feature_vlad.shape)
                print(name_hist, feature_hist.shape)
                
                # evaluation (or load result)
                for order in [1, 2]:
                    print(name_vlad, feature_vlad.shape)
                    print(name_hist, feature_hist.shape)
                    # vlad
                    name, result = evalLn(name_vlad, feature_vlad, ord=order)
                    results[name] = result
                    # hist
                    name, result = evalLn(name_hist, feature_hist, ord=order)
                    results[name] = result