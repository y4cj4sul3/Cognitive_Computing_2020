import cv2


def SIFT(img):
    # make sure the version of opencv-python >= 4.4.0
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return des


def ORB(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)

    return des


def VLAD(name, des, k, path='features/', overwrite=False):
    '''
        des: local descriptors (# images x # keypoints x d components)
        k: number of visual words

    '''
    # load from files
    filepath = os.path.join(path, name + '.npy')
    if not overwrite and os.path.isfile(filepath):
        print('load from file')
        return np.load(filepath, allow_pickle=True)

    # load visual words
    vw_filepath = os.path.join(path, name + '_vw.pickle')
    if not overwrite and os.path.isfile(vw_filepath):
        print('load VM from file')
        with open(vw_filepath, 'rb') as fp:
            VWs = pickle.load(fp)
    else:
        # flatten descriptors ( -> (# images x # keypoints) x d components)
        flat_des = list(itertools.chain.from_iterable(des))
        # k-means for visual words
        VWs = KMeans(n_clusters=k, verbose=1).fit(flat_des)
        # save VWs
        with open(vw_filepath, 'wb') as fp:
            pickle.dump(VWs, fp)

    # parameters
    d = des[0].shape[1]  # d-dimension
    # centroids
    c = VWs.cluster_centers_

    VLADs = []
    # for each image
    for X in des:
        # X (# keypoints x d components)
        # get NN(X)
        nn_x = VWs.predict(X.tolist())

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

    # save features into file
    VLADs = np.array(VLADs)
    np.save(filepath, VLADs)

    return VLADs
