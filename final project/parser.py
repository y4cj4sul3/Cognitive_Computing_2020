import pickle
import numpy as np

with open('data/UMS5/UMS5_lead_screw_hold_20201129000311/printJob_finish.pkl', 'rb') as fp:
    print(pickle.load(fp))

with open('data/UMS5/UMS5_lead_screw_hold_20201129000311/progress.npy', 'rb') as fp:
    print(np.load(fp, allow_pickle=True))

