import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from .detection import get_features

def get_matching_pairs(x, y, features, i, j, threshold = 0.5):
    pi, pj = least_error_ratio_match(features[i], features[j], threshold)
    xi, yi = x[i][pi], y[i][pi]
    xj, yj = x[j][pj], y[j][pj]
    i_coors = list(zip(xi, yi))
    j_coors = list(zip(xj, yj))
    return list(zip(i_coors, j_coors))

def least_error_ratio_match(f1, f2, threshold = 0.5):
    ''' Matching algorithm based on best and second best error ratio. '''

    assert f1.shape == f2.shape

    best_match = np.zeros((f1.shape[0], 2), dtype = np.int) # store the best and next best match
    p = []
    for i, f in enumerate(f1):
        d = np.sum(np.abs(f2 - f), axis = 1).ravel()
        indices = np.argsort(d)
        best_match[i] = indices[:2]
    
    for i, f in enumerate(f2):
        d = np.sum(np.abs(f1 - f), axis = 1).ravel()
        indices = np.argsort(d)
        if best_match[indices[0]][0] == i:
            p.append(indices[0])


    d1 = np.sum(np.abs(f1[p] - f2[best_match[p][:, 0]]), axis = 1)
    d2 = np.sum(np.abs(f1[p] - f2[best_match[p][:, 1]]), axis = 1)
    err_ratio = d1 / d2
    p = np.array(p, dtype = np.int)
    final_match = p[np.where(err_ratio < threshold)[0]]
    return final_match, best_match[final_match][:, 0]