import numpy as np
import os
import argparse
from skimage import io, transform
from skimage.util import crop
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import math
import random

# ===============
# Transform
# ===============

def FET_p(img, f):
    h,w,_ = img.shape
    def callback(xy_d):
        x_d = (xy_d[:, 0])
        y_d = (xy_d[:, 1])
        x_u = np.tan((x_d-w//2)/f)*f
        y_u = (y_d-h//2)/f*np.sqrt(np.square(x_u)+f**2)
        xy_u = np.column_stack((x_u+w//2, y_u+h//2))
        return xy_u

    out = transform.warp(img, callback, order = 1, mode = 'constant')
    return out
    
def project(img, f):
    return (FET_p(img, f) * 255).astype(np.uint8)

def FET_t(img, tx, ty):
    h, w, _ = img.shape
    
    def callback(xy_d):
        x_d = (xy_d[:, 0])
        y_d = (xy_d[:, 1])
        x_u = x_d + tx
        if ty <= 0:
            y_u = y_d + ty
        else :
            y_u = y_d
        xy_u = np.column_stack((x_u, y_u))
        return xy_u

    out = transform.warp(img, callback, order = 5, mode = 'constant', output_shape=(math.ceil(h+abs(ty)), math.ceil(w+abs(tx))))
    return out
    
def translate(img, txy):
    tx, ty = txy
    return (FET_t(img, tx, ty) * 255).astype(np.uint8)

def feature_project(features, f, h, w):
    new_feature_pairs = []
    for f_pair in features:
        (x1,y1), (x2,y2) = f_pair
        x1 -= w//2
        x2 -= w//2
        y1 -= h//2
        y2 -= h//2
        x1_w = f*np.arctan(x1/f)
        x2_w = f*np.arctan(x2/f)
        y1_w = f*y1/np.sqrt(x1**2+f**2)
        y2_w = f*y2/np.sqrt(x2**2+f**2)
        new_feature_pairs.append(((x1_w+w//2, y1_w+h//2), (x2_w+w//2, y2_w+h//2)))
    return new_feature_pairs

def ransac(pairs, k = 100, threshold = 5):
    max_vote_num = 0
    final_tx = 0
    final_ty = 0
    for _ in range(k):
        tx_sum = 0
        ty_sum = 0
        rand = random.randint(0, len(pairs)-1)
        (x2, y2), (x1, y1) = pairs[rand]
        tx = x2 - x1
        ty = y2 - y1
        vote_num = 0
        for _p in pairs:
            (x2, y2), (x1, y1) = _p
            _tx = x2 - x1
            _ty = y2 - y1
            dis = np.sqrt((tx-_tx)**2+(ty-_ty)**2)
            if dis <= threshold :
                vote_num += 1
                tx_sum += _tx
                ty_sum += _ty
        if vote_num > max_vote_num:
            max_vote_num = vote_num
            final_tx = tx_sum/vote_num
            final_ty = ty_sum/vote_num
    print(final_tx, final_ty, max_vote_num/len(pairs))
    return final_tx, final_ty