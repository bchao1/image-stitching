import numpy as np
import os
import argparse
from skimage import io, transform
from skimage.util import crop
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import math

# ===============
# Transform
# ===============

def FET_p(img, f, ratio):
    h,w,_ = img.shape
    f = f/ratio
    def callback(xy_d):
        x_d = (xy_d[:, 0])
        y_d = (xy_d[:, 1])
        x_u = np.tan((x_d-w//2)/f)*f
        y_u = (y_d-h//2)/f*np.sqrt(np.square(x_u)+f**2)
        xy_u = np.column_stack((x_u+w//2, y_u+h//2))
        return xy_u

    out = transform.warp(img, callback, order = 5, mode = 'constant')
    return out
    
def project(img, f, ratio):
    return (FET_p(img, f, ratio) * 255).astype(np.uint8)

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

    out = transform.warp(img, callback, order = 5, mode = 'constant', output_shape=(math.ceil(h+np.abs(ty)), math.ceil(w+np.abs(tx))))
    return out
    
def translate(img, txy):
    tx, ty = txy
    return (FET_t(img, tx, ty) * 255).astype(np.uint8)

def feature_project(features, f, ratio):
    f = f/ratio
    new_feature_pairs = []
    for f_pair in features:
        (x1,y1), (x2,y2) = f_pair
        x1_w = f*np.arctan(x1/f)
        x2_w = f*np.arctan(x2/f)
        y1_w = f*y1/np.sqrt(x1**2+f**2)
        y2_w = f*y2/np.sqrt(x2**2+f**2)
        new_feature_pairs.append(((x1_w, y1_w), (x2_w, y2_w)))
    return new_feature_pairs

def point_project(point, f, ratio):
    f = f/ratio
    (x1,y1) = point
    x1_w = f*np.arctan(x1/f)
    y1_w = f*y1/np.sqrt(x1**2+f**2)
    return (x1_w, y1_w)