import numpy as np
import os
import argparse
from skimage import io, transform
from skimage.util import crop
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# ===============
# Transform
# ===============

def FET(img, f, ratio):
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
    return (FET(img, 8000, 1) * 255).astype(np.uint8)