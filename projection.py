import numpy as np
import cv2
import os
import sys
import math
from matplotlib import pyplot as plt

def project( img, f, ratio ):
    h,w,_ = img.shape
    warped_img = np.zeros( img.shape )
    f = f/ratio
    for i in range( w ) :
        x = np.tan((i-w//2)/f)*f
        for j in range( h ) :
            y = (j-h//2)/f*np.sqrt(x**2+f**2)
            if( (x+w//2) < 1 or (x+w//2) >= w-1 ) :
                continue
            elif( (y+h//2) < 1 or (y+h//2) >= h-1 ) :
                continue
            else:
                x_f = math.floor( x )
                y_f = math.floor( y )
                a = x - x_f
                b = y - y_f
                warped_img[j, i, : ] = (1-a)*(1-b)*img[ y_f+h//2, x_f+w//2, : ] + (1-a)*(b)*img[ y_f+1+h//2, x_f+w//2, : ] + (1-b)*(a)*img[ y_f+h//2, x_f+1+w//2, : ] + (a)*(b)*img[ y_f+1+h//2, x_f+1+w//2, : ]
                # print(warped_img[j, i, :])
    return warped_img

