import numpy as np
from skimage import transform
import math
import random
import cv2
import os

def project(img, f):
    h,w,_ = img.shape
    def callback(xy_d):
        x_d = (xy_d[:, 0]) - w // 2
        y_d = (xy_d[:, 1]) - h // 2

        x_u = np.tan(x_d / f) * f
        y_u = y_d * np.sqrt(x_d**2 + f**2) / f
        xy_u = np.column_stack((x_u + w // 2, y_u + h // 2))
        return xy_u

    out = transform.warp(img, callback, order = 1, mode = 'edge')
    return (out * 255).astype(np.uint8)

def translate(img, tx, ty):
    h, w, _ = img.shape
    
    def callback(xy_d):
        x_d = (xy_d[:, 0])
        y_d = (xy_d[:, 1])
        x_u = x_d - tx
        y_u = y_d - ty
        xy_u = np.column_stack((x_u, y_u))
        return xy_u

    out = transform.warp(img, callback, order = 5, mode = 'edge', output_shape=(math.ceil(h+abs(ty)), math.ceil(w+abs(tx))))
    return (out * 255).astype(np.uint8)

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


def get_warped_images(run):
    image_dir = os.path.join('runs', run, 'images')
    warped_dir = os.path.join('runs', run, 'warped')
    image_files = sorted(os.listdir(image_dir), key = lambda x: int(x.split('.')[0]))
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    
    if not os.path.exists(warped_dir):
        os.mkdir(warped_dir)
    else:
        return [cv2.imread(os.path.join(warped_dir, '{}.jpg'.format(i))) for i in range(len(image_paths))]
    imgs = [transform.rescale(cv2.imread(impath), 1.0 / ratio, multichannel = True) for impath in image_paths]
    warped_imgs = [project(img, f) for img in imgs]
    for i, img in enumerate(warped_imgs):
        cv2.imwrite(os.path.join(warped_dir, '{}.jpg'.format(i)), img)
    return warped_imgs
    
if __name__ == '__main__':
    img = cv2.imread('../images/')