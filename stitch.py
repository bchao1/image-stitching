import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from lib.features.detection import harris_corner_detection, plot_detection
from lib.features.matching import least_error_ratio_match, get_matching_pairs
from lib.utils import read_image
from warp import project, feature_project, translate, ransac
import sys
import pickle

'''
run = 'lib1'
warped_dir = os.path.join('runs', run, 'warped')
n_image = len(os.listdir(warped_dir))
warped_imgs = [cv2.imread(os.path.join(warped_dir, '{}.jpg'.format(i))) for i in range(n_image)]
print(warped_imgs)
   #x, y, features = get_features(run)
   
shifts = pickle.load(open('shift.pickle', 'rb'))
ty = [shifts[k][1] for k in sorted(shifts.keys())]
print(shifts)
for i in range(1, len(ty)):
    ty[i] += ty[i - 1]
ty.insert(0, 0)
min_ty = min(ty)
if min_ty < 0:
    ty = [y - min_ty for y in ty]
assert len(ty) == len(warped_imgs)
assert all([y >= 0 for y in ty])
print(ty)
max_pad = max(ty)

global_x = 0
stitched_img = translate(warped_imgs[0], (global_x, ty[0]))
stitched_img = np.pad(stitched_img, ((0, max_pad - ty[0]), (0, 0), (0, 0)), mode = 'edge')


for i in range(0, len(warped_imgs) - 1):
    print('stiching ', i, i + 1)
    dx = shifts[i][0]
    global_x += dx

    warped1 = stitched_img
    warped2 = translate(warped_imgs[i + 1], (global_x, ty[i + 1]))
    warped2 = np.pad(warped2, ((0, max_pad - ty[i + 1]), (0, 0), (0, 0)), mode = 'edge')

    h1, w1, _ = warped1.shape
    h2, w2, _ = warped2.shape
    assert(h1 == h2)
    stitched_img = np.zeros((h1, w2, 3), dtype = np.float)
    #print(w1, w2, tx1, warped_imgs[stitch_idx+1].shape[1])
    stitched_img[:, :global_x, :] = warped1[:, :global_x, :]
    stitched_img[:, w1:, :] = warped2[:, w1:, :]
    ratio = np.arange(0, 1, 1.0 / (w1 - global_x))
    ratio_map = np.repeat(ratio[np.newaxis, :], h1, axis = 0)
    ratio_map = np.repeat(ratio_map[:, :, np.newaxis], 3, axis = 2)
    #print(ratio_map.shape)
    #print(ratio_map[:, :, 0])
    #warped1[:, tx1:, :][np.where(warped1[:, tx1:, :] == 0)] = 255
    #warped2[:, tx1:w1, :][np.where(warped2[:, tx1:w1, :] == 0)] = 255
    stitched_img[:, global_x:w1, :] = (1 - ratio_map) * warped1[:, global_x:, :] + ratio_map * warped2[:, global_x:w1, :]
    cv2.imwrite('test{}_new.jpg'.format(i), stitched_img)

'''
img = cv2.imread('test8_new.jpg')
img = img[100:-100, 100:-100, :]
cv2.imwrite('final.jpg', img)