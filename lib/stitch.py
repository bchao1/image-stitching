import os
import cv2
import numpy as np
import sys
import pickle
import random
import math
from .warp import translate, get_warped_images
from .utils import get_image_size
from .features.detection import get_features
from .features.matching import get_matching_pairs
from .warp import feature_project, pre_crop

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
    # print('Ransac accept ratio', max_vote_num/len(pairs))
    return final_tx, final_ty

def get_pairwise_alignments(run, f, ratio, use_cache = True):
    run_dir = os.path.join('runs', run)
    x, y, features = get_features(run, ratio, use_cache = use_cache)
    cache_file = os.path.join(run_dir, 'shift.pickle')

    if use_cache and os.path.exists(cache_file):
        print("Use cached pairwise alignments ...")
        return pickle.load(open(os.path.join(run_dir, 'shift.pickle'), 'rb'))
    
    shifts = {}
    h, w = get_image_size(run)
    h //= ratio
    w //= ratio
    print("Compute pairwise alignments ...")
    for stitch_idx in range(len(features) - 1):
        print('Computing pairwise aligments for', stitch_idx, stitch_idx + 1)
        pairs = get_matching_pairs(x, y, features, stitch_idx, stitch_idx + 1, threshold = 0.4)
        warped_pairs = feature_project(pairs, f, h, w)
        dx, dy = ransac(warped_pairs, k = 100, threshold = 3)
        shifts[stitch_idx] = (dx, dy)
    pickle.dump(shifts, open(os.path.join(run_dir, 'shift.pickle'), 'wb'))
    return shifts

def stitch_images(run, f, ratio, use_cache):
    f //= ratio
    run_dir = os.path.join('runs', run)
    shifts = get_pairwise_alignments(run, f, ratio, use_cache)

    warped_imgs = get_warped_images(run, f, ratio, use_cache)
    warped_imgs = pre_crop(warped_imgs, f)

    dy = np.add.accumulate([0]+[shifts[k][1] for k in sorted(shifts.keys())])
    dx = np.add.accumulate([0]+[shifts[k][0] for k in sorted(shifts.keys())])
    dx_dec, dx_int = np.modf(dx) # 取小數部分
    dx_int = dx_int.astype(int)

    warped_imgs = [translate(img, tx, ty) for img, tx, ty in zip(warped_imgs, dx_dec, dy)]

    min_dy, max_dy = np.min(dy), np.max(dy)

    if min_dy < 0: # normalize to make sure all dy shifts are positive
        dy -= min_dy
        max_dy -= min_dy
    assert all([y >= 0 for y in dy])
    
    N = len(warped_imgs) # num of images
    stitched_img = np.pad(warped_imgs[N-1], ((0, math.ceil(max_dy - dy[N-1])), (dx_int[N-1], 0), (0, 0)), mode = 'constant')
    for i in range(N-2, -1,-1):    
                                                    
        dx1 = dx_int[i]                             
        dx2 = dx_int[i+1]                           
        # Stitch stitched_img, warped_imgs[i]       
                                                    
        h1, w1, _ = stitched_img.shape              
        h2, w2, _ = warped_imgs[i].shape            
        assert(h1 >= h2)
        stitched_img[:h2, dx1:dx2, :] = warped_imgs[i][:, :dx2-dx1, :] # left part of image
        ratio = np.arange(0, (w2 + dx1 - dx2), 1.0) / (w2 + dx1 - dx2) # blending alpha

        # Make ratio to 3d
        ratio_map = np.repeat(ratio[:, np.newaxis], 3, axis = 1)

        # blend images
        stitched_img[:h2, dx2:dx1+w2, :] = ratio_map * stitched_img[:h2, dx2:dx1+w2, :] + (1-ratio_map) * warped_imgs[i][:, dx2-dx1:, :]
    stitched_img = stitched_img[math.ceil(max_dy):-math.ceil(max_dy), :, :] # crop out edges
    cv2.imwrite(os.path.join(run_dir, 'result.jpg'), stitched_img)