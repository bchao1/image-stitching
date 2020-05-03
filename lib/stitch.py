import os
import cv2
import numpy as np
import sys
import pickle
import random
from .warp import translate, get_warped_images
from .utils import get_image_size
from .features.detection import get_features
from .features.matching import get_matching_pairs
from .warp import feature_project

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
    #print(final_tx, final_ty, max_vote_num/len(pairs))
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
    print(h, w)
    print("Compute pairwise alignments ...")
    for stitch_idx in range(len(features) - 1):
        print('Computing pairwise aligments for', stitch_idx, stitch_idx + 1)
        pairs = get_matching_pairs(x, y, features, stitch_idx, stitch_idx + 1, threshold = 0.4)
        warped_pairs = feature_project(pairs, f, h, w)
        dx, dy = ransac(warped_pairs, k = 100, threshold = 3)
        dx, dy = int(dx), int(dy)
        print(dx, dy)
        shifts[stitch_idx] = (dx, dy)
    pickle.dump(shifts, open(os.path.join(run_dir, 'shift.pickle'), 'wb'))
    return shifts

def stitch_images(run, f, ratio, use_cache):
    f //= ratio
    run_dir = os.path.join('runs', run)
    warped_imgs = get_warped_images(run, f, ratio, use_cache)
    shifts = get_pairwise_alignments(run, f, ratio, use_cache)
    
    dy = [shifts[k][1] for k in sorted(shifts.keys())]

    for i in range(1, len(dy)): # compute global y shift for each image
        dy[i] += dy[i - 1]
    dy.insert(0, 0)

    min_dy, max_dy = min(dy), max(dy)

    if min_dy < 0: # normalize to make sure all dy shifts are positive
        dy = [y - min_dy for y in dy]
    assert all([y >= 0 for y in dy])
    
    global_dx = 0
    stitched_img = translate(warped_imgs[0], global_dx, dy[0])
    stitched_img = np.pad(stitched_img, ((0, max_dy - dy[0]), (0, 0), (0, 0)), mode = 'edge')

    for i in range(0, len(warped_imgs) - 1):
        print('Stiching images', i, i + 1)

        dx = shifts[i][0]
        global_dx += dx

        # Stitch warped1, warped2
        warped1 = stitched_img
        warped2 = translate(warped_imgs[i + 1], global_dx, dy[i + 1])
        warped2 = np.pad(warped2, ((0, max_dy - dy[i + 1]), (0, 0), (0, 0)), mode = 'edge')

        h1, w1, _ = warped1.shape
        h2, w2, _ = warped2.shape
        assert(h1 == h2)

        stitched_img = np.zeros((h1, w2, 3), dtype = np.float) # container for stitched image
        stitched_img[:, :global_dx, :] = warped1[:, :global_dx, :] # left part of image
        stitched_img[:, w1:, :] = warped2[:, w1:, :] # right part of image
        ratio = np.arange(0, 1, 1.0 / (w1 - global_dx)) # blending alpha

        # Make ratio to 3d
        ratio_map = np.repeat(ratio[np.newaxis, :], h1, axis = 0)
        ratio_map = np.repeat(ratio_map[:, :, np.newaxis], 3, axis = 2)

        # blend images
        stitched_img[:, global_dx:w1, :] = (1 - ratio_map) * warped1[:, global_dx:, :] + ratio_map * warped2[:, global_dx:w1, :]
    stitched_img = stitched_img[100:-100, 100:-100, :] # crop out edges
    cv2.imwrite(os.path.join(run_dir, 'result.jpg'), stitched_img)