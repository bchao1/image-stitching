import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.features.detection import harris_corner_detection, plot_detection
from lib.features.matching import least_error_ratio_match, get_matching_pairs
from lib.utils import read_image
from warp import project, feature_project, translate
import sys
import random

def get_features(img_dir, r_threshold = 1e7, max_features = 500, window = 20):
    img_files = sorted(os.listdir(img_dir))
    img_paths = [os.path.join(img_dir, f) for f in img_files]

    features = np.zeros((len(img_paths), max_features, 8 ** 2)) # 8 * 8 features
    x_coors = np.zeros((len(img_paths), max_features))
    y_coors = np.zeros((len(img_paths), max_features))

    for i, img in enumerate(img_paths):
        print("Detecting {} features ...".format(img))
        x, y, _, f = harris_corner_detection(img, 4, r_threshold = r_threshold,
            max_features = max_features, window = window)
        x_coors[i] = x
        y_coors[i] = y 
        features[i] = f
    np.save('x.npy', x_coors)
    np.save('y.npy', y_coors)
    np.save('features.npy', features)
    return x_coors, y_coors, features

def test(img_path):
    x, y, r, f = harris_corner_detection(img_path, 4, r_threshold = 1e7, max_features = 100, window = 3)
    plot_detection(x, y, img_path, r)

def plot_matching(img_dir = 'images', cache_dir = 'cache'):
    img_files = sorted(os.listdir(img_dir))
    img_paths = [os.path.join(img_dir, f) for f in img_files]
    # npz
    x_coors = np.load(os.path.join(cache_dir, 'x.npy'))
    y_coors = np.load(os.path.join(cache_dir, 'y.npy'))
    features = np.load(os.path.join(cache_dir, 'features.npy'))

    f, ax = plt.subplots(1, len(img_files))
    for i, img_path in enumerate(img_paths):
        ax[i].imshow(read_image(img_path, 4))
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].axes.get_yaxis().set_visible(False)

    colors = ['b', 'w', 'r', 'y', 'c', 'm', 'g']
    c = 0
    for i in range(len(features) - 1):
        f1 = features[i]
        f2 = features[i + 1]
        p1, p2 = least_error_ratio_match(f1, f2, 0.4)
        x1, y1 = x_coors[i][p1], y_coors[i][p1]
        x2, y2 = x_coors[i+1][p2], y_coors[i+1][p2]
        ax[i].scatter(x1, y1, marker = '+', color = colors[c])
        ax[i+1].scatter(x2, y2, marker = '+', color = colors[c])
        c = (c + 1) % len(colors)

    plt.subplots_adjust(top = 1, bottom = 0, right = 0.99, left = 0.01)
    plt.show()

if __name__ == '__main__':
    x = np.load('cache/x.npy').astype(np.int)
    y = np.load('cache/y.npy').astype(np.int)
    features = np.load('cache/features.npy')
    # get matching coodinates for images 0, 1
    pairs = get_matching_pairs(x, y, features, 2, 3, threshold = 0.5)
    f = int(sys.argv[2])
    image_dir = './images'
    # image_files = sorted(os.listdir(image_dir))
    image_files = ['3.JPG', '4.JPG']
    ratio = int(sys.argv[1])
    pairs = feature_project(pairs, f, ratio )
    imgs = []
    warped_imgs = []
    for i, file in enumerate(image_files):
        img = cv2.imread(os.path.join(image_dir, file))
        h,w,_ =  img.shape
        imgs.append( cv2.resize(img,(w//ratio, h//ratio)))

    for i, img in enumerate(imgs):
        warped_img = project( img, f, ratio)
        warped_imgs.append(warped_img)
    k = 100
    threshold = 10
    max_vote_num = 0
    final_tx = 0
    final_ty = 0
    for _ in range(k):
        rand = random.randint(0, len(pairs)-1)
        (x2, y2), (x1, y1) = pairs[rand]
        tx = x1 - x2
        ty = y1 - y2
        vote_num = 0
        for _p in pairs:
            (x2, y2), (x1, y1) = _p
            _tx = x1 - x2
            _ty = y1 - y2
            dis = np.sqrt((tx-_tx)**2+(ty-_ty)**2)
            vote_num += (dis <= threshold)
        if vote_num > max_vote_num:
            max_vote_num = vote_num
            final_tx = tx
            final_ty = ty
    print(final_tx, final_ty, max_vote_num/len(pairs))
    warped = []
    warped = translate(warped_imgs[1], (final_tx, final_ty))
    warped_imgs[1] = warped
    if final_ty > 0:
        warped = translate(warped_imgs[0], (0,-final_ty))
        warped_imgs[0] = warped
    cv2.imwrite('0.jpg', warped_imgs[0])
    cv2.imwrite('1.jpg', warped_imgs[1])

    h, w, _ = warped_imgs[0].shape
    for i in range(h):
        for j in range(w):
            if (warped_imgs[1][i,j] == [0,0,0]).all():
                warped_imgs[1][i,j] = warped_imgs[0][i,j]
            else:
                warped_imgs[1][i,j] = warped_imgs[1][i,j]/2 + warped_imgs[0][i,j]/2
    
    cv2.imwrite('test.jpg', warped_imgs[1])





    


