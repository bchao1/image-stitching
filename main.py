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

def get_features(run, downscale = 4, r_threshold = 1e8, max_features = 500, window = 20, override = False):
    img_dir = os.path.join('runs', run, 'images')
    detection_dir = os.path.join('runs', run, 'detection')
    if not os.path.exists(detection_dir):
        os.mkdir(detection_dir)
    elif not override:
        x_coors = np.load(os.path.join(detection_dir, 'x.npy'))
        y_coors = np.load(os.path.join(detection_dir, 'y.npy'))
        features = np.load(os.path.join(detection_dir, 'features.npy'))
        return x_coors, y_coors, features
    
    img_files = sorted(os.listdir(img_dir), key = lambda x: int(x.split('.')[0]))
    img_paths = [os.path.join(img_dir, f) for f in img_files]

    features = np.zeros((len(img_paths), max_features, 8 ** 2)) # 8 * 8 features
    x_coors = np.zeros((len(img_paths), max_features))
    y_coors = np.zeros((len(img_paths), max_features))

    for i, img in enumerate(img_paths):
        print("Detecting {} features ...".format(img))
        x, y, _, f = harris_corner_detection(img, downscale, r_threshold = r_threshold,
            max_features = max_features, window = window)
        x_coors[i] = x
        y_coors[i] = y 
        features[i] = f
    np.save(os.path.join(detection_dir, 'x.npy'), x_coors)
    np.save(os.path.join(detection_dir, 'y.npy'), y_coors)
    np.save(os.path.join(detection_dir, 'features.npy'), features)
    return x_coors, y_coors, features

def test(img_path, downscale = 4):
    x, y, r, f = harris_corner_detection(img_path, downscale, r_threshold = 1e7, max_features = 100, window = 3)
    plot_detection(x, y, img_path, r)

def plot_matching(run):
    img_dir = os.path.join('runs', run, 'images')
    img_files = sorted(os.listdir(img_dir))
    img_paths = [os.path.join(img_dir, f) for f in img_files]
    detection_dir = os.path.join('runs', run, 'detection')
    x_coors = np.load(os.path.join(detection_dir, 'x.npy'))
    y_coors = np.load(os.path.join(detection_dir, 'y.npy'))
    features = np.load(os.path.join(detection_dir, 'features.npy'))

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

def in_img(f, tx, ty, w, h, x_d, y_d):
    x_d = x_d - tx
    y_d = y_d - ty
    x_u = np.tan((x_d-w//2)/f)*f
    y_u = (y_d-h//2)/f*np.sqrt(np.square(x_u)+f**2)
    return(0 <= x_u+w//2 and x_u+w//2 <= w and 0 <= y_u+h//2 and y_u+h//2 <= h)


if __name__ == '__main__':
    get_features('lib1', override = True)
    plot_matching('lib1')
    '''
    ratio = int(sys.argv[1])
    f = int(sys.argv[2])

    f /= ratio

    run = 'lib1'
    image_dir = os.path.join('runs', run, 'images')
    warped_dir = os.path.join('runs', run, 'warped')
    if not os.path.exists(warped_dir):
        os.mkdir(warped_dir)

    image_files = sorted(os.listdir(image_dir))
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    imgs = [rescale(cv2.imread(impath), 1.0 / ratio, multichannel = True) for impath in image_paths]
    warped_imgs = [project(img, f) for img in imgs]
    
    x, y, features = get_features(run)
    
    stitch_idx = 7
    pairs = get_matching_pairs(x, y, features, stitch_idx, stitch_idx + 1, threshold = 0.4)


    h, w, _ = imgs[0].shape
    warped_pairs = feature_project(pairs, f, h, w)
    tx1, ty1 = ransac(warped_pairs, k = 100, threshold = 3)
    tx0 = 0
    ty0 = 0
    if( ty1 < 0 ):
        ty0 = -ty1
        ty1 = 0
    warped1 = translate(warped_imgs[stitch_idx], (0, ty0))
    warped2 = translate(warped_imgs[stitch_idx+1], (tx1, ty0)) #後面是圖片要增加多少邊長，所以兩張圖片的y方向都要增加一樣的邊長
    cv2.imwrite('warped1.jpg', warped1)
    cv2.imwrite('warped2.jpg', warped2)

    h1, w1, _ = warped1.shape
    h2, w2, _ = warped2.shape
    assert(h1 == h2)
    tx1 = int(tx1)
    stitched_img = np.zeros((h1, w2, 3), dtype = np.float)
    print(w1, w2, tx1, warped_imgs[stitch_idx+1].shape[1])
    stitched_img[:, :tx1, :] = warped1[:, :tx1, :]
    stitched_img[:, w1:, :] = warped2[:, w1:, :]
    ratio = np.arange(0, 1, 1.0 / (w1 - tx1))
    ratio_map = np.repeat(ratio[np.newaxis, :], h1, axis = 0)
    ratio_map = np.repeat(ratio_map[:, :, np.newaxis], 3, axis = 2)
    print(ratio_map.shape)
    print(ratio_map[:, :, 0])
    stitched_img[:, tx1:w1, :] = (1 - ratio_map) * warped1[:, tx1:, :] + ratio_map * warped2[:, tx1:w1, :]


    #h, w, _ = warped_imgs[stitch_idx].shape
    #for j in range(h):
    #    for i in range(w):
    #        if in_img( f, tx0, ty0, w, h, i, j) and in_img( f, tx1, ty1, w, h, i, j):
    #            boundary_width = 2*np.tan(w//2/f)*f
    #            blank_width = (w-boundary_width)/2
    #            boundary = w-blank_width
    #            boundary_dis0 = boundary - i
    #            ratio0 = boundary_dis0/(boundary_width-tx1)
    #            ratio1 = 1-ratio0
    #            warped2[j,i] = warped1[j,i]*ratio0 + warped2[j,i]*ratio1
    #            # print(ratio0)
    #        elif in_img( f, tx0, ty0, w, h, i, j):
    #            warped2[j,i] = warped1[j,i]

    cv2.imwrite('test{}.jpg'.format(stitch_idx), stitched_img)

    '''