import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.features.detection import harris_corner_detection, plot_detection
from lib.features.matching import least_error_ratio_match, get_matching_pairs
from lib.utils import read_image
from warp import project, feature_project, translate, ransac
import sys

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

def in_img(f, tx, ty, w, h, x_d, y_d):
    x_d = x_d - tx
    y_d = y_d - ty
    x_u = np.tan((x_d-w//2)/f)*f
    y_u = (y_d-h//2)/f*np.sqrt(np.square(x_u)+f**2)
    return(0 <= x_u+w//2 and x_u+w//2 <= w and 0 <= y_u+h//2 and y_u+h//2 <= h)


if __name__ == '__main__':
    get_features()
    plot_matching('images/library')
    # get matching coodinates for images 4, 3
    pairs = get_matching_pairs(x, y, features, 4, 3, threshold = 0.5)
    f = int(sys.argv[2])
    image_dir = './images/library'
    # image_files = sorted(os.listdir(image_dir))
    image_files = ['IMG_6584.JPG', 'IMG_6583.JPG']
    ratio = int(sys.argv[1])
    f = f/ratio
    imgs = []
    warped_imgs = []
    for i, file in enumerate(image_files):
        img = cv2.imread(os.path.join(image_dir, file))
        h,w,_ =  img.shape
        imgs.append( cv2.resize(img,(w//ratio, h//ratio)))

    for i, img in enumerate(imgs):
        warped_img = project( img, f )
        warped_imgs.append(warped_img)

    h, w, _ = imgs[0].shape
    warped_pairs = feature_project(pairs, f, h, w)
    tx1, ty1 = ransac(warped_pairs, k=100, threshold=3 )
    tx0 = 0
    ty0 = 0
    if( ty1 < 0 ):
        ty0 = -ty1
        ty1 = 0
    warped = translate(warped_imgs[0], (0, ty0))
    warped_imgs[0] = warped
    warped = translate(warped_imgs[1], (tx1, ty0))#後面是圖片要增加多少邊長，所以兩張圖片的y方向都要增加一樣的邊長
    warped_imgs[1] = warped

    # cv2.imwrite('0.jpg', warped_imgs[0])
    # cv2.imwrite('1.jpg', warped_imgs[1])
    h, w, _ = warped_imgs[0].shape
    for j in range(h):
        for i in range(w):
            if in_img( f, tx0, ty0, w, h, i, j) and in_img( f, tx1, ty1, w, h, i, j):
                boundary_width = 2*np.tan(w//2/f)*f
                blank_width = (w-boundary_width)/2
                boundary = w-blank_width
                boundary_dis0 = boundary - i
                ratio0 = boundary_dis0/(boundary_width-tx1)
                ratio1 = 1-ratio0
                warped_imgs[1][j,i] = warped_imgs[0][j,i]*ratio0 + warped_imgs[1][j,i]*ratio1
                # print(ratio0)
            elif in_img( f, tx0, ty0, w, h, i, j):
                warped_imgs[1][j,i] = warped_imgs[0][j,i]
    
    cv2.imwrite('haha.jpg', warped_imgs[1])





    


