import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from features.detection import harris_corner_detection
from features.matching import least_error_ratio_match

def read_image(img_path, downscale):
    img = cv2.imread(img_path)[:,:,::-1]
    h, w, _ = img.shape
    img = cv2.resize(img, (w // downscale, h // downscale), interpolation = cv2.INTER_CUBIC)
    return img

def plot_detection(x, y, img_file, corner_response):
    colored_img = cv2.imread(img_file)[:,:,::-1]
    h, w = corner_response.shape
    colored_img = cv2.resize(colored_img, (w, h), interpolation = cv2.INTER_CUBIC)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(corner_response, cmap = 'gray')
    ax[1].imshow(colored_img)
    ax[1].scatter(x, y, marker = '+', color = 'red')
    for i in range(2):
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def test_detection(img_path, downscale = 4):
    x, y, r, f = harris_corner_detection(img_path, downscale, r_threshold = 1e8, max_features = 100, window = 3)
    plot_detection(x, y, img_path, r)

def plot_matching(run, start, end, save = False):
    img_dir = os.path.join('runs', run, 'images')
    img_files = sorted(os.listdir(img_dir), key = lambda x: int(x.split('.')[0]))[start:end+1]
    print(img_files)
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
    for i in range(len(img_files) - 1):
        f1 = features[i]
        f2 = features[i + 1]
        p1, p2 = least_error_ratio_match(f1, f2, 0.4)
        x1, y1 = x_coors[i][p1], y_coors[i][p1]
        x2, y2 = x_coors[i+1][p2], y_coors[i+1][p2]
        ax[i].scatter(x1, y1, marker = '+', color = colors[c])
        ax[i+1].scatter(x2, y2, marker = '+', color = colors[c])
        c = (c + 1) % len(colors)

    plt.subplots_adjust(left=0.01, bottom=0, right=0.99, top=1)
    if save:
        plt.savefig(os.path.join('runs', run, 'matching_{}_{}.png'.format(start, end)))
    plt.show()

def plot_compare(img_file):
    x1, y1, corner_response1, _ = harris_corner_detection(img_file, 4, max_features = 100, non_maximal_suppression = False)
    x2, y2, corner_response2, _ = harris_corner_detection(img_file, 4, max_features = 100, non_maximal_suppression = True)

    colored_img = cv2.imread(img_file)[:,:,::-1]
    h, w = corner_response1.shape
    colored_img = cv2.resize(colored_img, (w, h), interpolation = cv2.INTER_CUBIC)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(colored_img)
    ax[0].scatter(x1, y1, marker = '.', color = 'red')
    ax[0].set_title("Without suppression")
    ax[1].imshow(colored_img)
    ax[1].scatter(x2, y2, marker = '.', color = 'red')
    ax[1].set_title("With suppression")
    for i in range(2):
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

def get_image_size(run):
    img_dir = os.path.join('runs', run, 'images')
    img_file = os.listdir(img_dir)[0]
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    return h, w

if __name__ == '__main__':
    img_file = '../runs/lib1/images/1.JPG'
    test_detection(img_file)