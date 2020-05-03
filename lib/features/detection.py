import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, sobel

''' Implementation of Feature detection '''

def sobel_filter(ori):
    ''' x, y direction sobel filter. '''
    if ori == 'x':
        return np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])
    else:
        return np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])

def compute_corner_response(S_xx, S_xy, S_yy, k = 0.04):
    det = S_xx * S_yy - S_xy ** 2
    trace = S_xx + S_yy
    response = det - k * (trace ** 2)
    return response

def adaptive_non_maximal_suppresion(sorted_indices, response, x, y, c_robust = 1):
    radius = np.zeros(sorted_indices.shape)
    radius[sorted_indices[0]] = np.inf
    for i, idx in enumerate(sorted_indices[1:], start = 1):
        neighbor_response = response[sorted_indices[:i]] * c_robust
        valid_neighbor = sorted_indices[:i][np.where(response[idx] < neighbor_response)]
        if len(valid_neighbor) == 0: # no valid neighbors
            continue
        delta_x = x[valid_neighbor] - x[idx]
        delta_y = y[valid_neighbor] - y[idx]
        r_min = np.min(np.sqrt(delta_x ** 2 + delta_y ** 2))
        radius[idx] = r_min
    return radius

def get_feature_discriptors(img, x_coors, y_coors, w):
    assert len(x_coors) == len(y_coors)

    features = np.zeros((len(x_coors), 8 ** 2))
    img = np.pad(img, w // 2, 'symmetric')
    for i, (x, y) in enumerate(zip(x_coors, y_coors)):
        patch = cv2.resize(img[y:y+w, x:x+w], (8, 8), interpolation = cv2.INTER_CUBIC)
        patch = patch.ravel() # flatten
        mean = np.mean(patch)
        std = np.std(patch)
        patch = (patch - mean) / std
        features[i] = patch
    return features

def harris_corner_detection(img_file, downscale, r_threshold = 1e8, max_features = 500, window = 20):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img = cv2.resize(img, (w // int(downscale), h // int(downscale)), interpolation = cv2.INTER_CUBIC)
    print("Image size: {}".format(img.shape))

    I_x = convolve2d(img, sobel_filter('x'), boundary = 'symm', mode = 'same')
    I_y = convolve2d(img, sobel_filter('y'), boundary = 'symm', mode = 'same')

    I_xx = I_x ** 2
    I_yy = I_y ** 2
    I_xy = I_x * I_y

    # Gaussian filter params
    sigma = 1.0
    window_size = 3
    truncate = ((window_size - 1) // 2 - 0.5) / sigma

    S_xx = gaussian_filter(I_xx, sigma = 1, truncate = truncate)
    S_yy = gaussian_filter(I_yy, sigma = 1, truncate = truncate)
    S_xy = gaussian_filter(I_xy, sigma = 1, truncate = truncate)

    corner_response = compute_corner_response(S_xx, S_xy, S_yy, k = 0.04)
    y, x = np.where(corner_response >= r_threshold)
    valid_response = corner_response[y, x] # those over threshold
    print(valid_response.shape)

    sorted_indices = (-valid_response).argsort() # sort indices by response value
    suppresion_radius = adaptive_non_maximal_suppresion(sorted_indices, valid_response, x, y)
    final_indices = (-suppresion_radius).argsort()[:max_features]
    y, x = y[final_indices], x[final_indices]

    features = get_feature_discriptors(img, x, y, w = window)
    return x, y, corner_response, features

def get_features(run, downscale, r_threshold = 1e8, max_features = 500, window = 20, use_cache = True):
    img_dir = os.path.join('runs', run, 'images')
    detection_dir = os.path.join('runs', run, 'detection')
    if os.path.exists(detection_dir) and use_cache:
        x_coors = np.load(os.path.join(detection_dir, 'x.npy'))
        y_coors = np.load(os.path.join(detection_dir, 'y.npy'))
        features = np.load(os.path.join(detection_dir, 'features.npy'))
        return x_coors, y_coors, features
    if not os.path.exists(detection_dir):
        os.mkdir(detection_dir)
    
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

if __name__ == '__main__':
    img_file = '../../images.jpg'
    x, y, corner_response, features = harris_corner_detection(img_file)
    plot_detection(x, y, img_file, corner_response)