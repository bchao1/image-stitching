import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

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

def compute_corner_response(I, k = 0.04):
    '''
        I is 3 * 1 vector, where:
        I[0] = I_xx
        I[1] = I_xy
        I[2] = I_xx
    '''
    M = np.array([
        [I[0], I[1]],
        [I[1], I[2]]
    ])
    eig, _ = np.linalg.eig(M)
    det = eig[0] * eig[1]
    trace = eig[0] + eig[1]
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

def get_feature_discriptors(img, x_coors, y_coors, w = 3):
    assert len(x_coors) == len(y_coors)

    features = np.zeros((len(x_coors), w ** 2))
    img = np.pad(img, w // 2, 'symmetric')
    for i, (x, y) in enumerate(zip(x_coors, y_coors)):
        patch = img[y:y+w, x:x+w].ravel()
        mean = np.mean(patch)
        std = np.std(patch)
        patch = (patch - mean) / std
        features[i] = patch
    return features

def harris_corner_detection(img_file, downscale = 1, r_threshold = 1e7, max_features = 100):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print("Image size: {}".format(img.shape))
    img = cv2.resize(img, (w // downscale, h // downscale), interpolation = cv2.INTER_CUBIC)

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

    derivatives = np.transpose(np.stack((S_xx, S_xy, S_yy)), (1, 2, 0))
    corner_response = np.apply_along_axis(compute_corner_response, 2, derivatives)

    y, x = np.where(corner_response >= r_threshold)
    valid_response = corner_response[y, x] # those over threshold

    sorted_indices = (-valid_response).argsort() # sort indices by response value
    suppresion_radius = adaptive_non_maximal_suppresion(sorted_indices, valid_response, x, y)
    final_indices = (-suppresion_radius).argsort()[:max_features]
    y, x = y[final_indices], x[final_indices]

    features = get_feature_discriptors(img, x, y)
    return x, y, corner_response, features

def plot_detection(x, y, img_file, corner_response):
    colored_img = cv2.imread(img_file)[:,:,::-1]

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(corner_response, cmap = 'gray')
    ax[1].imshow(colored_img)
    ax[1].scatter(x, y, marker = '+', color = 'red')
    plt.show()

if __name__ == '__main__':
    img_file = 'chessboard.jpg'
    x, y, corner_response, features = harris_corner_detection(img_file)
    plot_detection(x, y, img_file, corner_response)