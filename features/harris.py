import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

''' Implementation of Harris Corner Detector '''

def sobel_filter(ori):
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
    
img = cv2.imread('house.jpg', cv2.IMREAD_GRAYSCALE)
downscale = 1
h, w = img.shape
img = cv2.resize(img, (w // downscale, h // downscale), interpolation = cv2.INTER_CUBIC)

I_x = convolve2d(img, sobel_filter('x'), boundary = 'symm', mode = 'same')
I_y = convolve2d(img, sobel_filter('y'), boundary = 'symm', mode = 'same')
print(img.shape, I_x.shape, I_y.shape)

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
    res = np.array([sorted_indices[0]]) # global maximum must be included
    radius = np.zeros(sorted_indices.shape)
    radius[sorted_indices[0]] = np.inf
    ct = 0
    for i in sorted_indices[1:]:
        neighbor_response = response[res] * c_robust
        valid_neighbor = res[np.where(response[i] < neighbor_response)]
        if len(valid_neighbor) == 0:
            continue
        #print(valid_neighbor)
        delta_x = x[valid_neighbor] - x[i]
        delta_y = y[valid_neighbor] - y[i]
        r_min = np.min(np.sqrt(delta_x ** 2 + delta_y ** 2))
        radius[i] = r_min
        res = np.append(res, [i])
    return radius


threshold = 1e7
max_features = 100
corner_response = np.apply_along_axis(compute_corner_response, 2, derivatives)

y, x = np.where(corner_response >= threshold)
valid_response = corner_response[y, x] # those over threshold
print(valid_response.shape)
sorted_indices = (-valid_response).argsort()
suppresion_radius = adaptive_non_maximal_suppresion(sorted_indices, valid_response, x, y)
final_indices = (-suppresion_radius).argsort()[:max_features]
y, x = y[final_indices], x[final_indices]

f, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap = 'gray')
ax[1].imshow(corner_response, cmap = 'gray')
ax[1].scatter(x, y, marker = '+', color = 'red')
plt.show()