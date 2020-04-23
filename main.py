import os
import cv2
from lib.features.detection import harris_corner_detection

image_dir = 'images'
img_files = os.listdir(image_dir)