import cv2

def read_image(img_path, downscale):
    img = cv2.imread(img_path)[:,:,::-1]
    h, w, _ = img.shape
    img = cv2.resize(img, (w // downscale, h // downscale), interpolation = cv2.INTER_CUBIC)
    return img