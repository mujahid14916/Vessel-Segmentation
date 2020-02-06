import numpy as np
import cv2


def resize_image(img, target_shape=(500, 500)):
    im = np.copy(img)
    if im.shape[0] > im.shape[1]:
        h, w = target_shape[0], int(im.shape[1] * target_shape[1] / im.shape[0])
    else:
        h, w = int(im.shape[0] * target_shape[0] / im.shape[1]), target_shape[1]
    im = cv2.resize(im, (w, h))
    if h != target_shape[0]:
        t = np.zeros((target_shape[0]//2 - h//2, target_shape[1], 3), dtype=np.uint8)
        im = np.concatenate([t, im, t], axis=0)
    if w != target_shape[1]:
        t = np.zeros((target_shape[0], target_shape[0]//2 - w//2, 3), dtype=np.uint8)
        im = np.concatenate([t, im, t], axis=1)
    return im[:target_shape[0], :target_shape[1]]


def is_fundus(img, p=0.05, max_th=20):
    img = np.asarray(img)
    h, w = img.shape[:2]
    if p > 0.15 or p < 0:
        p = 0.05
    left_top_sum = np.sum(img[0:int(h*p), 0:int(w*p)])
    right_bottom_sum = np.sum(img[int(h*(1 - p)):h, int(w*(1 - p)):w])
    right_top_sum = np.sum(img[0:int(h*p), int(w*(1 - p)):w])
    left_bottom_sum = np.sum(img[int(h*(1 - p)):h, 0:int(w*p)])
    total_sum = left_top_sum + right_bottom_sum + left_bottom_sum + right_top_sum
    return total_sum < max_th