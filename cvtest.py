import numpy as np
from cv2.cv2 import grabCut
from cv2 import cv2


def grab(img, rect):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    grabCut(img, mask, rect, bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    _i = img * mask2[:, :, np.newaxis]
    cv2.imwrite('test\\grab.jpg', _i)
