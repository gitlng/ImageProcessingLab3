from Canny import toGray
import cv2
import numpy as np


def watershed(distances, input_picture):
    DIST_COEFF = 0.6
    inputed_picture_copy = np.copy(input_picture)
    gray_picture = toGray(inputed_picture_copy)
    rets, binare_picture = cv2.threshold(gray_picture,
                                         100, 255,
                                         cv2.THRESH_BINARY)
    back = cv2.dilate(binare_picture,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=3)
    ret, markers = cv2.connectedComponents(back)
    rets, fore = cv2.threshold(distances,
                               distances.max() * DIST_COEFF,
                               255, cv2.THRESH_BINARY)
    markers += 1
    a = cv2.subtract(back, np.uint8(fore))
    markers[a == 255] = 0
    markers = cv2.watershed(inputed_picture_copy, markers=markers)
    inputed_picture_copy[markers == -1] = (0, 0, 255)
    return inputed_picture_copy