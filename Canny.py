import cv2
from copy import copy
import numpy as np
import math

# https://habr.com/ru/post/114589/

def toGray(img):
    (img_width, img_height, img_channel) = img.shape
    for x in range(img_width):
        for y in range(img_height):
            b = int(img.item(x, y, 0))
            g = int(img.item(x, y, 1))
            r = int(img.item(x, y, 2))
            avg = (b + g + r) // 3

            for i in range(3):
                img.itemset((x, y, i), avg)
    return img

def gauss(img):
    res_img = copy(img)
    width, height = img.shape[:2]
    kernel = np.array([[0.09272289, 0.11905855, 0.09272289], [0.11905855, 0.1528742 , 0.11905855], [0.09272289, 0.11905855, 0.09272289]])
    for x in range(1, width-1):
        for y in range(1, height-1):
            color = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    color = color + img[x + i][y + j][0] * kernel[i + 1][j + 1]
            res_img[x, y, :] = color
    return res_img

def sobel(img):
    res_img = copy(img)
    width, height = img.shape[:2]
    kernelX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradientMatrix = np.zeros((width, height))
    for x in range(1, width-1):
        for y in range(1, height-1):
            gradientX = 0
            gradientY = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    gradientX = gradientX + img[x + i][y + j][0] * kernelX[i + 1][j + 1]
                    gradientY = gradientY + img[x + i][y + j][0] * kernelY[i + 1][j + 1]
            gradientAvg = math.sqrt(gradientX ** 2 + gradientY ** 2)
            res_img[x, y, :] = gradientAvg
            gradientMatrix[x, y] = int(math.atan2(gradientX, gradientY) / (math.pi / 4)) * (math.pi / 4) - math.pi / 2 if gradientAvg != 0 else -1
    return res_img, gradientMatrix

def isCorrectIndex(width, height, x, y):
    res = x >= 0 and x < width and y >= 0 and y < height
    return res

def Check(img, x, y, v):
    width, height = img.shape[:2]
    if not isCorrectIndex(width, height, x, y): return False
    if img[x][y][0] <= v: return True
    return False

def removeNonMax(img, gradientMatrix):
    res_img = copy(img)
    width, height = img.shape[:2]
    for x in range(width):
        for y in range(height):
            if gradientMatrix[x][y] != -1:
                dx = int(np.sign(math.cos(gradientMatrix[x][y])))
                dy = int(np.sign(math.sin(gradientMatrix[x][y])) * (-1))
                if Check(img, x+dx, y+dy, img[x][y][0]):
                    res_img[x+dx][y+dy][:] = 0
                if Check(img, x-dx, y-dy, img[x][y][0]):
                    res_img[x-dx][y-dy][:] = 0
                res_img[x][y][:] = img[x][y][0]
    return res_img


def threshold(img):
    bottom = 128
    top = 180
    avg = (bottom + top) // 2
    res_img = copy(img)
    width, height = img.shape[:2]
    for x in range(width):
        for y in range(height):
            if img[x][y][0] <= bottom:
                res_img[x][y][:] = 0
            elif img[x][y][0] >= top:
                res_img[x][y][:] = 255
            else:
                res_img[x][y][:] = avg
    return res_img

def Canny(img):
    res_img = copy(img)
    print('canny: toGray start')
    res_img = toGray(res_img)
    print('canny: toGray end')
    print('canny: gauss start')
    res_img = gauss(res_img)
    print('canny: gauss end')
    print('canny: sobel start')
    res_img, gradientMatrix = sobel(res_img)
    print('canny: sobel end')
    print('canny: removeNonMax start')
    res_img = removeNonMax(res_img, gradientMatrix)
    print('canny: removeNonMax end')
    print('canny: threshold start')
    res_img = threshold(res_img)
    print('canny: threshold end')
    return res_img
