import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def honglvdeng(file):
    '''
    1红2绿3黄
    :param file:文件路径
    :return:
    '''
    img = cv.imread(file)
    old_img = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 1)
    # 大津法二值化
    _, blur_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    # 膨胀/腐蚀/膨胀
    img = cv.dilate(blur_img, kernel, iterations=2)
    img = cv.erode(img, kernel, iterations=1)
    img = cv.dilate(img, kernel, iterations=2)

    # 轮廓信息
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    best_area = None
    best_cnt = None
    for i in range(len(contours)):  # 寻找最优可能的选区
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area < 500:
            continue

        (x, y), radius = cv.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        if area / 2 > circle_area or area * 2 < circle_area:
            continue

        abs_area = np.abs(circle_area - area)
        if best_area is None:
            best_area = abs_area
            best_cnt = cnt
        elif abs_area < best_area:
            best_area = abs_area
            best_cnt = cnt
    if best_area is None:
        return 0

    rect = cv.minAreaRect(best_cnt)
    box = cv.boxPoints(rect)
    box = np.int64(box)
    min_x, min_y = np.min(box, axis=0)
    max_x, max_y = np.max(box, axis=0)
    mask = np.zeros(shape=(np.shape(old_img)[0:2]), dtype=np.uint8)
    mask[min_y: max_y, min_x:max_x] = 255

    result = cv.bitwise_and(old_img, old_img, mask=mask)
    red_sum = np.sum(result[:, :, 2])
    green_sum = np.sum(result[:, :, 1])
    blue_sum = np.sum(result[:, :, 0])
    print(red_sum / 10000, green_sum / 10000, blue_sum / 10000)

    if red_sum / green_sum > 2:
        return 1
    if 0.5 < red_sum / blue_sum < 2 and 0.5 < green_sum / blue_sum < 2:
        return 2
    if red_sum / blue_sum > 2 and green_sum / blue_sum > 2:
        return 3
    cv.imshow('img2', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    print('red.jpg', honglvdeng("red.jpg"))
    print('red2.jpg', honglvdeng("red2.jpg"))
    print('green.jpg', honglvdeng("green.jpg"))
    print('green2.jpg', honglvdeng("green2.jpg"))
    print('yellow2.jpg', honglvdeng("yellow2.jpg"))
