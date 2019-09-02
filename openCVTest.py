import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time
import logging


@jit(nopython=True)
def get_corner_matrix(img3):
    corn1 = np.full(img3.shape, np.inf)
    corn2 = np.full(img3.shape, np.inf)
    corn3 = np.full(img3.shape, np.inf)
    corn4 = np.full(img3.shape, np.inf)

    for i in range(img3.shape[0]):
        for j in range(img3.shape[1]):
            if img3[i, j] == 1:
                corn1[i, j] = i+j
                corn2[i, j] = i + abs(j - img3.shape[1])
                corn3[i, j] = -i-j
                corn4[i, j] = abs(i - img3.shape[0]) + j

    return corn1, corn2, corn3, corn4

def ones_for_screen_matrix_light(img_grey):

    # average_brightness = img_grey[img_grey.shape[0]//7: 6*img_grey.shape[0]//7, img_grey.shape[1]//7:6*img_grey.shape[1]//7].mean()
    average_brightness = img_grey.mean()

    img2 = np.array(img_grey > average_brightness, dtype=np.float32)
    kernel = np.ones((10, 10), np.uint8)
    img3 = cv2.erode(img2, kernel, iterations=1)
    img3 = cv2.dilate(img3, kernel, iterations=1)
    return img3

def ones_for_screen_matrix_dark(img_grey):

    average_brightness = (img_grey[:, :img_grey.shape[1]//6].sum() + img_grey[:, 5*img_grey.shape[1]//6:].sum() +
                          img_grey[:img_grey.shape[0]//6, img_grey.shape[1]//6:5*img_grey.shape[1]//6].sum() +
                          img_grey[5*img_grey.shape[0]//6:, img_grey.shape[1]//6:5*img_grey.shape[1]//6].sum()) / \
                         (img_grey.shape[1]//6*img_grey.shape[0] * 2 + img_grey.shape[0]//6 * 2/3*img_grey.shape[1] * 2) * 1.2

    img2 = np.array(img_grey > average_brightness, dtype=np.float32)
    kernel = np.ones((10, 10), np.uint8)
    img3 = cv2.erode(img2, kernel, iterations=1)
    img3 = cv2.dilate(img3, kernel, iterations=1)
    return img3

def cut_image(img):
    """
    1) middle point
    2) brightness of middle-area
    3) treshold on brightness
    :param img:
    :return:
    """
    t1 = time.time()
    logging.info("Start photo cutting")

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_grey.mean() > 255/2:
        img3 = ones_for_screen_matrix_light(img_grey)
    else:
        img3 = ones_for_screen_matrix_dark(img_grey)

    logging.info(f"Photo black-white creating. {time.time() - t1}")
    t1 = time.time()


    corn_empty_stepX = img3.shape[0]//10
    corn_empty_stepY = img3.shape[1]//10

    img3[0:corn_empty_stepX,0:corn_empty_stepY] = np.zeros((corn_empty_stepX,corn_empty_stepY))
    img3[img3.shape[0]-corn_empty_stepX:, 0:corn_empty_stepY] = np.zeros((corn_empty_stepX, corn_empty_stepY))
    img3[img3.shape[0] - corn_empty_stepX:, img3.shape[1]-corn_empty_stepY:] = np.zeros((corn_empty_stepX, corn_empty_stepY))
    img3[0:corn_empty_stepX, img3.shape[1]-corn_empty_stepY:] = np.zeros((corn_empty_stepX, corn_empty_stepY))

    corn1, corn2, corn3, corn4 = get_corner_matrix(img3)

    c1 = np.unravel_index(corn1.argmin(), corn1.shape)
    c2 = np.unravel_index(corn2.argmin(), corn2.shape)
    c3 = np.unravel_index(corn3.argmin(), corn3.shape)
    c4 = np.unravel_index(corn4.argmin(), corn4.shape)

    logging.info(f"Corner matrix prep func finished {time.time() - t1}")
    t1 = time.time()
    # print(c1, c2, c3, c4)
    # print((img3.shape[1], img3.shape[0]))
    tl = list(c1)
    tr = list(c2)
    br = list(c3)
    bl = list(c4)

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    srcPoints = np.array([[tl[1], tl[0]], [tr[1], tr[0]], [br[1], br[0]], [bl[1], bl[0]]], dtype="float32")
    dstPoints = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    logging.debug(f"Source corner points is: \n{srcPoints}")
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        plt.imshow(img3)
        plt.scatter(srcPoints[:, 0], srcPoints[:, 1])
        plt.savefig(f"./logged_image/img_{time.time()}.jpg")

    logging.info(f"Additional point aloc finished: {time.time() - t1}")
    t1 = time.time()

    warp_mat = cv2.getPerspectiveTransform(srcPoints, dstPoints)

    logging.info(f"Warp matrix creation finished finished: {time.time() - t1}")
    t1 = time.time()
    img4 = cv2.warpPerspective(img, warp_mat, (maxWidth, maxHeight))
    logging.debug(f"Warp finished: {time.time() - t1}")
    # plt.imshow(img4)
    # plt.show()
    return img4

if __name__ == '__main__':
    img = cv2.imread(r".\data\photo_001.jpg")
    cut_image(img)