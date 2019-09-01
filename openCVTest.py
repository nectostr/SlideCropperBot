import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

@jit
def to_one_chanel(_img) -> np.array:
    img = np.empty((_img.shape[0], _img.shape[1]),dtype=np.int)
    img[:,:] = _img[:,:,:].mean()
    return img

@jit
def erosion(_img, mask=3):
    side = mask // 2
    img = np.append([_img[0]]*side, _img, axis=0)
    img = np.append(img, [_img[-1]]*side, axis=0)
    img2 = np.zeros((img.shape[0], img.shape[1] + side*2))
    img2[:, side:-side] = img.copy()
    a = np.array([list(img[:, 0])]*side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, :side] = a
    a = np.array([list(img[:, -1])] * side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, -side:] = a
    img = np.empty(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = i + side
            j1 = j + side
            img[i, j] = img2[i1 - side:i1 + side + 1, j1 - side:j1 + side + 1].min()
    return img

@jit
def dilatation(_img, mask=3):
    side = mask // 2
    img = np.append([_img[0]] * side, _img, axis=0)
    img = np.append(img, [_img[-1]] * side, axis=0)
    img2 = np.zeros((img.shape[0], img.shape[1] + side*2))
    img2[:, side:-side] = img.copy()
    a = np.array([list(img[:, 0])] * side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, :side] = a
    a = np.array([list(img[:, -1])] * side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, -side:] = a
    img = np.empty(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = i + side
            j1 = j + side
            img[i, j] = img2[i1 - side:i1 + side + 1, j1 - side:j1 + side + 1].max()
    return img

def try1(img):
    print(type(img))
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img_bw.shape)
    edj = cv2.Canny(img_bw, 30, 100, apertureSize=3)
    plt.imshow(edj)
    plt.figure()
    lines = cv2.HoughLines(edj, 3, np.pi/150, 100)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    plt.imshow(img)
    plt.show()

def try2(img):
    largest_area = 0
    largest_contour_index = 0
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plt.imshow(img2)
    plt.show()

def cut_image(img):
    """
    1) middle point
    2) brightness of middle-area
    3) treshold on brightness
    :param img:
    :return:
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # average_brightness = img_grey[img_grey.shape[0]//7: 6*img_grey.shape[0]//7, img_grey.shape[1]//7:6*img_grey.shape[1]//7].mean()
    average_brightness = img_grey.mean()

    img2 = np.array(img_grey > average_brightness, dtype=np.float32)
    kernel = np.ones((10, 10), np.uint8)
    img3 = cv2.erode(img2, kernel, iterations=1)
    img3 = cv2.dilate(img3, kernel, iterations=1)

    corn_empty_stepX = img3.shape[0]//10
    corn_empty_stepY = img3.shape[1]//10

    img3[0:corn_empty_stepX,0:corn_empty_stepY] = np.zeros((corn_empty_stepX,corn_empty_stepY))
    img3[img3.shape[0]-corn_empty_stepX:, 0:corn_empty_stepY] = np.zeros((corn_empty_stepX, corn_empty_stepY))
    img3[img3.shape[0] - corn_empty_stepX:, img3.shape[1]-corn_empty_stepY:] = np.zeros((corn_empty_stepX, corn_empty_stepY))
    img3[0:corn_empty_stepX, img3.shape[1]-corn_empty_stepY:] = np.zeros((corn_empty_stepX, corn_empty_stepY))

    corn1 = np.full(img3.shape, np.inf)
    corn2 = np.full(img3.shape, np.inf)
    corn3 = np.full(img3.shape, np.inf)
    corn4 = np.full(img3.shape, np.inf)

    for i in range(img3.shape[0]):
        for j in range(img3.shape[1]):
            if img3[i, j] == 1:
                corn1[i, j] = (i**2+j**2)**(1/2)
                corn2[i, j] = (i ** 2 + (j-img3.shape[1]) ** 2) ** (1 / 2)
                corn3[i, j] = ((i-img3.shape[0]) ** 2 + (j - img3.shape[1]) ** 2) ** (1 / 2)
                corn4[i, j] = ((i - img3.shape[0]) ** 2 + j ** 2) ** (1 / 2)

    c1 = np.unravel_index(corn1.argmin(), corn1.shape)
    c2 = np.unravel_index(corn2.argmin(), corn2.shape)
    c3 = np.unravel_index(corn3.argmin(), corn3.shape)
    c4 = np.unravel_index(corn4.argmin(), corn4.shape)

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

    print(srcPoints)
    plt.imshow(img3)
    plt.scatter(srcPoints[:, 0], srcPoints[:, 1])
    plt.savefig("image.jpg")

    warp_mat = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    img4 = cv2.warpPerspective(img, warp_mat, (maxWidth, maxHeight))
    # plt.imshow(img4)
    # plt.show()
    return img4

if __name__ == '__main__':
    img = cv2.imread(r".\data\photo_001.jpg")
    cut_image(img)