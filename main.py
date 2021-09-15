import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
from skimage.draw import ellipse, polygon
import cv2


init = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, ".", (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        init.append([x, y])


def creat_mask(img, init, type="ellipse"):
    shape = img.shape[0:2]
    mask = np.zeros([shape[0], shape[1]], dtype=bool)
    init = np.array(init)  # initial_points = init
    if type == "ellipse":
        r, c = init[0, 1], init[0, 0]
        rr, cc = np.abs(init[2, 1] - r), np.abs(init[1, 0] - c)
        rrr, ccc = ellipse(r, c, rr, cc)
    else:
        r, c = init[:, 1], init[:, 0]
        rrr, ccc = polygon(r, c)
    img2 = 0 * img
    img2[rrr, ccc] = img[rrr, ccc]
    mask[rrr, ccc] = True
    cv2.imwrite('masked.jpg', img2)
    return mask


def blend(img_target, img_source, mask, transfer=(0, 0)):
    y, x = np.where(mask == True)
    x_start = np.min(x) - 2
    x_end = np.max(x) + 2
    y_start = np.min(y) - 2
    y_end = np.max(y) + 2
    window_size = (y_end-y_start, x_end-x_start)
    mask = mask[y_start:y_end, x_start:x_end]

    n = window_size[0]*window_size[1]
    A = scipy.sparse.identity(n, format='lil')
    y, x = np.where(mask == True)
    ind = x + window_size[1] * y
    for i in ind:
        A[i, i] = 4
        if i + 1 < n:
            A[i, i + 1] = -1
        if i - 1 >= 0:
            A[i, i - 1] = -1
        if i + window_size[1] < n:
            A[i, i + window_size[1]] = -1
        if i - window_size[1] >= 0:
            A[i, i - window_size[1]] = -1
    A = A.tocsr()

    for channel in range(img_target.shape[2]):
        t = img_target[transfer[1]:window_size[0]+transfer[1],
                        transfer[0]:window_size[1]+transfer[0], channel]
        s = img_source[y_start:y_end, x_start:x_end, channel]
        t = t.flatten()
        grad_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        b = cv2.filter2D(s, cv2.CV_64F, grad_filter)
        b = b.flatten()
        y, x = np.where(mask==False)
        ind = x + window_size[1] * y
        b[ind] = t[ind]
        x = spsolve(A, b)
        x = np.reshape(x, window_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[transfer[1]:window_size[0]+transfer[1],
        transfer[0]:window_size[1]+transfer[0], channel] = x
    return img_target


def resize(img, coefficient = 1):
    if coefficient == 1:
        return img
    else:
        shape = img.shape
        shape = np.int_(np.array(shape)/coefficient)
        return cv2.resize(img, (shape[1], shape[0]))


def equalize_size(img1, img2, param="yes"):
    if param == "yes":
        if np.prod(img1.shape) < np.prod(img2.shape):
            shape = img1.shape
            img2 = cv2.resize(img2, (shape[1], shape[0]))
        else:
            shape = img2.shape
            img1 = cv2.resize(img1, (shape[1], shape[0]))
    return img1, img2


img_source = cv2.imread('1.source.jpg')
img_target = cv2.imread('2.target.jpg')
img_source, img_target = equalize_size(img_source, img_target, param="no")
img_source = resize(img_source, coefficient=1)
img_target = resize(img_target, coefficient=1)
img = np.copy(img_source)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyWindow("image")

mask = creat_mask(img_source, init, type="polygon")
init = []
img = np.copy(img_target)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyWindow("image")
init = np.array(init)
img_ret = blend(img_target, img_source, mask, transfer=init[0, :])
cv2.imwrite('res1.jpg', img_ret)
