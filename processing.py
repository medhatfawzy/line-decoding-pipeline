import cv2
import numpy as np
from scipy.spatial import distance


def warp_img(img):
    # https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html
    img_h = img.shape[0]
    img_w = img.shape[1]

    src = np.float32(
        [[0, img_h], [1207, img_h], [0, img_h // 10], [img_w, img_h // 10]]
    )
    dst = np.float32([[569, img_h], [711, img_h], [0, 0], [img_w, 0]])

    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix

    img = img[500:(img_h), 0:img_w]  # Apply np slicing for ROI crop
    img = cv2.warpPerspective(img, M, (img_w, img_h))  # Image warping
    img = img[0 : img_h - 150, 350:900]
    return img


def apply_region_of_interest(img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    return img[500:(img_h), 0:img_w]


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    return cv2.GaussianBlur(img, kernel_size, 0)


def apply_canny_edge(img, low_threshold=100, high_threshold=200):
    return cv2.Canny(img, low_threshold, high_threshold)


def apply_dilation(img, kernel_size=(3, 3)):
    return cv2.dilate(img, np.ones(kernel_size, np.uint8))


def detect_hough_lines(img):
    return cv2.HoughLinesP(
        img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=2, maxLineGap=5
    )


def draw_hough_lines(img, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), 255, 5)
    return img


def detect_correct_mark(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.minAreaRect(contour) for contour in contours]
    center_bottom = (img.shape[1] // 2, img.shape[0] // 1.5)

    distances = [distance.euclidean(rect[0], center_bottom) for rect in rects]
    center_rect = rects[np.argmin(distances)]

    return center_rect


def map_values(rect: tuple, img: np.ndarray) -> tuple[float, float]:
    """
    Map the values for the steer to (-1, 1)
    and the values for the throttle to (0, 1)

    Parameters
    ----------
    rect: tuple
        the rectangle that is used to determine the throttle and steering angle
    Return
    ------
    throttle: float
        the throttle for the car
    steer: float
        the steering angle for the car
    """
    center = rect[0]
    img_center = (img.shape[1] // 2, img.shape[0] // 1.5)

    offset = center[0] - img_center[0]  # offset from the center of the image
    offset = offset / img_center[0]  # normalize the offset

    d1 = rect[1][0]
    d2 = rect[1][1]
    width = min(rect[1][1], rect[1][0])
    height = max(rect[1][1], rect[1][0])
    angle = rect[2]

    # rounding to the nearest 5
    width = int(5 * round(width / 5))
    angle = int(5 * round(angle / 5))

    if angle in (0, 90, -0, -90):
        angle = 0

    elif d1 < d2:
        angle = 90 - angle

    else:
        angle = -angle

    throttle = width / 90
    steer = angle / (100 + throttle * 500) + offset

    return throttle, steer


def process_img(img: np.ndarray) -> tuple[float, float]:
    """
    Process image to find the angle and width of the rectangle in the image

    Parameters
    ----------
    img: np.ndarray
        image to process

    Return
    ------
    width: float
        width of the rectangle in the image representing the throttle
    angle: float
        angle of the rectangle in the image representing the steering angle
    """
    img = warp_img(img)
    img_w, img_h = img.shape[1], img.shape[0]

    img_blur = apply_gaussian_blur(img.copy())
    img_canny = apply_canny_edge(img_blur)
    img_canny = apply_dilation(img_canny)

    lines = detect_hough_lines(img_canny.copy())

    if lines is None:
        return 0, 0

    img_hou = np.zeros((img_h, img_w), dtype=np.uint8)
    draw_hough_lines(img_hou, lines)

    center_rect = detect_correct_mark(img_hou)

    throttle, steer = map_values(center_rect, img_hou)
    return throttle, steer
