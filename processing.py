import cv2
import numpy as np


def warp_img(img):
    # https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html
    img_h = img.shape[0]
    img_w = img.shape[1]

    src = np.float32([[0, img_h], [1207, img_h], [0, img_h // 10], [img_w, img_h // 10]])
    dst = np.float32([[569, img_h], [711, img_h], [0, 0], [img_w, 0]])

    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix

    img = img[500:(img_h), 0:img_w] # Apply np slicing for ROI crop
    img = cv2.warpPerspective(img, M, (img_w, img_h)) # Image warping
    img = img[0:img_h-150, 350:900]
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
        img,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=2,
        maxLineGap=5
    )

def draw_hough_lines(img, lines):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), 255, 5)
    return img

def find_largest_contour(img_lines):
    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea, default=None)


def map_values(width: float, height:float, angle:float) -> (float, float):
    """
    Map the values for the angle to (-1, 1)
    and the values for the width to (0, 1)

    Parameters
    ----------
    width: float
        width of the rectangle in the image representing the throttle
    angle: float
        angle of the rectangle in the image representing the steering angle

    Return
    ------
    throttle: float
        the throttle for the car
    steer: float
        the steering angle for the car
    """
    # rounding to the nearest 5
    width = int(5 * round(width/5))
    angle = int(5 * round(angle/5))

    if angle in (0, 90, -0, -90):
        angle = 0

    elif  15 < angle:
        angle = - angle

    throttle = width / 110
    steer = angle / 90

    return throttle, steer


def process_img(img: np.ndarray) -> (float, float):
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
    img_w, img_h = img.shape[1], img.shape[0]
    img = warp_img(img)

    img_blur = apply_gaussian_blur(img.copy())
    img_canny = apply_canny_edge(img_blur)
    img_canny = apply_dilation(img_canny)
    
    lines = detect_hough_lines(img_canny.copy())
    img_hou = np.zeros((img_h, img_w), dtype=np.uint8)
    draw_hough_lines(img_hou, lines)
    
    biggest_rectangle = find_largest_contour(img_hou)
    if biggest_rectangle is None:
        return 0.5, 0
    
    rect = cv2.minAreaRect(biggest_rectangle)
    width = min(rect[1][1], rect[1][0])
    height = max(rect[1][1], rect[1][0])
    angle = rect[2]
    throttle, steer = map_values(width, height, angle)
    return throttle, steer
