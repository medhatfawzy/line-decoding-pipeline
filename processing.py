import os
import time
import tempfile

import cv2
import numpy as np
from scipy.spatial import distance

import airsim


class Driver:
    def __init__(self, client, car_controls):
        """
        Parameters
        ----------
        client: airsim.CarClient
            the client to connect to the AirSim simulator
        car_controls: airsim.CarControls
            the controls for the car
        """
        self.client = client
        self.car_controls = car_controls
        self.processor = IMG_Processor()

    def drive(self, save_input=False):
        """
        Drive the car in the simulator

        Parameters
        ----------
        save_input: bool
            whether to save the input images to a temporary directory
        """

        self.save_input = save_input
        if self.save_input:
            self.img_counter = 0
            self.tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
            print(f"Saving images to {self.tmp_dir}")
            try:
                os.makedirs(self.tmp_dir)
            except OSError:
                if not os.path.isdir(self.tmp_dir):
                    raise f"Could not create directory {self.tmp_dir}"
        
        self.car_controls.throttle = 1
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)
        print("Go Forward")
        time.sleep(7)

        while self.client.getCarState().speed > 0:
            # get camera images from the car
            response = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        "front_center", airsim.ImageType.Scene, False, False
                    )
                ]
            )[0]  # scene vision image in uncompressed RGB array

            # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))

            img = np.frombuffer(
                response.image_data_uint8, dtype=np.uint8
            )  # get numpy array
            img = img.reshape(
                response.height, response.width, 3
            )  # reshape array to 3 channel image array H X W X 3
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            throttle, steer = self.processor.process_img(img_gray)
            print(f"Throttle: {throttle:.2f}, Steer: {steer:.2f}")

            if self.save_input:
                self._save_image(img_rgb, throttle, steer)

            if throttle < 0:
                self.car_controls.brake = 1.0
                self.car_controls.throttle = 0
                self.car_controls.steering = 0
                self.client.setCarControls(self.car_controls)
                break
            else:
                self.car_controls.throttle = throttle
                self.car_controls.steering = steer
                self.client.setCarControls(self.car_controls)

        # restore to original state
        # client.reset()
        self.client.enableApiControl(False)

    def _save_image(self, img_rgb, throttle=None, steer=None):
        # img_name = f"curved_{self.img_counter}_{throttle:.2f}_{steer:.2f}.png"
        img_name = f"curved_{self.img_counter}.png"
        file_path = os.path.join(self.tmp_dir, img_name)
        cv2.imwrite(os.path.normpath(file_path), img_rgb)
        self.img_counter += 1


class IMG_Processor:
    '''
    This class is used to process the image from the car
    '''
    def __init__(self):
        self.car_throttle = 0
        self.car_steer = 0

    @staticmethod
    def warp_img(img):
        img_h, img_w = img.shape[0], img.shape[1]

        new_w = 300
        new_h = 400
        src = np.float32(
            [
                [img_w // 3.8, img_h // 1.8], 
                [img_w // 1.5, img_h // 1.8], 
                [img_w, img_h], 
                [0, img_h]
            ]
        )
        dst = np.float32(
            [
                [0, 0],
                [new_w, 0],
                [new_w, new_h],
                [0, new_h],
            ]
        )


        M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix

        # Image warping
        img = cv2.warpPerspective(
            img,
            M,
            (new_w, new_h),
        )
        return img

    @staticmethod
    def apply_gaussian_blur(img, kernel_size=(3, 3)):
        return cv2.GaussianBlur(img, kernel_size, 0)

    @staticmethod
    def apply_canny_edge(img, low_threshold=100, high_threshold=200):
        return cv2.Canny(img, low_threshold, high_threshold)

    @staticmethod
    def apply_dilation(img, kernel_size=(3, 3)):
        return cv2.dilate(img, np.ones(kernel_size, np.uint8))

    @staticmethod
    def detect_hough_lines(img):
        return cv2.HoughLinesP(
            img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=2, maxLineGap=5
        )

    @staticmethod
    def draw_hough_lines(img, lines):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), 255, 3)
        return img

    @staticmethod
    def detect_correct_mark(img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.minAreaRect(contour) for contour in contours]
        center_bottom = (img.shape[1] // 2, img.shape[0] // 1.5)

        distances = [distance.euclidean(rect[0], center_bottom) for rect in rects]
        center_rect = rects[np.argmin(distances)]

        return center_rect

    def map_values(
        self,
        rect: tuple,
        img: np.ndarray,
    ) -> tuple[float, float]:
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

        throttle = max(width / (100 + self.car_throttle), 0.5)  # A trial and error value
        steer = angle / (90 + throttle * 10) + (offset)  # A trial and error value

        self.car_throttle = throttle
        self.car_steer = steer
        return throttle, steer

    def process_img(self, img: np.ndarray) -> tuple[float, float]:
        """
        Process image to find the angle and width of the rectangle in the image

        Parameters
        ----------
        img: np.ndarray
            image to process

        Return
        ------
        throttle: float
            the throttle for the car
        steer: float
            the steering angle for the car
        """
        img = IMG_Processor.warp_img(img)
        img = IMG_Processor.apply_gaussian_blur(img)
        img = IMG_Processor.apply_canny_edge(img)
        img = IMG_Processor.apply_dilation(img)
        img_w, img_h = img.shape[1], img.shape[0]

        lines = IMG_Processor.detect_hough_lines(img)
        if lines is None:
            return -1, 0

        img_hou = np.zeros((img_h, img_w), dtype=np.uint8)
        img_hou = IMG_Processor.draw_hough_lines(img_hou, lines)

        center_rect = IMG_Processor.detect_correct_mark(img_hou)

        throttle, steer = self.map_values(center_rect, img_hou)
        return throttle, steer
