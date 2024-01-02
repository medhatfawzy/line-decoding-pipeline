import os
import time
import tempfile

import cv2
import numpy as np
from scipy.spatial import distance

import airsim
import torch


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
        self.processor.set_precessor(Algo_Processor())

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
        time.sleep(5)

        while self.client.getCarState().speed > 0:
            # get camera images from the car
            response = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        "front_center", airsim.ImageType.Scene, False, False
                    )
                ]
            )[
                0
            ]  # scene vision image in uncompressed RGB array

            # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))

            img = np.frombuffer(
                response.image_data_uint8, dtype=np.uint8
            )  # get numpy array
            img = img.reshape(
                response.height, response.width, 3
            )  # reshape array to 3 channel image array H X W X 3
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            throttle, steer = self.processor.process_img(img_gray)
            print(f"Throttle: {throttle:.2f}, Steer: {steer:.2f}\n")

            if self.save_input:
                self._save_image(img, throttle, steer)

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

    def _save_image(self, img, throttle=None, steer=None):
        # img_name = f"curved_{self.img_counter}_{throttle:.2f}_{steer:.2f}.png"
        img_name = f"curved_{self.img_counter}.png"
        file_path = os.path.join(self.tmp_dir, img_name)
        cv2.imwrite(os.path.normpath(file_path), img)
        self.img_counter += 1


class IMG_Processor:
    """
    This is the standard interface for the image processor
    """

    def __init__(self):
        self.pressor = None

    def set_precessor(self, processor):
        self.processor = processor

    def process_img(self, img):
        return self.processor.process_img(img)


class DL_Processor(IMG_Processor):
    """
    This class is used to process the image from the car using the deep learning model
    """

    def __init__(self, model_path=r"reg model/model.pt"):
        self.model = torch.load(model_path).double()

    def process_img(self, img: np.ndarray) -> tuple[float, float]:
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.expand_dims(img, axis=0)
        img = img.transpose(0, 3, 1, 2)
        img = img / 255
        img = torch.from_numpy(img).double()

        steer, throttle = self.model.forward(img).tolist()[0]

        throttle = DL_Processor.round_pred(throttle, -0.5)
        return throttle, steer

    @staticmethod
    def round_pred(pred, threshold=0):
        if pred < 0.4 and pred > threshold:
            return 0.4
        elif pred < threshold:
            return -1
        else:
            return pred


class Algo_Processor(IMG_Processor):
    """
    This class is used to process the image from the car using the algorithm we have developed
    """

    def __init__(self):
        self.car_throttle = 0
        self.car_steer = 0
        self.center_bottom = 0

    @staticmethod
    def warp_img(img):
        IMAGE_H = img.shape[0]
        IMAGE_W = img.shape[1]
        new_h = 640
        new_w = 300
        src = np.float32(
            [
                [0, IMAGE_H],
                [IMAGE_W, IMAGE_H],
                [int(IMAGE_W // 2.2), int(IMAGE_H // 1.7)],
                [int(IMAGE_W // 1.8), int(IMAGE_H // 1.7)],
            ]
        )
        dst = np.float32(
            [
                [0, new_h],
                [new_w, new_h],
                [0, 0],
                [new_w, 0],
            ]
        )

        M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
        img = cv2.warpPerspective(img, M, (new_w, new_h))  # Image warping
        return img

    @staticmethod
    def gaussian_blur(img, kernel_size=(3, 3)):
        return cv2.GaussianBlur(img, kernel_size, 0)

    @staticmethod
    def canny_edge(img, low_threshold=100, high_threshold=200):
        return cv2.Canny(img, low_threshold, high_threshold)

    @staticmethod
    def dilation(img, kernel_size=(3, 3)):
        return cv2.dilate(img, np.ones(kernel_size, np.uint8))

    @staticmethod
    def detect_hough_lines(img):
        return cv2.HoughLinesP(
            img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=5, maxLineGap=10
        )

    @staticmethod
    def draw_hough_lines(img, lines):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), 255, 3)
        return img

    def detect_correct_mark(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.minAreaRect(contour) for contour in contours]
        self.center_bottom = (img.shape[1] // 2, img.shape[0] // 1.2)

        distances = [distance.euclidean(rect[0], self.center_bottom) for rect in rects]
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

        offset = (
            center[0] - self.center_bottom[0]
        )  # offset from the center of the image
        offset = offset / img.shape[1]  # normalize the offset

        # print(f"Offset: {offset:.2f}")
        d1 = rect[1][0]
        d2 = rect[1][1]
        width = min(d1, d2)
        angle = rect[2]

        # rounding to the nearest 5
        width = int(5 * round(width / 5))
        angle = int(5 * round(angle / 5))

        if int(angle) in {0, 90, -0, -90}:
            angle = 0

        elif d1 < d2:
            angle = 90 - angle

        else:
            angle = -angle

        steer = angle / ((self.car_throttle + self.car_steer) * 100 + 90) + (offset * 2)  # A trial and error value
        steer = max(min(steer, 1), -1)
        throttle = max(width / (120 + (np.abs(steer) * 50)), 0.3)  # A trial and error value

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
        img = Algo_Processor.warp_img(img)
        img = Algo_Processor.gaussian_blur(img)
        img = Algo_Processor.canny_edge(img)
        img = Algo_Processor.dilation(img)

        lines = Algo_Processor.detect_hough_lines(img)
        if lines is None:
            return -1, 0

        img_w, img_h = img.shape[1], img.shape[0]
        img_hou = np.zeros((img_h, img_w), dtype=np.uint8)
        img_hou = Algo_Processor.draw_hough_lines(img_hou, lines)

        center_rect = self.detect_correct_mark(img_hou)

        throttle, steer = self.map_values(center_rect, img_hou)
        return throttle, steer
