import airsim
import cv2
import numpy as np
import time
import asyncio
import os
import tempfile

from processing import process_img
from save_images import save_img


# async def drive():
def drive():
    tasks = []
    car_controls = airsim.CarControls()
    # go forward
    car_controls.throttle = 0.5
    car_controls.steering = 0
    client.setCarControls(car_controls)
    time.sleep(1)  # let car drive a bit
    print("Go Forward")

    idx = 0
    while client.getCarState().speed > 0:
        # get camera images from the car
        response = client.simGetImages(
            [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
        )[0]  # scene vision image in uncompressed RGB array

        # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))

        img1d = np.frombuffer(
            response.image_data_uint8, dtype=np.uint8
        )  # get numpy array
        img_rgb = img1d.reshape(
            response.height, response.width, 3
        )  # reshape array to 3 channel image array H X W X 3
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


        throttle, steer = process_img(
            img_gray,
            client.getCarState().speed,
            client.getCarState().kinematics_estimated.angular_velocity.z_val,
        )
        # task = asyncio.create_task(save_img(img_rgb.copy(), throttle, steer))
        # tasks.append(task)
        filename = os.path.join(tmp_dir, f"curved_{idx}_throttle_{throttle:.3f}_steer_{steer:.3f}")
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb)
        print(f"Throttle: {throttle:.2f}, Steer: {steer:.2f}")

        if throttle < 0:
            car_controls.brake = 1.0
            car_controls.throttle = 0
            car_controls.steering = 0
            client.setCarControls(car_controls)
            break
        else:
            car_controls.throttle = throttle
            car_controls.steering = steer
            client.setCarControls(car_controls)

        idx += 1
        # await asyncio.sleep(0.5)   # let car drive a bit

    # await asyncio.gather(*tasks)
    # restore to original state
    # client.reset()
    client.enableApiControl(False)


if __name__ == "__main__":
    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    print(f"API Control enabled: {client.isApiControlEnabled()}")

    # get state of the car
    car_state = client.getCarState()
    print(f"Speed {car_state.speed}, Gear {car_state.gear}")
    # print(f"Car state: {car_state}")

    tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
    print(f"Saving images to {tmp_dir}")
    try:
        os.makedirs(tmp_dir)
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    drive()
    # asyncio.run(drive()) 
