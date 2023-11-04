import airsim
import cv2
import numpy as np
import time
import asyncio

from processing import process_img
from save_images import save_img

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print(f"API Control enabled: {client.isApiControlEnabled()}")
car_controls = airsim.CarControls()


# get state of the car
car_state = client.getCarState()
print(f"Speed {car_state.speed}, Gear {car_state.gear}")
# print(f"Car state: {car_state}")

# go forward
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward")

time.sleep(1)


async def main():
    tasks = []
    while client.getCarState().speed > 0:
        # get camera images from the car
        responses = client.simGetImages(
            [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
        )  # scene vision image in uncompressed RGB array
        # print('Retrieved images: %d' % len(responses))

        for response in responses:
            # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))

            img1d = np.frombuffer(
                response.image_data_uint8, dtype=np.uint8
            )  # get numpy array
            img_rgb = img1d.reshape(
                response.height, response.width, 3
            )  # reshape array to 3 channel image array H X W X 3
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            # task = asyncio.create_task(save_img(img_rgb))
            # tasks.append(task)

            throttle, steer = process_img(img_gray)
            print(f"Throttle: {throttle}, Steer: {steer}")
            car_controls.throttle = throttle
            car_controls.steering = steer
            client.setCarControls(car_controls)

        # await asyncio.sleep(0.5)   # let car drive a bit

    await asyncio.gather(*tasks)
    # restore to original state
    client.reset()
    client.enableApiControl(False)


if __name__ == "__main__":
    asyncio.run(main())
