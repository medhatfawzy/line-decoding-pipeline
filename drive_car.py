import airsim
import cv2
import numpy as np
import os
import time
import tempfile

from processing import process_img

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print(f"API Control enabled: {client.isApiControlEnabled()}")
car_controls = airsim.CarControls()

tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
print(f"Saving images to {tmp_dir}")
try:
    os.makedirs(tmp_dir)
except OSError:
    if not os.path.isdir(tmp_dir):
        raise

# get state of the car
car_state = client.getCarState()
print(f"Speed {car_state.speed}, Gear {car_state.gear}")
print(f"Car state: {car_state}")

# go forward
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward")

time.sleep(0.5) 

for idx in range(90):

    # get camera images from the car
    responses = client.simGetImages([airsim.ImageRequest("front_center", 
                                                         airsim.ImageType.Scene, 
                                                         False, 
                                                         False)])  #scene vision image in uncompressed RGB array
    # print('Retrieved images: %d' % len(responses))
    
    for response_idx, response in enumerate(responses):
        # filename = os.path.join(tmp_dir, f"angle_-10_speed_5_{idx}")
        # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W X 3
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
        
        throttle, steer = process_img(img_gray)
        car_controls.throttle = throttle
        car_controls.steering = steer
        client.setCarControls(car_controls)
        print(f"Steer: {steer}, Throttle: {throttle}")

    
    time.sleep(0.5)   # let car drive a bit





#restore to original state
client.reset()

client.enableApiControl(False)
