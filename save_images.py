import tempfile
import os
import cv2

idx = 0


async def save_img(img_rgb, speed=5, angle=-10):
    tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
    # print(f"Saving images to {tmp_dir}")
    try:
        os.makedirs(tmp_dir)
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    filename = os.path.join(tmp_dir, f"angle_{angle}_speed_{speed}_{idx}")
    await cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

    idx += 1
