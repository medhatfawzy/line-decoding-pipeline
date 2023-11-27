import tempfile
import os
import cv2


def get_idx():
    idx = 0
    while True:
        yield idx
        idx += 1


async def save_img(img_rgb, throttle=5, steer=-10):
    tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_car")
    # print(f"Saving images to {tmp_dir}")
    try:
        os.makedirs(tmp_dir)
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise
    idx = next(get_idx())
    filename = os.path.join(tmp_dir, f"curved_{idx}_throttle_{throttle:.3f}_steer_{steer:.3f}")
    cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

    idx += 1
