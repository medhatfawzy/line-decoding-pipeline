First make sure that these installations are installed before running the .py file 
!pip install fastseg
!pip install fastai --upgrade
!pip install -U albumentations[imgaug]
second make sure that the following imports are in the same enviroment as the model 
Numpy
Pandas
os
cv2
gc
matplotlib.pyplot as plt
MAKE SURE THAT THE IMPORTS IN THE " detection_functions_final_version_2.py " are imported as well.
to run the .py file > %run ".py file"
MAKE SURE THAT THE ENVIROMENT USES GPU FOR SERIALIZATION OF THE MODEL OTHERWISE IT WON'T WORK.
# Specify the image, model, and save paths
image_path = '/content/curved_32.png' # change this path to the feed of images from the simulator.
model_path = '/content/fastai_model_latest.pth' # Change path of the model to the path u saved the included model in.
save_path = '/content/result_image.png' # Change the path of the saved imaged to where the simulator reeds images to get instructions from.
 
# Run the lane detection pipeline and save the result
lane_detection_pipeline(image_path, model_path, save_path) # run this code line to get predictions from the model included.