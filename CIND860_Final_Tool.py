import sys
import cv2
import numpy as np
import pandas as pd
import pydicom as dicom

import keras
from keras import models
from keras.models import model_from_json

from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.pixel_data_handlers.util import apply_voi
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.pixel_data_handlers.util import apply_windowing


image_base_dir= "/mnt/wd/CIND860/database/train_images"
dcm_name = sys.argv[1]




def lin_scale(img, new_w=1024, new_h=1024, colour='black'):
   aspect_ratio = img.shape[0]/img.shape[1]

   calc_h = int(new_w*(img.shape[0]/img.shape[1]))
   calc_w = int(new_h*(img.shape[1]/img.shape[0]))

   if colour == 'white':
      pad_colour = [255,255,255]
   else:
      pad_colour = [0,0,0]


   if calc_h > new_h:
      #calc height is larger than desired height.
      #therefore scale to x=calc_w, y=new_h
      #then pad right make x == new_w
      resize = cv2.resize(img, (calc_w,new_h))
      pad = new_w-calc_w
      resize2 = cv2.copyMakeBorder(resize,0,0,0,pad,cv2.BORDER_CONSTANT,None,pad_colour)

   else:
      #calc height is smaller than desired height.
      #therefore scale to x=new_w, y=calc_h
      #then pad top and bottom to make y == new_h
      resize = cv2.resize(img, (new_w,calc_h))

      pad = new_h-calc_h
      #Need to split pad up so we keep image centered top to bottom as cropped.
      #(ie pad half on bottom, half on top.
      pad1 = int(pad/2)
      pad2 = pad-pad1
      resize2 = cv2.copyMakeBorder(resize,pad1,pad2,0,0,cv2.BORDER_CONSTANT,None,pad_colour)

   return(resize2)



#For the automatic cropping of the image being performed below.
# set color bounds of white region
lower = (10,10,10) # lower bound for each channel(RGB)
upper = (255,255,255) # upper bound for each channel(RGB)

#Read dicom(mammogram/xray file in)
dcm = dicom.dcmread(dcm_name)

#Extract into pixels.
img = dcm.pixel_array.astype(np.float64)

#apply conversion to greyscale.
#only if the VOI LUT data is present in the dicom file.
if [0x0028, 0x1056] in dcm:
   img = apply_voi_lut(img, dcm, index=0)


#Rescale colour of each pixel into 8 bits
img = (np.maximum(img, 0) / img.max()) * 255.0

#Convert to uint8
img = img.astype(np.uint8)

#Convert from single channel colour into BGR format.
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


#Some dicom files were type "MONOCHROME1" and some "MONOCHROME2"
#The MONOCROME1 files are white based, whereas the MONOCHROME2 are the more customary black based.
#Invert colours if dicom was type "MONOCHROME1"
if dcm[0x0028, 0x0004].value == 'MONOCHROME1':
   img = 255-img


#this returns 0 or 255 for every pixel if it is in the range between lower and upper.
#for this purposes, we want to identify what part of the image is non black.
threshold = cv2.inRange(img, lower, upper)

#this finds contours within the image after the non black areas were identified.
contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

#find the main contour in the image, basically the area with the most infomation.
main_contour = max(contours, key=cv2.contourArea)

# get a rectangle bounding the main contour.
x,y,w,h = cv2.boundingRect(main_contour)

#add some padding to the values. (50 pixels to top and bottom, and 150 pixels to sides.)
#The min and max functions prevent going beyond the image's bounds.
x1 = max(0,x-150)
y1 = max(0,y-50)
x2 = min(img.shape[1],x+w+150)
y2 = min(img.shape[0],y+h+50)

# crop the image at the bounds
crop = img[y1:y2, x1:x2]

if x > 100:
   #breast image is left facing, therefore want to flip it on the horizontal axis.
   crop = cv2.flip(crop,1)

resize = lin_scale(crop,224,224,'black')

#Prep array for image.
X = np.zeros([0,224,224,3])

#convert images to floats, and rescale from 0-1
img = cv2.normalize(resize, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
X = np.append(X,[img],axis=0)


#Two methods to load a model:
# a) as one file, using keras.models.load_model()
# b) model_from_json() & model.load_weights()
#

#Load Model: (model and weights in one file)  (very slow  ~ 2.5 mins.)
#model_file = f"keras_cnn_model_RESNET50.keras"
#model = keras.models.load_model(model_file, compile=False)

#Load Model: (model and weigthts separately):  (significantly faster ~ 10 seconds) 
json_file = open(f"keras_cnn_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(f"keras_cnn_model.weights")


#Run image through CNN and obtain prediction.
Y_test = model.predict(X, verbose=0)
Y_test = np.round(Y_test).flatten() 
print(f"Predicted Y is {Y_test}")
