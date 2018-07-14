# USAGE
# python face_fetchers.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from PIL.Image import Image
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from PIL import ImageFilter , Image
from resizeimage import resizeimage

image_PIL = Image.open("../../../test_output/output_for_age/cnn_face_detection.png")
image_PIL = resizeimage.resize_width(image_PIL, 500,validate=False)

#Right EYE
croped_image = image_PIL.crop((110, 80, 250, 220))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=5))
image_PIL.paste(blurd_image, (110, 80, 250, 220))

#Left EYE
croped_image = image_PIL.crop((300, 80, 430, 220))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=5))
image_PIL.paste(blurd_image,(300, 80, 430, 220))

#Nose
croped_image = image_PIL.crop((220, 100, 320, 350))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=5))
image_PIL.paste(blurd_image, (220, 100, 320, 350))

#Mouth
croped_image = image_PIL.crop((160, 300, 380, 450))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=5))
image_PIL.paste(blurd_image,(160, 300, 380, 450))
image_PIL.show()



