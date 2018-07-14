# import required packages
from collections import OrderedDict

import cv2
import dlib
import argparse
import time
from PIL.Image import Image
from PIL import ImageFilter , Image
from resizeimage import resizeimage
import imutils
import numpy as np

# handle command line arguments
from xlwt.Utils import cellrange_to_rowcol_pair

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default="../../../test_data/test_Images/example_test_images_for_age/wrinkles-old-man.jpg", help='path to image file')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("../../../models/shape_predictor_68_face_landmarks.dat")
# load input image
image = cv2.imread(args.image)
image2 = image
if image is None:
    print("Could not read input image")
    exit()

# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# initialize cnn based face detector with the weights
# cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

# apply face detection (hog)
# image = imutils.resize(image, width=500,height=500)
faces_hog = hog_face_detector(image, 1)

end = time.time()


# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    # image2 = image.clone();
    cv2.rectangle(image, (x, y - 25), (x + w, y + h + 20), (0, 255, 0), 2)
    image2 = image2[y:y + h, x:x + w]

start = time.time()

# apply face detection (cnn)
# faces_cnn = cnn_face_detector(image, 1)


# write at the top left corner of the image
# for color identification
img_height, img_width = image.shape[:2]
# cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.10,(0,255,0), 2)
#cv2.putText(image, "CNN", (img_width - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.10, (0, 0, 255), 2)

# display output image

cv2.imshow("face detection with dlib", image)


cv2.imshow("croped face", image2)

# save output image
image2 = cv2.resize(image2,(500,500))
cv2.imwrite("../../../test_output/output_for_age/cnn_face_detection.png", image2)

image_PIL = Image.open("../../../test_output/output_for_age/cnn_face_detection.png")
# image_PIL = resizeimage.resize_width(image_PIL, 500,validate=False)

#Right EYE
croped_image = image_PIL.crop((110, 80, 250, 220))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
image_PIL.paste(blurd_image, (110, 80, 250, 220))

#Left EYE
croped_image = image_PIL.crop((300, 80, 430, 220))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
image_PIL.paste(blurd_image,(300, 80, 430, 220))

#Nose
croped_image = image_PIL.crop((220, 100, 320, 350))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
image_PIL.paste(blurd_image, (220, 100, 320, 350))

#Mouth
croped_image = image_PIL.crop((160, 300, 380, 450))
blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
image_PIL.paste(blurd_image,(160, 300, 380, 450))
#image_PIL.show()

lap = cv2.Canny(np.array(image_PIL), 120, 80)
lap = cv2.resize(lap,(300,300))
cv2.imshow("Laplacian", lap)

cv2.waitKey(0)

# close all windows
cv2.destroyAllWindows()



