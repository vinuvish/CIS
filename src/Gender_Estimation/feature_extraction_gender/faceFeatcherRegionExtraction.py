# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils

facial_features_cordinates = {}

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
  #  ("Mouth _Range", (48, 68)),
    ("TwoEyebrow_Range", (21, 23)),
    ("Right_Eyebrow_Range", (17, 22)),
    ("Left_Eyebrow_Range", (22, 27)),
    ("Upper_Lips_Range", (48, 55)),
    ("Lower_Lips_Range", (54, 61)),
    ("Jaw", (0, 17))
])


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())


def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    count = 0
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]

        pts = shape[j:k]
        pts2 = pts
        # print(pts[-1]
        regions = str(pts2[0, 0] - pts2[-1, 1])
        redion_in_pixel = regions[1:]
        facial_features_cordinates[name] = redion_in_pixel
        if name in "Left_Eyebrow_Range":
            facial_features_cordinates[name] = int(redion_in_pixel) +100
        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)


        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:

            hull = cv2.convexHull(pts)

            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
            cv2.putText(overlay, name + " : " + str(int(facial_features_cordinates.get(name))/50*10)[:5] +" mm", (10, 30 + count),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            count += 30
            print( name+" : " + str(facial_features_cordinates.get(name)))

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    # print(facial_features_cordinates)
    return output


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../../../models/shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread("../../../test_data/test_Images/example_test_images_for_gender/nandu.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
