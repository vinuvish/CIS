import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
image_path = sys.argv[1]
# image_path = "../../../test_data/test_Images/example_test_images_for_gender/natural_test.jpg"

detection_model_path = '../../models/emotion-recoganitation_trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../../models/emotion-recoganitation_trained_models/emotion_models/fer2013_mini_XCEPTION.110-0.65.hdf5'

emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    try:

        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_classifier_pridict =emotion_classifier.predict(gray_face)
    emotion_label_arg = np.argmax(emotion_classifier_pridict)
    emotion_text = emotion_labels[emotion_label_arg]
    print(emotion_text)


    accracy_data = str(emotion_classifier_pridict).split()
    for i in accracy_data:
        print(i)


    color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../../../test_output/output_for_emotion/predicted_test_image.png', bgr_image)
image = cv2.imread('../../../test_output/output_for_emotion/predicted_test_image.png')
cv2.imshow('window_frame', image)
cv2.waitKey(0)
