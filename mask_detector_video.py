from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import random
import shutil

filepath = './face_detector/'
face_classifier = './face_detector/haarcascade_frontalface_default.xml'
model_path = './model_test/mask_detector.h5'
MY_CONFIDENCE = 0.9
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

print('starting the final project')

src_cap = cv2.VideoCapture(-1)

while src_cap.isOpened():
    _, img = src_cap.read()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect MultiScale / faces
    faces = face_classifier.detectMultiScale(rgb, 1.3, 5)

    # Draw rectangles around each face
    for (x, y, w, h) in faces:
        # Save just the rectangle faces in SubRecFaces
        face_img = rgb[y:y + w, x:x + w]

        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (224, 224, 3))
        face_img = np.expand_dims(face_img, axis=0)

        pred = model_path.predict_on_batches(face_img)
        # print(pred)

        if pred[0][0] == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (0, 0, 255), -1)
            cv2.putText(img, 'NO MASK', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (0, 255, 0), -1)
            cv2.putText(img, 'MASK', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the image
        cv2.imshow('LIVE DETECTION', img)

        # if key 'q' is press then break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Stop video
src_cap.release()

# Close all started windows
cv2.destroyAllWindows()
