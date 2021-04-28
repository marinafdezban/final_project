from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from PIL import Image
import os, os.path

imgs = []
path = './images'
valid_images = [".jpg", ".gif", ".png", ".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path, f)))

filepath = './face_detector'
model_path = './model_test/mask_detector.h5'


def mask_image():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--confidence', type=float, default=0.9,
                    help='minimum probability to filter weak detections')
    args = vars(ap.parse_args())

    # load our serialized face detector model from disk
    print('[INFO] loading face detector model...')
    prototxtPath = os.path.sep.join([filepath, 'deploy.prototxt'])
    weightsPath = os.path.sep.join([filepath, 'res10_300x300_ssd_iter_140000.caffemodel'])
    face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print('[INFO] loading face mask detector model...')
    model = load_model(model_path)

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(imgs)
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print('[INFO] computing face detections...')
    face_model.setInput(blob)
    detections = face_model.forward()

    # detecting mask/without mask in every face
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= args['confidence']:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = image[startY:endY, startX:endX]

            if face != 0:
                face = cv2.resize(face, (160, 160))
                face = face[np.newaxis, ...]

                # predict mask/without mask with the model
                print('predicting the results')
                results = model.predict_on_batch(face)
                print(results)
                # print results
                label = 'Mask' if results[0][0] < 0 else 'No Mask'
                color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    cv2.imshow(image)
    cv2.waitKey(0)


if __name__ == "__main__":
    mask_image()
