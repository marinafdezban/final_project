import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from shutil import copyfile
from os import walk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = '/home/marina/Bootcamp/final_project/face_detector'
model_path = '/home/marina/Bootcamp/final_project/model_test/mask_detector.h5'
MY_CONFIDENCE = .9
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# Setting custom Page Title and Icon with changed layout and sidebar state
st.title('My first app')
st.beta_set_page_config(page_title='Face Mask Detector', page_icon='😷',
                        layout='centered', initial_sidebar_state='expanded')


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_images_with_faces():
    # face detector
    print("loading face detector model...")
    prototxtPath = os.path.sep.join([filepath, "deploy.prototxt"])
    weightsPath = os.path.sep.join([filepath, "res10_300x300_ssd_iter_140000.caffemodel"])
    face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

    # facemask detector model (trained in other notebook)
    print("loading face mask detector model...")
    model = load_model(model_path)

    # loading images
    mask_images = '/home/marina/Bootcamp/final_project/mask_detection/test/mask/'
    without_images = '/home/marina/Bootcamp/final_project/mask_detection/test/without_mask/'
    copy_images = '/home/marina/Bootcamp/final_project/mask_detection/mask_detection/copy/'
    f = []

    try:
        os.mkdir(copy_images)
    except OSError:
        print('File exists: continue')

    for (dirpath, dirnames, filenames) in walk(mask_images):
        for file in filenames:
            copyfile(mask_images + file, copy_images + 'mask_' + file)
            f.append(copy_images + 'mask_' + file)
        break

    for (dirpath, dirnames, filenames) in walk(without_images):
        for file in filenames:
            copyfile(without_images + file, copy_images + 'without_mask_' + file)
            f.append(copy_images + 'without_mask_' + file)
        break

    for img in f:
        image = cv2.imread(img)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # detecting faces in images
        print("computing face detections...")
        face_model.setInput(blob)
        detections = face_model.forward()

        # detecting mask/without mask in every face
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= MY_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = image[startY:endY, startX:endX]

                if len(face) != 0:
                    face = cv2.resize(face, IMG_SIZE)
                    face = face[np.newaxis, ...]

                    # predict mask/without mask with the model
                    results = model.predict_on_batch(face)
                    print(results)
                    # print results
                    label = "Mask" if results[0][0] < 0 else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    cv2.putText(image, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


get_images_with_faces()


def mask_detection() -> object:
    """

        :rtype: object
        """
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.sidebar.markdown("# Mask Detection on?")


mask_detection()