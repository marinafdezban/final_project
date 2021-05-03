# Face-Mask Detection

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/marinafernandezbanda/)


<p align="middle">
    <img src="./images_test/mask_moment.jpg" height=300 width=450>
    <img src="./images_test/prado_face_mask.jpeg" height=300 width=450>
    
### :woman_technologist: Introduction

In the COVID-19 crisis wearing masks is absolutely necessary for public health and in terms of controlling the spread of the pandemic. 
This project's aim is to develop a system that could detect masked and unmasked faces in images and real-time video. This can, for example, be used to alert people that do not wear a mask when using the public transport, airports or in a theatre.


### :raising_hand: Project Workflow 

Our pipeline consists of three steps:
  1. An AI model which detect all human faces in an image.
  2. An AI model which predict if that face wears mask/no_mask.
  3. The output is an annotated image with the prediction.
  
  
### 🎭 Model's performance

The face-mask model is trained with 900 images but in order to increase their volume it was used data augmentation and the weights of the MobileNetV2 model. More about this architecture can be found [here](https://arxiv.org/pdf/1801.04381.pdf). 

The overall performance of different metrics is shown below.
