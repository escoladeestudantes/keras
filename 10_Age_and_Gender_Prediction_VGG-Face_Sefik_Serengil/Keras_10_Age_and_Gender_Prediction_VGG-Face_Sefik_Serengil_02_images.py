#Usar
#python Keras_10_Age_and_Gender_Prediction_VGG-Face_Sefik_Serengil_02_images.py

#Documentation: https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/

import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt
from os import listdir
import glob
import os

### Keras 2.3.0 e TensorFlow 2.0
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
### Keras 2.3.0 e TensorFlow 2.0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    return model

def ageModel():
    model = loadVggFaceModel()
    
    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    
    age_model = Model(inputs=model.input, outputs=base_model_output)
    
    #you can find the pre-trained weights for age prediction here: https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing
    age_model.load_weights("age_model_weights.h5")
    
    return age_model

def genderModel():
    model = loadVggFaceModel()
    
    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)
    
    #you can find the pre-trained weights for gender prediction here: https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view?usp=sharing
    gender_model.load_weights("gender_model_weights.h5")
    
    return gender_model
    
age_model = ageModel()
gender_model = genderModel()

#age model has 101 outputs and its outputs will be multiplied by its index label. sum will be apparent age
output_indexes = np.array([i for i in range(0, 101)])

#------------------------

path = ''
for infile in glob.glob(os.path.join(path, '*.jpg')):
	img = cv2.imread(infile)
	faces = face_cascade.detectMultiScale(img, 1.13, 5)
	for (x,y,w,h) in faces:
		if w > 130: #ignore small faces
		    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		    try:
		        #age gender data set has 40% margin around the face. expand detected face.
		        margin = 30
		        margin_x = int((w * margin)/100)
		        margin_y = int((h * margin)/100)
		        detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
		
		    except:
		        print("detected face has no margin")
		    
		    try:
		        #vgg-face expects inputs (224, 224, 3)
		        detected_face = cv2.resize(detected_face, (224, 224))
		        
		        img_pixels = image.img_to_array(detected_face)
		        img_pixels = np.expand_dims(img_pixels, axis = 0)
		        img_pixels /= 255
		        
		        #find out age and gender
		        age_distributions = age_model.predict(img_pixels)
		        apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
		        
		        gender_distribution = gender_model.predict(img_pixels)[0]
		        gender_index = np.argmax(gender_distribution)
		        
		        if gender_index == 0: gender = "Female"
		        else: gender = "Male"
		        
		        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),5)
		        #labels for age and gender
		        cv2.putText(img, apparent_age, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 10)
		        cv2.putText(img, apparent_age, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
		        
		        cv2.putText(img, str(gender), (x, y+w), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 10)
		        cv2.putText(img, str(gender), (x, y+w), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
		        cv2.imshow("Detected face", detected_face)		        
		    except Exception as e:
		        print("exception",str(e))
		    
		    cv2.imshow("Output", img)
	cv2.waitKey(0)
cv2.destroyAllWindows()
