#Usar
#python 02_keras_vgg-face_face_x_video.py

#Sefik Ilkin Serengil -> you can find the documentation of this code from the following link: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

import numpy as np
import cv2

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

from os import listdir

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#-----------------------

color = (0,255,0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
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
    
    #you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
    from keras.models import model_from_json
    model.load_weights('vgg_face_weights.h5')
    
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    return vgg_face_descriptor

model = loadVggFaceModel()

#------------------------

#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "faces/"

employees = dict()

for file in listdir(employee_pictures):
    employee, extension = file.split(".")
    employees[employee] = model.predict(preprocess_image('faces/%s.jpg' % (employee)))[0,:]
    
print("employee representations retrieved successfully")

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

#------------------------

#cap = cv2.VideoCapture(0) #webcam
#cap = cv2.VideoCapture('AtentadoNapalm_Tokyo.mp4') #video
cap = cv2.VideoCapture('Gigante_no_Mic_Apendice.mp4') #video
ret=True
while(ret==True):
    ret, img = cap.read()
    if (ret==True):
        img = cv2.resize(img, (int(0.5*img.shape[1]), int(0.5*img.shape[0])), interpolation = cv2.INTER_AREA)
        #img = cv2.resize(img, (640, 360))
        faces = face_cascade.detectMultiScale(img, 1.2, 5)
        
        for (x,y,w,h) in faces:
            if w > 130: 
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4) #draw rectangle to main image
                
                detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
                
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                #img_pixels /= 255
                #employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                img_pixels /= 127.5
                img_pixels -= 1
                
                captured_representation = model.predict(img_pixels)[0,:]
                
                found = 0
                for i in employees:
                    employee_name = i
                    representation = employees[i]
                    similarity = findCosineSimilarity(representation, captured_representation)
                    print(similarity)
                    if(similarity < 0.30):
                        label_name = "%s (%s)" % (employee_name, str(round(similarity,2)))
                        cv2.putText(img, label_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 14)
                        cv2.putText(img, label_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)                    
                        found = 1
                        break
                        
                #connect face and text
                cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),color,3)
                cv2.line(img,(x+w,y-20),(x+w+10,y-20),color,3)
            
                if(found == 0): #if found image is not in employee database
                    cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
        cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break
    
#kill open cv things        
cap.release()
cv2.destroyAllWindows()
