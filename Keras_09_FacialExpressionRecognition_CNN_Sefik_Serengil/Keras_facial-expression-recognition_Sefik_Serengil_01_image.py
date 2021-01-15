#https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/
#https://sefiks.com/2018/01/10/real-time-facial-expression-recognition-on-streaming-data/

#Usar
#python Keras_facial-expression-recognition_Sefik_Serengil_01_image.py

import numpy as np
import cv2
from keras.preprocessing import image

#TF 2.0.0 e Keras 2.3.0
import tensorflow
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
#https://github.com/serengil/tensorflow-101/blob/master/model/facial_expression_model_structure.json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
#https://github.com/serengil/tensorflow-101/blob/master/model/facial_expression_model_weights.h5
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------

#emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotions = ('bravo', 'nojo', 'medo', 'feliz', 'triste', 'surpreso', 'neutro')

img = cv2.imread('ADL_DK_47.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.15, 5)

#print(faces) #locations of detected faces

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),5) #draw rectangle to main image
    
    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    
    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    
    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    
    predictions = model.predict(img_pixels) #store probabilities of 7 expressions
    
    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
    max_index = np.argmax(predictions[0])
    
    emotion = emotions[max_index]
    
    #write emotion text above rectangle
    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 10)
    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)
    
    #process on detected face end
    #-------------------------

cv2.imshow('img',img)
cv2.waitKey(0)
#kill open cv things        
cv2.destroyAllWindows()
