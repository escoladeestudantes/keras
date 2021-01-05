#Usar
#python keras_eye_blink_detector_MMOD_Eetveldt.py

#https://github.com/Guarouba/face_rec

import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
from eye_status import *
from tqdm import tqdm
from collections import defaultdict

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def draw_eyes(img_bgr, x_left,y_top,x_right,y_bottom, state):
    if (state == 'open'):
        color = (0,255,0)
    elif (state == 'closed'):
        color = (0,0,255)
    else:
        color = (255,0,0)
    cv2.rectangle(img_bgr,(x_left,y_top),(x_right,y_bottom),color,2)
    cv2.putText(img_bgr, state, (x_left,int(0.9*y_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img_bgr

def crop_eye(img_bgr, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    deltax = x2 - x1
    deltay = y2 - y1
    x_left = int(x1 - 0.3*deltax)
    y_top = int(y1 - 0.5*deltay)
    x_right = int(x2 + 0.3*deltax)
    y_bottom = int(y2 + 0.5*deltay)
    eye = img_bgr[y_top:y_bottom, x_left:x_right]
    return eye, x_left, y_top, x_right, y_bottom

def isBlinking(history, maxFrames):
    """ @history: A string containing the history of eyes status 
         where a '1' means that the eyes were closed and '0' open.
        @maxFrames: The maximal number of successive frames where an eye is closed """
    blink = False
    n = 0
    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        n = history.count(pattern)
        if n>0:
            blink = True
    return blink, n

model = load_model()
eyes_detected = defaultdict(str)
detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture('2.mp4')
#cap = cv2.VideoCapture('../FabioBrazza03.mp4')
ret=True

while(ret==True):
    ret, frame = cap.read()
    if(ret==True):
        #frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(frame, 0)
        idx = len(faces)
        i=0
        for face in faces:
            shapes = predictor(gray, face.rect)
            shapes = face_utils.shape_to_np(shapes)

            eye_status = '1' # we suppose the eyes are open
            eye_left, x1, y1, x2, y2 = crop_eye(frame, eye_points=shapes[36:42])
            pred = predict(eye_left, model)
            #print(pred)
            if(pred == 'closed'):
                eye_status = '0'
            frame = draw_eyes(frame, x1, y1, x2, y2, pred)
            eye_right, x1, y1, x2, y2 = crop_eye(frame, eye_points=shapes[42:48])
            pred = predict(eye_right, model)
            if(pred == 'closed'):
                eye_status = '0'
            frame = draw_eyes(frame, x1, y1, x2, y2, pred)
            eyes_detected[i] += eye_status
            historico = eyes_detected[i]
            blink, number = isBlinking(historico,3)
            if (blink == True):
                cv2.rectangle(frame, (face.rect.left(),face.rect.top()), (face.rect.right(),face.rect.bottom()), (255,0,0), 2)
                cv2.putText(frame, 'Piscada: {}'.format(number), (int(1.1*face.rect.left()), int(0.9*face.rect.bottom())), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

        cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
