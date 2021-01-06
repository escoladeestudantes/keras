#Usar
#python Regressor-Face_Pose.py

#https://github.com/arnaldog12/Deep-Learning/tree/master/problems/Regressor-Face%20Pose

import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%matplotlib inline

#=====================

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import math
#=====================

x, y = pkl.load(open('data/samples.pkl', 'rb'))

print(x.shape, y.shape)

roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

print(roll.min(), roll.max(), roll.mean(), roll.std())
print(pitch.min(), pitch.max(), pitch.mean(), pitch.std())
print(yaw.min(), yaw.max(), yaw.mean(), yaw.std())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_val = std.transform(x_val)
x_test = std.transform(x_test)

#=====================

def detect_face_points(image, rect, landmarks_detector):
    #dlib_points = landmarks_detector(image, rect[0]) #HOG
    dlib_points = landmarks_detector(image, rect[0].rect) #MMOD
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
        if i==33:
            xcenter, ycenter = x, y
    return face_points, xcenter, ycenter
        
def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)

def draw_orientation(img, roll, pitch, yaw, xc, yc):
    x_org = 150
    y_org = 100
    comprimento_linha = 80
    img = cv2.circle(img, (x_org, y_org), 4, (255,0,0),-1)
    img = cv2.circle(img, (xc, yc), 4, (255,0,0),-1)
    
    py_x = int(comprimento_linha * math.sin(math.radians(yaw)))
    py_y = int(comprimento_linha * math.sin(math.radians(pitch)))
    
    img = cv2.line(img, (x_org, y_org), (x_org + py_x, y_org - py_y), (0,0,255),2)
    
    img = cv2.line(img, (xc, yc), (xc + py_x, yc - py_y), (0,0,255),2)
    
    r_x = int(comprimento_linha * math.sin(math.radians(roll)))
    r_y = int(comprimento_linha * math.cos(math.radians(roll)))
    img = cv2.line(img, (x_org, y_org), (x_org + r_x, y_org - r_y), (255,0,0),2)
    img = cv2.line(img, (xc, yc), (xc + r_x, yc - r_y), (255,0,0),2)
    
    return img

#detector = dlib.get_frontal_face_detector() #HOG
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat") #MMOD
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

model = load_model('models/model.h5')
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Gigante_no_Mic_Apendice.mp4')
ret=True
stop=0
while (ret==True):
    if (stop == 0):
        ret, frame = cap.read()
        if (ret==True):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
            face_rect = detector(frame, 0)
            if len(face_rect) > 0:
                face_points, xc, yc = detect_face_points(frame, face_rect, predictor)
                for x, y in face_points:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                features = compute_features(face_points)
                features = std.transform(features)
                y_pred = model.predict(features)
                roll_pred, pitch_pred, yaw_pred = y_pred[0]
                frame = draw_orientation(frame, roll_pred, pitch_pred, yaw_pred, xc, yc)
                print(' Roll: {:.2f}°'.format(roll_pred))
                print('Pitch: {:.2f}°'.format(pitch_pred))
                print('  Yaw: {:.2f}°'.format(yaw_pred))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('result', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        stop = not(stop)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
