#Usar
#python 00_crop_face.py

import cv2
import dlib

cnn_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

#imagem = cv2.imread('Gigante_no_Mic_01.jpg')
imagem = cv2.imread('Gigante_no_Mic_02.jpg')

faces = cnn_face_detector(imagem, 0)
for face in faces:
    face_detectada = imagem[face.rect.top():face.rect.bottom(), face.rect.left():face.rect.right()]
cv2.imshow('Face Detectada', face_detectada)
#cv2.imwrite('faces/Gigante_no_Mic1.jpg', face_detectada)
cv2.imwrite('faces/Gigante_no_Mic2.jpg', face_detectada)
cv2.waitKey(0)

