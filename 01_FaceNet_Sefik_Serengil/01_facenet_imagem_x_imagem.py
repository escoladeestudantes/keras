#Usar
#python 01_facenet_imagem_x_imagem.py

import numpy as np

from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt
from keras.preprocessing import image

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#model = model_from_json(open("facenet_model.json", "r").read())

from inception_resnet_v1 import *
model = InceptionResNetV1()

#you can find the pre-trained weights at https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
model.load_weights('facenet_weights.h5')

#model.summary()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#metric = "euclidean" #euclidean or cosine
metric = "cosine"
threshold = 0
if metric == "euclidean":
    threshold = 0.35
elif metric == "cosine":
    threshold = 0.07

def verifyFace(img1, img2):
    #produce 128-dimensional representation
    img1_representation = model.predict(preprocess_image(img1))[0,:]
    img2_representation = model.predict(preprocess_image(img2))[0,:]
    
    if metric == "euclidean":
        img1_representation = l2_normalize(img1_representation)
        img2_representation = l2_normalize(img2_representation)

        euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
        print("euclidean distance (l2 norm): ",euclidean_distance)

        if euclidean_distance < threshold:
            print("verified... they are same person")
        else:
            print("unverified! they are not same person!")
            
    elif metric == "cosine":
        cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
        print("cosine similarity: ",cosine_similarity)

        if cosine_similarity < 0.14:
            print("verified... they are same person")
        else:
            print("unverified! they are not same person!")
    
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img(img1))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img(img2))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)
    print("-----------------------------------------")

verifyFace("faces/Gigante_no_Mic1.jpg", "faces/Gigante_no_Mic2.jpg")

