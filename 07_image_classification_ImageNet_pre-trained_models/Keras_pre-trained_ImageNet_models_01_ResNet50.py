#Usar
#python Keras_pre-trained_ImageNet_models_01_ResNet50.py

#https://keras.io/api/applications/
#https://www.tensorflow.org/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

import tensorflow
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

import time

model = ResNet50(weights='imagenet')

img_path = 'horse.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
start_time = time.time()
preds = model.predict(x)
print('Time: {}'.format(time.time() - start_time))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
result = decode_predictions(preds, top=3)[0]
print('Predicted:', result[0])
print('{}: {:.2f} %'.format(result[0][1], 100*result[0][2]))

