#Usar
#python Keras_pre-trained_ImageNet_models_02_VGG-16.py

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

import tensorflow
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

import time

model = VGG16(weights='imagenet', include_top=True)

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


