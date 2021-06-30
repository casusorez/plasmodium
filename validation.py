
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn import metrics
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import models

mon_model = tf.keras.models.load_model('toto_model.h5')
img_path = 'results/boxes/combined_4045_2_297.png'
img = image.load_img(img_path, target_size=(50, 50))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)

print("prédiction brute :", mon_model.predict(img_tensor))
print("prédiction arrondie :", np.round(mon_model.predict(img_tensor)))

layer_outputs = [layer.output for layer in mon_model.layers[:12]] 
# on extrait les sorties de chaque couche de notre réseau
activation_model = models.Model(inputs=mon_model.input, outputs=layer_outputs) 

#toutes les transformations que subit notre image dans les couches
activations = activation_model.predict(img_tensor) 

#on teste si ça marche bien 
first_layer_activation = activations[3]
print(first_layer_activation.shape)

for i in range (first_layer_activation.shape[1]):
  plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')