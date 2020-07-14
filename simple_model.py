from keras import layers, models ,Input
from keras_preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import numpy as np
import matplotlib as plt
#necessary resources




# 自建 model
input_tensor = Input(shape = (240, 240, 3))
x = layers.Conv2D(32, (3, 3), activation = 'relu' )(input_tensor)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation = 'relu' )(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation = 'relu' )(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3))(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation = 'relu')(x)
output_tensor = layers.Dense(1 ,activation = 'sigmoid')(x)

model = models.Model(input_tensor, output_tensor)

print(model.summary())




