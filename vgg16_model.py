from keras import layers, models ,Input
from keras_preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import numpy as np
import matplotlib as plt


input_tensor = Input(shape = (240, 240, 3))
conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (240,240,3))

x = conv_base(input_tensor)
x = layers.Flatten()(x)
x = layers.Dense(1024,activation='relu')(x)
output_tensor = layers.Dense(1,activation='sigmoid')(x)
model_vgg16 = models.Model(input_tensor,output_tensor)

print(model_vgg16.summary())
