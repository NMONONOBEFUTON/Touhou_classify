from keras import layers, models ,Input
from keras_preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import numpy as np
import matplotlib as plt

class Vgg16:
    @staticmethod
    def build(width, height, depth, classes):
        input_tensor = Input(shape = (width, height, depth))
        conv_base = VGG16(weights = 'imagenet',
            include_top = False,
            input_shape = (width, height, depth))

        x = conv_base(input_tensor)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        output_tensor = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(input_tensor, output_tensor)

        return model