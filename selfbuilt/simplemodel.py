from keras import layers, models ,Input
# necessary resources


# 自建 model
class Simple:
    @staticmethod
    def build(width, height, depth, classes):
        input_tensor = Input(shape = (width, height, depth))
        x = layers.Conv2D(32, (3, 3), activation = 'relu' )(input_tensor)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(64, (3, 3), activation = 'relu' )(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(128, (3, 3), activation = 'relu' )(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(128, (3, 3))(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation = 'relu' )(x)
        output_tensor = layers.Dense(classes ,activation = 'sigmoid' )(x)

        model = models.Model(input_tensor, output_tensor)
        
        return model