import tensorflow as tf
# from tensorflow import ImageDataGenerator

def ToAugmantate(train_augmentate: bool = True):
    if train_augmentate == True:
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=0,
            zoom_range=0.0,
            horizontal_flip= True ,
            vertical_flip = False,
            rescale=None
        )
        return train_generator
    
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0,
        zoom_range=0.0,
        horizontal_flip= True ,
        vertical_flip = False,
        rescale=None
    )
    return test_generator