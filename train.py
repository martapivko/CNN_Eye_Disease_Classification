
import os
from shutil import copyfile, rmtree
from typing import Union, List, Tuple, Set, Dict
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array
from keras_preprocessing.image.image_data_generator import ImageDataGenerator
from keras_applications.vgg16 import VGG16
from keras_applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import SGD
from folder import KFolder

def train_model():

    train_image_directory = os.path.join(os.getcwd(), "dataset")

    val_image_directory = os.path.join(os.getcwd(), "dataset")

    test_image_directory = os.path.join(os.getcwd(), "dataset")



    X_train, y_train = get_data(image_path=train_image_directory)
    X_valid, y_valid = get_data(image_path=val_image_directory)
    X_test, y_test = get_data(image_path=test_image_directory)

    X,y = np.concatenate([X_train, X_valid], axis=0), np.concatenate([y_train, y_valid], axis=0)

    del X_train, y_train, X_valid, y_valid
    
    img_size = (256,256)
    num_classes = 1
    batch_size = 16
    epochs = 100
    optimizer = tf.keras.optimizers.Adam(0.0003)
    callbacks = [tf.keras.callbacks.ModelCheckpoint("./best_model.h5", save_best_only=True, verbose=1)]
    metrics = []
    
    kfolder = KFolder(k=5, x=X, y=y)

    for i, fold in enumerate(kfolder.folds):

        callbacks = [tf.keras.callbacks.ModelCheckpoint(f"./best_model_fold_{i}.h5", save_best_only=True, verbose=1)]
        model = get_model(img_size=img_size, num_classes=num_classes)
        model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
     

        X_train, y_train, X_valid, y_valid = fold

        model_history = model.fit_generator(generator=train_generator, steps_per_epoch=int(np.ceil(TRAIN_SIZE / 128)),
                              epochs=100, verbose=1, validation_data=validation_generator,
                              validation_steps=int(np.ceil(VALIDATION_SIZE / 128)))

        del model
        best_model = tf.keras.models.load_model(f"./best_model_fold_{i}.h5", custom_objects={"binary_crossentropy_plus_dice_loss":sm.losses.bce_dice_loss, "f1-score":sm.metrics.FScore()})

        evaluation = best_model.evaluate(X_test, y_test)

        with open(f"./results_fold_{i}.txt",'w') as f:
            f.write(f"test loss: {evaluation[0]}\n")
            f.write(f"accuracy: {evaluation[1]}\n")
            f.write(f"dice-score: {evaluation[2]}\n")
        print(100*'--')


if __name__=='__main__':
    train_model()
