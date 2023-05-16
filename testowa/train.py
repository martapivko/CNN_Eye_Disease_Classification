import subprocess
from subprocess import STDOUT
import os
# print(subprocess.run(["apt-get", "update"],capture_output=True))
# print(subprocess.run(["apt-get", "install", "ffmpeg libsm6 libxext6 -y"],capture_output=True))
# check_call(['apt-get', 'install', '-y', 'ffmpeg libsm6 libxext6'],
#      stdout=open(os.devnull,'wb'), stderr=STDOUT)

proc = subprocess.Popen('apt-get install -y libgl1-mesa-glx', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=STDOUT, executable="/bin/bash") 
proc.wait()

from sklearn.metrics import confusion_matrix
from model import Net
# Import libs
import numpy as np
import glob2 as glob
import matplotlib.pyplot as plt
from sklearn import model_selection
import tensorflow as tf
import cv2

# Config


# Functions
def adjustImg(img):
    mean = np.mean(img)
    for row in img:
        for pixel in row:
            if pixel > mean:
                pixel = 255
    return img


def convert_imgs2arr(dir_name, dicom=False) -> list:
    arr: list = []
    dir = glob.glob(dir_name + '/*')
    for filename in dir:
        im = np.array(cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), (96, 96)))
        # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_TOZERO, 11, 2)
        im = adjustImg(im)
        arr.append(im)
    return arr



# Main script
normal_list = convert_imgs2arr('NORMAL')
pneumo_list = convert_imgs2arr('PNG', True)
all_imgs = np.concatenate((normal_list, pneumo_list), axis=0)
dataset = np.arange(0, len(all_imgs))
normal_labels = [0 for i in range(0, len(normal_list))]
pneumo_labels = [1 for i in range(0, len(pneumo_list))]
all_labels = np.concatenate((normal_labels, pneumo_labels), axis=0)

# im2 = PILimg.fromarray(normal_list[0])
# plt.pyplot.imshow(im2, cmap=plt.pyplot.get_cmap('gray'))
# plt.pyplot.show()
# im3 = PILimg.fromarray(pneumo_list[0])
# plt.pyplot.imshow(im3, cmap=plt.pyplot.get_cmap('gray'))
# plt.pyplot.show()
labels = np.arange(0, len(all_imgs))
for i in range(0, len(all_imgs)):
    dataset[i] = i
ind_train, ind_test = model_selection.train_test_split(dataset, test_size=.3, random_state=1)


def getImgsArrs(source):
    labels = []
    vals = []
    for i in source:
        labels.append(all_labels[i])
        vals.append(all_imgs[i])
    return np.array(vals), np.array(labels)

## Train ##
imgs_train, labels_train = getImgsArrs(ind_train)
## Test ##
imgs_test, labels_test = getImgsArrs(ind_test)
# network
# print(imgs_train[0].shape)

if __name__=="__main__":
    net = Net()
    nOfEpochs = 3

    net.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    #training our model
    our_model = net.model.fit(imgs_train, labels_train, epochs=nOfEpochs, validation_split=0.1, verbose=1)
    # eval
    eval = net.model.evaluate(imgs_test, labels_test)
    plt.plot(our_model.history['accuracy'])
    plt.plot(our_model.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.xlim(0, 3)
    plt.show()

    #macierz pomyłek
    predict = net.model.predict(imgs_test, steps=33)

    # from sklearn.metrics import confusion_matrix
    print(confusion_matrix(labels_test, predict.argmax(axis=1)))