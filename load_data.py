import pickle
import os
from keras import models
from tensorflow import keras
from keras import layers, optimizers as opt

DEF_MODEL_PATH = "./Model/"

def serialize_model(model, file_name, history_model, def_path = DEF_MODEL_PATH):
    # serialize model to JSON
    model_json = model.to_json()
    with open(def_path+file_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(def_path+file_name+".h5")
    with open(def_path+file_name, 'wb') as file_pi:
        pickle.dump(history_model, file_pi)
    print("Saved model to disk")

def deserialize_model(file_name, def_path = DEF_MODEL_PATH):
    # load json and create model
    def_path = os.path.abspath(def_path)+"\\"
    json_file = open(def_path+file_name+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(def_path+file_name+".h5")
    print("Loaded model from disk")
    loaded_model = compile_model(loaded_model)
    # history
    with open(def_path+file_name, "rb") as file_pi:
        history = pickle.load(file_pi)
    return loaded_model, history

def compile_model(model: keras.Sequential) -> keras.Sequential:
    optimizer = keras.optimizers.Adam(0.0003)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model