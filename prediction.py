#### sentiment classification prediction file

import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Model,Input
from tensorflow.keras.layers import *
from model import sentiment_model
import numpy as np


def predict(x):
    input_len = 170
    model = sentiment_model()
    from_disk = pickle.load(open("tv_layer.pkl", "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])
    
    pkl_file = open('LE.pkl', 'rb')
    le_departure = pickle.load(pkl_file) 
    pkl_file.close()

    # y = new_v(x)
    model.load_weights('sentiment_model.h5')
    test_sent = new_v(x)
    test_sent = tf.reshape(test_sent, shape = (1, input_len))
    y = np.argmax(model.predict(test_sent))

    return (le_departure.inverse_transform([y]))[0]