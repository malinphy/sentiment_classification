#### sentiment classification model

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, Model,Input
from tensorflow.keras.layers import *


def sentiment_model():
    input_len = 170
    vocab_size = 45000
    embed_dim = 32 
    input_layer = Input(shape=(input_len,), name = 'input_layer')
    emb_layer = Embedding(vocab_size, embed_dim, name = 'embedding_layer')(input_layer)
    flat_layer = Flatten(name = 'Flatten_layer')(emb_layer)
    d1_layer = Dense(128,activation = 'relu',name = 'd1_layer')(flat_layer)
    d2_layer = Dense(64,activation = 'relu',name = 'd2_layer')(d1_layer)
    d3_layer = Dense(32,activation = 'relu',name = 'd3_layer')(d2_layer)
    final_layer = Dense(4,activation = 'softmax',name = 'final_layer')(d3_layer)

    return Model(inputs = input_layer, outputs = final_layer)