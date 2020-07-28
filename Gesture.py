from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from numpy import array
import tools
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib import style
from tensorflow import keras
from keras import initializers as init
from keras import layers
import time
import os

global model
tf.random.set_seed(123)
np.random.seed(123)

def GetData():
    data = tools.read_csv('data/GroupedDeltas.csv')
    labels = tools.read_csv('data/GroupedLabels.csv')
    return data, labels 

def BuildData():
    data = tools.read_csv('data/GestureDeltasTrain.csv').values
    labels = tools.read_csv('data/GestureLabelsTrain.csv').values
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas00.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels00.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas01.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels01.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas02.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels02.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas03.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels03.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas04.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels04.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas05.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels05.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas06.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels06.csv').values], axis=0)
    data = np.concatenate([data,tools.read_csv('data/CorrectionsDeltas07.csv').values], axis=0)
    labels = np.concatenate([labels, tools.read_csv('data/CorrectionsLabels07.csv').values], axis=0)
    data = np.concatenate([data,data*0.5], axis=0)
    labels = np.concatenate([labels,labels], axis = 0)
    data = np.concatenate([data,data*0.75], axis=0)
    labels = np.concatenate([labels,labels], axis = 0)
    data = np.concatenate([data,data*1.25], axis=0)
    labels = np.concatenate([labels,labels], axis = 0)
    tf.random.shuffle(data,seed = 999)
    tf.random.shuffle(labels,seed = 999)
    np.savetxt('data/GroupedDeltas.csv', data, delimiter=',')
    np.savetxt('data/GroupedLabels.csv', labels, delimiter=',')
    return data, labels 

def AddNoise(matrix):
    varience = np.average(matrix) * 0.001
    return matrix + (np.random.uniform(low= -varience, high= varience, size=(matrix.shape)))

regularizer = tf.keras.regularizers.l2(0.0001)
#X, y = GetData()
X, y = BuildData()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(X.shape[1],), activation=tf.nn.tanh, kernel_regularizer=regularizer),
    tf.keras.layers.Dense(50, activation=tf.nn.tanh, kernel_regularizer=regularizer),
    tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0025),loss=tf.keras.losses.MSE)

start = time.time()
def Go():
    tb_callback, log_dir = tools.GetTensorboardCallback('Gesture-')
    print(len(X))
    model.fit(X,y, epochs=10000, batch_size =len(X), verbose=1, callbacks=[tb_callback])

    testdata = tools.read_csv('data/GestureDeltas_test.csv').values
    testlabels = tools.read_csv('data/GestureLabels_test.csv').values
    #testdata = np.concatenate([testdata,np.square(testdata)], axis=1)
    results = model.evaluate(testdata, testlabels)
    tools.WriteJson(model, "Gesture-Weights")
    #tools.LaunchTensorboard(log_dir)

Go()

