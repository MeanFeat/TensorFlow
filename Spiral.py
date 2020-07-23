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

global model
tf.random.set_seed(123)
np.random.seed(123)

def GetData():
    data = tools.read_csv('SpiralData.csv').values[:,:-1]
    labels = tools.read_csv('SpiralLables.csv').values
    data = np.concatenate([data,AddNoise(data)], axis=0)
    labels = np.concatenate([labels,labels], axis=0)
    return data, labels

def DoubleData(data):
    return np.concatenate((data, AddNoise(data)), axis=0)

def AddNoise(matrix):
    varience = np.average(matrix) * 0.01
    return matrix + (np.random.uniform(low= -varience, high= varience, size=(matrix.shape)))

regularizer = tf.keras.regularizers.l2(0.0002)
X, y = GetData()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, input_shape=(X.shape[1],), activation=tf.nn.tanh),
    tf.keras.layers.Dense(8, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1,activation=tf.nn.tanh)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.015),loss=tf.keras.losses.MSE)

start = time.time()
def Go():
    tb_callback, log_dir = tools.GetTensorboardCallback('Spiral-')
    model.fit(X,y, epochs=2000, batch_size =len(X), verbose=1, callbacks=[tb_callback])
    results = model.evaluate(X, y)
    tools.WriteJson(model, "weights", results = results)
    tools.LaunchTensorboard(log_dir)

def GoAnim():
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    px = np.arange(-2, 2, 0.04)
    py = np.arange(-2, 2, 0.04)
    pxx, pyy = np.meshgrid(px,py)
    pxx = pxx.reshape(10000,1)
    pyy = pyy.reshape(10000,1)
    P = np.concatenate([pxx,pyy], axis=1)
    P = np.concatenate([P,np.square(P)], axis=1)
    def Animate(i):
        pz = model.predict(P).reshape(100,100)
        plt.contourf(px,py,pz)
    animation = anim.FuncAnimation(fig, Animate)
    for i in range(0,100):
        model.fit(X,y, epochs=1000, batch_size = 5000, verbose=1)
        plt.pause(0.1)
        plt.show(block=False)  

Go()

