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

tf.random.set_seed(123)
np.random.seed(123)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

X = tools.read_csv('SpiralData.csv').values
y = tools.read_csv('SpiralLables.csv').values
X = X[:,:-1]

def AddNoise(matrix):
    varience = np.average(matrix) * 0.01
    return matrix + (np.random.uniform(low= -varience, high= varience, size=(matrix.shape)))

X = np.concatenate([X,AddNoise(X)], axis=0)
y = np.concatenate([y,y], axis=0)


global model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, input_shape=(4,), activation=tf.nn.tanh, use_bias=True, bias_initializer=init.Ones()),
    tf.keras.layers.Dense(8, activation=tf.nn.tanh, use_bias=True, bias_initializer=init.Ones()),
    tf.keras.layers.Dense(1,activation=tf.nn.tanh)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MSE)

tb_callback, log_dir = tools.GetTensorboardCallback('Spiral-')
model.fit(X,y, epochs=5000, batch_size=len(X), verbose=0, callbacks=[tb_callback])

model.evaluate(X, y)


tools.LaunchTensorboard(log_dir)


# px = np.arange(-2, 2, 0.04)
# py = np.arange(-2, 2, 0.04)
# pxx, pyy = np.meshgrid(px,py)
# pxx = pxx.reshape(10000,1)
# pyy = pyy.reshape(10000,1)
# P = np.concatenate([pxx,pyy], axis=1)
# P = np.concatenate([P,np.square(P)], axis=1)

# def Animate(i):
#     pz = model.predict(P).reshape(100,100)
#     plt.contourf(px,py,pz)

# animation = anim.FuncAnimation(fig, Animate, interval=500)
# for i in range(0,1000):
#     model.fit(X,y, epochs=500, batch_size=len(X), verbose=1)
#     plt.pause(0.1)
#     plt.show(block=False)